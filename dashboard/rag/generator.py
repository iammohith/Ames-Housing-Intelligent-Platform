"""
RAG Generator — flan-t5-base inference with agentic self-reflection and memory.
"""

from __future__ import annotations

import os
import re

MODEL_CACHE = os.getenv("TRANSFORMERS_CACHE", "/app/model_cache")
_model = None
_tokenizer = None
_model_lock = None


def _get_model_lock():
    """Get or create the model loading lock (thread-safe)."""
    global _model_lock
    if _model_lock is None:
        import threading
        _model_lock = threading.Lock()
    return _model_lock


def _load_model():
    global _model, _tokenizer
    # Use lock to prevent race conditions when multiple requests try to load model simultaneously
    lock = _get_model_lock()
    with lock:
        # Double-check pattern: verify again after acquiring lock
        if _model is None:
            from transformers import T5ForConditionalGeneration, T5Tokenizer

            _tokenizer = T5Tokenizer.from_pretrained(
                "google/flan-t5-base", cache_dir=MODEL_CACHE
            )
            _model = T5ForConditionalGeneration.from_pretrained(
                "google/flan-t5-base", cache_dir=MODEL_CACHE
            )
    return _model, _tokenizer


def compute_grounding_score(answer: str, context: str) -> float:
    """
    Compute how well-grounded the answer is in the context.
    
    MAANG-level: Semantic grounding using TF-IDF-like overlap scoring.
    Returns 0.0-1.0: higher = more grounded in context.
    """
    if not answer or not context:
        return 0.0
    
    import re
    
    # Tokenize both texts
    context_words = set(re.findall(r'\b\w+\b', context.lower()))
    answer_words = re.findall(r'\b\w+\b', answer.lower())
    
    if not answer_words:
        return 0.0
    
    # Compute overlap (Jaccard similarity variant)
    # Only count "content words" (filter stopwords)
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'is', 'was', 'are', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might',
        'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
    }
    
    content_answer = [w for w in answer_words if w not in stopwords]
    content_context = context_words - stopwords
    
    if not content_answer:
        return 0.0
    
    # Compute overlap score
    matches = sum(1 for w in content_answer if w in content_context)
    score = matches / len(content_answer)
    
    return min(1.0, score)



def generate_answer(
    question: str, context: str, chat_history: str = "", max_length: int = 256
) -> str:
    """
    Generate answer using flan-t5-base with context grounding and self-reflection.
    MAANG-level: Strong hallucination prevention, confidence scoring, explicit uncertainty.
    """
    if not context or not context.strip():
        return "I don't have enough context to answer that question. Please run the pipeline first."

    model, tokenizer = _load_model()

    # Validate context has sufficient semantic content
    if len(context.split()) < 20:
        return "I don't have enough context to answer that question. Please run the pipeline first."
    
    # Strip citation/header markers BEFORE word budget truncation
    import re as _re
    context = _re.sub(r'\[Source:[^\]]*\]\n*', '', context)
    context = context.replace('[Live System State]\n', '')
    context = _re.sub(r'DOCUMENT:\s*\S+\s*', '', context).strip()
    
    # Smart context truncation (by words) to fit in T5 512 token limit while PRESERVING sentences
    sentences = [s.strip() for s in _re.split(r'(?<!\b\d)(?<=[.!?])\s+', context.replace('\n', ' ')) if s.strip()]
    
    truncated_context = ""
    word_count = 0
    for sentence in sentences:
        sentence_words = sentence.split()
        if word_count + len(sentence_words) <= 350:
            truncated_context += sentence + "\n"
            word_count += len(sentence_words)
        else:
            break
            
    if not truncated_context.strip():
        return "I don't have enough context to answer that question. Please run the pipeline first."

    # Standard strict extraction prompt works best for small models like flan-t5
    prompt = "Answer the question based ONLY on the provided facts.\n"
    if chat_history and chat_history.strip():
        prompt += f"[Conversation History]\n{chat_history}\n\n"
    prompt += "Facts:\n"
    for fact in truncated_context.split("\n"):
        if fact.strip():
            prompt += f"- {fact.strip()}\n"
    prompt += f"\nQuestion: {question}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # 1. Grounding score check (0.0 to 1.0, higher is better)
    grounding = compute_grounding_score(answer, truncated_context)

    # 2. Agentic Self-Reflection (LLM-as-a-judge) with HIGHER thresholds
    # STRICT hallucination prevention: only accept high-confidence answers
    
    # Gate 1: immediate fallback for structurally ungrounded answers (< 0.30)
    if grounding < 0.30:
        return _extractive_fallback(question, truncated_context)

    # Gate 2: LLM-as-a-judge for borderline answers (0.30 - 0.50)
    if grounding < 0.50:
        verification_prompt = (
            f"Context: {truncated_context}\n"
            f"Answer: {answer}\n"
            f"Question: {question}\n"
            f"Is this answer completely and accurately supported by the context? "
            f"Reply strictly YES or NO only."
        )
        v_inputs = tokenizer(
            verification_prompt, return_tensors="pt", max_length=512, truncation=True
        )
        v_outputs = model.generate(**v_inputs, max_length=10)
        verification = (
            tokenizer.decode(v_outputs[0], skip_special_tokens=True).strip().upper()
        )

        if "NO" in verification or "CANNOT" in verification:
            # Model rejected its own answer - use extractive fallback
            return _extractive_fallback(question, truncated_context)

    # 4. Check for hallucination markers
    hallucination_markers = [
        "i assume", "probably", "might be", "could be", "seems like",
        "likely", "perhaps", "apparently", "supposedly", "allegedly",
        "according to my knowledge", "based on general knowledge"
    ]
    if any(marker in answer.lower() for marker in hallucination_markers):
        # Model used uncertain language - try extractive fallback
        return _extractive_fallback(question, truncated_context)
    
    return answer


def _extractive_fallback(question: str, context: str) -> str:
    """Return best matching sentence using Semantic Cosine Similarity when LLM fails."""
    # Split by common sentence delimiters, but ignore periods after digits (e.g., "1. ")
    import re
    sentences = [s.strip() for s in re.split(r'(?<!\b\d)(?<=[.!?])\s+', context.replace('\n', ' ')) if s.strip()]
    
    if not sentences:
        return "I don't have enough information to answer that question."

    try:
        from rag.retriever import _get_embedding_fn
        import numpy as np
        
        # We reuse the same local embedding model used for retrieval
        emb_fn = _get_embedding_fn()
        
        # Embed question and sentences
        q_emb = np.array(emb_fn([question])[0])
        s_embs = np.array(emb_fn(sentences))
        
        # Calculate cosine similarity
        norms = np.linalg.norm(s_embs, axis=1) * np.linalg.norm(q_emb)
        norms[norms == 0] = 1e-10
        similarities = np.dot(s_embs, q_emb) / norms
        
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        # Strict semantic overlap threshold to avoid hallucination
        if best_score < 0.35:
            return "I don't have enough information to answer that question."
            
        return sentences[best_idx]
    except Exception:
        # Fallback to naive keyword match if embeddings fail to load
        q_words = set(re.findall(r"\w+", question.lower()))
        best, best_score = "", 0
        for s in sentences:
            s_words = re.findall(r"\w+", s.lower())
            score = sum(1 for w in q_words if w in s_words)
            if score > best_score:
                best_score = score
                best = s

        if best_score < 2:
            return "I don't have enough information to answer that question."
        return best
