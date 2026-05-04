"""
RAG Generator — flan-t5-base inference with agentic self-reflection and memory.
"""

from __future__ import annotations

import os
import re

MODEL_CACHE = os.getenv("TRANSFORMERS_CACHE", "/app/model_cache")
_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is None:
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        _tokenizer = T5Tokenizer.from_pretrained(
            "google/flan-t5-base", cache_dir=MODEL_CACHE
        )
        _model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-base", cache_dir=MODEL_CACHE
        )
    return _model, _tokenizer


def generate_answer(
    question: str, context: str, chat_history: str = "", max_length: int = 256
) -> str:
    """Generate answer using flan-t5-base with context grounding and self-reflection."""
    if not context.strip():
        return "I don't have enough context to answer that question. Please run the pipeline first."

    model, tokenizer = _load_model()

    # Smart context truncation (by words) to fit in T5 512 token limit
    context_words = context.split()
    truncated_context = " ".join(context_words[:350])

    # Build prompt with chat history and instruction for citations
    prompt = "Answer this question using only the context below. Include [Source: Title] citations if possible.\n"
    if chat_history:
        prompt += f"Recent Chat History:\n{chat_history}\n\n"
    prompt += f"Context: {truncated_context}\nQuestion: {question}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # 1. Grounding score check
    grounding = compute_grounding_score(answer, truncated_context)

    # 2. Agentic Self-Reflection (LLM-as-a-judge)
    # If the grounding score is borderline, we ask the model to double check its own work.
    if 0.2 < grounding < 0.7:
        verification_prompt = (
            f"Context: {truncated_context}\n"
            f"Answer: {answer}\n"
            f"Is this answer completely factually supported by the context? Reply strictly YES or NO."
        )
        v_inputs = tokenizer(
            verification_prompt, return_tensors="pt", max_length=512, truncation=True
        )
        v_outputs = model.generate(**v_inputs, max_length=10)
        verification = (
            tokenizer.decode(v_outputs[0], skip_special_tokens=True).strip().upper()
        )

        if "NO" in verification:
            # Model rejected its own answer
            return _extractive_fallback(question, truncated_context)

    elif grounding <= 0.2:
        # Heavily ungrounded, skip reflection and fallback
        return _extractive_fallback(question, truncated_context)

    # Validate answer length
    if len(answer.split()) < 15:
        fallback = _extractive_fallback(question, truncated_context)
        if (
            fallback != "I don't have enough information to answer that question."
            and len(fallback.split()) > len(answer.split())
        ):
            return fallback

    return answer


def _extractive_fallback(question: str, context: str) -> str:
    """Return best matching sentence when flan-t5 response is ungrounded or rejected."""
    # Split by common sentence delimiters, preserving [Source: ...] if possible
    sentences = [s.strip() for s in context.replace("\n", ". ").split(".") if s.strip()]
    q_words = set(re.findall(r"\w+", question.lower()))
    best, best_score = "", 0
    for s in sentences:
        s_words = re.findall(r"\w+", s.lower())
        score = sum(1 for w in q_words if w in s_words)
        if score > best_score:
            best_score = score
            best = s

    # Strict overlap threshold to avoid hallucination
    if best_score < 2:  # Must match at least 2 keywords
        return "I don't have enough information to answer that question."

    return best


def compute_grounding_score(answer: str, context: str) -> float:
    """Compute grounding score: fraction of answer words found in context."""
    if not answer or not context:
        return 0.0
    # exclude stop words and short words
    answer_words = set(w for w in re.findall(r"\w+", answer.lower()) if len(w) > 3)
    context_words = set(w for w in re.findall(r"\w+", context.lower()))

    if not answer_words:
        return 1.0  # If answer is just small words, assume grounded

    return len(answer_words & context_words) / len(answer_words)
