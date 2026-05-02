"""
RAG Generator — flan-t5-small inference with answer validation and extractive fallback.
"""
from __future__ import annotations
import os

MODEL_CACHE = os.getenv("TRANSFORMERS_CACHE", "/app/model_cache")
_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is None:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        _tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small", cache_dir=MODEL_CACHE)
        _model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", cache_dir=MODEL_CACHE)
    return _model, _tokenizer


def generate_answer(question: str, context: str, max_length: int = 256) -> str:
    """Generate answer using flan-t5-small with context grounding."""
    if not context.strip():
        return "I don't have enough context to answer that question. Please run the pipeline first."

    model, tokenizer = _load_model()

    prompt = (
        f"Answer this question using only the context below.\n"
        f"Context: {context[:2000]}\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        **inputs, max_length=max_length, num_beams=4,
        early_stopping=True, no_repeat_ngram_size=3,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Validate answer length
    if len(answer.split()) < 15:
        # Extractive fallback
        fallback = _extractive_fallback(question, context)
        if len(fallback.split()) > len(answer.split()):
            return fallback

    return answer


def _extractive_fallback(question: str, context: str) -> str:
    """Return best matching sentence when flan-t5 response is too short."""
    sentences = [s.strip() for s in context.replace("\n", ". ").split(".") if s.strip()]
    q_words = set(question.lower().split())
    best, best_score = "", 0
    for s in sentences:
        score = sum(1 for w in q_words if w in s.lower())
        if score > best_score:
            best_score = score
            best = s
    return best if best else "Unable to find a relevant answer in the knowledge base."


def compute_grounding_score(answer: str, context: str) -> float:
    """Compute grounding score: fraction of answer words found in context."""
    if not answer or not context:
        return 0.0
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    if not answer_words:
        return 0.0
    return len(answer_words & context_words) / len(answer_words)
