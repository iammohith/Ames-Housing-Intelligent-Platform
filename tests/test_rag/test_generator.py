"""Tests for RAG Generator."""
import pytest


def test_extractive_fallback():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "dashboard"))
    from rag.generator import _extractive_fallback

    context = "The top feature is Overall Quality. House age also matters. Garage area is important."
    result = _extractive_fallback("What is the top feature?", context)
    assert "Overall Quality" in result or "feature" in result.lower()


def test_grounding_score():
    from rag.generator import compute_grounding_score

    score = compute_grounding_score(
        "Overall Quality is the top feature",
        "The top feature influencing house prices is Overall Quality"
    )
    assert score > 0.5

    zero_score = compute_grounding_score("completely unrelated words xyz", "context about housing")
    assert zero_score < 0.5


def test_grounding_score_empty():
    from rag.generator import compute_grounding_score
    assert compute_grounding_score("", "context") == 0.0
    assert compute_grounding_score("answer", "") == 0.0
