"""Tests for RAG Retriever."""
import pytest


def test_deduplication():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "dashboard"))
    from rag.retriever import _deduplicate

    chunks = [
        "The neighborhood NAmes has a median price of $145,000",
        "The neighborhood NAmes has a median price of $145,000 approximately",
        "XGBoost achieved an R² of 0.921 on the test set",
    ]
    unique = _deduplicate(chunks, threshold=0.8)
    assert len(unique) <= len(chunks)
    assert len(unique) >= 2  # At least the two distinct topics


def test_fallback_context(tmp_path):
    from rag.retriever import _fallback_context
    # Should return empty when no artifacts exist
    result = _fallback_context("test query")
    assert isinstance(result, str)
