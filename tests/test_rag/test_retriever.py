"""Tests for RAG Retriever."""

import pytest


def test_bm25_search():
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "dashboard"))
    from rag.retriever import _bm25_search

    chunks = [
        "The neighborhood NAmes has a median price of $145,000",
        "The neighborhood NAmes has a median price of $145,000 approximately",
        "XGBoost achieved an R² of 0.921 on the test set",
    ]
    unique = _bm25_search("XGBoost", chunks, top_n=1)
    assert len(unique) == 1
    assert "XGBoost" in unique[0]


def test_fallback_context(tmp_path):
    from rag.retriever import _fallback_context

    # Should return empty when no artifacts exist
    result = _fallback_context("test query")
    assert isinstance(result, str)
