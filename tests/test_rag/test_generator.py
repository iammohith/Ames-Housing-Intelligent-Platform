"""Tests for RAG Generator."""

import pytest


def test_extractive_fallback():
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "dashboard"))
    from rag.generator import _extractive_fallback

    context = "The top feature is Overall Quality. House age also matters. Garage area is important."
    result = _extractive_fallback("What is the top feature?", context)
    assert "Overall Quality" in result or "feature" in result.lower()


def test_grounding_score():
    from rag.generator import compute_grounding_score

    score = compute_grounding_score(
        "Overall Quality is the top feature",
        "The top feature influencing house prices is Overall Quality",
    )
    assert score > 0.5

    zero_score = compute_grounding_score(
        "completely unrelated words xyz", "context about housing"
    )
    assert zero_score < 0.5


def test_grounding_score_empty():
    from rag.generator import compute_grounding_score

    assert compute_grounding_score("", "context") == 0.0
    assert compute_grounding_score("answer", "") == 0.0

def test_grounding_gates_ordering(monkeypatch):
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "dashboard"))
    
    from rag import generator
    
    # Mock extractive fallback
    def mock_fallback(q, c):
        return "[FALLBACK TRIGGERED]"
    monkeypatch.setattr(generator, "_extractive_fallback", mock_fallback)
    
    # Mock compute_grounding_score to return 0.2 (should trigger immediate fallback)
    monkeypatch.setattr(generator, "compute_grounding_score", lambda a, c: 0.2)
    
    # Mock the LLM so if it's called we'll know
    class MockModel:
        def generate(self, **kwargs):
            if kwargs.get("max_length") == 10:
                raise Exception("Judge model should not be called for score < 0.3")
            return [[1]]
            
    class MockTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1]}
        def decode(self, *args, **kwargs):
            return "Generated answer"
    
    def mock_load_model():
        return MockModel(), MockTokenizer()
    monkeypatch.setattr(generator, "_load_model", mock_load_model)
    
    long_context = "This is a long enough context string to bypass the twenty word limit check in the generator so it can proceed. " * 2
    
    # Ensure answer doesn't have hallucination markers to avoid failing Gate 3
    result = generator.generate_answer("question", long_context, chat_history="")
    assert result == "[FALLBACK TRIGGERED]"

def test_chat_history_injection(monkeypatch):
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "dashboard"))
    
    from rag import generator
    
    # Mock model and tokenizer
    class MockTokenizer:
        def __call__(self, text, **kwargs):
            # Verify chat history is in the prompt
            assert "[Conversation History]" in text
            assert "User: previous question" in text
            assert "Assistant: previous answer" in text
            return {"input_ids": [1]}
        
        def decode(self, *args, **kwargs):
            return "Generated answer"
            
    class MockModel:
        def generate(self, **kwargs):
            return [[1]]
            
    def mock_load_model():
        return MockModel(), MockTokenizer()
    monkeypatch.setattr(generator, "_load_model", mock_load_model)
    
    monkeypatch.setattr(generator, "compute_grounding_score", lambda a, c: 0.9)
    
    long_context = "This is a long enough context string to bypass the twenty word limit check in the generator so it can proceed. " * 2
    result = generator.generate_answer("question", long_context, chat_history="[Turn 1] User: previous question\nAssistant: previous answer")
    assert result == "Generated answer"
