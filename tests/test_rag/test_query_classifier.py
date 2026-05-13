"""Tests for RAG Query Classifier."""

import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "dashboard"))
from rag.query_classifier import classify_query


def test_classify_query_classes():
    # Test temporal
    intent, conf = classify_query("which year had the most home sales?")
    assert intent == "temporal"
    assert conf > 0.0

    # Test spatial (neighborhood)
    intent, conf = classify_query("neighborhoods in ames with high prices")
    assert intent == "neighborhood"
    assert conf > 0.0

    # Test anomaly
    intent, conf = classify_query("how many outliers were detected?")
    assert intent == "anomaly"
    assert conf > 0.0

    # Test model_performance
    intent, conf = classify_query("what is the r2 score of the xgboost model?")
    assert intent == "model_performance"
    assert conf > 0.0

    # Test feature
    intent, conf = classify_query("what are the top features influencing price?")
    assert intent == "feature"
    assert conf > 0.0

def test_classify_query_margin_of_dominance():
    # Test high signal query (one clear intent)
    intent1, conf1 = classify_query("rmse and r2 of the model")
    assert intent1 == "model_performance"
    
    # Test low signal query (no clear intent)
    intent2, conf2 = classify_query("tell me about the dataset")
    assert intent2 == "general"
    assert conf2 == 0.0
    
    # The margin of dominance confidence should be higher for the high signal query
    assert conf1 > conf2

def test_classify_query_empty_input():
    intent, conf = classify_query("")
    assert intent == "general"
    assert conf == 0.0
    
    intent, conf = classify_query(None)
    assert intent == "general"
    assert conf == 0.0
