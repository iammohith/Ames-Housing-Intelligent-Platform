"""
Query Classifier — Route queries to the right document collection.
"""

from __future__ import annotations

from typing import Literal

QueryIntent = Literal[
    "neighborhood",
    "model_performance",
    "data_quality",
    "anomaly",
    "feature",
    "temporal",
    "general",
]

INTENT_KEYWORDS = {
    "neighborhood": [
        "neighborhood",
        "area",
        "location",
        "where",
        "region",
        "nridg",
        "names",
        "meadow",
    ],
    "model_performance": [
        "r2",
        "r²",
        "rmse",
        "mae",
        "model",
        "accuracy",
        "score",
        "predict",
        "xgboost",
        "ridge",
    ],
    "data_quality": ["null", "missing", "clean", "quality", "impute", "data issue"],
    "anomaly": ["anomal", "outlier", "unusual", "flag", "isolation", "zscore"],
    "feature": ["feature", "importance", "shap", "correlation", "variable", "column"],
    "temporal": [
        "year",
        "month",
        "time",
        "trend",
        "season",
        "2006",
        "2007",
        "2008",
        "2009",
        "2010",
    ],
}


def classify_query(query: str) -> QueryIntent:
    """Classify query intent for better retrieval routing."""
    q_lower = query.lower()
    scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        scores[intent] = sum(1 for kw in keywords if kw in q_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"
