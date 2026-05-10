"""
Query Classifier — Route queries to the right document collection.
MAANG-level: Semantic + keyword hybrid classification with confidence scoring.
"""

from __future__ import annotations

import re
from typing import Dict, Literal, Tuple

QueryIntent = Literal[
    "neighborhood",
    "model_performance",
    "data_quality",
    "anomaly",
    "feature",
    "temporal",
    "general",
]

# Semantic intent definitions (not just keywords)
INTENT_PATTERNS = {
    "neighborhood": {
        "keywords": [
            "neighborhood", "area", "location", "where", "region", 
            "nridgheights", "nridgehts", "stonebrook", "stonebr",
            "northridge", "noridge", "names", "meadowv", "brddale",
            "idotrr", "price by neighborhood", "market segments"
        ],
        "patterns": [
            r"which.*neighborhood.*(?:best|most|popular|expensive|affordable)",
            r"neighborhood.*(?:comparison|stats|analysis)",
            r"(?:best|worst).*neighborhood",
            r"area.*(?:analysis|comparison|price)",
        ],
        "weight": 1.0,
    },
    "model_performance": {
        "keywords": [
            "r2", "r²", "rmse", "mae", "mape", "model", "accuracy", "score", 
            "predict", "xgboost", "ridge", "lightgbm", "performance", "evaluation",
            "train", "test", "validation", "metric", "error", "regression"
        ],
        "patterns": [
            r"model.*(?:performance|accuracy|evaluation|score)",
            r"how.*(?:accurate|good|well).*model",
            r"(?:xgboost|ridge|lightgbm).*(?:performs|score|metric)",
            r"rmse|r2|mae.*(?:test|validation|score)",
        ],
        "weight": 1.0,
    },
    "data_quality": {
        "keywords": [
            "null", "missing", "clean", "quality", "impute", "data issue",
            "incomplete", "valid", "validation", "integrity", "schema",
            "na", "nan", "null rate", "data cleaning"
        ],
        "patterns": [
            r"(?:data|column).*(?:quality|integrity|validation)",
            r"(?:null|missing|na).*(?:rate|count|handling)",
            r"(?:clean|impute|fix).*(?:data|values|nulls)",
        ],
        "weight": 1.0,
    },
    "anomaly": {
        "keywords": [
            "anomal", "outlier", "unusual", "flag", "isolation", "zscore",
            "anomalous", "detection", "outliers", "anomalies", "suspicious",
            "rare", "unexpected", "abnormal"
        ],
        "patterns": [
            r"(?:detect|find|identify).*(?:anomal|outlier|unusual)",
            r"anomaly.*(?:report|detection|count)",
            r"(?:isolation|zscore).*(?:forest|analysis|method)",
        ],
        "weight": 1.0,
    },
    "feature": {
        "keywords": [
            "feature", "importance", "shap", "correlation", "variable", "column",
            "engineered", "engineering", "top", "most important", "relevance",
            "influential", "contribution", "impact"
        ],
        "patterns": [
            r"(?:top|most|least).*(?:important|relevant).*(?:feature|variable)",
            r"feature.*(?:importance|ranking|correlation)",
            r"which.*(?:features|variables).*(?:matter|important|influence)",
            r"shap|feature.*importance.*report",
        ],
        "weight": 1.0,
    },
    "temporal": {
        "keywords": [
            "year", "month", "time", "trend", "season", "seasonal",
            "2006", "2007", "2008", "2009", "2010", "historical",
            "time series", "temporal", "over time", "forecast"
        ],
        "patterns": [
            r"(?:trend|temporal|over time).*(?:analysis|change|pattern)",
            r"(?:2006|2007|2008|2009|2010).*(?:price|trend|sales)",
            r"(?:seasonal|monthly|yearly).*(?:pattern|trend|variation)",
        ],
        "weight": 1.0,
    },
}


def classify_query(query: str) -> Tuple[QueryIntent, float]:
    """
    Classify query intent with confidence scoring (0.0-1.0).
    Returns (intent, confidence) tuple.
    
    High confidence (>0.7): Strong semantic signals
    Medium confidence (0.3-0.7): Mixed signals
    Low confidence (<0.3): Ambiguous/general
    """
    if not query or not query.strip():
        return "general", 0.0
    
    q_lower = query.lower()
    q_clean = re.sub(r'\W+', ' ', q_lower).strip()
    
    scores: Dict[str, float] = {intent: 0.0 for intent in INTENT_PATTERNS.keys()}
    
    # Keyword matching (count-based)
    for intent, config in INTENT_PATTERNS.items():
        keyword_score = 0
        for keyword in config["keywords"]:
            if keyword in q_lower:
                keyword_score += 1
        scores[intent] += keyword_score * 0.4  # 40% weight for keywords
    
    # Pattern matching (regex-based)
    for intent, config in INTENT_PATTERNS.items():
        pattern_score = 0
        for pattern in config["patterns"]:
            if re.search(pattern, q_lower, re.IGNORECASE):
                pattern_score += 1
        scores[intent] += pattern_score * 0.6  # 60% weight for patterns
    
    best_intent = max(scores, key=scores.get)
    max_score = scores[best_intent]
    
    if max_score == 0:
        return "general", 0.0
        
    total_possible = (len(INTENT_PATTERNS[best_intent]["keywords"]) * 0.4 + 
                      len(INTENT_PATTERNS[best_intent]["patterns"]) * 0.6)
    confidence = max_score / max(total_possible, 1.0)
    
    return best_intent, min(confidence, 1.0)
