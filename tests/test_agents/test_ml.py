"""Tests for ML Training Agent."""

import numpy as np
import pytest


@pytest.mark.asyncio
async def test_ml_trains_three_models(sample_df, mock_event_bus, run_id):
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pipeline"))
    from unittest.mock import MagicMock

    from agents.cleaning_agent import CleaningAgent
    from agents.encoding_agent import EncodingAgent
    from agents.feature_agent import FeatureAgent
    from agents.ml_agent import MLAgent

    # Build pipeline
    clean = CleaningAgent(mock_event_bus, run_id)
    _, df_clean = await clean.execute({}, df=sample_df)
    
    feat = FeatureAgent(mock_event_bus, run_id)
    _, df_feat = await feat.execute({}, df=df_clean)
    
    enc = EncodingAgent(mock_event_bus, run_id)
    _, df_enc = await enc.execute({}, df=df_feat)

    # Train
    ml = MLAgent(mock_event_bus, run_id)
    result, _ = await ml.execute({}, df=df_enc)

    assert result.models_trained == 3
    assert result.best_model_name in ["ridge", "xgboost", "lightgbm"]
    assert result.best_test_r2 > 0  # Should have some predictive power
