"""Tests for ML Training Agent."""
import pytest
import numpy as np

@pytest.mark.asyncio
async def test_ml_trains_three_models(sample_df, mock_event_bus, run_id):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pipeline"))
    from agents.cleaning_agent import CleaningAgent
    from agents.feature_agent import FeatureAgent
    from agents.encoding_agent import EncodingAgent
    from agents.ml_agent import MLAgent
    from unittest.mock import MagicMock

    # Build pipeline
    mock_up = MagicMock(); mock_up._df = sample_df
    clean = CleaningAgent(mock_event_bus, run_id)
    await clean.execute({"ingestion_agent": mock_up})
    feat = FeatureAgent(mock_event_bus, run_id)
    await feat.execute({"cleaning_agent": clean})
    enc = EncodingAgent(mock_event_bus, run_id)
    await enc.execute({"feature_agent": feat})

    # Train
    ml = MLAgent(mock_event_bus, run_id)
    result = await ml.execute({"encoding_agent": enc})

    assert result.models_trained == 3
    assert result.best_model_name in ["ridge", "xgboost", "lightgbm"]
    assert result.best_test_r2 > 0  # Should have some predictive power
