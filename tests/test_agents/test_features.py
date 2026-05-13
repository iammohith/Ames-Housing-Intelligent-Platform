"""Tests for Feature Engineering Agent."""

import pytest


@pytest.mark.asyncio
async def test_features_creates_12(sample_df, mock_event_bus, run_id):
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pipeline"))
    from unittest.mock import MagicMock

    from agents.cleaning_agent import CleaningAgent
    from agents.feature_agent import FeatureAgent

    # Clean first
    clean_agent = CleaningAgent(mock_event_bus, run_id)
    _, df_clean = await clean_agent.execute({}, df=sample_df)

    # Feature engineering
    agent = FeatureAgent(mock_event_bus, run_id)
    result, df_out = await agent.execute({}, df=df_clean)

    assert result.features_created == 12
    assert "TotalSF" in df_out.columns
    assert "HouseAge" in df_out.columns
    assert "TotalBathrooms" in df_out.columns
    assert (df_out["HouseAge"] >= 0).all()
