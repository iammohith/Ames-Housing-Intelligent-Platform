"""Tests for Feature Engineering Agent."""
import pytest

@pytest.mark.asyncio
async def test_features_creates_12(sample_df, mock_event_bus, run_id):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pipeline"))
    from agents.cleaning_agent import CleaningAgent
    from agents.feature_agent import FeatureAgent
    from unittest.mock import MagicMock

    # Clean first
    clean_agent = CleaningAgent(mock_event_bus, run_id)
    mock_up = MagicMock(); mock_up._df = sample_df
    await clean_agent.execute({"ingestion_agent": mock_up})

    # Feature engineering
    agent = FeatureAgent(mock_event_bus, run_id)
    result = await agent.execute({"cleaning_agent": clean_agent})

    assert result.features_created == 12
    assert "TotalSF" in agent._df.columns
    assert "HouseAge" in agent._df.columns
    assert "TotalBathrooms" in agent._df.columns
    assert (agent._df["HouseAge"] >= 0).all()
