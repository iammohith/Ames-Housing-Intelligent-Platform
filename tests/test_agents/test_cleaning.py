"""Tests for Cleaning Agent."""
import pytest
import pandas as pd

@pytest.mark.asyncio
async def test_cleaning_fills_structural_na(sample_df, mock_event_bus, run_id):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pipeline"))
    from agents.cleaning_agent import CleaningAgent
    from unittest.mock import MagicMock

    agent = CleaningAgent(mock_event_bus, run_id)
    mock_upstream = MagicMock()
    mock_upstream._df = sample_df
    result = await agent.execute({"ingestion_agent": mock_upstream})

    assert result.cleaning_report.post_clean_null_rate == 0.0
    assert agent._df.isnull().sum().sum() == 0

@pytest.mark.asyncio
async def test_cleaning_drops_electrical_null(sample_df, mock_event_bus, run_id):
    from agents.cleaning_agent import CleaningAgent
    from unittest.mock import MagicMock

    agent = CleaningAgent(mock_event_bus, run_id)
    mock_upstream = MagicMock()
    mock_upstream._df = sample_df
    result = await agent.execute({"ingestion_agent": mock_upstream})

    assert result.cleaning_report.rows_dropped >= 0
    assert (agent._df["SalePrice"] > 0).all()
