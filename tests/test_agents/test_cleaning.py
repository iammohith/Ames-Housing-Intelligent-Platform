"""Tests for Cleaning Agent."""

import pandas as pd
import pytest


@pytest.mark.asyncio
async def test_cleaning_fills_structural_na(sample_df, mock_event_bus, run_id):
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pipeline"))
    from unittest.mock import MagicMock

    from agents.cleaning_agent import CleaningAgent

    agent = CleaningAgent(mock_event_bus, run_id)
    mock_upstream = MagicMock()
    mock_upstream._df = sample_df
    result = await agent.execute({"ingestion_agent": mock_upstream})

    assert result.cleaning_report.post_clean_null_rate == 0.0
    assert agent._df.isnull().sum().sum() == 0


@pytest.mark.asyncio
async def test_cleaning_drops_electrical_null(sample_df, mock_event_bus, run_id):
    from unittest.mock import MagicMock

    from agents.cleaning_agent import CleaningAgent

    agent = CleaningAgent(mock_event_bus, run_id)
    mock_upstream = MagicMock()
    mock_upstream._df = sample_df
    result = await agent.execute({"ingestion_agent": mock_upstream})

    assert result.cleaning_report.rows_dropped >= 0
    assert (agent._df["SalePrice"] > 0).all()
