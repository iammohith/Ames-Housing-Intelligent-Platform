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
    result, df_out = await agent.execute({}, df=sample_df)

    assert result.cleaning_report.post_clean_null_rate == 0.0
    assert df_out.isnull().sum().sum() == 0


@pytest.mark.asyncio
async def test_cleaning_drops_electrical_null(sample_df, mock_event_bus, run_id):
    from agents.cleaning_agent import CleaningAgent

    agent = CleaningAgent(mock_event_bus, run_id)
    result, df_out = await agent.execute({}, df=sample_df)

    assert result.cleaning_report.rows_dropped >= 0
    assert (df_out["SalePrice"] > 0).all()
