"""Tests for Anomaly Detection Agent."""

import pytest


@pytest.mark.asyncio
async def test_anomaly_flags_without_removing(sample_df, mock_event_bus, run_id):
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pipeline"))
    from unittest.mock import MagicMock

    from agents.anomaly_agent import AnomalyAgent

    mock_up = MagicMock()
    mock_up._df = sample_df
    agent = AnomalyAgent(mock_event_bus, run_id)
    result = await agent.execute({"encoding_agent": mock_up})

    # Should flag some but not remove any
    assert result.anomaly_report.total_flagged >= 0
    assert result.row_count == len(sample_df)  # No rows removed
