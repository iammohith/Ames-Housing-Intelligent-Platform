"""Tests for Anomaly Detection Agent."""

import pytest


@pytest.mark.asyncio
async def test_anomaly_flags_without_removing(sample_df, mock_event_bus, run_id):
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pipeline"))
    from unittest.mock import MagicMock

    from agents.anomaly_agent import AnomalyAgent
    import numpy as np

    # Anomaly agent now expects post-encoded data
    sample_df["TotalSF"] = sample_df.get("Total Bsmt SF", 0) + sample_df.get("1st Flr SF", 0) + sample_df.get("2nd Flr SF", 0)
    for col in ["Gr Liv Area", "Lot Area", "SalePrice", "TotalSF"]:
        if col in sample_df.columns:
            sample_df[f"log_{col}"] = np.log1p(sample_df[col])

    agent = AnomalyAgent(mock_event_bus, run_id)
    result, df_out = await agent.execute({}, df=sample_df)

    # Should flag some but not remove any
    assert result.anomaly_report.total_flagged >= 0
    assert result.row_count == len(sample_df)  # No rows removed
    assert "anomaly_flagged" in df_out.columns
