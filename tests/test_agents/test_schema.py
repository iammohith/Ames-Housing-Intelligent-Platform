"""Tests for Schema Validation Agent."""

import pytest


@pytest.mark.asyncio
async def test_schema_validates_columns(sample_df, mock_event_bus, run_id):
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pipeline"))
    from unittest.mock import MagicMock

    from agents.schema_agent import SchemaAgent

    agent = SchemaAgent(mock_event_bus, run_id)
    result, df_out = await agent.execute({}, df=sample_df)

    assert result.schema_confidence_score > 0.5
    assert result.row_count == len(sample_df)
    assert len(result.null_rates) == len(sample_df.columns)


@pytest.mark.asyncio
async def test_schema_identifies_structural_na(sample_df, mock_event_bus, run_id):
    from agents.schema_agent import SchemaAgent

    agent = SchemaAgent(mock_event_bus, run_id)
    result, _ = await agent.execute({}, df=sample_df)

    # Alley, Pool QC, etc. should be structural NA candidates
    assert len(result.structural_na_candidates) > 0
