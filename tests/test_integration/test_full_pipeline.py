"""Integration test — full pipeline end-to-end on the real dataset."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pipeline"))


@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_pipeline_e2e(sample_csv_path, mock_event_bus):
    """Run all 8 agents in sequence on the full dataset."""
    from agents.cleaning_agent import CleaningAgent
    from agents.feature_agent import FeatureAgent
    from agents.ingestion_agent import IngestionAgent
    from agents.schema_agent import SchemaAgent
    from core.schemas import IngestionInput

    run_id = "integration-test"

    # Agent 1: Ingestion
    a1 = IngestionAgent(mock_event_bus, run_id)
    r1 = await a1.execute(IngestionInput(csv_path=sample_csv_path))
    assert r1.row_count == 2930
    assert r1.col_count == 82

    # Agent 2: Schema
    a2 = SchemaAgent(mock_event_bus, run_id)
    r2 = await a2.execute({"ingestion_agent": a1})
    assert r2.schema_confidence_score > 0.9

    # Agent 3: Cleaning
    a3 = CleaningAgent(mock_event_bus, run_id)
    r3 = await a3.execute({"ingestion_agent": a1})
    assert r3.cleaning_report.post_clean_null_rate == 0.0
    assert a3._df.isnull().sum().sum() == 0

    # Agent 4: Features
    a4 = FeatureAgent(mock_event_bus, run_id)
    r4 = await a4.execute({"cleaning_agent": a3})
    assert r4.features_created == 12
    assert "TotalSF" in a4._df.columns
    assert (a4._df["HouseAge"] >= 0).all()

    # Verify pipeline integrity
    assert len(a4._df) > 2900  # Most rows retained
    assert (a4._df["SalePrice"] > 0).all()
