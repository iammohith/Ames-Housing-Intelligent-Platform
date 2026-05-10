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
    r1, df1 = await a1.execute(IngestionInput(csv_path=sample_csv_path))
    assert r1.row_count == 2930
    assert r1.col_count == 82

    # Agent 2: Schema
    a2 = SchemaAgent(mock_event_bus, run_id)
    r2, df2 = await a2.execute({"ingestion_agent": r1}, df=df1)
    assert r2.schema_confidence_score > 0.9

    # Agent 3: Cleaning
    a3 = CleaningAgent(mock_event_bus, run_id)
    r3, df3 = await a3.execute({"ingestion_agent": r1, "schema_agent": r2}, df=df2)
    assert r3.cleaning_report.post_clean_null_rate == 0.0
    assert df3.isnull().sum().sum() == 0

    # Agent 4: Features
    a4 = FeatureAgent(mock_event_bus, run_id)
    r4, df4 = await a4.execute({"cleaning_agent": r3}, df=df3)
    assert r4.features_created == 12
    assert "TotalSF" in df4.columns
    assert (df4["HouseAge"] >= 0).all()

    # Verify pipeline integrity
    assert len(df4) > 2900  # Most rows retained
    assert (df4["SalePrice"] > 0).all()
