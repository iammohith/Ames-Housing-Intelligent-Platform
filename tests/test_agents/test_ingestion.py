"""Tests for Ingestion Agent."""

import hashlib

import pytest


@pytest.mark.asyncio
async def test_ingestion_reads_csv(sample_csv_path, mock_event_bus, run_id):
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pipeline"))
    from agents.ingestion_agent import IngestionAgent
    from core.schemas import IngestionInput

    agent = IngestionAgent(mock_event_bus, run_id)
    result, df = await agent.execute(IngestionInput(csv_path=sample_csv_path))

    assert result.row_count == 2930
    assert result.col_count == 82
    assert len(result.dataset_hash) == 64  # SHA-256 hex
    assert result.file_size_bytes > 0
    assert result.encoding_detected is not None
    assert df is not None


@pytest.mark.asyncio
async def test_ingestion_hash_deterministic(sample_csv_path, mock_event_bus, run_id):
    from agents.ingestion_agent import IngestionAgent
    from core.schemas import IngestionInput

    agent1 = IngestionAgent(mock_event_bus, run_id)
    r1, _ = await agent1.execute(IngestionInput(csv_path=sample_csv_path))

    agent2 = IngestionAgent(mock_event_bus, "run-2")
    r2, _ = await agent2.execute(IngestionInput(csv_path=sample_csv_path))

    assert r1.dataset_hash == r2.dataset_hash


@pytest.mark.asyncio
async def test_ingestion_rejects_wrong_shape(tmp_path, mock_event_bus, run_id):
    import pandas as pd
    from agents.ingestion_agent import IngestionAgent, IngestionError
    from core.schemas import IngestionInput

    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(bad_csv, index=False)

    agent = IngestionAgent(mock_event_bus, run_id)
    with pytest.raises(IngestionError):
        await agent.execute(IngestionInput(csv_path=str(bad_csv)))
