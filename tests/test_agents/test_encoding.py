"""Tests for Encoding Agent."""

import pytest


@pytest.mark.asyncio
async def test_encoding_produces_numeric(sample_df, mock_event_bus, run_id, tmp_path):
    import os
    import sys

    os.environ["ARTIFACTS_DIR"] = str(tmp_path)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pipeline"))
    from unittest.mock import MagicMock

    from agents.cleaning_agent import CleaningAgent
    from agents.encoding_agent import EncodingAgent
    from agents.feature_agent import FeatureAgent

    clean = CleaningAgent(mock_event_bus, run_id)
    _, df_clean = await clean.execute({}, df=sample_df)

    feat = FeatureAgent(mock_event_bus, run_id)
    _, df_feat = await feat.execute({}, df=df_clean)

    enc = EncodingAgent(mock_event_bus, run_id)
    result, df_enc = await enc.execute({}, df=df_feat)

    assert result.ordinal_cols_encoded > 0
    assert len(result.artifacts_saved) > 0
    assert result.row_count > 0
    assert df_enc.select_dtypes(include=['object']).columns.empty
