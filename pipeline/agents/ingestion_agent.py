"""
Agent 1 — Ingestion Agent
Verify the raw file matches its contract before any byte is processed.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime

import chardet
import pandas as pd
from agents.base_agent import BaseAgent
from core.schemas import AgentStatus, IngestionInput, IngestionOutput


class IngestionError(Exception):
    pass


class IngestionAgent(BaseAgent[IngestionInput, IngestionOutput]):
    name = "ingestion_agent"
    version = "1.0.0"

    async def execute(self, input_data: IngestionInput) -> IngestionOutput:
        csv_path = input_data.csv_path

        # Read raw bytes for hash and encoding detection
        await self.emit(AgentStatus.PROGRESS, f"Reading {os.path.basename(csv_path)}")
        with open(csv_path, "rb") as f:
            raw_bytes = f.read()

        file_size = len(raw_bytes)

        # Detect encoding
        detection = chardet.detect(raw_bytes[:10000])
        encoding = detection.get("encoding", "utf-8") or "utf-8"
        await self.emit(AgentStatus.PROGRESS, f"Encoding detected: {encoding}")

        # SHA-256 hash before parsing
        dataset_hash = hashlib.sha256(raw_bytes).hexdigest()
        await self.emit(
            AgentStatus.PROGRESS,
            f"SHA-256: {dataset_hash[:12]}...",
        )

        # Check for rerun
        is_rerun = False
        force_rerun = os.getenv("FORCE_RERUN", "false").lower() == "true"
        if not force_rerun:
            try:
                is_rerun = await self._check_hash_exists(dataset_hash)
                if is_rerun:
                    await self.emit(
                        AgentStatus.PROGRESS,
                        f"SHA-256: {dataset_hash[:12]}... (cache hit — previously processed)",
                    )
            except Exception:
                pass  # DB not available, continue as new run

        # Parse CSV
        df = pd.read_csv(csv_path, encoding=encoding)
        row_count, col_count = df.shape

        await self.emit(
            AgentStatus.PROGRESS,
            f"Parsed {row_count:,} rows × {col_count} columns",
            rows_in=row_count,
        )

        # Assert shape
        if row_count != input_data.expected_rows:
            raise IngestionError(
                f"Row count mismatch: expected {input_data.expected_rows}, got {row_count}"
            )
        if col_count != input_data.expected_cols:
            raise IngestionError(
                f"Column count mismatch: expected {input_data.expected_cols}, got {col_count}"
            )

        # Store DataFrame in shared state
        self._df = df

        return IngestionOutput(
            row_count=row_count,
            col_count=col_count,
            dataset_hash=dataset_hash,
            is_rerun=is_rerun,
            file_size_bytes=file_size,
            encoding_detected=encoding,
            ingestion_ts=datetime.utcnow(),
            columns=list(df.columns),
        )

    async def _check_hash_exists(self, dataset_hash: str) -> bool:
        """Check if this dataset hash was successfully processed before."""
        db_url = os.getenv("DATABASE_URL_SYNC")
        if not db_url:
            return False
        try:
            import psycopg2

            conn = psycopg2.connect(db_url)
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM pipeline_runs WHERE dataset_hash = %s AND status = 'SUCCESS'",
                (dataset_hash,),
            )
            count = cur.fetchone()[0]
            conn.close()
            return count > 0
        except Exception:
            return False
