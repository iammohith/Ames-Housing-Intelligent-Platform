"""
Agent 8 — Orchestration Agent
Post-pipeline tasks: knowledge base building and database persistence.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any, Dict

import pandas as pd
from agents.base_agent import BaseAgent
from core.metrics import (anomalies_detected_total, data_drift_score,
                          knowledge_base_chunks_total, model_mae, model_r2,
                          model_rmse, rows_processed_last_run)
from core.schemas import AgentStatus, PipelineResult


class OrchestrationAgent(BaseAgent):
    name = "orchestration_agent"
    version = "1.0.0"

    async def execute(self, input_data: Dict[str, Any], df: pd.DataFrame = None) -> tuple[PipelineResult, pd.DataFrame]:
        results = input_data
        await self.emit(
            AgentStatus.PROGRESS, "All agents complete | Building RAG knowledge base"
        )

        ml_result = results.get("ml_agent")
        best_model = (
            getattr(ml_result, "best_model_name", "unknown") if ml_result else "unknown"
        )
        best_r2 = getattr(ml_result, "best_test_r2", 0.0) if ml_result else 0.0
        best_rmse = getattr(ml_result, "best_test_rmse", 0.0) if ml_result else 0.0
        rows = getattr(ml_result, "row_count", 0) if ml_result else 0
        rows_processed_last_run.set(rows)

        feature_result = results.get("feature_agent")
        features_count = (
            getattr(feature_result, "total_columns", 0) if feature_result else 0
        )

        anomaly_result = results.get("anomaly_agent")
        anomalies_count = (
            getattr(anomaly_result.anomaly_report, "total_flagged", 0)
            if anomaly_result and hasattr(anomaly_result, "anomaly_report")
            else 0
        )
        anomalies_detected_total.set(anomalies_count)

        knowledge_chunks = 0
        try:
            from core.knowledge_builder import KnowledgeBuilder

            kb = KnowledgeBuilder()
            # Run knowledge builder in thread pool to avoid blocking event loop
            loop = asyncio.get_running_loop()
            knowledge_chunks = await loop.run_in_executor(
                None, kb.build, results, self.run_id
            )
            knowledge_base_chunks_total.set(knowledge_chunks)
            await self.emit(
                AgentStatus.PROGRESS,
                f"Knowledge base: {knowledge_chunks} chunks indexed",
            )
        except Exception as e:
            await self.emit(AgentStatus.WARNING, f"Knowledge base build skipped: {e}")

        # ML Metrics
        if ml_result and hasattr(ml_result, "model_results"):
            for mr in ml_result.model_results:
                model_rmse.labels(model_name=mr.model_name, split="val").set(
                    mr.val_metrics.rmse_dollars
                )
                model_rmse.labels(model_name=mr.model_name, split="test").set(
                    mr.test_metrics.rmse_dollars
                )
                model_r2.labels(model_name=mr.model_name, split="val").set(
                    mr.val_metrics.r2
                )
                model_r2.labels(model_name=mr.model_name, split="test").set(
                    mr.test_metrics.r2
                )
                model_mae.labels(model_name=mr.model_name, split="val").set(
                    mr.val_metrics.mae_dollars
                )
                model_mae.labels(model_name=mr.model_name, split="test").set(
                    mr.test_metrics.mae_dollars
                )

        # Schema Metrics (Data Drift)
        schema_result = results.get("schema_agent")
        if schema_result and hasattr(schema_result, "schema_report"):
            for col in schema_result.schema_report.columns:
                data_drift_score.labels(column_name=col.name).set(col.null_rate)

        try:
            await self._persist_run(results)
        except Exception as e:
            await self.emit(AgentStatus.WARNING, f"DB persistence skipped: {e}")

        output = PipelineResult(
            run_id=self.run_id,
            status="SUCCESS",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            agents_completed=list(results.keys()),
            best_model=best_model,
            best_r2=best_r2,
            best_rmse=best_rmse,
            rows_processed=rows,
            knowledge_chunks=knowledge_chunks,
            features_count=features_count,
            anomalies_count=anomalies_count,
        )
        return output, df

    async def _persist_run(self, results: dict):
        db_url = os.getenv("DATABASE_URL_SYNC")
        if not db_url:
            return
        try:
            import json

            import psycopg2

            conn = psycopg2.connect(db_url)
            cur = conn.cursor()
            ingestion = results.get("ingestion_agent")
            ml_result = results.get("ml_agent")
            dataset_hash = (
                getattr(ingestion, "dataset_hash", "unknown")
                if ingestion
                else "unknown"
            )
            rows = getattr(ml_result, "row_count", 0) if ml_result else 0
            cur.execute(
                "INSERT INTO pipeline_runs (run_id, dataset_hash, status, rows_out) VALUES (%s,%s,%s,%s) ON CONFLICT (run_id) DO UPDATE SET status='SUCCESS', completed_at=NOW()",
                (self.run_id, dataset_hash, "SUCCESS", rows),
            )

            # Persist model results
            if ml_result and hasattr(ml_result, "model_results"):
                for mr in ml_result.model_results:
                    cur.execute(
                        "INSERT INTO model_results (run_id,model_name,val_rmse,test_rmse,test_r2,test_mae,test_mape,is_best) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                        (
                            self.run_id,
                            mr.model_name,
                            mr.val_metrics.rmse_dollars,
                            mr.test_metrics.rmse_dollars,
                            mr.test_metrics.r2,
                            mr.test_metrics.mae_dollars,
                            mr.test_metrics.mape,
                            mr.is_best,
                        ),
                    )

            # Persist anomaly log
            anomaly_result = results.get("anomaly_agent")
            if anomaly_result and hasattr(anomaly_result, "anomaly_report"):
                for record in anomaly_result.anomaly_report.flagged_records:
                    cur.execute(
                        "INSERT INTO anomaly_log (run_id, pid, neighborhood, methods, isolation_score, severity, anomalous_features) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                        (
                            self.run_id,
                            record.pid,
                            record.neighborhood,
                            record.methods,
                            record.isolation_score,
                            record.overall_severity.value,
                            json.dumps(
                                {
                                    k: v.model_dump()
                                    for k, v in record.anomalous_features.items()
                                }
                            ),
                        ),
                    )

            # Persist schema history
            schema_result = results.get("schema_agent")
            if schema_result and hasattr(schema_result, "schema_report"):
                for col in schema_result.schema_report.columns:
                    cur.execute(
                        "INSERT INTO schema_history (run_id, column_name, data_type, null_rate, is_structural_na) VALUES (%s,%s,%s,%s,%s)",
                        (
                            self.run_id,
                            col.name,
                            col.data_type.value,
                            col.null_rate,
                            col.is_structural_na,
                        ),
                    )

            conn.commit()
            conn.close()
        except Exception as e:
            self.log.warning("DB persistence failed", error=str(e))
    
