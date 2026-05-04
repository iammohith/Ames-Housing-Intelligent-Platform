"""
Pipeline API Routes — trigger, status, cancel, history.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Dict

from api.middleware import verify_api_key
from core.event_bus import event_bus
from core.metrics import pipeline_currently_running, pipeline_runs_total
from core.schemas import (AgentStatus, PipelineStatusResponse,
                          RunPipelineResponse)
from fastapi import APIRouter, BackgroundTasks, Depends

router = APIRouter()

# In-memory pipeline state (in production, use Redis)
_running_pipelines: Dict[str, dict] = {}


async def _execute_pipeline(run_id: str):
    """Execute the full 8-agent pipeline."""
    from agents.anomaly_agent import AnomalyAgent
    from agents.cleaning_agent import CleaningAgent
    from agents.encoding_agent import EncodingAgent
    from agents.feature_agent import FeatureAgent
    from agents.ingestion_agent import IngestionAgent
    from agents.ml_agent import MLAgent
    from agents.orchestration_agent import OrchestrationAgent
    from agents.schema_agent import SchemaAgent
    from core.schemas import IngestionInput

    pipeline_currently_running.set(1)
    _running_pipelines[run_id] = {"status": "RUNNING", "started_at": datetime.utcnow()}

    try:
        from core.dag import DAGOrchestrator

        agents_map = {
            "ingestion_agent": IngestionAgent(event_bus, run_id),
            "schema_agent": SchemaAgent(event_bus, run_id),
            "cleaning_agent": CleaningAgent(event_bus, run_id),
            "feature_agent": FeatureAgent(event_bus, run_id),
            "encoding_agent": EncodingAgent(event_bus, run_id),
            "anomaly_agent": AnomalyAgent(event_bus, run_id),
            "ml_agent": MLAgent(event_bus, run_id),
            "orchestration_agent": OrchestrationAgent(event_bus, run_id),
        }

        orchestrator = DAGOrchestrator(agents_map, event_bus)
        final = await orchestrator.run_pipeline(run_id)

        if final.status == "SUCCESS":
            pipeline_runs_total.labels(status="success").inc()
            _running_pipelines[run_id]["status"] = "SUCCESS"
        else:
            pipeline_runs_total.labels(status="failure").inc()
            _running_pipelines[run_id]["status"] = "FAILED"

        _running_pipelines[run_id]["result"] = final

    except Exception as e:
        pipeline_runs_total.labels(status="failure").inc()
        _running_pipelines[run_id]["status"] = "FAILED"
        _running_pipelines[run_id]["error"] = str(e)
    finally:
        pipeline_currently_running.set(0)


@router.post("/run-pipeline", response_model=RunPipelineResponse)
async def run_pipeline(
    background_tasks: BackgroundTasks, api_key: str = Depends(verify_api_key)
):
    run_id = str(uuid.uuid4())[:12]
    background_tasks.add_task(_execute_pipeline, run_id)
    return RunPipelineResponse(run_id=run_id, message="Pipeline started")


@router.get("/status/{run_id}", response_model=PipelineStatusResponse)
async def get_status(run_id: str):
    pipeline = _running_pipelines.get(run_id, {})
    result = pipeline.get("result")
    metrics = {}
    if result:
        metrics = {
            "rows_processed": result.rows_processed,
            "features_count": result.features_count,
            "anomalies_count": result.anomalies_count,
            "best_rmse": result.best_rmse,
            "best_r2": result.best_r2,
            "knowledge_chunks": result.knowledge_chunks,
        }
    return PipelineStatusResponse(
        run_id=run_id,
        status=pipeline.get("status", "UNKNOWN"),
        progress_pct=event_bus.get_progress(run_id),
        agents=event_bus.get_agent_status(run_id),
        started_at=pipeline.get("started_at"),
        metrics=metrics,
    )


@router.delete("/run/{run_id}")
async def cancel_pipeline(run_id: str, api_key: str = Depends(verify_api_key)):
    if run_id in _running_pipelines:
        _running_pipelines[run_id]["status"] = "CANCELLED"
        return {"message": f"Pipeline {run_id} cancelled"}
    return {"message": f"Pipeline {run_id} not found"}


@router.get("/pipeline-runs")
async def get_pipeline_runs():
    try:
        import os

        import psycopg2

        conn = psycopg2.connect(os.getenv("DATABASE_URL_SYNC", ""))
        cur = conn.cursor()
        cur.execute(
            "SELECT run_id, status, started_at FROM pipeline_runs ORDER BY started_at DESC LIMIT 20"
        )
        rows = cur.fetchall()
        conn.close()
        runs = []
        for r in rows:
            runs.append(
                {
                    "run_id": r[0],
                    "status": r[1],
                    "started_at": r[2].isoformat() if r[2] else None,
                }
            )
        return {"runs": runs}
    except Exception as e:
        # Fallback to memory if DB fails
        runs = []
        for rid, data in _running_pipelines.items():
            runs.append(
                {
                    "run_id": rid,
                    "status": data.get("status"),
                    "started_at": (
                        data.get("started_at", "").isoformat()
                        if data.get("started_at")
                        else None
                    ),
                }
            )
        return {"runs": runs}
