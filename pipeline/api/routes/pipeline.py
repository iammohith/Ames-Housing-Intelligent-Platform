"""
Pipeline API Routes — trigger, status, cancel, history.
"""
from __future__ import annotations
import asyncio
import uuid
from datetime import datetime
from typing import Dict
from fastapi import APIRouter, Depends, BackgroundTasks
from core.event_bus import event_bus
from core.schemas import RunPipelineResponse, PipelineStatusResponse, AgentStatus
from core.metrics import pipeline_runs_total, pipeline_currently_running
from api.middleware import verify_api_key

router = APIRouter()

# In-memory pipeline state (in production, use Redis)
_running_pipelines: Dict[str, dict] = {}


async def _execute_pipeline(run_id: str):
    """Execute the full 8-agent pipeline."""
    from agents.ingestion_agent import IngestionAgent
    from agents.schema_agent import SchemaAgent
    from agents.cleaning_agent import CleaningAgent
    from agents.feature_agent import FeatureAgent
    from agents.encoding_agent import EncodingAgent
    from agents.anomaly_agent import AnomalyAgent
    from agents.ml_agent import MLAgent
    from agents.orchestration_agent import OrchestrationAgent
    from core.schemas import IngestionInput

    pipeline_currently_running.set(1)
    _running_pipelines[run_id] = {"status": "RUNNING", "started_at": datetime.utcnow()}

    try:
        agents_map = {}

        # Agent 1: Ingestion
        agent = IngestionAgent(event_bus, run_id)
        result = await agent.run(IngestionInput())
        agents_map["ingestion_agent"] = agent

        # Agent 2: Schema
        agent2 = SchemaAgent(event_bus, run_id)
        result2 = await agent2.run(agents_map)
        agents_map["schema_agent"] = agent2

        # Agent 3: Cleaning
        agent3 = CleaningAgent(event_bus, run_id)
        result3 = await agent3.run(agents_map)
        agents_map["cleaning_agent"] = agent3

        # Agent 4: Features
        agent4 = FeatureAgent(event_bus, run_id)
        result4 = await agent4.run(agents_map)
        agents_map["feature_agent"] = agent4

        # Agent 5: Encoding
        agent5 = EncodingAgent(event_bus, run_id)
        result5 = await agent5.run(agents_map)
        agents_map["encoding_agent"] = agent5

        # Agent 6 & 7: Anomaly + ML (parallel)
        agent6 = AnomalyAgent(event_bus, run_id)
        agent7 = MLAgent(event_bus, run_id)

        await event_bus.emit(
            __import__("core.schemas", fromlist=["AgentEvent"]).AgentEvent(
                run_id=run_id, agent="orchestration_agent",
                status=AgentStatus.PROGRESS,
                message="Agents anomaly_agent and ml_agent running in parallel",
                timestamp=datetime.utcnow(),
            )
        )

        result6, result7 = await asyncio.gather(
            agent6.run(agents_map),
            agent7.run(agents_map),
        )
        agents_map["anomaly_agent"] = agent6
        agents_map["ml_agent"] = agent7

        # Agent 8: Orchestration
        agent8 = OrchestrationAgent(event_bus, run_id)
        # Pass result objects instead of agent objects for orchestration
        results_dict = {
            "ingestion_agent": result,
            "schema_agent": result2,
            "cleaning_agent": result3,
            "feature_agent": result4,
            "encoding_agent": result5,
            "anomaly_agent": result6,
            "ml_agent": result7,
        }
        final = await agent8.run(results_dict)

        pipeline_runs_total.labels(status="success").inc()
        _running_pipelines[run_id]["status"] = "SUCCESS"
        _running_pipelines[run_id]["result"] = final

    except Exception as e:
        pipeline_runs_total.labels(status="failure").inc()
        _running_pipelines[run_id]["status"] = "FAILED"
        _running_pipelines[run_id]["error"] = str(e)
    finally:
        pipeline_currently_running.set(0)


@router.post("/run-pipeline", response_model=RunPipelineResponse)
async def run_pipeline(background_tasks: BackgroundTasks, api_key: str = Depends(verify_api_key)):
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
        import psycopg2
        import os
        conn = psycopg2.connect(os.getenv("DATABASE_URL_SYNC", ""))
        cur = conn.cursor()
        cur.execute("SELECT run_id, status, started_at FROM pipeline_runs ORDER BY started_at DESC LIMIT 20")
        rows = cur.fetchall()
        conn.close()
        runs = []
        for r in rows:
            runs.append({
                "run_id": r[0],
                "status": r[1],
                "started_at": r[2].isoformat() if r[2] else None,
            })
        return {"runs": runs}
    except Exception as e:
        # Fallback to memory if DB fails
        runs = []
        for rid, data in _running_pipelines.items():
            runs.append({
                "run_id": rid,
                "status": data.get("status"),
                "started_at": data.get("started_at", "").isoformat() if data.get("started_at") else None,
            })
        return {"runs": runs}
