"""
FastAPI Application — Main entrypoint with WebSocket hub, SSE, and routes.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from api.routes import analytics, pipeline, predict, rag
from core.event_bus import event_bus
from core.metrics import (api_request_duration_seconds,
                          pipeline_currently_running, pipeline_runs_total)
from core.schemas import AgentEvent, AgentStatus, PipelineResult
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sse_starlette.sse import EventSourceResponse
from starlette.responses import Response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    yield


app = FastAPI(
    title="Ames Housing Intelligence Platform",
    description="Production-grade ML pipeline API with real-time monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include route modules
app.include_router(pipeline.router, prefix="/api", tags=["Pipeline"])
app.include_router(predict.router, prefix="/api", tags=["Prediction"])
app.include_router(analytics.router, prefix="/api", tags=["Analytics"])
app.include_router(rag.router, prefix="/api", tags=["RAG"])


# ── Request timing middleware ────────────────────────────────────────────
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    api_request_duration_seconds.labels(
        endpoint=request.url.path,
        method=request.method,
        status_code=str(response.status_code),
    ).observe(duration)
    return response


# ── WebSocket endpoint ───────────────────────────────────────────────────
@app.websocket("/ws/pipeline/{run_id}")
async def pipeline_websocket(websocket: WebSocket, run_id: str):
    await event_bus.connect(run_id, websocket)
    try:
        while True:
            await asyncio.sleep(30)
            await websocket.send_text('{"type":"ping"}')
    except WebSocketDisconnect:
        await event_bus.disconnect(run_id, websocket)
    except Exception:
        await event_bus.disconnect(run_id, websocket)


# ── SSE fallback ─────────────────────────────────────────────────────────
@app.get("/api/pipeline/{run_id}/events")
async def pipeline_events_sse(run_id: str, request: Request):
    async def event_generator():
        async for event in event_bus.subscribe(run_id):
            if await request.is_disconnected():
                break
            yield {"data": event.model_dump_json()}

    return EventSourceResponse(event_generator())


# ── Prometheus metrics ───────────────────────────────────────────────────
@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── Health check ─────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
