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
    """Startup and shutdown lifecycle with proper resource management."""
    import asyncio
    import structlog
    logger = structlog.get_logger()
    
    # Startup
    logger.info("🚀 Starting Ames Housing API...")
    cleanup_task = None
    try:
        # Validate environment and config (fail-fast)
        from core.startup import validate_environment
        validate_environment()
        
        # Verify event bus is ready
        event_bus_health = event_bus._history  # Simple health check
        logger.info("✓ Event bus initialized")
        
        # Warm up metrics
        pipeline_currently_running.set(0)
        logger.info("✓ Prometheus metrics initialized")
        
        # Schedule periodic event bus cleanup
        cleanup_task = asyncio.create_task(_cleanup_event_bus_periodically())
        
        # Log startup
        logger.info("✓ API startup complete | Ready to accept requests")
    except Exception as e:
        logger.error("Failed to initialize API resources", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Ames Housing API...")
    try:
        # Cancel cleanup task safely
        if cleanup_task is not None:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clear all event bus state for clean restart
        event_bus._history.clear()
        event_bus._connections.clear()
        event_bus._subscribers.clear()
        logger.info("✓ Event bus cleaned up")
        
        # Final metrics flush would happen here if using external sink
        logger.info("✓ API shutdown complete")
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))


async def _cleanup_event_bus_periodically():
    """Periodically clean up old run history from event bus to prevent memory leaks."""
    import asyncio
    import structlog
    logger = structlog.get_logger()
    
    while True:
        try:
            await asyncio.sleep(3600)  # Cleanup every hour
            # Keep only the last 100 runs in memory
            if len(event_bus._history) > 100:
                oldest_runs = sorted(event_bus._history.keys())[:len(event_bus._history) - 100]
                for run_id in oldest_runs:
                    event_bus.clear_run(run_id)
                logger.info(f"Cleaned up {len(oldest_runs)} old runs from event bus history")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Error during event bus cleanup", error=str(e))


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
    keepalive_task = None
    try:
        # Keepalive task to prevent connection timeout
        async def send_keepalive():
            while True:
                try:
                    await asyncio.sleep(25)
                    # Send proper JSON event (client expects this format)
                    ping_event = AgentEvent(
                        run_id=run_id,
                        agent="event_bus",
                        status=AgentStatus.PROGRESS,
                        message="[keepalive]",
                        timestamp=datetime.utcnow(),
                    )
                    await websocket.send_text(ping_event.model_dump_json())
                except Exception:
                    break
        
        keepalive_task = asyncio.create_task(send_keepalive())
        
        # Listen for client messages (optional pong)
        while True:
            try:
                data = await websocket.receive_text()
                # Client may send pong/ping for diagnostics; ignore
                if not data.startswith("{"):
                    continue
            except WebSocketDisconnect:
                break
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    finally:
        if keepalive_task:
            keepalive_task.cancel()
            try:
                await keepalive_task
            except asyncio.CancelledError:
                pass
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
