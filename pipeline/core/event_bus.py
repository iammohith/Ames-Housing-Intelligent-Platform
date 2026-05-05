"""
EventBus — Central real-time event broadcasting system.
Agents emit events here; the hub broadcasts to all connected WebSocket clients.
Supports event history replay for late joiners (handles page refresh mid-pipeline).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from core.schemas import AgentEvent, AgentStatus
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class EventBus:
    """
    Central event bus for real-time pipeline event broadcasting.

    - Agents call `emit()` during execution
    - WebSocket clients receive events in real-time
    - Late joiners get full event history replay
    - SSE subscribers get events via async generator
    """

    def __init__(self, max_history_per_run: int = 500, max_total_runs: int = 50):
        self._connections: Dict[str, List[WebSocket]] = defaultdict(list)
        self._history: Dict[str, List[AgentEvent]] = defaultdict(list)
        self._subscribers: Dict[str, List[asyncio.Queue]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self.max_history_per_run = max_history_per_run
        self.max_total_runs = max_total_runs

    async def connect(self, run_id: str, websocket: WebSocket) -> None:
        """Accept a WebSocket connection and replay all past events for this run."""
        await websocket.accept()
        async with self._lock:
            self._connections[run_id].append(websocket)

        # Replay history for late joiners
        for event in self._history.get(run_id, []):
            try:
                await websocket.send_text(event.model_dump_json())
            except Exception:
                break

    async def disconnect(self, run_id: str, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            if run_id in self._connections:
                try:
                    self._connections[run_id].remove(websocket)
                except ValueError:
                    pass

    async def emit(self, event: AgentEvent) -> None:
        """
        Broadcast an event to all connected WebSocket clients and SSE subscribers.
        Called by agents during execution. Maintains bounded history.
        """
        run_id = event.run_id

        # Store in history with FIFO eviction per run
        self._history[run_id].append(event)
        if len(self._history[run_id]) > self.max_history_per_run:
            self._history[run_id].pop(0)
        
        # Trim old runs if too many accumulated
        if len(self._history) > self.max_total_runs:
            oldest_run = min(self._history.keys(), key=lambda r: self._history[r][0].timestamp if self._history[r] else datetime.utcnow())
            del self._history[oldest_run]

        event_json = event.model_dump_json()

        # Broadcast to WebSocket clients
        dead_connections: List[WebSocket] = []
        for ws in self._connections.get(run_id, []):
            try:
                await ws.send_text(event_json)
            except Exception:
                dead_connections.append(ws)

        # Clean up dead connections
        for ws in dead_connections:
            try:
                self._connections[run_id].remove(ws)
            except ValueError:
                pass

        # Notify SSE subscribers
        dead_queues: List[asyncio.Queue] = []
        for queue in self._subscribers.get(run_id, []):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                dead_queues.append(queue)

        for q in dead_queues:
            try:
                self._subscribers[run_id].remove(q)
            except ValueError:
                pass

        logger.info(f"[{run_id}] {event.agent} → {event.status.value}: {event.message}")

    async def subscribe(self, run_id: str) -> AsyncGenerator[AgentEvent, None]:
        """
        Async generator for SSE fallback.
        Yields events as they are emitted for a given run_id.
        Closes stream when orchestration_agent completes OR when all agents have tried and failed.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subscribers[run_id].append(queue)

        # Replay history first
        for event in self._history.get(run_id, []):
            yield event

        # Track which agents have completed
        completed_agents = set()
        failed_agents = set()
        all_agents = {
            "ingestion_agent",
            "schema_agent",
            "cleaning_agent",
            "feature_agent",
            "encoding_agent",
            "anomaly_agent",
            "ml_agent",
            "orchestration_agent",
        }

        try:
            while True:
                event = await queue.get()
                yield event
                # Track completion events
                if event.status == AgentStatus.SUCCESS:
                    completed_agents.add(event.agent)
                elif event.status == AgentStatus.FAILED:
                    failed_agents.add(event.agent)
                
                # Stop when orchestration_agent completes (success or failure)
                # OR when we've seen failures that cascade (e.g., critical agent fails)
                if event.agent == "orchestration_agent" and event.status in (
                    AgentStatus.SUCCESS,
                    AgentStatus.FAILED,
                ):
                    break
        finally:
            try:
                self._subscribers[run_id].remove(queue)
            except ValueError:
                pass

    def clear_run(self, run_id: str) -> None:
        """Clear all history for a specific run (for cleanup)."""
        if run_id in self._history:
            del self._history[run_id]
        if run_id in self._connections:
            del self._connections[run_id]
        if run_id in self._subscribers:
            del self._subscribers[run_id]

    def get_history(self, run_id: str) -> List[AgentEvent]:
        """Get all events for a given run."""
        return list(self._history.get(run_id, []))

    def get_agent_status(self, run_id: str) -> Dict[str, str]:
        """Get the latest status for each agent in a run."""
        statuses: Dict[str, str] = {}
        for event in self._history.get(run_id, []):
            statuses[event.agent] = event.status.value
        return statuses

    def get_progress(self, run_id: str) -> float:
        """Calculate overall pipeline progress (0.0 to 100.0).
        
        Counts both SUCCESS and FAILED agents as completed for progress tracking.
        Progress reaches 100% when all agents have been attempted (completed or failed).
        """
        all_agents = [
            "ingestion_agent",
            "schema_agent",
            "cleaning_agent",
            "feature_agent",
            "encoding_agent",
            "anomaly_agent",
            "ml_agent",
            "orchestration_agent",
        ]
        completed = 0
        statuses = self.get_agent_status(run_id)
        for agent in all_agents:
            agent_status = statuses.get(agent, "")
            # Count both success and failed as "done" for progress purposes
            if agent_status in ("SUCCESS", "FAILED"):
                completed += 1
        return (completed / len(all_agents)) * 100.0

    def clear_run(self, run_id: str) -> None:
        """Clear all data for a completed/cancelled run with proper locking."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._async_clear_run(run_id))
        except RuntimeError:
            # No event loop in current thread, use sync version
            self._history.pop(run_id, None)
            self._connections.pop(run_id, None)
            self._subscribers.pop(run_id, None)
    
    async def _async_clear_run(self, run_id: str) -> None:
        """Async version of clear_run with proper locking to prevent race conditions."""
        async with self._lock:
            self._history.pop(run_id, None)
            self._connections.pop(run_id, None)
            self._subscribers.pop(run_id, None)


# Singleton event bus instance
event_bus = EventBus()
