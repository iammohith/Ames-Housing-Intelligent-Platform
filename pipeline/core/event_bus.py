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

    def __init__(self):
        self._connections: Dict[str, List[WebSocket]] = defaultdict(list)
        self._history: Dict[str, List[AgentEvent]] = defaultdict(list)
        self._subscribers: Dict[str, List[asyncio.Queue]] = defaultdict(list)
        self._lock = asyncio.Lock()

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
        Called by agents during execution.
        """
        run_id = event.run_id

        # Store in history
        self._history[run_id].append(event)

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
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subscribers[run_id].append(queue)

        # Replay history first
        for event in self._history.get(run_id, []):
            yield event

        try:
            while True:
                event = await queue.get()
                yield event
                # Stop if pipeline completed or failed
                if event.status in (AgentStatus.SUCCESS, AgentStatus.FAILED):
                    if event.agent == "orchestration_agent":
                        break
        finally:
            try:
                self._subscribers[run_id].remove(queue)
            except ValueError:
                pass

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
        """Calculate overall pipeline progress (0.0 to 100.0)."""
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
            if statuses.get(agent) == "SUCCESS":
                completed += 1
        return (completed / len(all_agents)) * 100.0

    def clear_run(self, run_id: str) -> None:
        """Clear all data for a completed/cancelled run."""
        self._history.pop(run_id, None)
        self._connections.pop(run_id, None)
        self._subscribers.pop(run_id, None)


# Singleton event bus instance
event_bus = EventBus()
