"""
BaseAgent — Abstract base class for all pipeline agents.
Handles logging, metrics, event emission, and timing.
"""
from __future__ import annotations
import traceback
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Generic, TypeVar
import structlog
from pydantic import BaseModel
from core.event_bus import EventBus
from core.metrics import agent_duration_seconds, agent_runs_total
from core.schemas import AgentEvent, AgentStatus

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)

class BaseAgent(ABC, Generic[InputT, OutputT]):
    """Every agent implements this contract. Override execute(), never run()."""
    name: str = "base_agent"
    version: str = "1.0.0"

    def __init__(self, event_bus: EventBus, run_id: str):
        self.log = structlog.get_logger(agent=self.name)
        self.event_bus = event_bus
        self.run_id = run_id

    async def emit(self, status: AgentStatus, message: str, **kwargs) -> None:
        event = AgentEvent(
            run_id=self.run_id, agent=self.name, status=status,
            message=message, timestamp=datetime.utcnow(), **kwargs,
        )
        await self.event_bus.emit(event)

    @abstractmethod
    async def execute(self, input_data: InputT) -> OutputT:
        """Core agent logic — implemented by each agent."""
        ...

    async def run(self, input_data: InputT) -> OutputT:
        """Wrapper: timing, logging, metrics, event broadcasting."""
        await self.emit(AgentStatus.STARTED, f"{self.name} initialising")
        start_time = time.time()
        try:
            result = await self.execute(input_data)
            duration_ms = int((time.time() - start_time) * 1000)
            agent_duration_seconds.labels(agent_name=self.name).observe(duration_ms / 1000.0)
            agent_runs_total.labels(agent_name=self.name, status="success").inc()
            await self.emit(AgentStatus.SUCCESS, f"{self.name} completed successfully",
                          duration_ms=duration_ms, rows_out=getattr(result, "row_count", None))
            self.log.info("Agent completed", agent=self.name, duration_ms=duration_ms)
            return result
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            agent_runs_total.labels(agent_name=self.name, status="failure").inc()
            tb = traceback.format_exc()
            await self.emit(AgentStatus.FAILED, f"{self.name} failed: {str(e)}",
                          duration_ms=duration_ms, traceback=tb)
            self.log.error("Agent failed", agent=self.name, error=str(e))
            raise
