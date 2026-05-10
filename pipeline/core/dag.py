"""
DAG Orchestrator — Async topological execution with parallelism and retry logic.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import structlog
from core.event_bus import EventBus
from core.schemas import AgentEvent, AgentStatus, PipelineResult

logger = structlog.get_logger()

# Agent dependency graph
DAG = {
    "ingestion_agent": [],
    "schema_agent": ["ingestion_agent"],
    "cleaning_agent": ["schema_agent"],
    "feature_agent": ["cleaning_agent"],
    "encoding_agent": ["feature_agent"],
    "anomaly_agent": ["encoding_agent"],
    "ml_agent": ["encoding_agent"],  # parallel with anomaly
    "orchestration_agent": ["anomaly_agent", "ml_agent"],
}


class DeadlockError(Exception):
    pass


class DAGOrchestrator:
    """
    Topological execution with parallelism where dependencies allow.
    anomaly_agent and ml_agent run concurrently.
    Retry logic: 3 attempts, exponential backoff (5s -> 10s -> 20s).
    """

    def __init__(self, agents: Dict[str, Any], event_bus: EventBus):
        # Validate all required agents are present
        required_agents = set(DAG.keys())
        provided_agents = set(agents.keys())
        missing_agents = required_agents - provided_agents
        
        if missing_agents:
            raise ValueError(
                f"Missing required agents: {missing_agents}. "
                f"Expected: {required_agents}"
            )
        
        # Validate DAG is acyclic
        self._validate_dag_is_acyclic()
        
        self.agents = agents
        self.event_bus = event_bus
        self.completed: Set[str] = set()
        self.results: Dict[str, Any] = {}
        self.latest_df: Optional[pd.DataFrame] = None
        self._df_lock = asyncio.Lock()
    
    def _validate_dag_is_acyclic(self) -> None:
        """Ensure the DAG has no cycles using depth-first search."""
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in DAG.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in DAG:
            if node not in visited:
                if has_cycle(node):
                    raise ValueError(f"Circular dependency detected at {node}")

    async def run_pipeline(self, run_id: str) -> PipelineResult:
        start_time = time.time()
        started_at = datetime.utcnow()

        await self.event_bus.emit(
            AgentEvent(
                run_id=run_id,
                agent="orchestration_agent",
                status=AgentStatus.STARTED,
                message=f"DAG Orchestrator initialized | 8 agents | run_id={run_id}",
            )
        )

        failed_agents: List[str] = []
        pending_tasks: Set[asyncio.Task] = set()
        running_agents: Set[str] = set()

        try:
            while len(self.completed) < len(self.agents):
                ready = [
                    name
                    for name, deps in DAG.items()
                    if name not in self.completed
                    and name not in failed_agents
                    and name not in running_agents
                    and all(d in self.completed for d in deps)
                ]

                if not ready and not pending_tasks:
                    if len(self.completed) + len(failed_agents) < len(self.agents):
                        raise DeadlockError("No runnable agents remaining")
                    break

                for name in ready:
                    task = asyncio.create_task(self._run_with_retry(name, run_id))
                    task.set_name(name)
                    pending_tasks.add(task)
                    running_agents.add(name)

                if pending_tasks:
                    done, pending_tasks = await asyncio.wait(
                        pending_tasks, return_when=asyncio.FIRST_COMPLETED
                    )

                    for task in done:
                        name = task.get_name()
                        running_agents.remove(name)
                        try:
                            task.result()
                        except Exception as e:
                            failed_agents.append(name)
                            logger.error("Agent failed permanently", agent=name, error=str(e))

        except Exception as e:
            await self.event_bus.emit(
                AgentEvent(
                    run_id=run_id,
                    agent="orchestration_agent",
                    status=AgentStatus.FAILED,
                    message=str(e),
                )
            )

        duration_ms = int((time.time() - start_time) * 1000)
        status = "FAILED" if failed_agents else "SUCCESS"

        orch_res = self.results.get("orchestration_agent")
        rows_processed = getattr(orch_res, "rows_processed", None) if orch_res else None
        features_count = getattr(orch_res, "features_count", None) if orch_res else None
        anomalies_count = getattr(orch_res, "anomalies_count", None) if orch_res else None
        knowledge_chunks = getattr(orch_res, "knowledge_chunks", None) if orch_res else None

        return PipelineResult(
            run_id=run_id,
            status=status,
            started_at=started_at,
            completed_at=datetime.utcnow(),
            duration_ms=duration_ms,
            agents_completed=list(self.completed),
            agents_failed=failed_agents,
            best_model=getattr(self.results.get("ml_agent"), "best_model_name", None),
            best_r2=getattr(self.results.get("ml_agent"), "best_test_r2", None),
            best_rmse=getattr(self.results.get("ml_agent"), "best_test_rmse", None),
            rows_processed=rows_processed,
            features_count=features_count,
            anomalies_count=anomalies_count,
            knowledge_chunks=knowledge_chunks,
        )

    # Agents that run in parallel — their DataFrame outputs are discarded
    # (only their typed result objects matter for downstream orchestration)
    PARALLEL_AGENTS = {"anomaly_agent", "ml_agent"}

    async def _run_with_retry(self, agent_name: str, run_id: str, max_retries: int = 3, base_delay: float = 5.0):
        for attempt in range(max_retries):
            try:
                agent = self.agents[agent_name]
                input_data = self._build_input(agent_name)
                
                # Each agent gets an independent copy of the DataFrame
                # This prevents race conditions between parallel agents
                async with self._df_lock:
                    current_df = self.latest_df.copy() if self.latest_df is not None else None
                
                result, df_out = await agent.run(input_data, df=current_df)
                
                # Only sequential agents update the shared DataFrame
                # Parallel agents (anomaly, ml) consume the encoded df but
                # their outputs are typed result objects — df changes are discarded
                if agent_name not in self.PARALLEL_AGENTS:
                    async with self._df_lock:
                        if df_out is not None:
                            self.latest_df = df_out
                
                self.results[agent_name] = result
                self.completed.add(agent_name)
                return result
            except Exception as e:
                delay = base_delay * (2**attempt)
                if attempt < max_retries - 1:
                    await self.event_bus.emit(AgentEvent(run_id=run_id, agent=agent_name, status=AgentStatus.RETRYING, message=f"Attempt {attempt+1} failed: {e}. Retrying in {delay}s"))
                    await asyncio.sleep(delay)
                else:
                    await self.event_bus.emit(AgentEvent(run_id=run_id, agent=agent_name, status=AgentStatus.FAILED, message=f"Exhausted retries: {e}"))
                    raise

    def _build_input(self, agent_name: str) -> Any:
        from core.schemas import IngestionInput
        if agent_name == "ingestion_agent":
            return IngestionInput()
        return self.results
