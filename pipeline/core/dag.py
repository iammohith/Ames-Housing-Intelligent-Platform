"""
DAG Orchestrator — Async topological execution with parallelism and retry logic.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Set

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
        extra_agents = provided_agents - required_agents
        
        if missing_agents:
            raise ValueError(
                f"Missing required agents: {missing_agents}. "
                f"Expected: {required_agents}"
            )
        
        if extra_agents:
            logger.warning(f"Extra agents provided (will be ignored): {extra_agents}")
        
        # Validate DAG is acyclic using topological sort (DFS with recursion stack)
        self._validate_dag_is_acyclic()
        
        self.agents = agents
        self.event_bus = event_bus
        self.completed: Set[str] = set()
        self.results: Dict[str, Any] = {}
    
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
                    raise ValueError(
                        f"Circular dependency detected in DAG at node {node}. "
                        f"Ensure all agent dependencies form an acyclic graph."
                    )

    async def run_pipeline(self, run_id: str) -> PipelineResult:
        start_time = time.time()
        started_at = datetime.utcnow()

        await self.event_bus.emit(
            AgentEvent(
                run_id=run_id,
                agent="orchestration_agent",
                status=AgentStatus.STARTED,
                message=f"DAG Orchestrator initialised | 8 agents | run_id={run_id}",
                timestamp=datetime.utcnow(),
            )
        )

        failed_agents: List[str] = []
        pending_tasks: Set[asyncio.Task] = set()
        running_agents: Set[str] = set()

        try:
            while len(self.completed) < len(self.agents):
                # Find agents whose dependencies are satisfied
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
                        raise DeadlockError(
                            "No runnable agents — possible cycle or upstream failure"
                        )
                    break

                if len(ready) > 1:
                    await self.event_bus.emit(
                        AgentEvent(
                            run_id=run_id,
                            agent="orchestration_agent",
                            status=AgentStatus.PROGRESS,
                            message=f"Agents {', '.join(ready)} running in parallel",
                            timestamp=datetime.utcnow(),
                        )
                    )

                # Run ready agents
                for name in ready:
                    task = asyncio.create_task(self._run_with_retry(name, run_id))
                    task.set_name(name)
                    pending_tasks.add(task)
                    running_agents.add(name)

                # Wait for at least one task to finish
                if pending_tasks:
                    done, pending_tasks = await asyncio.wait(
                        pending_tasks, return_when=asyncio.FIRST_COMPLETED
                    )

                    for task in done:
                        name = task.get_name()
                        running_agents.remove(name)

                        try:
                            task.result()  # Will raise if agent failed permanently
                        except Exception as e:
                            failed_agents.append(name)
                            logger.error(
                                "Agent permanently failed", agent=name, error=str(e)
                            )

        except DeadlockError as e:
            await self.event_bus.emit(
                AgentEvent(
                    run_id=run_id,
                    agent="orchestration_agent",
                    status=AgentStatus.FAILED,
                    message=str(e),
                    timestamp=datetime.utcnow(),
                )
            )

        duration_ms = int((time.time() - start_time) * 1000)
        status = "FAILED" if failed_agents else "SUCCESS"

        if "orchestration_agent" in self.results and status == "SUCCESS":
            return self.results["orchestration_agent"]

        return PipelineResult(
            run_id=run_id,
            status=status,
            started_at=started_at,
            completed_at=datetime.utcnow(),
            duration_ms=duration_ms,
            agents_completed=list(self.completed),
            agents_failed=failed_agents,
        )

    async def _run_with_retry(
        self,
        agent_name: str,
        run_id: str,
        max_retries: int = 3,
        base_delay: float = 5.0,
    ):
        for attempt in range(max_retries):
            try:
                agent = self.agents[agent_name]
                input_data = self._build_input(agent_name)
                result = await agent.run(input_data)
                self.results[agent_name] = result
                self.completed.add(agent_name)
                return result
            except Exception as e:
                delay = base_delay * (2**attempt)
                if attempt < max_retries - 1:
                    await self.event_bus.emit(
                        AgentEvent(
                            run_id=run_id,
                            agent=agent_name,
                            status=AgentStatus.RETRYING,
                            message=f"Attempt {attempt+1} failed: {e}. Retrying in {delay}s",
                            timestamp=datetime.utcnow(),
                        )
                    )
                    await asyncio.sleep(delay)
                else:
                    await self.event_bus.emit(
                        AgentEvent(
                            run_id=run_id,
                            agent=agent_name,
                            status=AgentStatus.FAILED,
                            message=f"All {max_retries} attempts exhausted: {e}",
                            timestamp=datetime.utcnow(),
                        )
                    )
                    raise

    def _build_input(self, agent_name: str) -> Any:
        """Build input for an agent from previous agent results.

        Data flow pattern:
        - ingestion_agent: receives IngestionInput (CSV path, expected shape)
        - orchestration_agent: receives self.results dict (Pydantic model outputs)
          so it can read structured fields like .anomaly_report, .best_model_name
        - All other agents: receive self.agents dict (agent instances).
          Each agent's _get_df() traverses this dict to find the upstream agent
          whose _df attribute was set during its execute() call.
          This is a mutable shared-state pattern — agent instances persist
          across the pipeline run and accumulate state (e.g., _df, _metadata_cols).
        """
        from core.schemas import IngestionInput

        if agent_name == "ingestion_agent":
            return IngestionInput()
        if agent_name == "orchestration_agent":
            return self.results
        # All other agents receive the agent instances dict.
        # Each agent's _get_df() finds the right upstream agent's _df.
        return self.agents
