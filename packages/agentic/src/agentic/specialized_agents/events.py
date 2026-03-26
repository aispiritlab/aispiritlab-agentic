from __future__ import annotations

from dataclasses import dataclass

from agentic.workflow.messages import Event


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskDelegated(Event):
    """Emitted when a PlannerAgent delegates a task to another agent."""

    kind: str = "task_delegated"
    name: str = "task_delegated"
    target_agent: str = ""
    task_description: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskCompleted(Event):
    """Emitted when a delegated task has been completed by an agent."""

    kind: str = "task_completed"
    name: str = "task_completed"
    target_agent: str = ""
    task_description: str = ""
    result: str = ""
