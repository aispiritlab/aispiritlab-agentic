"""Output handlers — event-driven dispatch of TaskDelegated to worker agents.

This replaces the for-loop pattern from lab2 with declarative event routing.
Each TaskDelegated event is matched by the handler and dispatched to the
correct worker agent via run_workflow.
"""
from __future__ import annotations

from collections.abc import Callable

from agentic.specialized_agents.events import TaskCompleted, TaskDelegated
from agentic.workflow import (
    TechnicalRoutingFn,
    UserMessage,
    WorkflowOutputHandler,
    run_workflow,
    workflow_output_handler,
)
from agentic.workflow.messages import Message

from .deciders import worker_decider


def build_worker_dispatch_handler(
    worker_routings: dict[str, TechnicalRoutingFn],
    *,
    on_completed: Callable[[TaskCompleted], None] | None = None,
) -> tuple[WorkflowOutputHandler, list[TaskCompleted]]:
    """Build an output handler that dispatches TaskDelegated events to worker agents.

    Args:
        worker_routings: Map of agent_name → routing function.
        on_completed: Optional callback fired after each task completes.

    Returns:
        A tuple of (handler, completed_tasks list).
    """
    completed_tasks: list[TaskCompleted] = []

    def _handle_task_delegated(message: Message) -> str | None:
        if not isinstance(message, TaskDelegated):
            return None

        routing = worker_routings.get(message.target_agent)
        if routing is None:
            return f"[Unknown agent: {message.target_agent}]"

        execution = run_workflow(
            message=UserMessage(
                text=message.task_description,
                source="planner",
            ),
            decider=worker_decider,
            routing_fn=routing,
        )

        task = TaskCompleted(
            source=message.target_agent,
            target_agent=message.target_agent,
            task_description=message.task_description,
            result=execution.text,
        )
        completed_tasks.append(task)
        if on_completed is not None:
            on_completed(task)

        return f"[{message.target_agent.title()}]: {execution.text}"

    handler = workflow_output_handler(
        can_handle=(TaskDelegated,),
        each_message=_handle_task_delegated,
        name="worker_dispatch",
    )

    return handler, completed_tasks
