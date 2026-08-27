"""Output handlers — event-driven dispatch registered on the runtime bus.

When the runtime publishes TaskDelegated events, the handler automatically
routes them to the correct worker workflow. No manual dispatch needed.
"""

from __future__ import annotations

from collections.abc import Callable

from agentic.specialized_agents.events import TaskCompleted, TaskDelegated
from agentic.workflow import (
    AgenticWorkflow,
    UserMessage,
    WorkflowOutputHandler,
    coerce_execution,
    reply_metadata,
    user_message,
    workflow_output_handler,
)
from agentic.workflow.messages import Message


def build_worker_output_handler(
    worker_workflows: dict[str, AgenticWorkflow | Callable[[UserMessage], object]],
    *,
    on_completed: Callable[[TaskCompleted], None] | None = None,
) -> WorkflowOutputHandler:
    """Build an output handler that dispatches TaskDelegated to worker workflows.

    Registered on the runtime bus. When the planner emits TaskDelegated events
    and the runtime publishes them, this handler fires automatically.

    Args:
        worker_workflows: Map of agent_name → workflow handler.
        on_completed: Optional callback fired after each task completes.
    """

    def _handle(message: Message) -> str | None:
        if not isinstance(message, TaskDelegated):
            return None

        workflow = worker_workflows.get(message.target_agent)
        if workflow is None:
            return f"[Unknown agent: {message.target_agent}]"

        handler = workflow.handle if isinstance(workflow, AgenticWorkflow) else workflow
        execution = coerce_execution(
            handler(
                user_message(
                    message.task_description,
                    source="planner",
                    reply_to=message,
                )
            )
        )

        task = TaskCompleted(
            target_agent=message.target_agent,
            task_description=message.task_description,
            result=execution.text,
            metadata=reply_metadata(message, source=message.target_agent),
        )
        if on_completed is not None:
            on_completed(task)

        return f"[{message.target_agent.title()}]: {execution.text}"

    return workflow_output_handler(
        can_handle=(TaskDelegated,),
        each_message=_handle,
        name="worker_dispatch",
    )
