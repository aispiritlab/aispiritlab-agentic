"""Workflow wrappers — adapt agents to the runtime's handler interface.

Each workflow wraps an agent and returns a WorkflowExecution.
The runtime calls handler(UserMessage) and gets back structured results.
"""
from __future__ import annotations

from agentic.core_agent import CoreAgentic
from agentic.specialized_agents import PlannerAgent, TaskDelegated
from agentic.workflow import WorkflowBuilder
from agentic.workflow.execution import WorkflowExecution
from agentic.workflow.messages import UserMessage


def make_planner_workflow(planner: PlannerAgent):
    """Wrap PlannerAgent as a workflow handler.

    Returns WorkflowExecution with emitted_events = list of TaskDelegated.
    The runtime publishes these events, triggering output handlers.
    """

    def handle(message: UserMessage) -> WorkflowExecution:
        tasks = planner.plan(message.text or "")
        return WorkflowExecution(
            text=f"Plan created with {len(tasks)} step(s).",
            emitted_events=tuple(tasks),
        )

    return handle


def make_worker_workflow(name: str, agent: CoreAgentic):
    """Build a worker workflow through WorkflowBuilder."""

    return WorkflowBuilder(name).agent(agent).build()
