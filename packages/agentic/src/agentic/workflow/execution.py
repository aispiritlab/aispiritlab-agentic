from __future__ import annotations

from dataclasses import dataclass

from agentic.agent import AgentResult
from agentic.tools import ToolRunResult

from agentic.workflow.messages import Message


@dataclass(frozen=True, slots=True)
class ExecutionTurnRecord:
    agent_result: AgentResult
    tool_results: tuple[ToolRunResult, ...] = ()


@dataclass(frozen=True, slots=True)
class WorkflowExecution:
    text: str
    agent_result: AgentResult | None = None
    tool_results: tuple[ToolRunResult, ...] = ()
    emitted_events: tuple[Message, ...] = ()
    recorded_turns: tuple[ExecutionTurnRecord, ...] = ()
