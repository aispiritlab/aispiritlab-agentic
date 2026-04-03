from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from agentic.core_agent import CoreAgentic
from agentic.message import ToolMessage
from agentic.usage import UNLIMITED, RunUsage, UsageLimits

from agentic.workflow.messages import AssistantMessage, ConversationData, Message, RecordedMessageMetadata


class Reactor(Protocol):
    """Side effect po fakcie biznesowym: input message -> invocation -> output message."""

    def can_handle(self, command: Message) -> bool: ...

    def invoke(self, command: Message) -> Message: ...


MessageRouter = Callable[[Message], Sequence[Message]]
"""Stateless message router: takes a message, returns commands/events for the stream."""

TechnicalRoutingFn = Callable[[Message], Reactor | None]
"""Mapowanie komend na Reactors: przyjmuje komende, zwraca ktory Reactor obsluguje."""


@dataclass(frozen=True, kw_only=True)
class LLMResponse(AssistantMessage):
    """Response from LLM with tool execution details for MessageRouter inspection."""

    kind: str = "llm_response"
    tool_calls: tuple[tuple[str, dict[str, Any]], ...] = ()
    _agent_result: Any = None  # AgentResult for observability (prompt_snapshot, usage)
    _tool_results: tuple[Any, ...] = ()  # ToolRunResult objects

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMReactor:
    """LLM call jako Reactor. Wraps CoreAgentic. Single-turn."""

    def __init__(self, agent: CoreAgentic) -> None:
        self._agent = agent

    def can_handle(self, command: Message) -> bool:
        return isinstance(command.data, ConversationData) and command.data.text is not None and len(command.data.text) > 0

    def invoke(self, command: Message) -> Message:
        text = command.data.text if isinstance(command.data, ConversationData) and command.data.text else ""
        response = self._agent.respond(text)
        tool_calls = tuple(response.result.tool_calls) if response.result.tool_calls else ()
        return LLMResponse(
            data=ConversationData(role="assistant", text=response.output),
            metadata=RecordedMessageMetadata(
                runtime_id=command.metadata.runtime_id,
                session_id=command.metadata.session_id,
                turn_id=command.metadata.turn_id,
                reply_to_message_id=getattr(command.metadata, "message_id", "") or None,
                domain=command.metadata.domain,
                source="llm",
                target=command.metadata.source or None,
                agent_run_id=response.result.run_id,
                trace=command.metadata.trace,
                attempt_no=command.metadata.attempt_no,
                loop_iteration=command.metadata.loop_iteration,
            ),
            tool_calls=tool_calls,
            _agent_result=response.result,
            _tool_results=response.tool_results,
        )


class MultiTurnLLMReactor:
    """LLM call z multi-turn tool cycle. Dla Sage i innych agentow z toolami.

    Encapsulates the agent's internal tool-calling loop:
    agent.respond() -> tool_results -> format -> agent.respond(tool_msg) -> ... until no tools.
    """

    def __init__(
        self,
        agent: CoreAgentic,
        *,
        max_turns: int = 10,
        post_process: Callable[[str], str] | None = None,
        usage_limits: UsageLimits | None = None,
    ) -> None:
        self._agent = agent
        self._max_turns = max_turns
        self._post_process = post_process
        self._usage_limits = usage_limits or UNLIMITED
        self._run_usage = RunUsage()

    def can_handle(self, command: Message) -> bool:
        return isinstance(command.data, ConversationData) and command.data.text is not None and len(command.data.text) > 0

    @property
    def run_usage(self) -> RunUsage:
        return self._run_usage

    def invoke(self, command: Message) -> Message:
        text = command.data.text if isinstance(command.data, ConversationData) and command.data.text else ""
        self._run_usage = RunUsage()
        self._usage_limits.check_before_request(self._run_usage)
        response = self._agent.respond(text)
        self._run_usage.add(response.result.request_usage)
        if response.result.tool_calls:
            self._run_usage.add_tool_calls(len(response.result.tool_calls))
        self._usage_limits.check_after_request(self._run_usage)
        response.result.loop_iteration = 0

        turn = 0
        while response.tool_results and turn < self._max_turns:
            turn += 1
            self._usage_limits.check_before_request(self._run_usage)
            tool_msg = ToolMessage(self._format_tool_results(response))
            response = self._agent.respond(tool_msg)
            self._run_usage.add(response.result.request_usage)
            if response.result.tool_calls:
                self._run_usage.add_tool_calls(len(response.result.tool_calls))
            self._usage_limits.check_after_request(self._run_usage)
            response.result.loop_iteration = turn

        output = response.output
        if self._post_process:
            output = self._post_process(output)

        return LLMResponse(
            data=ConversationData(role="assistant", text=output),
            metadata=RecordedMessageMetadata(
                runtime_id=command.metadata.runtime_id,
                session_id=command.metadata.session_id,
                turn_id=command.metadata.turn_id,
                reply_to_message_id=getattr(command.metadata, "message_id", "") or None,
                domain=command.metadata.domain,
                source="llm",
                target=command.metadata.source or None,
                agent_run_id=response.result.run_id,
                trace=command.metadata.trace,
                attempt_no=command.metadata.attempt_no,
                loop_iteration=command.metadata.loop_iteration,
            ),
            tool_calls=tuple(response.result.tool_calls) if response.result.tool_calls else (),
            _agent_result=response.result,
            _tool_results=response.tool_results,
        )

    @staticmethod
    def _format_tool_results(response: Any) -> str:
        parts: list[str] = []
        for tool_result in response.tool_results:
            tool_name, tool_args = tool_result.tool_call
            parts.append(
                "\n".join(
                    [
                        f"Tool: {tool_name}",
                        f"Arguments: {json.dumps(tool_args, ensure_ascii=False)}",
                        "Output:",
                        tool_result.output,
                    ]
                )
            )
        return "\n\n".join(parts)
