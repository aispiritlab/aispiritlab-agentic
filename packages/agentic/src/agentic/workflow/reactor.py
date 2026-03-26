from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from agentic.core_agent import CoreAgentic
from agentic.message import ToolMessage

from agentic.workflow.messages import AssistantMessage, Message


class Reactor(Protocol):
    """Side effect po fakcie biznesowym: input message -> invocation -> output message."""

    def can_handle(self, command: Message) -> bool: ...

    def invoke(self, command: Message) -> Message: ...


Decider = Callable[[Message], Sequence[Message]]
"""Workflow archetype: przyjmuje wiadomosc, zwraca liste komend/eventow do streama."""

TechnicalRoutingFn = Callable[[Message], Reactor | None]
"""Mapowanie komend na Reactors: przyjmuje komende, zwraca ktory Reactor obsluguje."""


@dataclass(frozen=True, kw_only=True)
class LLMResponse(AssistantMessage):
    """Response from LLM with tool execution details for Decider inspection."""

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
        return command.text is not None and len(command.text) > 0

    def invoke(self, command: Message) -> Message:
        text = command.text or ""
        response = self._agent.respond(text)
        tool_calls = tuple(response.result.tool_calls) if response.result.tool_calls else ()
        return LLMResponse(
            text=response.output,
            domain=command.domain,
            source="llm",
            reply_to_message_id=command.message_id or None,
            runtime_id=command.runtime_id,
            turn_id=command.turn_id,
            agent_run_id=response.result.run_id,
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
    ) -> None:
        self._agent = agent
        self._max_turns = max_turns
        self._post_process = post_process

    def can_handle(self, command: Message) -> bool:
        return command.text is not None and len(command.text) > 0

    def invoke(self, command: Message) -> Message:
        text = command.text or ""
        response = self._agent.respond(text)

        turn = 0
        while response.tool_results and turn < self._max_turns:
            turn += 1
            tool_msg = ToolMessage(self._format_tool_results(response))
            response = self._agent.respond(tool_msg)

        output = response.output
        if self._post_process:
            output = self._post_process(output)

        return LLMResponse(
            text=output,
            domain=command.domain,
            source="llm",
            reply_to_message_id=command.message_id or None,
            runtime_id=command.runtime_id,
            turn_id=command.turn_id,
            agent_run_id=response.result.run_id,
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
