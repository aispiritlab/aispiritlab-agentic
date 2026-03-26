from __future__ import annotations

from typing import Any

from agentic.agent import AgentResult
from agentic.core_agent import CoreAgentResponse
from agentic.tools import ToolRunResult

from agentic_runtime.messaging.messages import AssistantMessage, Message, UserMessage
from agentic_runtime.reactor import LLMReactor, LLMResponse, MultiTurnLLMReactor


def _make_agent_result(
    content: str = "ok",
    run_id: str | None = "run-1",
    tool_calls: list[tuple[str, dict[str, Any]]] | None = None,
) -> AgentResult:
    return AgentResult(content=content, run_id=run_id, tool_calls=tool_calls)


def _make_response(
    output: str = "ok",
    run_id: str | None = "run-1",
    tool_calls: list[tuple[str, dict[str, Any]]] | None = None,
    tool_results: tuple[ToolRunResult, ...] = (),
) -> CoreAgentResponse:
    return CoreAgentResponse(
        result=_make_agent_result(content=output, run_id=run_id, tool_calls=tool_calls),
        output=output,
        tool_results=tool_results,
    )


def _make_tool_result(
    name: str = "add_note",
    args: dict[str, Any] | None = None,
    output: str = "done",
) -> ToolRunResult:
    return ToolRunResult(tool_call=(name, args or {}), output=output)


class FakeAgent:
    """Fake CoreAgentic for testing."""

    def __init__(self, responses: list[CoreAgentResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def respond(self, message: str | Any) -> CoreAgentResponse:
        self.calls.append(str(message))
        return self._responses.pop(0)


class TestLLMReactor:
    def test_can_handle_with_text(self) -> None:
        agent = FakeAgent([])
        reactor = LLMReactor(agent=agent)  # type: ignore[arg-type]
        msg = UserMessage(text="hello")
        assert reactor.can_handle(msg) is True

    def test_can_handle_without_text(self) -> None:
        agent = FakeAgent([])
        reactor = LLMReactor(agent=agent)  # type: ignore[arg-type]
        msg = Message()
        assert reactor.can_handle(msg) is False

    def test_invoke_returns_llm_response(self) -> None:
        response = _make_response(output="world", run_id="run-42")
        agent = FakeAgent([response])
        reactor = LLMReactor(agent=agent)  # type: ignore[arg-type]

        command = UserMessage(
            text="hello",
            domain="test",
            runtime_id="rt-1",
            turn_id="turn-1",
            message_id="msg-1",
        )
        result = reactor.invoke(command)

        assert isinstance(result, LLMResponse)
        assert isinstance(result, AssistantMessage)  # LLMResponse IS AssistantMessage
        assert result.text == "world"
        assert result.domain == "test"
        assert result.source == "llm"
        assert result.reply_to_message_id == "msg-1"
        assert result.runtime_id == "rt-1"
        assert result.turn_id == "turn-1"
        assert result.agent_run_id == "run-42"

    def test_invoke_carries_tool_calls(self) -> None:
        tool_calls = [("add_note", {"name": "test", "content": "hello"})]
        response = _make_response(output="ok", tool_calls=tool_calls)
        agent = FakeAgent([response])
        reactor = LLMReactor(agent=agent)  # type: ignore[arg-type]

        result = reactor.invoke(UserMessage(text="create note"))
        assert isinstance(result, LLMResponse)
        assert result.has_tool_calls is True
        assert result.tool_calls == (("add_note", {"name": "test", "content": "hello"}),)

    def test_invoke_no_tool_calls(self) -> None:
        response = _make_response(output="just text")
        agent = FakeAgent([response])
        reactor = LLMReactor(agent=agent)  # type: ignore[arg-type]

        result = reactor.invoke(UserMessage(text="hello"))
        assert isinstance(result, LLMResponse)
        assert result.has_tool_calls is False
        assert result.tool_calls == ()

    def test_invoke_passes_text_to_agent(self) -> None:
        response = _make_response(output="ok")
        agent = FakeAgent([response])
        reactor = LLMReactor(agent=agent)  # type: ignore[arg-type]

        reactor.invoke(UserMessage(text="what is 2+2"))
        assert agent.calls == ["what is 2+2"]

    def test_invoke_with_empty_text(self) -> None:
        response = _make_response(output="fallback")
        agent = FakeAgent([response])
        reactor = LLMReactor(agent=agent)  # type: ignore[arg-type]

        result = reactor.invoke(Message())
        assert result.text == "fallback"
        assert agent.calls == [""]


class TestMultiTurnLLMReactor:
    def test_single_turn_no_tools(self) -> None:
        response = _make_response(output="answer")
        agent = FakeAgent([response])
        reactor = MultiTurnLLMReactor(agent=agent)  # type: ignore[arg-type]

        result = reactor.invoke(UserMessage(text="question"))
        assert isinstance(result, LLMResponse)
        assert result.text == "answer"
        assert len(agent.calls) == 1

    def test_multi_turn_with_tools(self) -> None:
        tool_result = _make_tool_result(name="search", output="found it")
        first = _make_response(
            output="calling search",
            tool_calls=[("search", {"query": "test"})],
            tool_results=(tool_result,),
        )
        second = _make_response(output="final answer")
        agent = FakeAgent([first, second])
        reactor = MultiTurnLLMReactor(agent=agent)  # type: ignore[arg-type]

        result = reactor.invoke(UserMessage(text="find something"))
        assert isinstance(result, LLMResponse)
        assert result.text == "final answer"
        assert len(agent.calls) == 2
        assert "Tool: search" in agent.calls[1]

    def test_max_turns_limit(self) -> None:
        """Agent keeps calling tools — reactor stops at max_turns."""
        tool_result = _make_tool_result(output="loop")
        looping = _make_response(
            output="still going",
            tool_calls=[("loop_tool", {})],
            tool_results=(tool_result,),
        )
        agent = FakeAgent([looping] * 5)
        reactor = MultiTurnLLMReactor(agent=agent, max_turns=3)  # type: ignore[arg-type]

        result = reactor.invoke(UserMessage(text="start"))
        assert isinstance(result, LLMResponse)
        # 1 initial + 3 tool turns = 4 calls
        assert len(agent.calls) == 4

    def test_post_process(self) -> None:
        response = _make_response(output="<think>reasoning</think> answer")
        agent = FakeAgent([response])

        def strip_think(text: str) -> str:
            return text.replace("<think>", "").replace("</think>", "").strip()

        reactor = MultiTurnLLMReactor(agent=agent, post_process=strip_think)  # type: ignore[arg-type]
        result = reactor.invoke(UserMessage(text="q"))
        assert result.text == "reasoning answer"
