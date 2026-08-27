from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agentic.agent import AgentResult
from agentic.core_agent import CoreAgentResponse
from agentic.tools import ToolRunResult
from agentic_runtime.messaging.messages import (
    ConversationData,
    Event,
    Message,
    RecordedMessageMetadata,
    UserMessage,
)
from agentic_runtime.reactor import LLMReactor, LLMResponse, MultiTurnLLMReactor
from agentic_runtime.routing import make_llm_routing
from agentic_runtime.workflow_runner import run_workflow


@dataclass(frozen=True, slots=True, kw_only=True)
class CreatedNote(Event):
    kind: str = "created_note"
    type: str = "created_note"
    note_name: str = ""
    note_content: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class NoteUpdated(Event):
    kind: str = "note_updated"
    type: str = "note_updated"
    note_name: str = ""
    note_path: str = ""


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


class FakeAgent:
    def __init__(self, responses: list[CoreAgentResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def respond(self, message: str | Any) -> CoreAgentResponse:
        self.calls.append(str(message))
        return self._responses.pop(0)


def _passthrough_decider(msg: Message) -> Sequence[Message]:
    if isinstance(msg, UserMessage):
        return [msg]
    return []


def _user_message(text: str, runtime_id: str = "rt", turn_id: str = "") -> UserMessage:
    return UserMessage(
        data=ConversationData(role="user", text=text),
        metadata=RecordedMessageMetadata(runtime_id=runtime_id, turn_id=turn_id),
    )


class TestRunWorkflowSimple:
    def test_passthrough_single_turn(self) -> None:
        """Simplest case: UserMessage -> LLMReactor -> AssistantMessage -> done."""
        agent = FakeAgent([_make_response(output="hello world", run_id="r-1")])
        reactor = LLMReactor(agent=agent)  # type: ignore[arg-type]
        routing = make_llm_routing(reactor)

        execution = run_workflow(
            message=_user_message("hi", runtime_id="rt", turn_id="t1"),
            decider=_passthrough_decider,
            routing_fn=routing,
        )

        assert execution.text == "hello world"
        assert execution.agent_result is not None
        assert execution.agent_result.run_id == "r-1"
        assert len(execution.recorded_turns) == 1

    def test_no_llm_response_returns_empty(self) -> None:
        """If decider produces no commands -> no LLM call -> empty execution."""

        def noop_decider(msg: Message) -> Sequence[Message]:
            return []

        execution = run_workflow(
            message=_user_message("hi"),
            decider=noop_decider,
            routing_fn=lambda cmd: None,
        )

        assert execution.text == ""


class TestRunWorkflowWithDomainEvents:
    def test_manage_notes_decider_emits_events(self) -> None:
        """ManageNotes pattern: LLM responds with tool_calls -> decider emits domain events."""
        agent = FakeAgent(
            [
                _make_response(
                    output="Note created",
                    run_id="r-1",
                    tool_calls=[("add_note", {"note_name": "test", "note": "content"})],
                ),
            ]
        )
        reactor = LLMReactor(agent=agent)  # type: ignore[arg-type]
        routing = make_llm_routing(reactor)

        def notes_decider(msg: Message) -> Sequence[Message]:
            if isinstance(msg, UserMessage):
                return [msg]
            if isinstance(msg, LLMResponse) and msg.has_tool_calls:
                return [
                    CreatedNote(
                        note_name="test",
                        note_content="content",
                        metadata=RecordedMessageMetadata(
                            runtime_id=msg.metadata.runtime_id,
                            source="manage_notes",
                        ),
                    ),
                    NoteUpdated(
                        note_name="test",
                        note_path="/notes/test.md",
                        metadata=RecordedMessageMetadata(
                            runtime_id=msg.metadata.runtime_id,
                            source="manage_notes",
                        ),
                    ),
                ]
            return []

        execution = run_workflow(
            message=_user_message("create note", runtime_id="rt"),
            decider=notes_decider,
            routing_fn=routing,
        )

        assert execution.text == "Note created"
        assert len(execution.emitted_events) == 2
        assert isinstance(execution.emitted_events[0], CreatedNote)
        assert isinstance(execution.emitted_events[1], NoteUpdated)


class TestRunWorkflowMultiTurn:
    def test_multi_turn_reactor(self) -> None:
        """Sage pattern: MultiTurnLLMReactor handles tool cycle internally."""
        tool_result = ToolRunResult(tool_call=("search", {"q": "test"}), output="found")
        agent = FakeAgent(
            [
                _make_response(
                    output="searching...",
                    tool_calls=[("search", {"q": "test"})],
                    tool_results=(tool_result,),
                ),
                _make_response(output="final answer", run_id="r-2"),
            ]
        )
        reactor = MultiTurnLLMReactor(agent=agent)  # type: ignore[arg-type]
        routing = make_llm_routing(reactor)

        execution = run_workflow(
            message=_user_message("find something"),
            decider=_passthrough_decider,
            routing_fn=routing,
        )

        assert execution.text == "final answer"
        assert execution.agent_result is not None
        assert execution.agent_result.run_id == "r-2"


class TestRunWorkflowOrganizer:
    def test_organizer_transforms_created_note(self) -> None:
        """Organizer pattern: CreatedNote -> format -> LLM -> response."""
        agent = FakeAgent([_make_response(output="organized", run_id="r-1")])
        reactor = LLMReactor(agent=agent)  # type: ignore[arg-type]
        routing = make_llm_routing(reactor)

        def organizer_decider(msg: Message) -> Sequence[Message]:
            if isinstance(msg, CreatedNote):
                return [
                    _user_message(
                        f"Note: {msg.note_name}\n{msg.note_content}",
                        runtime_id=msg.metadata.runtime_id,
                    )
                ]
            if isinstance(msg, UserMessage) and not isinstance(msg, CreatedNote):
                return [msg]
            return []

        from agentic_runtime.messaging.consumer import MessageConsumer
        from agentic_runtime.messaging.message_stream import InMemoryMessageStream

        stream = InMemoryMessageStream()
        consumer = MessageConsumer()
        stream.append(
            CreatedNote(
                note_name="shopping",
                note_content="buy milk",
                metadata=RecordedMessageMetadata(runtime_id="rt"),
            )
        )
        consumer.consume(stream, organizer_decider, routing)

        assert len(agent.calls) == 1
        assert "shopping" in agent.calls[0]
        assert "buy milk" in agent.calls[0]
