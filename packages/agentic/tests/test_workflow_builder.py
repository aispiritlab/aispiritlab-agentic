from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic.agent import AgentResult
from agentic.core_agent import CoreAgentResponse
from agentic.workflow import WorkflowBuilder, WorkflowRuntime
from agentic.workflow.messages import Event, UserCommand, UserMessage


class FakeAgent:
    def __init__(self, responses: list[CoreAgentResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []
        self.started = 0
        self.reset_count = 0
        self.closed = 0

    def respond(self, message: str | Any) -> CoreAgentResponse:
        self.calls.append(str(message))
        return self._responses.pop(0)

    def start(self) -> str:
        self.started += 1
        return "started"

    def reset(self) -> None:
        self.reset_count += 1

    def close(self) -> None:
        self.closed += 1


def _response(
    text: str,
    *,
    tool_calls: list[tuple[str, dict[str, Any]]] | None = None,
) -> CoreAgentResponse:
    return CoreAgentResponse(
        result=AgentResult(content=text, run_id="run-1", tool_calls=tool_calls),
        output=text,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Triggered(Event):
    kind: str = "triggered"
    name: str = "triggered"


def test_workflow_builder_creates_simple_request_response_workflow() -> None:
    agent = FakeAgent([_response("hello world")])
    workflow = WorkflowBuilder("helper").agent(agent).build()

    result = workflow.handle(UserMessage(text="hi"))

    assert result.text == "hello world"
    assert agent.calls == ["hi"]


def test_workflow_builder_uses_start_and_reset_hooks() -> None:
    agent = FakeAgent([_response("unused")])
    workflow = WorkflowBuilder("helper").agent(agent).build()

    started = workflow.handle(UserCommand(name="start"))
    reset = workflow.handle(UserCommand(name="reset"))

    assert started == "started"
    assert reset == ""
    assert agent.started == 1
    assert agent.reset_count == 1


def test_workflow_builder_emits_events_without_custom_decider() -> None:
    agent = FakeAgent([_response("done", tool_calls=[("delegate", {"step": 1})])])
    workflow = (
        WorkflowBuilder("planner")
        .agent(agent)
        .emit_events(lambda response: [Event(name="delegated", payload={"count": len(response.tool_calls)})])
        .build()
    )

    result = workflow.handle(UserMessage(text="plan"))

    assert result.text == "done"
    assert len(result.emitted_events) == 1
    assert result.emitted_events[0].name == "delegated"
    assert result.emitted_events[0].payload == {"count": 1}


def test_workflow_builder_maps_external_event_inputs() -> None:
    agent = FakeAgent([_response("mapped")])
    workflow = (
        WorkflowBuilder("organizer")
        .agent(agent)
        .inputs("Triggered", "UserCommand", "UserMessage")
        .map_input(
            lambda message: UserMessage(text=message.text or "")
            if isinstance(message, Triggered)
            else message if isinstance(message, UserMessage) else None
        )
        .build()
    )

    result = workflow.handle(Triggered(text="external input"))

    assert result.text == "mapped"
    assert agent.calls == ["external input"]


def test_workflow_runtime_runs_builder_workflow_and_publishes_turn_messages() -> None:
    agent = FakeAgent([_response("runtime reply")])
    workflow = WorkflowBuilder("helper").agent(agent).build()
    runtime = WorkflowRuntime()
    runtime.register_workflow("helper", workflow)

    reply = runtime.run_text("hello", "helper")

    assert reply == "runtime reply"
    assert runtime.message_log[0].kind == "turn_started"
    assert runtime.message_log[-1].kind == "turn_completed"
