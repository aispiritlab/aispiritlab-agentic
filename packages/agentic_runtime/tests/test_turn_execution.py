from contextlib import contextmanager
from typing import Any

import pytest

from agentic.agent import AgentResult
from agentic.observability import NoopLLMTracer

from agentic_runtime.execution import WorkflowExecution
from agentic_runtime.messaging.message_bus import InMemoryMessageBus
from agentic_runtime.messaging.messages import AssistantMessage, Event, TurnCompleted, TurnStarted, UserMessage
from agentic_runtime.turn_execution import TurnExecutor, TurnPlan


class _TraceHandle:
    def update(self, **kwargs: Any) -> None:
        del kwargs


class _TraceTracer(NoopLLMTracer):
    @contextmanager
    def workflow(self, **kwargs: Any):
        del kwargs
        yield _TraceHandle()

    @property
    def current_trace_id(self) -> str | None:
        return "trace-turn-1"


def test_turn_executor_emits_routed_turn_messages_in_order() -> None:
    bus = InMemoryMessageBus()
    executor = TurnExecutor(bus=bus, tracer=NoopLLMTracer())

    reply = executor.execute(
        TurnPlan(
            incoming=UserMessage(
                runtime_id="rt-1",
                turn_id="turn-1",
                domain="manage_notes",
                source="user",
                target="manage_notes",
                text="Dodaj notatkę Projekt",
            ),
            handler=lambda message: "Notatka Projekt dodana.",
            trace_name="manage_notes",
            lifecycle_domain="manage_notes",
            lifecycle_target="manage_notes",
            lifecycle_workflow_name="manage_notes",
            output_agent_name="manage_notes",
            selected_workflow="manage_notes",
        )
    )

    assert reply == "Notatka Projekt dodana."
    assert isinstance(bus.messages[0], TurnStarted)
    assert isinstance(bus.messages[1], Event)
    assert bus.messages[1].name == "workflow_selected"
    assert isinstance(bus.messages[2], UserMessage)

    assistant_messages = [m for m in bus.messages if isinstance(m, AssistantMessage)]
    assert [m.scope for m in assistant_messages] == ["transport", "canonical"]
    assert [m.text for m in assistant_messages] == [
        "Notatka Projekt dodana.",
        "Notatka Projekt dodana.",
    ]
    assert isinstance(bus.messages[-1], TurnCompleted)


def test_turn_executor_preserves_trace_and_run_id_for_fallback_execution() -> None:
    bus = InMemoryMessageBus()
    executor = TurnExecutor(bus=bus, tracer=_TraceTracer())

    reply = executor.execute(
        TurnPlan(
            incoming=UserMessage(
                runtime_id="rt-1",
                turn_id="turn-1",
                domain="general",
                source="user",
                text="Hej fallback",
            ),
            handler=lambda message: WorkflowExecution(
                text=f"fallback:{message.text}",
                agent_result=AgentResult(
                    content=f"fallback:{message.text}",
                    run_id="run-fallback-1",
                    trace_id="trace-turn-1",
                ),
            ),
            trace_name="general",
            lifecycle_domain="general",
            lifecycle_target=None,
            lifecycle_workflow_name="general",
            output_agent_name="assistant",
        )
    )

    assert reply == "fallback:Hej fallback"
    turn_started = bus.messages[0]
    assert isinstance(turn_started, TurnStarted)
    assert turn_started.trace_id == "trace-turn-1"

    user_message = bus.messages[1]
    assert isinstance(user_message, UserMessage)
    assert user_message.trace_id == "trace-turn-1"

    assistants = [m for m in bus.messages if isinstance(m, AssistantMessage)]
    assert assistants[-1].agent_run_id == "run-fallback-1"

    turn_completed = bus.messages[-1]
    assert isinstance(turn_completed, TurnCompleted)
    assert turn_completed.trace_id == "trace-turn-1"


def test_turn_executor_reuses_existing_targeted_turn_id() -> None:
    bus = InMemoryMessageBus()
    executor = TurnExecutor(bus=bus, tracer=NoopLLMTracer())

    executor.execute(
        TurnPlan(
            incoming=UserMessage(
                runtime_id="rt-1",
                turn_id="turn-existing",
                domain="sage",
                source="user",
                target="sage",
                text="Pomóż mi podjąć decyzję",
            ),
            handler=lambda message: "Decyzja",
            trace_name="sage",
            lifecycle_domain="sage",
            lifecycle_target="sage",
            lifecycle_workflow_name="sage",
            output_agent_name="sage",
        )
    )

    turn_started = bus.messages[0]
    targeted_message = bus.messages[1]
    assert isinstance(turn_started, TurnStarted)
    assert isinstance(targeted_message, UserMessage)
    assert turn_started.turn_id == "turn-existing"
    assert targeted_message.turn_id == "turn-existing"


def test_turn_executor_publishes_error_turn_completed() -> None:
    bus = InMemoryMessageBus()
    executor = TurnExecutor(bus=bus, tracer=NoopLLMTracer())

    def _explode(message: UserMessage) -> str:
        del message
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        executor.execute(
            TurnPlan(
                incoming=UserMessage(
                    runtime_id="rt-1",
                    turn_id="turn-1",
                    domain="manage_notes",
                    source="user",
                    target="manage_notes",
                    text="Dodaj notatkę",
                ),
                handler=_explode,
                trace_name="manage_notes",
                lifecycle_domain="manage_notes",
                lifecycle_target="manage_notes",
                lifecycle_workflow_name="manage_notes",
                output_agent_name="manage_notes",
            )
        )

    turn_completed = bus.messages[-1]
    assert isinstance(turn_completed, TurnCompleted)
    assert turn_completed.status == "error"
    assert turn_completed.payload == {
        "workflow": "manage_notes",
        "error_type": "RuntimeError",
        "error_message": "boom",
    }
