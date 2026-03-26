from contextlib import contextmanager
import threading
from types import SimpleNamespace
from typing import Any
from uuid import UUID

from agentic.agent import AgentResult
from agentic.image_generation_call import ImageGenerationResult
from agentic.observability import NoopLLMTracer
from agentic.workflow.messages import (
    AssistantMessage,
    Event,
    ToolCallEvent,
    ToolResultMessage,
    TurnCompleted,
    TurnStarted,
    UserMessage,
)
from agentic_runtime.execution import ExecutionTurnRecord, WorkflowExecution
from agentic_runtime.messaging.message_bus import InMemoryMessageBus
from agentic_runtime.output_handler import workflow_output_handler

from personal_assistant.messaging.events import CreatedNote
from personal_assistant.output_handlers import build_organizer_output_handler
import personal_assistant.runtime as runtime_module
from personal_assistant.runtime import PARuntime as AgenticRuntime


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
        return "trace-general-1"


def _build_runtime(
    *,
    note_handle=None,  # noqa: ANN001
    organizer_handle=None,  # noqa: ANN001
) -> AgenticRuntime:
    runtime = AgenticRuntime.__new__(AgenticRuntime)
    runtime._stop_lock = threading.Lock()
    runtime._stopped = False
    runtime.runtime_id = "runtime-1"
    runtime.bus = InMemoryMessageBus()
    runtime._tracer = NoopLLMTracer()
    runtime.router = SimpleNamespace(
        route=lambda text, summary: "manage_notes",
        start=lambda: "greeting",
        close=lambda: None,
    )
    runtime._kb_task_runner = SimpleNamespace(close=lambda: None)
    runtime.llm_call = None
    runtime.image_call = SimpleNamespace(close=lambda: None)

    runtime.personalize_workflow = SimpleNamespace(
        description=SimpleNamespace(
            agent_name="personalize",
            description="personalize workflow",
            capabilities=("personalization",),
        ),
        handle=lambda message: "personalize",
        close=lambda: None,
    )
    runtime.discovery_notes_workflow = SimpleNamespace(
        description=SimpleNamespace(
            agent_name="discovery_notes",
            description="discovery workflow",
            capabilities=("search",),
        ),
        handle=lambda message: "discovery",
        close=lambda: None,
    )
    runtime.sage_workflow = SimpleNamespace(
        description=SimpleNamespace(
            agent_name="sage",
            description="sage workflow",
            capabilities=("decision-making",),
        ),
        handle=lambda message: "sage",
        close=lambda: None,
    )
    runtime.note_workflow = SimpleNamespace(
        description=SimpleNamespace(
            agent_name="manage_notes",
            description="manage workflow",
            capabilities=("notes",),
        ),
        handle=note_handle or (lambda message: "Notatka Projekt dodana."),
        close=lambda: None,
    )
    runtime.organizer_workflow = SimpleNamespace(
        description=SimpleNamespace(
            agent_name="organizer",
            description="organizer workflow",
            capabilities=("organize",),
        ),
        handle=organizer_handle or (lambda message: "organizer"),
        close=lambda: None,
    )
    runtime.workflows = {
        "personalize": runtime.personalize_workflow,
        "manage_notes": runtime.note_workflow,
        "discovery_notes": runtime.discovery_notes_workflow,
        "sage": runtime.sage_workflow,
        "organizer": runtime.organizer_workflow,
    }
    runtime.bus.register_output_handler(build_organizer_output_handler(runtime.organizer_workflow))
    return runtime


def test_new_runtime_id_uses_uuidv7() -> None:
    runtime_id = AgenticRuntime._new_runtime_id()

    assert UUID(runtime_id).version == 7


def test_given_user_message_when_runtime_handles_then_routes_to_workflow() -> None:
    # given
    runtime = _build_runtime()
    user_message = UserMessage(
        runtime_id=runtime.runtime_id,
        domain="general",
        source="user",
        text="Dodaj notatkę Projekt",
    )

    # when
    reply = runtime.handle(user_message)

    # then
    assert reply == "Notatka Projekt dodana."
    assert len(runtime.bus.messages) == 6

    turn_started = runtime.bus.messages[0]
    assert isinstance(turn_started, TurnStarted)
    assert turn_started.domain == "manage_notes"
    assert turn_started.payload == {"workflow": "manage_notes"}

    routing_event = runtime.bus.messages[1]
    assert isinstance(routing_event, Event)
    assert routing_event.name == "workflow_selected"
    assert routing_event.payload == {"workflow": "manage_notes"}

    targeted_message = runtime.bus.messages[2]
    assert isinstance(targeted_message, UserMessage)
    assert targeted_message.domain == "manage_notes"
    assert targeted_message.text == "Dodaj notatkę Projekt"

    assistant_messages = [m for m in runtime.bus.messages if isinstance(m, AssistantMessage)]
    assert [m.scope for m in assistant_messages] == ["transport", "canonical"]
    assert [m.text for m in assistant_messages] == [
        "Notatka Projekt dodana.",
        "Notatka Projekt dodana.",
    ]
    assert assistant_messages[0].message_id == assistant_messages[1].message_id

    turn_completed = runtime.bus.messages[-1]
    assert isinstance(turn_completed, TurnCompleted)
    assert turn_completed.status == "success"
    assert turn_completed.payload["workflow"] == "manage_notes"
    assert turn_completed.payload["final_message_id"] == assistant_messages[-1].message_id


def test_given_fallback_llm_call_when_runtime_handles_then_trace_and_run_id_are_preserved() -> None:
    runtime = _build_runtime()
    runtime._tracer = _TraceTracer()
    runtime.router = SimpleNamespace(route=lambda text, summary: None, start=lambda: "greeting")
    runtime.llm_call = SimpleNamespace(
        respond=lambda text: SimpleNamespace(
            output=f"fallback:{text}",
            result=AgentResult(
                content=f"fallback:{text}",
                run_id="run-fallback-1",
                trace_id="trace-general-1",
            ),
            tool_results=(),
        )
    )
    user_message = UserMessage(
        runtime_id=runtime.runtime_id,
        domain="general",
        source="user",
        text="Hej fallback",
    )

    reply = runtime.handle(user_message)

    assert reply == "fallback:Hej fallback"
    turn_started = runtime.bus.messages[0]
    assert isinstance(turn_started, TurnStarted)
    assert turn_started.trace_id == "trace-general-1"

    targeted_message = runtime.bus.messages[1]
    assert isinstance(targeted_message, UserMessage)
    assert targeted_message.trace_id == "trace-general-1"

    assistants = [m for m in runtime.bus.messages if isinstance(m, AssistantMessage)]
    assert assistants[-1].agent_run_id == "run-fallback-1"

    turn_completed = runtime.bus.messages[-1]
    assert isinstance(turn_completed, TurnCompleted)
    assert turn_completed.trace_id == "trace-general-1"


def test_given_tuple_reply_when_runtime_handles_then_coerces_to_text() -> None:
    # given
    runtime = _build_runtime(
        note_handle=lambda message: ("Notatka Projekt dodana.", object()),
    )

    # when
    reply = runtime.handle_message("Dodaj notatkę Projekt")

    # then
    assert reply == "Notatka Projekt dodana."
    assistant_messages = [m for m in runtime.bus.messages if isinstance(m, AssistantMessage)]
    assert assistant_messages[-1].text == "Notatka Projekt dodana."
    assert isinstance(runtime.bus.messages[-1], TurnCompleted)


def test_run_generate_image_uses_visual_call() -> None:
    runtime = _build_runtime()
    runtime.image_call = SimpleNamespace(
        generate_image=lambda text: ImageGenerationResult(
            image_path="/tmp/bird.png",
            prompt=text,
            seed=42,
            width=1024,
            height=1024,
            steps=4,
        ),
    )

    reply = runtime.run_generate_image("Stwórz obraz ptaka")

    assert reply.image_path == "/tmp/bird.png"
    assert reply.prompt == "Stwórz obraz ptaka"


def test_reset_chat_clears_direct_chat_and_visual_histories() -> None:
    runtime = _build_runtime()
    llm_reset_calls: list[str] = []
    runtime.llm_call = SimpleNamespace(reset=lambda: llm_reset_calls.append("llm"))

    runtime.reset_chat()

    assert llm_reset_calls == ["llm"]


def test_runtime_stop_closes_resources_and_is_idempotent(monkeypatch) -> None:
    runtime = _build_runtime()
    close_steps: list[str] = []

    runtime._kb_task_runner = SimpleNamespace(close=lambda: close_steps.append("kb_tasks"))
    runtime.router = SimpleNamespace(
        route=lambda text, summary: "manage_notes",
        start=lambda: "greeting",
        close=lambda: close_steps.append("router"),
    )
    runtime.personalize_workflow.close = lambda: close_steps.append("workflow:personalize")
    runtime.note_workflow.close = lambda: close_steps.append("workflow:manage_notes")
    runtime.discovery_notes_workflow.close = lambda: close_steps.append("workflow:discovery_notes")
    runtime.sage_workflow.close = lambda: close_steps.append("workflow:sage")
    runtime.organizer_workflow.close = lambda: close_steps.append("workflow:organizer")
    runtime.llm_call = SimpleNamespace(close=lambda: close_steps.append("llm"))
    runtime.image_call = SimpleNamespace(close=lambda: close_steps.append("image"))
    runtime.bus = SimpleNamespace(close=lambda: close_steps.append("bus"))
    runtime._tracer = SimpleNamespace(shutdown=lambda: close_steps.append("tracer"))

    monkeypatch.setattr(runtime_module, "close_knowledge_base", lambda: close_steps.append("knowledge_base"))

    runtime.stop()
    runtime.stop()

    assert close_steps == [
        "kb_tasks",
        "router",
        "workflow:personalize",
        "workflow:manage_notes",
        "workflow:discovery_notes",
        "workflow:sage",
        "workflow:organizer",
        "llm",
        "image",
        "knowledge_base",
        "bus",
        "tracer",
    ]


def test_given_created_note_when_published_then_organizer_handles_it() -> None:
    # given
    runtime = _build_runtime(
        organizer_handle=lambda message: f"organizer:{message.note_name}:{message.name}",
    )

    # when
    replies = runtime.bus.publish(
        CreatedNote(
            runtime_id=runtime.runtime_id,
            source="manage_notes",
            note_name="Projekt",
            note_content="Plan sprintu",
        )
    )

    # then
    assert replies == ["organizer:Projekt:created_note"]
    assert isinstance(runtime.bus.messages[0], CreatedNote)


def test_bus_close_flushes_pending_batch_output_handlers() -> None:
    class FakeStore:
        def __init__(self) -> None:
            self.closed = False

        def enqueue(self, message) -> None:  # noqa: ANN001
            del message

        def close(self) -> None:
            self.closed = True

    flushed_batches: list[list[str]] = []
    store = FakeStore()
    bus = InMemoryMessageBus(store=store)
    bus.register_output_handler(
        workflow_output_handler(
            can_handle=(CreatedNote,),
            each_batch=lambda messages: flushed_batches.append(
                [message.note_name for message in messages]
            ) or [None],
            batch_size=2,
        )
    )

    replies = bus.publish(
        CreatedNote(
            runtime_id="runtime-1",
            source="manage_notes",
            note_name="Projekt",
            note_content="Plan sprintu",
        )
    )
    bus.close()

    assert replies == []
    assert flushed_batches == [["Projekt"]]
    assert store.closed is True


def test_bus_flush_output_handlers_returns_trailing_batch_results() -> None:
    bus = InMemoryMessageBus()
    bus.register_output_handler(
        workflow_output_handler(
            can_handle=(CreatedNote,),
            each_batch=lambda messages: [f"batch:{len(messages)}"],
            batch_size=2,
        )
    )

    replies = bus.publish(
        CreatedNote(
            runtime_id="runtime-1",
            source="manage_notes",
            note_name="Projekt",
            note_content="Plan sprintu",
        )
    )
    flushed_replies = bus.flush_output_handlers()

    assert replies == []
    assert flushed_replies == ["batch:1"]


def test_given_user_command_reset_when_clear_history_then_all_workflows_reset() -> None:
    # given
    runtime = _build_runtime()
    previous_runtime_id = runtime.runtime_id
    received_commands: list[tuple[str, str]] = []

    runtime.personalize_workflow.handle = lambda message: received_commands.append(
        ("personalize", message.name)
    ) or ""
    runtime.note_workflow.handle = lambda message: received_commands.append(
        ("manage_notes", message.name)
    ) or ""
    runtime.discovery_notes_workflow.handle = lambda message: received_commands.append(
        ("discovery_notes", message.name)
    ) or ""
    runtime.sage_workflow.handle = lambda message: received_commands.append(
        ("sage", message.name)
    ) or ""
    runtime.organizer_workflow.handle = lambda message: received_commands.append(
        ("organizer", message.name)
    ) or ""
    runtime.bus.publish(
        UserMessage(
            runtime_id=runtime.runtime_id,
            domain="general",
            source="user",
            text="hej",
        )
    )

    # when
    runtime.clear_personalization_history()

    # then
    assert runtime.bus.messages == []
    assert runtime.runtime_id != previous_runtime_id
    assert received_commands == [
        ("personalize", "reset"),
        ("manage_notes", "reset"),
        ("discovery_notes", "reset"),
        ("sage", "reset"),
        ("organizer", "reset"),
    ]


def test_given_personalization_finished_when_summary_then_hides_organizer_and_personalize(
    monkeypatch,
) -> None:
    # given
    runtime = AgenticRuntime.__new__(AgenticRuntime)
    runtime.workflows = {
        "personalize": SimpleNamespace(
            description=SimpleNamespace(
                agent_name="personalize",
                description="personalize workflow",
                capabilities=("personalization",),
            )
        ),
        "manage_notes": SimpleNamespace(
            description=SimpleNamespace(
                agent_name="manage_notes",
                description="manage workflow",
                capabilities=("notes",),
            )
        ),
        "discovery_notes": SimpleNamespace(
            description=SimpleNamespace(
                agent_name="discovery_notes",
                description="discovery workflow",
                capabilities=("search",),
            )
        ),
        "sage": SimpleNamespace(
            description=SimpleNamespace(
                agent_name="sage",
                description="sage workflow",
                capabilities=("decision-making",),
            )
        ),
        "organizer": SimpleNamespace(
            description=SimpleNamespace(
                agent_name="organizer",
                description="organizer workflow",
                capabilities=("organize",),
            )
        ),
    }
    monkeypatch.setattr(runtime_module, "is_personalization_finished", lambda: True)

    # when
    summary = runtime._available_workflows_summary()

    # then
    assert "manage_notes" in summary
    assert "discovery_notes" in summary
    assert "sage" in summary
    assert "personalize" not in summary
    assert "organizer" not in summary


def test_given_personalization_not_finished_when_resolve_then_allows_personalize(
    monkeypatch,
) -> None:
    # given
    runtime = AgenticRuntime.__new__(AgenticRuntime)
    personalize_workflow = SimpleNamespace(
        description=SimpleNamespace(
            agent_name="personalize",
            description="personalize workflow",
            capabilities=("personalization",),
        )
    )
    runtime.workflows = {
        "personalize": personalize_workflow,
        "manage_notes": SimpleNamespace(
            description=SimpleNamespace(
                agent_name="manage_notes",
                description="manage workflow",
                capabilities=("notes",),
            )
        ),
    }
    runtime.router = SimpleNamespace(route=lambda text, summary: "personalize")

    # when / then
    monkeypatch.setattr(runtime_module, "is_personalization_finished", lambda: False)
    assert runtime._resolve_workflow("hej") is personalize_workflow

    monkeypatch.setattr(runtime_module, "is_personalization_finished", lambda: True)
    assert runtime._resolve_workflow("hej") is None


def test_given_runtime_when_start_then_returns_greeting() -> None:
    # given
    runtime = _build_runtime()

    # when
    greeting = runtime.start()

    # then
    assert greeting == "greeting"


def test_given_runtime_when_run_then_delegates_to_handle() -> None:
    # given
    runtime = _build_runtime()

    # when
    reply = runtime.run("Dodaj notatkę Projekt")

    # then
    assert reply == "Notatka Projekt dodana."


# --- Multi-turn publishing tests ---


def _make_agent_result(
    run_id: str = "run-1",
    tool_calls: list[tuple[str, dict[str, Any]]] | None = None,
) -> AgentResult:
    return AgentResult(content="ok", run_id=run_id, tool_calls=tool_calls)


def test_multi_turn_tool_messages_publish_in_order() -> None:
    from agentic.tools import ToolRunResult

    runtime = _build_runtime()
    turn1_result = _make_agent_result(
        run_id="run-1",
        tool_calls=[("add_note", {"note_name": "X"})],
    )
    turn1_tr = ToolRunResult(tool_call=("add_note", {"note_name": "X"}), output="created")
    turn2_result = _make_agent_result(run_id="run-2")

    execution = WorkflowExecution(
        text="Done.",
        recorded_turns=(
            ExecutionTurnRecord(agent_result=turn1_result, tool_results=(turn1_tr,)),
            ExecutionTurnRecord(agent_result=turn2_result),
        ),
    )
    incoming = UserMessage(
        runtime_id="rt-1",
        turn_id="t-1",
        domain="test",
        source="user",
        message_id="msg-user",
    )

    runtime.bus.clear()
    runtime._publish_workflow_execution(
        incoming=incoming,
        workflow_name="test_agent",
        execution=execution,
    )

    tool_calls = [m for m in runtime.bus.messages if isinstance(m, ToolCallEvent)]
    tool_results = [m for m in runtime.bus.messages if isinstance(m, ToolResultMessage)]
    assistants = [m for m in runtime.bus.messages if isinstance(m, AssistantMessage)]

    assert len(tool_calls) == 1
    assert tool_calls[0].payload["name"] == "add_note"
    assert len(tool_results) == 1
    assert tool_results[0].text == "created"
    assert len(assistants) >= 1
    assert assistants[-1].text == "Done."


def test_reply_to_chains_across_turns() -> None:
    from agentic.tools import ToolRunResult

    runtime = _build_runtime()
    turn1_result = _make_agent_result(
        run_id="run-1",
        tool_calls=[("step1", {})],
    )
    turn1_tr = ToolRunResult(tool_call=("step1", {}), output="out1")
    turn2_result = _make_agent_result(
        run_id="run-2",
        tool_calls=[("step2", {})],
    )
    turn2_tr = ToolRunResult(tool_call=("step2", {}), output="out2")
    turn3_result = _make_agent_result(run_id="run-3")

    execution = WorkflowExecution(
        text="Final.",
        recorded_turns=(
            ExecutionTurnRecord(agent_result=turn1_result, tool_results=(turn1_tr,)),
            ExecutionTurnRecord(agent_result=turn2_result, tool_results=(turn2_tr,)),
            ExecutionTurnRecord(agent_result=turn3_result),
        ),
    )
    incoming = UserMessage(
        runtime_id="rt-1",
        turn_id="t-1",
        domain="test",
        source="user",
        message_id="msg-user",
    )

    runtime.bus.clear()
    runtime._publish_workflow_execution(
        incoming=incoming,
        workflow_name="test_agent",
        execution=execution,
    )

    # First tool_call should reply to user message
    first_tool_call = [m for m in runtime.bus.messages if isinstance(m, ToolCallEvent)][0]
    assert first_tool_call.reply_to_message_id == "msg-user"

    # Each subsequent tool message should chain reply_to from previous
    all_chained = [
        m for m in runtime.bus.messages
        if isinstance(m, (ToolCallEvent, ToolResultMessage))
    ]
    for i in range(1, len(all_chained)):
        assert all_chained[i].reply_to_message_id is not None


def test_final_assistant_uses_last_turn_run_id() -> None:
    runtime = _build_runtime()
    turn1_result = _make_agent_result(run_id="run-first")
    turn2_result = _make_agent_result(run_id="run-last")

    execution = WorkflowExecution(
        text="Result.",
        recorded_turns=(
            ExecutionTurnRecord(agent_result=turn1_result),
            ExecutionTurnRecord(agent_result=turn2_result),
        ),
    )
    incoming = UserMessage(
        runtime_id="rt-1",
        turn_id="t-1",
        domain="test",
        source="user",
        message_id="msg-user",
    )

    runtime.bus.clear()
    runtime._publish_workflow_execution(
        incoming=incoming,
        workflow_name="test_agent",
        execution=execution,
    )

    assistants = [m for m in runtime.bus.messages if isinstance(m, AssistantMessage)]
    assert assistants[-1].agent_run_id == "run-last"


def test_backward_compat_no_recorded_turns_with_agent_result() -> None:
    runtime = _build_runtime()
    agent_result = _make_agent_result(run_id="run-single")

    execution = WorkflowExecution(
        text="Simple.",
        agent_result=agent_result,
    )
    incoming = UserMessage(
        runtime_id="rt-1",
        turn_id="t-1",
        domain="test",
        source="user",
        message_id="msg-user",
    )

    runtime.bus.clear()
    runtime._publish_workflow_execution(
        incoming=incoming,
        workflow_name="test_agent",
        execution=execution,
    )

    assistants = [m for m in runtime.bus.messages if isinstance(m, AssistantMessage)]
    assert len(assistants) >= 1
    assert assistants[-1].text == "Simple."
    assert assistants[-1].agent_run_id == "run-single"


def test_backward_compat_text_only_execution() -> None:
    runtime = _build_runtime()

    execution = WorkflowExecution(text="Plain text.")
    incoming = UserMessage(
        runtime_id="rt-1",
        turn_id="t-1",
        domain="test",
        source="user",
        message_id="msg-user",
    )

    runtime.bus.clear()
    runtime._publish_workflow_execution(
        incoming=incoming,
        workflow_name="test_agent",
        execution=execution,
    )

    assistants = [m for m in runtime.bus.messages if isinstance(m, AssistantMessage)]
    assert len(assistants) >= 1
    assert assistants[-1].text == "Plain text."
    tool_calls = [m for m in runtime.bus.messages if isinstance(m, ToolCallEvent)]
    assert tool_calls == []
