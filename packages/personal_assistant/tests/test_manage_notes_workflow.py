from types import SimpleNamespace
from typing import Any

from agentic.agent import AgentResult
from agentic.core_agent import CoreAgentResponse
from agentic.tools import ToolRunResult
from agentic.workflow import WorkflowBuilder
from agentic_runtime.execution import WorkflowExecution
from agentic_runtime.messaging.messages import (
    ConversationData,
    RecordedMessageMetadata,
    UserCommand,
    UserMessage,
)
from agentic_runtime.reactor import LLMReactor
from agentic_runtime.routing import make_llm_routing
from personal_assistant.agents.manage_notes.commands import AddNoteCommand, EditNoteCommand
from personal_assistant.agents.manage_notes.manage_notes_workflow import ManageNotesWorkflow
from personal_assistant.deciders import build_note_events, make_manage_notes_decider
from personal_assistant.messaging.events import CreatedNote, NoteUpdated


def _build_workflow(
    fake_respond: Any,
    parse_tool_fn: Any = None,
    resolve_note_path: Any = None,
    reset_hook: Any = None,
) -> ManageNotesWorkflow:
    """Build a ManageNotesWorkflow with fake agent for testing."""
    workflow = ManageNotesWorkflow.__new__(ManageNotesWorkflow)
    workflow.description = SimpleNamespace(agent_name="manage_notes")

    fake_agent = SimpleNamespace(
        respond=fake_respond,
        start=lambda: "start",
        reset=reset_hook or (lambda: None),
        _agent=SimpleNamespace(
            toolsets=SimpleNamespace(
                parse_tool=parse_tool_fn or (lambda tc: None),
            ),
        ),
        _resolve_note_path=resolve_note_path or (lambda name: f"/vault/{name}.md"),
        description=SimpleNamespace(agent_name="manage_notes"),
    )
    workflow._agent = fake_agent
    workflow._reactor = LLMReactor(agent=fake_agent)
    workflow._routing = make_llm_routing(workflow._reactor)
    workflow._decider = make_manage_notes_decider(
        toolsets=fake_agent._agent.toolsets,
        resolve_note_path=fake_agent._resolve_note_path,
        agent_name="manage_notes",
    )

    def _emit_events(response: Any) -> tuple[Any, ...]:
        if not response.has_tool_calls:
            return ()
        command = fake_agent._agent.toolsets.parse_tool(response.tool_calls[0])
        if command is None:
            return ()
        return build_note_events(
            command,
            resolve_note_path=fake_agent._resolve_note_path,
            agent_name="manage_notes",
            metadata=response.metadata,
        )

    # Build the real inner workflow, the same way __init__ does, so these tests
    # exercise the production handle() path.
    workflow._workflow = (
        WorkflowBuilder("manage_notes")
        .agent(fake_agent)
        .inputs("UserMessage", "UserCommand")
        .emit_events(_emit_events)
        .build()
    )
    workflow.description = workflow._workflow.description
    return workflow


def test_manage_notes_workflow_returns_text_and_publishes_created_note_event() -> None:
    tool_call = ("add_note", {"note_name": "Projekt", "note": "Plan sprintu"})
    tool_run_result = ToolRunResult(tool_call=tool_call, output="Notatka dodana.")
    response = CoreAgentResponse(
        result=AgentResult(
            content="Notatka dodana.",
            tool_calls=[tool_call],
            run_id="run-1",
        ),
        output="Notatka dodana.",
        tool_results=(tool_run_result,),
    )

    workflow = _build_workflow(
        fake_respond=lambda msg: response,
        parse_tool_fn=lambda tc: AddNoteCommand(note_name="Projekt", note="Plan sprintu"),
    )

    result = workflow.handle(
        UserMessage(
            data=ConversationData(role="user", text="Dodaj notatkę"),
            metadata=RecordedMessageMetadata(
                runtime_id="runtime-1",
                domain="manage_notes",
                source="user",
                target="manage_notes",
            ),
        ),
    )

    assert isinstance(result, WorkflowExecution)
    assert result.text == "Notatka dodana."
    assert any(isinstance(e, CreatedNote) for e in result.emitted_events)
    assert any(isinstance(e, NoteUpdated) for e in result.emitted_events)
    created = next(e for e in result.emitted_events if isinstance(e, CreatedNote))
    assert created.note_name == "Projekt"
    assert created.note_content == "Plan sprintu"
    assert created.metadata.source == "manage_notes"


def test_manage_notes_workflow_publishes_note_updated_for_edit_note() -> None:
    tool_call = ("edit_note", {"note_name": "Projekt"})
    tool_run_result = ToolRunResult(tool_call=tool_call, output="Notatka zaktualizowana.")
    response = CoreAgentResponse(
        result=AgentResult(
            content="Notatka zaktualizowana.",
            tool_calls=[tool_call],
            run_id="run-1",
        ),
        output="Notatka zaktualizowana.",
        tool_results=(tool_run_result,),
    )

    workflow = _build_workflow(
        fake_respond=lambda msg: response,
        parse_tool_fn=lambda tc: EditNoteCommand(note_name="Projekt", note="updated"),
    )

    result = workflow.handle(
        UserMessage(
            data=ConversationData(role="user", text="Edytuj notatkę"),
            metadata=RecordedMessageMetadata(
                runtime_id="runtime-1",
                domain="manage_notes",
                source="user",
                target="manage_notes",
            ),
        ),
    )

    assert isinstance(result, WorkflowExecution)
    assert result.text == "Notatka zaktualizowana."
    updated = [e for e in result.emitted_events if isinstance(e, NoteUpdated)]
    assert len(updated) == 1
    assert updated[0].note_name == "Projekt"
    assert updated[0].note_path == "/vault/Projekt.md"


def test_manage_notes_workflow_returns_plain_text_without_publishing_event() -> None:
    response = CoreAgentResponse(
        result=AgentResult(content="To jest zwykła odpowiedź.", run_id="run-1"),
        output="To jest zwykła odpowiedź.",
    )

    workflow = _build_workflow(fake_respond=lambda msg: response)

    result = workflow.handle(
        UserMessage(
            data=ConversationData(role="user", text="Pokaż notatki"),
            metadata=RecordedMessageMetadata(
                runtime_id="runtime-1",
                domain="manage_notes",
                source="user",
                target="manage_notes",
            ),
        ),
    )

    assert isinstance(result, WorkflowExecution)
    assert result.text == "To jest zwykła odpowiedź."
    assert result.emitted_events == ()


def test_manage_notes_workflow_resets_agent_with_command() -> None:
    reset_calls: list[str] = []

    def _unused_respond(message: Any) -> Any:  # pragma: no cover - reset never responds
        raise AssertionError("reset must not call the model")

    workflow = _build_workflow(
        _unused_respond,
        reset_hook=lambda: reset_calls.append("reset"),
    )

    response = workflow.handle(
        UserCommand(
            type="reset",
            metadata=RecordedMessageMetadata(
                runtime_id="runtime-1",
                domain="manage_notes",
                source="runtime",
                target="manage_notes",
            ),
        ),
    )

    assert response == ""
    assert reset_calls == ["reset"]


def test_manage_notes_workflow_respond_path_emits_domain_events_single_pass() -> None:
    tool_call = ("add_note", {"note_name": "Projekt", "note": "Plan sprintu"})
    tool_run_result = ToolRunResult(tool_call=tool_call, output="Notatka dodana.")
    response = CoreAgentResponse(
        result=AgentResult(
            content="Notatka dodana.",
            tool_calls=[tool_call],
            run_id="run-1",
        ),
        output="Notatka dodana.",
        tool_results=(tool_run_result,),
    )

    respond_calls: list[str] = []

    def fake_respond(msg: str | Any) -> CoreAgentResponse:
        respond_calls.append(str(msg))
        return response

    workflow = _build_workflow(
        fake_respond=fake_respond,
        parse_tool_fn=lambda tc: AddNoteCommand(note_name="Projekt", note="Plan sprintu"),
    )

    result = workflow.handle(
        UserMessage(
            data=ConversationData(role="user", text="Dodaj notatkę Projekt"),
            metadata=RecordedMessageMetadata(
                runtime_id="runtime-1",
                domain="manage_notes",
                source="user",
                target="manage_notes",
            ),
        ),
    )

    assert isinstance(result, WorkflowExecution)
    assert result.text == "Notatka dodana."
    assert respond_calls == ["Dodaj notatkę Projekt"]
    assert any(isinstance(e, CreatedNote) for e in result.emitted_events)
    assert any(isinstance(e, NoteUpdated) for e in result.emitted_events)
