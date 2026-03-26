from __future__ import annotations

from typing import Any

from personal_assistant.deciders import (
    make_manage_notes_decider,
    make_organizer_decider,
    passthrough_decider,
    sage_decider,
)
from agentic.workflow.messages import (
    AssistantMessage,
    Event,
    UserMessage,
)
from personal_assistant.messaging.events import CreatedNote, NoteUpdated
from agentic_runtime.reactor import LLMResponse


class TestPassthroughDecider:
    def test_user_message_passes_through(self) -> None:
        msg = UserMessage(text="hello")
        result = passthrough_decider(msg)
        assert len(result) == 1
        assert result[0] == msg

    def test_assistant_message_terminates(self) -> None:
        msg = AssistantMessage(text="response")
        result = passthrough_decider(msg)
        assert len(result) == 0

    def test_event_terminates(self) -> None:
        msg = Event(name="something")
        result = passthrough_decider(msg)
        assert len(result) == 0

    def test_llm_response_terminates(self) -> None:
        msg = LLMResponse(text="done")
        result = passthrough_decider(msg)
        assert len(result) == 0


class TestSageDecider:
    def test_user_message_passes_through(self) -> None:
        msg = UserMessage(text="question")
        result = sage_decider(msg)
        assert len(result) == 1
        assert result[0] == msg

    def test_llm_response_terminates(self) -> None:
        msg = LLMResponse(text="answer")
        result = sage_decider(msg)
        assert len(result) == 0


class FakeToolsets:
    """Fake Toolsets that returns pre-configured commands."""

    def __init__(self, parsed_commands: dict[str, Any]) -> None:
        self._parsed = parsed_commands

    def parse_tool(self, tool_call: tuple[str, dict[str, Any]]) -> Any:
        name = tool_call[0]
        return self._parsed.get(name)


class TestManageNotesDecider:
    def test_user_message_passes_through(self) -> None:
        decider = make_manage_notes_decider(
            toolsets=FakeToolsets({}),  # type: ignore[arg-type]
            resolve_note_path=lambda name: f"/notes/{name}.md",
        )
        msg = UserMessage(text="create note")
        result = decider(msg)
        assert len(result) == 1
        assert result[0] == msg

    def test_add_note_emits_created_and_updated(self) -> None:
        from personal_assistant.agents.manage_notes.commands import AddNoteCommand

        decider = make_manage_notes_decider(
            toolsets=FakeToolsets({  # type: ignore[arg-type]
                "add_note": AddNoteCommand(note_name="test", note="content"),
            }),
            resolve_note_path=lambda name: f"/notes/{name}.md",
            agent_name="manage_notes",
        )
        msg = LLMResponse(
            text="note created",
            tool_calls=(("add_note", {"note_name": "test", "note": "content"}),),
            runtime_id="rt-1",
            turn_id="turn-1",
        )
        result = decider(msg)

        assert len(result) == 2
        assert isinstance(result[0], CreatedNote)
        assert result[0].note_name == "test"
        assert result[0].note_content == "content"
        assert result[0].runtime_id == "rt-1"
        assert result[0].source == "manage_notes"

        assert isinstance(result[1], NoteUpdated)
        assert result[1].note_name == "test"
        assert result[1].note_path == "/notes/test.md"

    def test_edit_note_emits_updated(self) -> None:
        from personal_assistant.agents.manage_notes.commands import EditNoteCommand

        decider = make_manage_notes_decider(
            toolsets=FakeToolsets({  # type: ignore[arg-type]
                "edit_note": EditNoteCommand(note_name="existing", note="updated"),
            }),
            resolve_note_path=lambda name: f"/notes/{name}.md",
        )
        msg = LLMResponse(
            text="note updated",
            tool_calls=(("edit_note", {"note_name": "existing"}),),
            runtime_id="rt-1",
            turn_id="turn-1",
        )
        result = decider(msg)

        assert len(result) == 1
        assert isinstance(result[0], NoteUpdated)
        assert result[0].note_name == "existing"
        assert result[0].note_path == "/notes/existing.md"

    def test_unknown_tool_returns_empty(self) -> None:
        decider = make_manage_notes_decider(
            toolsets=FakeToolsets({"unknown": object()}),  # type: ignore[arg-type]
            resolve_note_path=lambda name: "",
        )
        msg = LLMResponse(
            text="something",
            tool_calls=(("unknown", {}),),
        )
        result = decider(msg)
        assert len(result) == 0

    def test_no_tool_calls_returns_empty(self) -> None:
        decider = make_manage_notes_decider(
            toolsets=FakeToolsets({}),  # type: ignore[arg-type]
            resolve_note_path=lambda name: "",
        )
        msg = LLMResponse(text="just text")
        result = decider(msg)
        assert len(result) == 0


class TestOrganizerDecider:
    def test_created_note_formats_as_user_message(self) -> None:
        decider = make_organizer_decider()
        msg = CreatedNote(
            note_name="shopping",
            note_content="buy milk",
            runtime_id="rt-1",
            turn_id="turn-1",
            source="manage_notes",
        )
        result = decider(msg)

        assert len(result) == 1
        assert isinstance(result[0], UserMessage)
        assert "shopping" in result[0].text
        assert "buy milk" in result[0].text
        assert result[0].runtime_id == "rt-1"

    def test_user_message_passes_through(self) -> None:
        decider = make_organizer_decider()
        msg = UserMessage(text="organize this")
        result = decider(msg)
        assert len(result) == 1
        assert result[0] == msg

    def test_llm_response_terminates(self) -> None:
        decider = make_organizer_decider()
        msg = LLMResponse(text="organized")
        result = decider(msg)
        assert len(result) == 0
