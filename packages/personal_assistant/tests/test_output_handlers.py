from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from personal_assistant.messaging.events import CreatedNote, NoteDeleted, NoteUpdated
from agentic_runtime.output_handler import dispatch_output_handlers
from personal_assistant.output_handlers import (
    KnowledgeBaseTaskRunner,
    build_organizer_output_handler,
    build_rag_output_handler,
)


def test_organizer_handler_delegates_created_note_to_workflow() -> None:
    workflow = SimpleNamespace(handle=MagicMock(return_value="organized"))
    handler = build_organizer_output_handler(workflow)
    message = CreatedNote(
        runtime_id="r1", source="manage_notes", note_name="Foo", note_content="bar",
    )

    results = dispatch_output_handlers([handler], message)

    assert results == ["organized"]
    workflow.handle.assert_called_once_with(message)


def test_rag_handler_calls_update_note_in_kb_for_note_updated() -> None:
    runner = SimpleNamespace(submit_update=MagicMock(), submit_delete=MagicMock())
    handler = build_rag_output_handler(runner)
    message = NoteUpdated(
        runtime_id="r1", source="manage_notes", note_name="Foo", note_path="/notes/foo.md",
    )
    results = dispatch_output_handlers([handler], message)

    assert results == []
    runner.submit_update.assert_called_once_with("/notes/foo.md")
    runner.submit_delete.assert_not_called()


def test_rag_handler_calls_delete_note_from_kb_for_note_deleted() -> None:
    runner = SimpleNamespace(submit_update=MagicMock(), submit_delete=MagicMock())
    handler = build_rag_output_handler(runner)
    message = NoteDeleted(
        runtime_id="r1", source="manage_notes", note_name="Foo", note_path="/notes/foo.md",
    )
    results = dispatch_output_handlers([handler], message)

    assert results == []
    runner.submit_delete.assert_called_once_with("/notes/foo.md")
    runner.submit_update.assert_not_called()


def test_knowledge_base_task_runner_close_is_idempotent() -> None:
    runner = KnowledgeBaseTaskRunner(resync=lambda: None)

    runner.close()
    runner.close()
