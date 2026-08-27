from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from agentic.workflow import InMemoryCheckpointStore, InMemoryEventStore
from agentic.workflow.messages import RecordedMessageMetadata
from agentic_runtime.output_handler import dispatch_output_handlers
from personal_assistant.messaging.events import CreatedNote, NoteDeleted, NoteUpdated
from personal_assistant.output_handlers import (
    DurableKnowledgeBaseProjector,
    KnowledgeBaseTaskRunner,
    build_organizer_output_handler,
    build_rag_output_handler,
)


def test_organizer_handler_delegates_created_note_to_workflow() -> None:
    workflow = SimpleNamespace(handle=MagicMock(return_value="organized"))
    handler = build_organizer_output_handler(workflow)
    message = CreatedNote(
        note_name="Foo",
        note_content="bar",
        metadata=RecordedMessageMetadata(runtime_id="r1", source="manage_notes"),
    )

    results = dispatch_output_handlers([handler], message)

    assert results == ["organized"]
    workflow.handle.assert_called_once_with(message)


def test_rag_handler_calls_update_note_in_kb_for_note_updated() -> None:
    runner = SimpleNamespace(submit_update=MagicMock(), submit_delete=MagicMock())
    handler = build_rag_output_handler(runner)
    message = NoteUpdated(
        note_name="Foo",
        note_path="/notes/foo.md",
        metadata=RecordedMessageMetadata(runtime_id="r1", source="manage_notes"),
    )
    results = dispatch_output_handlers([handler], message)

    assert results == []
    runner.submit_update.assert_called_once_with("/notes/foo.md")
    runner.submit_delete.assert_not_called()


def test_rag_handler_calls_delete_note_from_kb_for_note_deleted() -> None:
    runner = SimpleNamespace(submit_update=MagicMock(), submit_delete=MagicMock())
    handler = build_rag_output_handler(runner)
    message = NoteDeleted(
        note_name="Foo",
        note_path="/notes/foo.md",
        metadata=RecordedMessageMetadata(runtime_id="r1", source="manage_notes"),
    )
    results = dispatch_output_handlers([handler], message)

    assert results == []
    runner.submit_delete.assert_called_once_with("/notes/foo.md")
    runner.submit_update.assert_not_called()


def test_knowledge_base_task_runner_close_is_idempotent() -> None:
    runner = KnowledgeBaseTaskRunner(resync=lambda: None)

    runner.close()
    runner.close()


def test_task_runner_exposes_background_failure() -> None:
    def fail(_: str) -> None:
        raise RuntimeError("index unavailable")

    runner = KnowledgeBaseTaskRunner(update_note=fail)
    runner.submit_update("/notes/foo.md")

    failures = runner.flush()
    runner.close()

    assert len(failures) == 1
    assert str(failures[0]) == "index unavailable"


def test_durable_kb_projector_resumes_from_global_checkpoint() -> None:
    event_store = InMemoryEventStore()
    checkpoints = InMemoryCheckpointStore()
    updated: list[str] = []
    event_store.append_to_stream(
        "note:workspace:foo",
        [NoteUpdated(note_name="Foo", note_path="/notes/foo.md")],
    )
    first = DurableKnowledgeBaseProjector(
        event_store=event_store,
        checkpoint_store=checkpoints,
        update_note=updated.append,
    )
    first.start()
    first.run_to_end()
    first.close()

    event_store.append_to_stream(
        "note:workspace:bar",
        [NoteUpdated(note_name="Bar", note_path="/notes/bar.md")],
    )
    second = DurableKnowledgeBaseProjector(
        event_store=event_store,
        checkpoint_store=checkpoints,
        update_note=updated.append,
    )
    second.start()
    second.run_to_end()

    assert updated == ["/notes/foo.md", "/notes/bar.md"]
