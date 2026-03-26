from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
import threading
from typing import TYPE_CHECKING

from agentic.workflow.messages import Message
from agentic_runtime.output_handler import WorkflowOutputHandler, workflow_output_handler

from personal_assistant.messaging.events import CreatedNote, NoteDeleted, NoteUpdated

if TYPE_CHECKING:
    from personal_assistant.agents.organizer.organizer_workflow import OrganizerWorkflow


def _lazy_update_note_in_kb(note_path: str) -> None:
    from knowledge_base import update_note_in_kb

    update_note_in_kb(note_path)


def _lazy_delete_note_from_kb(note_path: str) -> None:
    from knowledge_base import delete_note_from_kb

    delete_note_from_kb(note_path)


update_note_in_kb = _lazy_update_note_in_kb
delete_note_from_kb = _lazy_delete_note_from_kb


class KnowledgeBaseTaskRunner:
    def __init__(
        self,
        *,
        resync: Callable[[], object] | None = None,
        update_note: Callable[[str], None] | None = None,
        delete_note: Callable[[str], None] | None = None,
    ) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag")
        self._lock = threading.Lock()
        self._closed = False
        self._resync = resync
        self._update_note = update_note or update_note_in_kb
        self._delete_note = delete_note or delete_note_from_kb

    def _submit(self, fn: Callable[..., object], *args: object) -> Future[object] | None:
        with self._lock:
            if self._closed:
                return None
            return self._executor.submit(fn, *args)

    def submit_resync(self) -> Future[object] | None:
        if self._resync is None:
            return None
        return self._submit(self._resync)

    def submit_update(self, note_path: str) -> Future[object] | None:
        return self._submit(self._update_note, note_path)

    def submit_delete(self, note_path: str) -> Future[object] | None:
        return self._submit(self._delete_note, note_path)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            executor = self._executor

        executor.shutdown(wait=True, cancel_futures=False)


def build_organizer_output_handler(
    organizer_workflow: OrganizerWorkflow,
) -> WorkflowOutputHandler:
    """Replaces _on_workflow_event: delegates CreatedNote to organizer."""

    def _handle(message: Message) -> str | None:
        return organizer_workflow.handle(message)

    return workflow_output_handler(
        can_handle=(CreatedNote,),
        each_message=_handle,
        name="organizer",
    )


def build_rag_output_handler(task_runner: KnowledgeBaseTaskRunner) -> WorkflowOutputHandler:
    """Schedule KB sync work on the runtime-owned task runner."""

    def _handle(message: Message) -> str | None:
        if isinstance(message, NoteUpdated):
            task_runner.submit_update(message.note_path)
            return None
        if isinstance(message, NoteDeleted):
            task_runner.submit_delete(message.note_path)
            return None
        return None

    return workflow_output_handler(
        can_handle=(NoteUpdated, NoteDeleted),
        each_message=_handle,
        name="rag",
    )
