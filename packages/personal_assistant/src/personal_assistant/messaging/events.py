from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from agentic.workflow.messages import Event, Message, RecordedMessageMetadata
from agentic_runtime.distributed.serialization import register_record_types


def _metadata_with_updates(metadata: RecordedMessageMetadata, **updates: Any) -> RecordedMessageMetadata:
    values = {field.name: getattr(metadata, field.name) for field in fields(RecordedMessageMetadata)}
    values.update(updates)
    return RecordedMessageMetadata(**values)


@dataclass(frozen=True, slots=True, kw_only=True, init=False)
class CreatedNote(Event):
    def __init__(
        self,
        *,
        data: dict[str, Any] | None = None,
        note_name: str = "",
        note_content: str = "",
        metadata: RecordedMessageMetadata | None = None,
    ) -> None:
        super().__init__(
            kind="created_note",
            type="created_note",
            data=data or {"note_name": note_name, "note_content": note_content},
            metadata=metadata or RecordedMessageMetadata(domain="manage_notes", target="organizer"),
        )

    @property
    def note_name(self) -> str:
        return str(self.data.get("note_name", ""))

    @property
    def note_content(self) -> str:
        return str(self.data.get("note_content", ""))

    def with_metadata(self, **updates: Any) -> Message:
        return CreatedNote(
            data=dict(self.data),
            metadata=_metadata_with_updates(self.metadata, **updates),
        )

    def with_data(self, **updates: Any) -> Message:
        return CreatedNote(data={**self.data, **updates}, metadata=self.metadata)


@dataclass(frozen=True, slots=True, kw_only=True, init=False)
class NoteUpdated(Event):
    def __init__(
        self,
        *,
        data: dict[str, Any] | None = None,
        note_name: str = "",
        note_path: str = "",
        metadata: RecordedMessageMetadata | None = None,
    ) -> None:
        super().__init__(
            kind="note_updated",
            type="note_updated",
            data=data or {"note_name": note_name, "note_path": note_path},
            metadata=metadata or RecordedMessageMetadata(domain="manage_notes", target="rag"),
        )

    @property
    def note_name(self) -> str:
        return str(self.data.get("note_name", ""))

    @property
    def note_path(self) -> str:
        return str(self.data.get("note_path", ""))

    def with_metadata(self, **updates: Any) -> Message:
        return NoteUpdated(
            data=dict(self.data),
            metadata=_metadata_with_updates(self.metadata, **updates),
        )

    def with_data(self, **updates: Any) -> Message:
        return NoteUpdated(data={**self.data, **updates}, metadata=self.metadata)


@dataclass(frozen=True, slots=True, kw_only=True, init=False)
class NoteDeleted(Event):
    def __init__(
        self,
        *,
        data: dict[str, Any] | None = None,
        note_name: str = "",
        note_path: str = "",
        metadata: RecordedMessageMetadata | None = None,
    ) -> None:
        super().__init__(
            kind="note_deleted",
            type="note_deleted",
            data=data or {"note_name": note_name, "note_path": note_path},
            metadata=metadata or RecordedMessageMetadata(domain="manage_notes", target="rag"),
        )

    @property
    def note_name(self) -> str:
        return str(self.data.get("note_name", ""))

    @property
    def note_path(self) -> str:
        return str(self.data.get("note_path", ""))

    def with_metadata(self, **updates: Any) -> Message:
        return NoteDeleted(
            data=dict(self.data),
            metadata=_metadata_with_updates(self.metadata, **updates),
        )

    def with_data(self, **updates: Any) -> Message:
        return NoteDeleted(data={**self.data, **updates}, metadata=self.metadata)


register_record_types(CreatedNote, NoteUpdated, NoteDeleted)
