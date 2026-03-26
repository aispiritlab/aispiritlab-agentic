from __future__ import annotations

from dataclasses import dataclass

from agentic.workflow.messages import Event


@dataclass(frozen=True, slots=True, kw_only=True)
class CreatedNote(Event):
    kind: str = "created_note"
    domain: str = "manage_notes"
    target: str | None = "organizer"
    name: str = "created_note"
    note_name: str = ""
    note_content: str = ""

    def __post_init__(self) -> None:
        if not self.payload:
            object.__setattr__(
                self,
                "payload",
                {
                    "note_name": self.note_name,
                    "note_content": self.note_content,
                },
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class NoteUpdated(Event):
    kind: str = "note_updated"
    domain: str = "manage_notes"
    target: str | None = "rag"
    name: str = "note_updated"
    note_name: str = ""
    note_path: str = ""

    def __post_init__(self) -> None:
        if not self.payload:
            object.__setattr__(
                self,
                "payload",
                {
                    "note_name": self.note_name,
                    "note_path": self.note_path,
                },
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class NoteDeleted(Event):
    kind: str = "note_deleted"
    domain: str = "manage_notes"
    target: str | None = "rag"
    name: str = "note_deleted"
    note_name: str = ""
    note_path: str = ""

    def __post_init__(self) -> None:
        if not self.payload:
            object.__setattr__(
                self,
                "payload",
                {
                    "note_name": self.note_name,
                    "note_path": self.note_path,
                },
            )
