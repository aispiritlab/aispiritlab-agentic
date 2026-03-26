from __future__ import annotations

from dataclasses import dataclass

from agentic.tools import Command


@dataclass(frozen=True)
class AddNoteCommand(Command):
    note_name: str
    note: str


@dataclass(frozen=True)
class EditNoteCommand(Command):
    note_name: str
    note: str


@dataclass(frozen=True)
class GetNoteCommand(Command):
    note_name: str


@dataclass(frozen=True)
class ListNotesCommand(Command):
    pass
