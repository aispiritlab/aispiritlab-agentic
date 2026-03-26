from __future__ import annotations

from agentic.history import History
from agentic.models import ModelProvider
from agentic.message import SystemMessage
from agentic.metadata import Description
from agentic.core_agent import CoreAgentic
from .commands import AddNoteCommand, EditNoteCommand
from .events import CreatedNote, NoteUpdated
from .tools import toolset as manage_notes_toolset, _get_vault_path, _normalize_note_name
from personal_assistant.settings import settings
from structlog import get_logger

logger = get_logger(__name__)


class ManageNotesAgent(CoreAgentic):
    """Notes workflow for managing note CRUD + list operations."""

    description = Description(
        agent_name="manage_notes",
        description="Handles note operations: create, read, edit and list notes.",
        capabilities=(
            "notes",
            "add-note",
            "create-note",
            "read-note",
            "edit-note",
            "list-notes",
        ),
    )

    _WELCOME_MESSAGE = "Cześć! Mogę dodać notatkę, edytować notatkę, odczytać notatkę albo wyświetlić listę notatek. Co chcesz zrobić?"
    _EMPTY_MESSAGE_RESPONSE = "Proszę wpisać wiadomość."
    _model_provider = ModelProvider(settings.model_name)

    def __init__(self, *args, **kwargs) -> None:
        if "toolsets" not in kwargs:
            kwargs["toolsets"] = [manage_notes_toolset]
        super().__init__(*args, **kwargs)

    def start(self) -> str:
        self._agent.history = History()
        self._agent.history.add(SystemMessage(self._WELCOME_MESSAGE))
        return self._WELCOME_MESSAGE

    @staticmethod
    def _resolve_note_path(note_name: str) -> str:
        vault_path = _get_vault_path()
        if vault_path is None:
            return ""
        normalized = _normalize_note_name(note_name)
        return str(vault_path / f"{normalized}.md")

    def call(
        self, user_message: str
    ) -> tuple[str, CreatedNote, NoteUpdated] | tuple[str, NoteUpdated] | str:
        if not user_message.strip():
            return self._EMPTY_MESSAGE_RESPONSE
        response = self.respond(user_message)
        if not response.result.tool_calls:
            return response.output

        tool_call = response.result.tool_calls[0]
        command = self._agent.toolsets.parse_tool(tool_call)
        if command is None:
            return response.output

        tool_result = response.tool_results[0] if response.tool_results else self._agent.toolsets.execute(command)

        match command:
            case AddNoteCommand(note_name=name, note=content):
                logger.info(f"Add note result: {command}")
                note_updated = NoteUpdated(
                    note_name=name,
                    note_path=self._resolve_note_path(name),
                )
                return tool_result.output, CreatedNote(
                    note_name=name,
                    note_content=content,
                ), note_updated
            case EditNoteCommand(note_name=name):
                logger.info(f"Edit note result: {command}")
                return tool_result.output, NoteUpdated(
                    note_name=name,
                    note_path=self._resolve_note_path(name),
                )
            case _:
                return tool_result.output
