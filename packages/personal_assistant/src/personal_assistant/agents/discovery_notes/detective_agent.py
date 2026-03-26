from __future__ import annotations

from agentic.history import History
from agentic.models import ModelProvider
from agentic.message import SystemMessage
from agentic.metadata import Description
from agentic.core_agent import CoreAgentic
from .tools import toolset as detective_toolset
from personal_assistant.settings import settings


class DiscoveryNotesAgent(CoreAgentic):
    """Dedicated note workflow for semantic search."""

    description = Description(
        agent_name="discovery_notes",
        description="Dedicated note workflow for semantic search and note retrieval.",
        capabilities=("search", "semantic-search", "list-notes", "get-note"),
    )

    _WELCOME_MESSAGE = "Cześć! Pomogę Ci znaleźć notatki semantycznie. Jakie hasło mam wyszukać?"
    _EMPTY_MESSAGE_RESPONSE = "Proszę wpisać hasło do wyszukania."
    _model_provider = ModelProvider(settings.model_name)

    def __init__(self, *args, **kwargs) -> None:
        if "toolsets" not in kwargs:
            kwargs["toolsets"] = [detective_toolset]
        super().__init__(*args, **kwargs)

    def start(self) -> str:
        self._agent.history = History()
        self._agent.history.add(SystemMessage(self._WELCOME_MESSAGE))
        return self._WELCOME_MESSAGE

    def call(self, user_message: str) -> str:
        if not user_message.strip():
            return self._EMPTY_MESSAGE_RESPONSE

        return self.respond(user_message).output
