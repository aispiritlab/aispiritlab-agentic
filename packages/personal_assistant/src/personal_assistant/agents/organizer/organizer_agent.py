from __future__ import annotations

from agentic.history import History
from agentic.models import ModelProvider
from agentic.message import SystemMessage
from agentic.metadata import Description
from agentic.core_agent import CoreAgentic
from .tools import toolset as organizer_toolset
from personal_assistant.settings import settings


class OrganizerAgent(CoreAgentic):
    """Dedicated workflow for PARA note organization."""

    description = Description(
        agent_name="organizer",
        description="Classifies notes with PARA heuristics and applies the proper tag.",
        capabilities=("organize", "classify", "tag-note", "para"),
    )

    _WELCOME_MESSAGE = "Cześć! Sklasyfikuję notatkę metodą PARA i dodam odpowiedni tag."
    _EMPTY_MESSAGE_RESPONSE = "Brak danych notatki do sklasyfikowania."
    _model_provider = ModelProvider(settings.model_name)

    def __init__(self, *args, **kwargs) -> None:
        if "toolsets" not in kwargs:
            kwargs["toolsets"] = [organizer_toolset]
        super().__init__(*args, **kwargs)

    def start(self) -> str:
        self._agent.history = History()
        self._agent.history.add(SystemMessage(self._WELCOME_MESSAGE))
        return self._WELCOME_MESSAGE

    def call(self, user_message: str) -> str:
        if not user_message.strip():
            return self._EMPTY_MESSAGE_RESPONSE

        return self.respond(user_message).output
