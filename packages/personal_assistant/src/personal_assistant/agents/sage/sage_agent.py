from __future__ import annotations

from agentic.core_agent import CoreAgentic
from agentic.history import History
from agentic.message import SystemMessage
from agentic.metadata import Description
from agentic.models import ModelProvider

from personal_assistant.settings import settings


class SageAgent(CoreAgentic):
    """Decision-support workflow based on a 7-step decision process."""

    description = Description(
        agent_name="sage",
        description="Wspiera podejmowanie decyzji metodą 7 kroków.",
        capabilities=(
            "decision-making",
            "7-step-process",
            "alternatives-analysis",
            "implementation-plan",
        ),
    )

    _WELCOME_MESSAGE = (
        "Cześć! Jestem Sage. Pomogę Ci podjąć decyzję metodą 7 kroków. "
        "Jaki temat chcesz przeanalizować?"
    )
    _EMPTY_MESSAGE_RESPONSE = "Opisz decyzję, którą chcesz podjąć."
    _model_provider = ModelProvider(settings.thinkink_model)

    def start(self) -> str:
        self._agent.history = History()
        self._agent.history.add(SystemMessage(self._WELCOME_MESSAGE))
        return self._WELCOME_MESSAGE

    def call(self, user_message: str) -> str:
        if not user_message.strip():
            return self._EMPTY_MESSAGE_RESPONSE

        response = self.respond(user_message)
        return response.output.replace("<think>", "").replace("</think>", "").strip()
