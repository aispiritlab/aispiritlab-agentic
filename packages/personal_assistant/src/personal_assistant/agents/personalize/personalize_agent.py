from __future__ import annotations

from agentic.core_agent import CoreAgentic
from agentic.metadata import Description


class WrongPath(ValueError):
    pass

def extract_posix_token(text: str) -> str | None:
    stop = set(" \t\r\n\"'<>[](){};,")
    n = len(text)

    i = text.find("/")
    if i == -1:
        return None

    j = i + 1
    while j < n and text[j] not in stop:
        j += 1

    token = text[i:j]  # e.g. "/home/ts/trst" or "/ania" or "/matsrt/"

    # if it's just one segment (only one "/"), drop the leading slash
    if token.count("/") < 2:
        stripped = token.lstrip("/")
        # but if it ends with "/" (e.g. "/matsrt/"), treat as invalid
        if stripped.endswith("/"):
            raise WrongPath(f"WrongPath: {token}")
        return stripped

    return token


class PersonalizeAgent(CoreAgentic):
    """Minimal model-first onboarding agent."""

    description = Description(
        agent_name="personalize",
        description="Handles user onboarding and personalization setup.",
        capabilities=("personalization", "onboarding", "profile"),
    )
    _START_SIGNAL = "[START_PERSONALIZATION]"
    _EMPTY_MESSAGE_RESPONSE = "Proszę wpisać wiadomość."
    _FINISH_MESSAGE = "Dziękuję za onbording!"

    def start(self) -> str:
        response = self.respond(self._START_SIGNAL)
        return response.output

    def call(self, user_message: str) -> str:
        if not user_message.strip():
            return self._EMPTY_MESSAGE_RESPONSE
        return self.respond(user_message).output
