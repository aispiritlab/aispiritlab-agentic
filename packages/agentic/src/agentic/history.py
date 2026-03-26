from typing import Sequence

from agentic.message import SystemMessage, Message, UserMessage


class History:
    """Conversation history stored as role-tagged turns."""

    def __init__(self) -> None:
        self._history: list[Message] = []

    def add(self, message: Message) -> None:
        self._history.append(message)

    def get(self, length: int) -> Sequence[Message]:
        return self._history[-length:]

    def store(self, user_message: str, assistant_message: str) -> str:
        self._history.append(UserMessage(user_message))
        self._history.append(SystemMessage(assistant_message))
        return assistant_message

    def conversation_text(self, last_messages: int = 20) -> str:
        return "\n".join(msg.as_turn() for msg in self._history[-last_messages:])

    def clear(self) -> None:
        self._history.clear()
