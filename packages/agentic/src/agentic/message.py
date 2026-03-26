from __future__ import annotations

from typing import Any


class Message:
    def __init__(
        self,
        content: str | None = None,
        structural_message: dict[str, Any] | None = None,
    ):
        self.content = content
        self.structural_message = structural_message

    def get_text(self) -> str:
        if self.content is not None:
            return self.content
        if isinstance(self.structural_message, dict):
            content = self.structural_message.get("content")
            if isinstance(content, str):
                return content
        return ""

    def __str__(self) -> str:
        return self.get_text()

    def __repr__(self) -> str:
        return self.get_text()


class SystemMessage(Message):
    role = "model"

    def __init__(self, content: str | None = None):
        super().__init__(content)

    def as_turn(self) -> str:
        return f"<start_of_turn>{self.role}\n{self.content}\n<end_of_turn>"


class ToolMessage(Message):
    role = "tool"

    def __init__(self, content: str | None = None):
        super().__init__(content)

    def as_turn(self) -> str:
        return f"<tool_call_response>{self.content}\n</tool_call_response>"


class UserMessage(Message):
    role = "user"

    def __init__(self, content: str | None = None):
        super().__init__(content)

    def as_turn(self) -> str:
        return f"<start_of_turn>{self.role}\n{self.content}\n<end_of_turn>"
