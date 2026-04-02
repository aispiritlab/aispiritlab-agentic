"""OpenAI-compatible API schema types."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ChatCompletionRequest:
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 512
    presence_penalty: float = 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "presence_penalty": self.presence_penalty,
        }


@dataclass(frozen=True, slots=True)
class UsageInfo:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ChatChoice:
    index: int = 0
    message: ChatMessage = field(default_factory=lambda: ChatMessage(role="assistant", content=""))
    finish_reason: str = "stop"


@dataclass(frozen=True, slots=True)
class ChatCompletionResponse:
    id: str = ""
    choices: list[ChatChoice] = field(default_factory=list)
    usage: UsageInfo = field(default_factory=UsageInfo)

    @classmethod
    def from_dict(cls, data: dict) -> ChatCompletionResponse:
        choices = [
            ChatChoice(
                index=c.get("index", 0),
                message=ChatMessage(
                    role=c.get("message", {}).get("role", "assistant"),
                    content=c.get("message", {}).get("content", ""),
                ),
                finish_reason=c.get("finish_reason", "stop"),
            )
            for c in data.get("choices", [])
        ]
        usage_data = data.get("usage", {})
        usage = UsageInfo(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )
        return cls(
            id=data.get("id", ""),
            choices=choices,
            usage=usage,
        )
