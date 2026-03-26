from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True, kw_only=True)
class Message:
    kind: str = "message"
    runtime_id: str = ""
    turn_id: str = ""
    event_id: str = ""
    message_id: str = ""
    reply_to_message_id: str | None = None
    domain: str = ""
    source: str = ""
    target: str | None = None
    role: str = ""
    scope: str = "canonical"
    name: str | None = None
    text: str | None = None
    payload: dict[str, Any] | None = None
    sequence_no: int | None = None
    chunk_index: int | None = None
    chunk_count: int | None = None
    tool_call_id: str | None = None
    agent_run_id: str | None = None
    prompt_name: str | None = None
    prompt_hash: str | None = None
    status: str | None = None
    content_sha256: str | None = None
    trace_id: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class UserMessage(Message):
    kind: str = "conversation"
    role: str = "user"
    text: str = ""


Conversation = UserMessage


@dataclass(frozen=True, slots=True, kw_only=True)
class AssistantMessage(Message):
    kind: str = "assistant_message"
    role: str = "assistant"
    text: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class PromptSnapshot(Message):
    kind: str = "prompt_snapshot"
    role: str = "system"
    text: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolResultMessage(Message):
    kind: str = "tool_result"
    role: str = "tool"
    name: str = ""
    text: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class UserCommand(Message):
    kind: str = "command"
    name: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def type(self) -> str:
        return self.name

    @property
    def data(self) -> dict[str, Any]:
        return dict(self.payload)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "runtime_id": self.runtime_id,
            "turn_id": self.turn_id,
            "message_id": self.message_id,
            "domain": self.domain,
            "source": self.source,
            "scope": self.scope,
            "sequence_no": self.sequence_no,
        }


Command = UserCommand


@dataclass(frozen=True, slots=True, kw_only=True)
class Event(Message):
    kind: str = "event"
    name: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def type(self) -> str:
        return self.name

    @property
    def data(self) -> dict[str, Any]:
        return dict(self.payload)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "runtime_id": self.runtime_id,
            "turn_id": self.turn_id,
            "message_id": self.message_id,
            "reply_to_message_id": self.reply_to_message_id,
            "domain": self.domain,
            "source": self.source,
            "target": self.target,
            "role": self.role,
            "scope": self.scope,
            "sequence_no": self.sequence_no,
            "tool_call_id": self.tool_call_id,
            "agent_run_id": self.agent_run_id,
            "prompt_name": self.prompt_name,
            "prompt_hash": self.prompt_hash,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCallEvent(Event):
    kind: str = "tool_call"
    role: str = "assistant"
    name: str = "tool_call"
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class TurnStarted(Event):
    kind: str = "turn_started"
    scope: str = "transport"
    name: str = "turn_started"
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class TurnCompleted(Event):
    kind: str = "turn_completed"
    scope: str = "transport"
    name: str = "turn_completed"
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class MessageStarted(Event):
    kind: str = "message_started"
    scope: str = "transport"
    name: str = "message_started"
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class MessageChunk(Message):
    kind: str = "message_chunk"
    scope: str = "transport"
    text: str = ""
    chunk_index: int = 0
    chunk_count: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class MessageCompleted(Event):
    kind: str = "message_completed"
    scope: str = "transport"
    name: str = "message_completed"
    status: str = "completed"
    payload: dict[str, Any] = field(default_factory=dict)
