from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, replace
from typing import Any

from agentic.observability import TraceSnapshot, build_trace_snapshot


@dataclass(frozen=True, slots=True, kw_only=True)
class MessageMetadata:
    runtime_id: str = ""
    session_id: str = ""
    turn_id: str = ""
    reply_to_message_id: str | None = None
    domain: str = ""
    source: str = ""
    target: str | None = None
    role: str = ""
    scope: str = "canonical"
    chunk_index: int | None = None
    chunk_count: int | None = None
    tool_call_id: str | None = None
    agent_run_id: str | None = None
    prompt_name: str | None = None
    prompt_hash: str | None = None
    status: str | None = None
    stream_name: str = ""
    stream_position: int | None = None
    global_position: int | None = None
    trace: TraceSnapshot | None = None
    attempt_no: int | None = None
    loop_iteration: int | None = None

    @property
    def trace_id(self) -> str | None:
        return None if self.trace is None or not self.trace.trace_id else self.trace.trace_id

    @property
    def span_id(self) -> str | None:
        return None if self.trace is None or not self.trace.span_id else self.trace.span_id

    @property
    def parent_span_id(self) -> str | None:
        return None if self.trace is None or not self.trace.parent_span_id else self.trace.parent_span_id

    @property
    def span_name(self) -> str | None:
        return None if self.trace is None or not self.trace.span_name else self.trace.span_name

    @property
    def span_type(self) -> str | None:
        return None if self.trace is None or not self.trace.span_type else self.trace.span_type


@dataclass(frozen=True, slots=True, kw_only=True)
class RecordedMessageMetadata(MessageMetadata):
    event_id: str = ""
    message_id: str = ""
    sequence_no: int | None = None
    content_sha256: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationData:
    role: str = ""
    text: str | None = None
    name: str | None = None
    payload: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Message:
    kind: str = "message"
    type: str = "message"
    data: Any = field(default_factory=dict)
    metadata: MessageMetadata = field(default_factory=RecordedMessageMetadata)

    def with_metadata(self, **updates: Any) -> Message:
        trace = updates.pop("trace", self.metadata.trace)
        if any(
            key in updates
            for key in ("session_id", "trace_id", "span_id", "parent_span_id", "span_name", "span_type")
        ):
            trace = build_trace_snapshot(
                trace,
                session_id=str(updates.pop("session_id", self.metadata.session_id) or ""),
                trace_id=updates.pop("trace_id", self.metadata.trace_id),
                span_id=updates.pop("span_id", self.metadata.span_id),
                parent_span_id=updates.pop("parent_span_id", self.metadata.parent_span_id),
                span_name=updates.pop("span_name", self.metadata.span_name),
                span_type=updates.pop("span_type", self.metadata.span_type),
            )
        return replace(self, metadata=replace(self.metadata, trace=trace, **updates))

    def with_data(self, **updates: Any) -> Message:
        if isinstance(self.data, dict):
            data = dict(self.data)
            data.update(updates)
            return replace(self, data=data)
        if not dataclasses.is_dataclass(self.data):
            raise TypeError(
                f"with_data() requires data to be a dict or dataclass, got {type(self.data).__name__}"
            )
        return replace(self, data=replace(self.data, **updates))


@dataclass(frozen=True, slots=True, kw_only=True)
class UserMessage(Message):
    kind: str = "conversation"
    type: str = "user_message"
    data: ConversationData = field(default_factory=lambda: ConversationData(role="user", text=""))
    metadata: RecordedMessageMetadata = field(default_factory=RecordedMessageMetadata)


Conversation = UserMessage


@dataclass(frozen=True, slots=True, kw_only=True)
class AssistantMessage(Message):
    kind: str = "assistant_message"
    type: str = "assistant_message"
    data: ConversationData = field(default_factory=lambda: ConversationData(role="assistant", text=""))
    metadata: RecordedMessageMetadata = field(default_factory=RecordedMessageMetadata)


@dataclass(frozen=True, slots=True, kw_only=True)
class PromptSnapshot(Message):
    kind: str = "prompt_snapshot"
    type: str = "prompt_snapshot"
    data: ConversationData = field(default_factory=lambda: ConversationData(role="system", text="", payload={}))
    metadata: RecordedMessageMetadata = field(default_factory=RecordedMessageMetadata)


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolResultMessage(Message):
    kind: str = "tool_result"
    type: str = "tool_result"
    data: ConversationData = field(default_factory=lambda: ConversationData(role="tool", text="", payload={}))
    metadata: RecordedMessageMetadata = field(default_factory=RecordedMessageMetadata)


@dataclass(frozen=True, slots=True, kw_only=True)
class UserCommand(Message):
    kind: str = "command"
    type: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    metadata: RecordedMessageMetadata = field(default_factory=RecordedMessageMetadata)


Command = UserCommand


@dataclass(frozen=True, slots=True, kw_only=True)
class Event(Message):
    kind: str = "event"
    type: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    metadata: RecordedMessageMetadata = field(default_factory=RecordedMessageMetadata)


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCallEvent(Event):
    kind: str = "tool_call"
    type: str = "tool_call"


@dataclass(frozen=True, slots=True, kw_only=True)
class TurnStarted(Event):
    kind: str = "turn_started"
    type: str = "turn_started"
    metadata: RecordedMessageMetadata = field(default_factory=lambda: RecordedMessageMetadata(scope="transport"))


@dataclass(frozen=True, slots=True, kw_only=True)
class TurnCompleted(Event):
    kind: str = "turn_completed"
    type: str = "turn_completed"
    metadata: RecordedMessageMetadata = field(default_factory=lambda: RecordedMessageMetadata(scope="transport"))


@dataclass(frozen=True, slots=True, kw_only=True)
class MessageStarted(Event):
    kind: str = "message_started"
    type: str = "message_started"
    metadata: RecordedMessageMetadata = field(default_factory=lambda: RecordedMessageMetadata(scope="transport"))


@dataclass(frozen=True, slots=True, kw_only=True)
class MessageChunk(Message):
    kind: str = "message_chunk"
    type: str = "message_chunk"
    data: ConversationData = field(default_factory=lambda: ConversationData(role="assistant", text=""))
    metadata: RecordedMessageMetadata = field(default_factory=lambda: RecordedMessageMetadata(scope="transport"))


@dataclass(frozen=True, slots=True, kw_only=True)
class MessageCompleted(Event):
    kind: str = "message_completed"
    type: str = "message_completed"
    metadata: RecordedMessageMetadata = field(
        default_factory=lambda: RecordedMessageMetadata(scope="transport", status="completed")
    )
