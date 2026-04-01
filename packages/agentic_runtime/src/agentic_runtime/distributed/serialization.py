from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from typing import Any

from agentic_runtime.distributed.contracts import AgentHeartbeat, AgentRegistration
from agentic_runtime.messaging.messages import (
    AssistantMessage,
    Command,
    ConversationData,
    Event,
    Message,
    MessageMetadata,
    MessageChunk,
    MessageCompleted,
    MessageStarted,
    PromptSnapshot,
    RecordedMessageMetadata,
    ToolCallEvent,
    ToolResultMessage,
    TurnCompleted,
    TurnStarted,
    UserCommand,
    UserMessage,
)
from agentic.observability import TraceSnapshot

_BASE_SERIALIZABLE_TYPES = (
    AgentHeartbeat,
    AgentRegistration,
    AssistantMessage,
    Command,
    Event,
    Message,
    MessageChunk,
    MessageCompleted,
    MessageStarted,
    PromptSnapshot,
    ToolCallEvent,
    ToolResultMessage,
    TurnCompleted,
    TurnStarted,
    UserCommand,
    UserMessage,
)
_TYPE_MAP = {record_type.__name__: record_type for record_type in _BASE_SERIALIZABLE_TYPES}


def register_record_types(*record_types: type[object]) -> None:
    for record_type in record_types:
        _TYPE_MAP[record_type.__name__] = record_type


def serialize_record(record: object) -> str:
    if not is_dataclass(record):
        raise TypeError(f"Cannot serialize non-dataclass record: {type(record)!r}")

    payload = asdict(record)
    payload["__type__"] = type(record).__name__
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def deserialize_record(value: str | bytes) -> Any:
    raw = value.decode("utf-8") if isinstance(value, bytes) else value
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("Serialized record must decode to a JSON object")

    record_type_name = payload.pop("__type__", None)
    if not isinstance(record_type_name, str):
        raise KeyError("Serialized record is missing __type__")

    record_type = _TYPE_MAP[record_type_name]
    if isinstance(payload.get("metadata"), dict):
        metadata_payload = dict(payload["metadata"])
        trace_payload = metadata_payload.get("trace")
        if isinstance(trace_payload, dict):
            metadata_payload["trace"] = TraceSnapshot(**trace_payload)
        metadata_type = MessageMetadata
        if any(key in metadata_payload for key in ("event_id", "message_id", "sequence_no", "content_sha256")):
            metadata_type = RecordedMessageMetadata
        payload["metadata"] = metadata_type(**metadata_payload)
    if record_type in {
        AssistantMessage,
        MessageChunk,
        PromptSnapshot,
        ToolResultMessage,
        UserMessage,
    } and isinstance(payload.get("data"), dict):
        payload["data"] = ConversationData(**payload["data"])
    return record_type(**payload)
