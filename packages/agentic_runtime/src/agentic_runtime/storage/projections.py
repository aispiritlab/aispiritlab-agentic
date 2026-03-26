from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import orjson

from agentic_runtime.messaging.messages import Message


@dataclass(frozen=True, slots=True)
class MessageRow:
    event_id: str
    message_id: str
    runtime_id: str
    turn_id: str
    reply_to_message_id: str | None
    kind: str
    event_type: str
    role: str | None
    scope: str
    domain: str
    source: str
    target: str | None
    name: str | None
    text: str | None
    payload_json: bytes | None
    sequence_no: int | None
    chunk_index: int | None
    chunk_count: int | None
    tool_call_id: str | None
    agent_run_id: str | None
    prompt_name: str | None
    prompt_hash: str | None
    status: str | None
    content_sha256: str | None
    trace_id: str | None
    created_at_ns: int


@dataclass(frozen=True, slots=True)
class ConversationRecordRow:
    message_id: str
    runtime_id: str
    turn_id: str
    reply_to_message_id: str | None
    kind: str
    event_type: str
    role: str
    domain: str
    source: str
    target: str | None
    name: str | None
    text: str | None
    payload_json: bytes | None
    sequence_no: int | None
    tool_call_id: str | None
    agent_run_id: str | None
    prompt_name: str | None
    prompt_hash: str | None
    status: str | None
    content_sha256: str | None
    trace_id: str | None
    created_at_ns: int


class Projection(Protocol):
    can_handle: tuple[type[Message], ...]

    def handle(self, event: Message) -> MessageRow: ...


def handle_projections(
    projections: list[Projection],
    message: Message,
) -> MessageRow | None:
    for projection in projections:
        if isinstance(message, projection.can_handle):
            return projection.handle(message)
    return None


def _to_payload_json(payload: object) -> bytes | None:
    if payload in (None, {}):
        return None
    return orjson.dumps(payload)


def row_to_conversation_record(row: MessageRow) -> ConversationRecordRow | None:
    if row.scope != "canonical":
        return None
    if row.role not in {"system", "user", "assistant", "tool"}:
        return None
    return ConversationRecordRow(
        message_id=row.message_id,
        runtime_id=row.runtime_id,
        turn_id=row.turn_id,
        reply_to_message_id=row.reply_to_message_id,
        kind=row.kind,
        event_type=row.event_type,
        role=row.role,
        domain=row.domain,
        source=row.source,
        target=row.target,
        name=row.name,
        text=row.text,
        payload_json=row.payload_json,
        sequence_no=row.sequence_no,
        tool_call_id=row.tool_call_id,
        agent_run_id=row.agent_run_id,
        prompt_name=row.prompt_name,
        prompt_hash=row.prompt_hash,
        status=row.status,
        content_sha256=row.content_sha256,
        trace_id=row.trace_id,
        created_at_ns=row.created_at_ns,
    )


@dataclass(frozen=True)
class GenericProjection:
    can_handle: tuple[type[Message], ...] = (Message,)

    def handle(self, event: Message) -> MessageRow:
        return MessageRow(
            event_id=event.event_id,
            message_id=event.message_id,
            runtime_id=event.runtime_id,
            turn_id=event.turn_id,
            reply_to_message_id=event.reply_to_message_id,
            kind=event.kind,
            event_type=type(event).__name__,
            role=event.role or None,
            scope=event.scope,
            domain=event.domain,
            source=event.source,
            target=event.target,
            name=event.name,
            text=event.text,
            payload_json=_to_payload_json(event.payload),
            sequence_no=event.sequence_no,
            chunk_index=event.chunk_index,
            chunk_count=event.chunk_count,
            tool_call_id=event.tool_call_id,
            agent_run_id=event.agent_run_id,
            prompt_name=event.prompt_name,
            prompt_hash=event.prompt_hash,
            status=event.status,
            content_sha256=event.content_sha256,
            trace_id=event.trace_id,
            created_at_ns=time.time_ns(),
        )


DEFAULT_PROJECTIONS: list[Projection] = [GenericProjection()]
