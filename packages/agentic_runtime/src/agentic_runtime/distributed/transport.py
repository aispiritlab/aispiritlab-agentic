from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any

from redis import Redis
from redis.exceptions import ResponseError

from agentic.workflow.messages import normalize_recorded_message
from agentic_runtime.distributed.contracts import AgentHeartbeat, AgentRegistration
from agentic_runtime.distributed.serialization import deserialize_record, serialize_record
from agentic_runtime.messaging.messages import Message


@dataclass(frozen=True, slots=True)
class ConsumedRecord:
    stream: str
    entry_id: str
    record: object


@dataclass(frozen=True, slots=True)
class MalformedRecord:
    raw_payload: str
    error_type: str
    error_message: str


def normalize_distributed_message(message: Message) -> Message:
    return normalize_recorded_message(message)


class RedisStreamsTransport:
    def __init__(
        self,
        redis_url: str,
        *,
        prefix: str = "agentic",
        stream_maxlen: int | None = None,
    ) -> None:
        if stream_maxlen is not None and stream_maxlen < 1:
            raise ValueError("stream_maxlen must be positive when configured")
        self._client = Redis.from_url(redis_url, decode_responses=True)
        self._prefix = prefix.rstrip(":")
        self._stream_maxlen = stream_maxlen

    @property
    def client(self) -> Redis:
        return self._client

    @property
    def prefix(self) -> str:
        return self._prefix

    def control_stream(self) -> str:
        return f"{self._prefix}:control"

    def health_stream(self) -> str:
        return f"{self._prefix}:health"

    def message_stream(self, target: str) -> str:
        resolved_target = target.strip()
        if not resolved_target:
            raise ValueError("Distributed message target must not be empty")
        return f"{self._prefix}:messages:{resolved_target}"

    def dead_letter_stream(self, target: str) -> str:
        resolved_target = target.strip()
        if not resolved_target:
            raise ValueError("Dead-letter target must not be empty")
        return f"{self._prefix}:dead-letter:{resolved_target}"

    def publish_control(self, registration: AgentRegistration) -> str:
        return self._xadd(
            self.control_stream(),
            {"payload": serialize_record(registration)},
        )

    def publish_health(self, heartbeat: AgentHeartbeat) -> str:
        return self._xadd(
            self.health_stream(),
            {"payload": serialize_record(heartbeat)},
        )

    def publish_message(self, message: Message) -> str:
        normalized = normalize_distributed_message(message)
        if not normalized.metadata.target:
            raise ValueError("Distributed messages must have a target")

        return self._xadd(
            self.message_stream(normalized.metadata.target),
            {"payload": serialize_record(normalized)},
        )

    def publish_dead_letter(
        self,
        *,
        target: str,
        source_stream: str,
        group: str,
        entry_id: str,
        record: object,
        error: str,
        attempts: int,
    ) -> str:
        if isinstance(record, MalformedRecord):
            payload = record.raw_payload
        else:
            try:
                payload = serialize_record(record)
            except (TypeError, ValueError):
                payload = repr(record)
        return self._xadd(
            self.dead_letter_stream(target),
            {
                "source_stream": source_stream,
                "group": group,
                "entry_id": entry_id,
                "attempts": str(attempts),
                "error": error,
                "payload": payload,
                "recorded_at_ns": str(time.time_ns()),
            },
        )

    def last_message_id(self, target: str) -> str:
        entries = self._client.xrevrange(self.message_stream(target), count=1)
        if not entries:
            return "0-0"
        return entries[0][0]

    def read_messages(
        self,
        target: str,
        *,
        after_id: str = "0-0",
        block_ms: int = 1_000,
        count: int = 10,
    ) -> list[ConsumedRecord]:
        response = self._client.xread(
            {self.message_stream(target): after_id},
            block=block_ms,
            count=count,
        )
        return self._deserialize_records(response)

    def ensure_consumer_group(self, target: str, group: str) -> None:
        try:
            self._client.xgroup_create(
                self.message_stream(target),
                group,
                id="0",
                mkstream=True,
            )
        except ResponseError as error:
            if "BUSYGROUP" not in str(error):
                raise

    def consume_target(
        self,
        target: str,
        *,
        group: str,
        consumer: str,
        block_ms: int = 1_000,
        count: int = 10,
    ) -> list[ConsumedRecord]:
        self.ensure_consumer_group(target, group)
        response = self._client.xreadgroup(
            groupname=group,
            consumername=consumer,
            streams={self.message_stream(target): ">"},
            block=block_ms,
            count=count,
        )
        return self._deserialize_records(response)

    def autoclaim_pending(
        self,
        target: str,
        *,
        group: str,
        consumer: str,
        min_idle_ms: int = 5_000,
        count: int = 10,
    ) -> list[ConsumedRecord]:
        """Claim pending messages from any idle consumer in the group.

        Uses XAUTOCLAIM to transfer ownership of messages that have been
        idle for at least *min_idle_ms* to the specified *consumer*.
        After a crash the new process has a different PID (consumer name),
        so ``xreadgroup("0")`` would return nothing — ``xautoclaim`` steals
        messages from *any* idle consumer, solving the PID-change problem.
        """
        self.ensure_consumer_group(target, group)
        _next_id, entries, _deleted = self._client.xautoclaim(
            name=self.message_stream(target),
            groupname=group,
            consumername=consumer,
            min_idle_time=min_idle_ms,
            start_id="0-0",
            count=count,
        )
        stream_name = self.message_stream(target)
        records: list[ConsumedRecord] = []
        for entry_id, payload in entries:
            if not isinstance(payload, dict):
                record: object = MalformedRecord(
                    raw_payload=repr(payload),
                    error_type="InvalidEnvelope",
                    error_message="Redis stream entry must be a mapping",
                )
            else:
                record = self._record_from_payload(payload)
            records.append(
                ConsumedRecord(
                    stream=stream_name,
                    entry_id=entry_id,
                    record=record,
                )
            )
        return records

    def ack(self, stream: str, group: str, entry_id: str) -> int:
        return int(self._client.xack(stream, group, entry_id))

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _deserialize_records(
        response: list[tuple[str, list[tuple[str, dict[str, Any]]]]],
    ) -> list[ConsumedRecord]:
        records: list[ConsumedRecord] = []
        for stream, entries in response:
            for entry_id, payload in entries:
                record = RedisStreamsTransport._record_from_payload(payload)
                records.append(
                    ConsumedRecord(
                        stream=stream,
                        entry_id=entry_id,
                        record=record,
                    )
                )
        return records

    def _xadd(self, stream: str, fields: dict[str, str]) -> str:
        if self._stream_maxlen is None:
            return str(self._client.xadd(stream, fields))
        return str(
            self._client.xadd(
                stream,
                fields,
                maxlen=self._stream_maxlen,
                approximate=True,
            )
        )

    @staticmethod
    def _record_from_payload(payload: dict[str, Any]) -> object:
        serialized = payload.get("payload")
        if not isinstance(serialized, str):
            return MalformedRecord(
                raw_payload=repr(serialized),
                error_type="InvalidEnvelope",
                error_message="Redis stream entry is missing a string payload",
            )
        return RedisStreamsTransport._deserialize_record(serialized)

    @staticmethod
    def _deserialize_record(serialized: str) -> object:
        try:
            return deserialize_record(serialized)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            return MalformedRecord(
                raw_payload=serialized,
                error_type=type(error).__name__,
                error_message=str(error),
            )
