from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from redis import Redis
from redis.exceptions import ResponseError

from agentic_runtime.distributed.contracts import AgentHeartbeat, AgentRegistration
from agentic_runtime.distributed.serialization import deserialize_record, serialize_record
from agentic_runtime.messaging.messages import Message


@dataclass(frozen=True, slots=True)
class ConsumedRecord:
    stream: str
    entry_id: str
    record: object


class RedisStreamsTransport:
    def __init__(self, redis_url: str, *, prefix: str = "agentic") -> None:
        self._client = Redis.from_url(redis_url, decode_responses=True)
        self._prefix = prefix.rstrip(":")

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

    def publish_control(self, registration: AgentRegistration) -> str:
        return self._client.xadd(
            self.control_stream(),
            {"payload": serialize_record(registration)},
        )

    def publish_health(self, heartbeat: AgentHeartbeat) -> str:
        return self._client.xadd(
            self.health_stream(),
            {"payload": serialize_record(heartbeat)},
        )

    def publish_message(self, message: Message) -> str:
        if not message.target:
            raise ValueError("Distributed messages must have a target")

        return self._client.xadd(
            self.message_stream(message.target),
            {"payload": serialize_record(message)},
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
                serialized = payload.get("payload")
                if not isinstance(serialized, str):
                    continue
                records.append(
                    ConsumedRecord(
                        stream=stream,
                        entry_id=entry_id,
                        record=deserialize_record(serialized),
                    )
                )
        return records
