from __future__ import annotations

from dataclasses import dataclass
import time

from agentic_runtime.distributed.contracts import AgentHeartbeat, AgentRegistration
from agentic_runtime.distributed.serialization import deserialize_record, serialize_record
from agentic_runtime.distributed.transport import RedisStreamsTransport


@dataclass(frozen=True, slots=True)
class AgentSnapshot:
    agent_name: str
    capabilities: tuple[str, ...]
    role: str
    consumer_group: str
    status: str
    last_seen_ns: int


class RedisServiceRegistry:
    def __init__(self, transport: RedisStreamsTransport) -> None:
        self._transport = transport

    def register(self, registration: AgentRegistration) -> None:
        self._transport.publish_control(registration)
        self._transport.client.hset(
            self._descriptors_key(),
            registration.agent_name,
            serialize_record(registration),
        )

    def heartbeat(self, heartbeat: AgentHeartbeat) -> None:
        self._transport.publish_health(heartbeat)
        self._transport.client.hset(
            self._heartbeats_key(),
            heartbeat.agent_name,
            serialize_record(heartbeat),
        )

    def live_agents(self, *, max_age_seconds: float) -> list[AgentSnapshot]:
        now_ns = time.time_ns()
        max_age_ns = int(max_age_seconds * 1_000_000_000)
        descriptors = self._transport.client.hgetall(self._descriptors_key())
        heartbeats = self._transport.client.hgetall(self._heartbeats_key())

        snapshots: list[AgentSnapshot] = []
        for agent_name, descriptor_payload in descriptors.items():
            registration = deserialize_record(descriptor_payload)
            if not isinstance(registration, AgentRegistration):
                continue

            heartbeat_payload = heartbeats.get(agent_name)
            if heartbeat_payload is None:
                continue

            heartbeat = deserialize_record(heartbeat_payload)
            if not isinstance(heartbeat, AgentHeartbeat):
                continue

            age_ns = now_ns - heartbeat.emitted_at_ns
            if age_ns > max_age_ns:
                continue

            snapshots.append(
                AgentSnapshot(
                    agent_name=registration.agent_name,
                    capabilities=registration.capabilities,
                    role=registration.role,
                    consumer_group=registration.consumer_group,
                    status=heartbeat.status,
                    last_seen_ns=heartbeat.emitted_at_ns,
                )
            )

        return sorted(snapshots, key=lambda snapshot: snapshot.agent_name)

    def find_by_capability(
        self,
        capability: str,
        *,
        max_age_seconds: float,
    ) -> AgentSnapshot | None:
        for agent in self.live_agents(max_age_seconds=max_age_seconds):
            if capability in agent.capabilities:
                return agent
        return None

    def _descriptors_key(self) -> str:
        return f"{self._transport.prefix}:registry:descriptors"

    def _heartbeats_key(self) -> str:
        return f"{self._transport.prefix}:registry:heartbeats"
