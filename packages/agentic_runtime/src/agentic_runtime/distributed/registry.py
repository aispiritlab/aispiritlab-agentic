"""Who is running, folded from the control and health topics.

Redis gave this two hashes and let every agent overwrite its own field in
place. Apache Iggy has no key-value surface — Laser's `kv` answers only against
a managed backend, and this runs against plain Iggy — so what was already on
the wire is now the whole story: a registration is appended to the control
topic, a heartbeat to the health topic, and the current picture is a fold over
both.

Two consequences are worth naming. The health topic takes one record per agent
per heartbeat interval, so it is created with a message expiry: a heartbeat
older than the liveness TTL cannot make an agent live, and keeping it only
lengthens the replay at start-up. And a fold is asynchronous, so `live_agents`
waits for the tail to reach what *this* process published before it answers —
registering and then not appearing in your own listing is the one confusing
thing a log-backed registry would otherwise do.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time

from agentic_runtime.distributed.contracts import AgentHeartbeat, AgentRegistration
from agentic_runtime.distributed.serialization import deserialize_record
from agentic_runtime.distributed.transport import LaserTransport

# How long `live_agents` waits for the fold to catch up with this process.
SETTLE_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class AgentSnapshot:
    agent_name: str
    capabilities: tuple[str, ...]
    role: str
    consumer_group: str
    status: str
    last_seen_ns: int


class LaserServiceRegistry:
    """The agent registry, over the Laser transport's control and health topics."""

    def __init__(
        self,
        transport: LaserTransport,
        *,
        settle_timeout_seconds: float = SETTLE_TIMEOUT_SECONDS,
    ) -> None:
        self._transport = transport
        self._settle_timeout_seconds = settle_timeout_seconds
        self._lock = threading.Lock()
        self._registrations: dict[str, AgentRegistration] = {}
        self._heartbeats: dict[str, AgentHeartbeat] = {}
        self._published: dict[str, str] = {}

    def register(self, registration: AgentRegistration) -> None:
        self._remember(
            self._transport.control_stream(),
            self._transport.publish_control(registration),
        )

    def heartbeat(self, heartbeat: AgentHeartbeat) -> None:
        self._remember(
            self._transport.health_stream(),
            self._transport.publish_health(heartbeat),
        )

    def live_agents(self, *, max_age_seconds: float) -> list[AgentSnapshot]:
        self._settle()
        now_ns = time.time_ns()
        max_age_ns = int(max_age_seconds * 1_000_000_000)

        with self._lock:
            registrations = dict(self._registrations)
            heartbeats = dict(self._heartbeats)

        snapshots: list[AgentSnapshot] = []
        for agent_name, registration in registrations.items():
            heartbeat = heartbeats.get(agent_name)
            if heartbeat is None:
                continue
            if now_ns - heartbeat.emitted_at_ns > max_age_ns:
                continue
            snapshots.append(
                AgentSnapshot(
                    agent_name=registration.agent_name,
                    # A tuple went onto the log and JSON hands back a list, so
                    # the snapshot's declared type is only true if it says so.
                    capabilities=tuple(registration.capabilities),
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

    # ------------------------------------------------------------------
    # The fold
    # ------------------------------------------------------------------

    def _remember(self, topic: str, entry_id: str) -> None:
        with self._lock:
            self._published[topic] = entry_id

    def _settle(self) -> None:
        control = self._transport.control_stream()
        health = self._transport.health_stream()
        self._transport.tail(control, sink=self._fold)
        self._transport.tail(health, sink=self._fold)
        with self._lock:
            published = dict(self._published)
        for topic in (control, health):
            self._transport.tail_settled(
                topic,
                through=published.get(topic),
                timeout=self._settle_timeout_seconds,
            )

    def _fold(self, entry_id: str, payload: str) -> None:
        """Apply one record. Called on the transport's loop, never by a caller."""
        del entry_id
        try:
            record = deserialize_record(payload)
        # One unreadable record must not stop the fold.
        except Exception:
            return
        with self._lock:
            if isinstance(record, AgentRegistration):
                self._registrations[record.agent_name] = record
            elif isinstance(record, AgentHeartbeat):
                known = self._heartbeats.get(record.agent_name)
                if known is None or record.emitted_at_ns >= known.emitted_at_ns:
                    self._heartbeats[record.agent_name] = record
