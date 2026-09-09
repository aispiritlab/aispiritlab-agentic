"""The distributed transport, over Apache Iggy through the Laser SDK.

What every caller above this module sees is unchanged from the Redis Streams
transport it replaces: publish a record, consume a target as a named group,
ack one entry, reclaim what an idle consumer never acked. Underneath, the
mapping is:

| Redis Streams                    | Iggy / Laser                                |
|----------------------------------|---------------------------------------------|
| a key prefix                     | one Iggy **stream**, named for the prefix   |
| `<prefix>:messages:<target>`     | topic `messages.<target>` on that stream    |
| `XADD`                           | a producer send, keyed by conversation      |
| `XREADGROUP` with `>`            | a consumer group poll                       |
| entry id `<ms>-<seq>`            | `<partition>-<offset + 1>`                  |
| `XACK` of one entry              | a stored offset — see `PartitionLedger`    |
| `XAUTOCLAIM`                     | redelivery from the ledger, then the group  |
| `MAXLEN` trimming                | the topic's message expiry                  |

Three of those rows are not a rename, and each is a decision:

**An offset is not an ack.** Redis acks one entry and leaves the rest of the
pending list alone; Iggy stores one scalar offset per partition, and storing
the offset of the message just handled would step straight over one that failed
and was left for a retry. `PartitionLedger` keeps every delivered offset until
it is acked and stores only the contiguous run below the lowest unacked one.
Everything above that watermark is redelivered — the same at-least-once outcome
`XAUTOCLAIM` produced for a consumer that came back under a new name.

**There is no key-value side.** Laser's `kv` needs a managed backend, and this
runs against plain Apache Iggy, so the agent registry is a fold over the
control and health topics rather than two hashes. See `registry.py`.

**One partition, on purpose.** A target's messages are one ordered sequence,
which is what a Redis stream gave the service handling them. Raising
`partitions` buys parallelism per target and gives up that order; the ledger is
already per-partition, but `read_messages` reads partition 0 only, so the topic
a chat client tails must stay at one.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import CancelledError
import contextlib
from dataclasses import dataclass, field
import json
import threading
import time
from typing import Any
import uuid

from laser_sdk import Consumer, Laser, Producer, Topic

from agentic.workflow.messages import normalize_recorded_message
from agentic_runtime.distributed.contracts import AgentHeartbeat, AgentRegistration
from agentic_runtime.distributed.loop import BackgroundLoop, LoopClosedError
from agentic_runtime.distributed.serialization import deserialize_record, serialize_record
from agentic_runtime.messaging.messages import Message

DEFAULT_CONNECTION_STRING = "iggy:iggy@127.0.0.1:8090"

# Before every entry, in the spelling every caller already passes.
BEGINNING = "0-0"

# The registry's two topics are folded in order, so they are never partitioned.
_REGISTRY_PARTITIONS = 1

# How long a poll that already holds a record waits for the next one.
_DRAIN_GRACE_SECONDS = 0.02

# Slack above a poll's own deadline, so a caller waits on the loop rather than
# forever if the loop stops answering.
_CALL_MARGIN_SECONDS = 5.0

# How long "what is at the end of this topic" waits before answering "nothing".
_HEAD_TIMEOUT_SECONDS = 0.5


__all__ = [
    "BEGINNING",
    "DEFAULT_CONNECTION_STRING",
    "ConsumedRecord",
    "JoinedGroup",
    "LaserTransport",
    "MalformedRecord",
    "PartitionLedger",
    "PendingEntry",
    "TopicFold",
    "TopicReader",
    "TransportClosedError",
    "entry_id_of",
    "normalize_distributed_message",
    "parse_entry_id",
]


class TransportClosedError(RuntimeError):
    """Raised when a transport is used after `close`.

    Worth its own name. A consume loop driven from a thread that outlives the
    transport would otherwise surface a bare `CancelledError` from inside the
    Laser SDK, which reads like a bug in the broker client rather than a
    service that was stopped in the wrong order.
    """


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


def entry_id_of(partition: int, offset: int) -> str:
    """Iggy's `(partition, offset)` in Redis' `<ms>-<seq>` shape.

    The offset is written one-based so `"0-0"` keeps the meaning every caller
    gives it — before every entry — which partition 0 at offset 0 would
    otherwise be indistinguishable from.
    """
    return f"{partition}-{offset + 1}"


def parse_entry_id(entry_id: str) -> tuple[int, int]:
    """The `(partition, offset)` behind an entry id. Raises on anything else."""
    partition, _, sequence = entry_id.partition("-")
    return int(partition), int(sequence) - 1


@dataclass(slots=True)
class PendingEntry:
    """One delivered, not-yet-committed record."""

    payload: str
    delivered_at_ns: int
    acked: bool = False


class PartitionLedger:
    """Redis' pending-entries list for one Iggy partition.

    Delivery records an offset here; `ack` marks it; `watermark` advances over
    the contiguous acked run and reports the offset that may be stored. A gap —
    a message that failed and is waiting for its retry — holds the watermark
    where it is, which is the whole reason this exists.
    """

    def __init__(self) -> None:
        self._entries: dict[int, PendingEntry] = {}
        self._next: int | None = None

    def deliver(self, offset: int, payload: str, *, now_ns: int) -> None:
        if self._next is None:
            self._next = offset
        self._entries[offset] = PendingEntry(payload=payload, delivered_at_ns=now_ns)

    def ack(self, offset: int) -> bool:
        """Mark *offset* handled. False when it was already acked or unknown."""
        entry = self._entries.get(offset)
        if entry is None or entry.acked:
            return False
        entry.acked = True
        if self._next is not None and offset < self._next:
            # A redelivery of something already committed — the group rebalanced,
            # or a crash replayed it. It must not hold the watermark.
            del self._entries[offset]
        return True

    def watermark(self) -> int | None:
        """The offset to store now, or `None` when the run has not moved."""
        if self._next is None:
            return None
        committed: int | None = None
        while (entry := self._entries.get(self._next)) is not None and entry.acked:
            del self._entries[self._next]
            committed = self._next
            self._next += 1
        return committed

    def idle(self, *, min_idle_ns: int, now_ns: int, limit: int) -> list[tuple[int, str]]:
        """Delivered, unacked and quiet for long enough — and now redelivered."""
        due = sorted(
            offset
            for offset, entry in self._entries.items()
            if not entry.acked and now_ns - entry.delivered_at_ns >= min_idle_ns
        )
        reclaimed: list[tuple[int, str]] = []
        for offset in due[:limit]:
            entry = self._entries[offset]
            entry.delivered_at_ns = now_ns
            reclaimed.append((offset, entry.payload))
        return reclaimed


@dataclass(slots=True)
class JoinedGroup:
    """One joined consumer group and the ledger of what it owes an ack."""

    consumer: Consumer
    ledgers: dict[int, PartitionLedger] = field(default_factory=dict)

    def ledger(self, partition: int) -> PartitionLedger:
        return self.ledgers.setdefault(partition, PartitionLedger())


@dataclass(slots=True)
class TopicReader:
    """A group-less consumer tailing a topic for one caller's cursor."""

    consumer: Consumer
    next_entry_id: str


@dataclass(slots=True)
class TopicFold:
    """A background fold of one topic, for the registry.

    It is stopped by asking rather than by cancelling. `cancel` on the future
    `run_coroutine_threadsafe` hands back reports success the instant it is
    called, so a `close` that waited on it would not wait at all — and would
    stop the loop out from under a fold that still had a consumer open.
    """

    task: Any
    through: str | None = None
    caught_up: bool = False
    stopping: bool = False
    finished: threading.Event = field(default_factory=threading.Event)


class LaserTransport:
    """The Laser-backed transport. Synchronous, over a loop it owns."""

    def __init__(
        self,
        connection_string: str = DEFAULT_CONNECTION_STRING,
        *,
        prefix: str = "agentic",
        partitions: int = 1,
        message_expiry: str = "server_default",
        health_expiry: str = "server_default",
        connect_timeout: float = 15.0,
    ) -> None:
        if partitions < 1:
            raise ValueError("partitions must be positive")
        self._prefix = prefix.strip().rstrip(":")
        if not self._prefix:
            raise ValueError("prefix must not be empty")
        self._partitions = partitions
        self._message_expiry = message_expiry
        # Only the health topic. A registration must outlive every heartbeat
        # after it, or an agent that has been up for an hour drops out of a
        # fold that can still see it saying "alive".
        self._health_expiry = health_expiry
        self._lock = threading.RLock()
        self._loop = BackgroundLoop(name=f"{self._prefix}-laser")
        self._producers: dict[str, Producer] = {}
        self._topics: dict[str, Topic] = {}
        self._groups: dict[tuple[str, str], JoinedGroup] = {}
        self._readers: dict[str, TopicReader] = {}
        self._tails: dict[str, TopicFold] = {}
        self._tail_changed = threading.Condition(self._lock)
        self._closed = False
        try:
            self._laser: Laser = self._run(
                lambda: Laser.connect(connection_string, stream=self._prefix),
                timeout=connect_timeout,
            )
            self._run(lambda: self._laser.stream(self._prefix).ensure(), timeout=connect_timeout)
        except BaseException:
            self._loop.close()
            raise

    # ------------------------------------------------------------------
    # Names
    # ------------------------------------------------------------------

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def laser(self) -> Laser:
        return self._laser

    def control_stream(self) -> str:
        return "control"

    def health_stream(self) -> str:
        return "health"

    def message_stream(self, target: str) -> str:
        return f"messages.{_require_target(target, 'Distributed message target')}"

    def dead_letter_stream(self, target: str) -> str:
        return f"dead-letter.{_require_target(target, 'Dead-letter target')}"

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish_control(self, registration: AgentRegistration) -> str:
        return self._publish(
            self.control_stream(),
            serialize_record(registration),
            key=registration.agent_name,
            partitions=_REGISTRY_PARTITIONS,
        )

    def publish_health(self, heartbeat: AgentHeartbeat) -> str:
        return self._publish(
            self.health_stream(),
            serialize_record(heartbeat),
            key=heartbeat.agent_name,
            partitions=_REGISTRY_PARTITIONS,
            expiry=self._health_expiry,
        )

    def publish_message(self, message: Message) -> str:
        normalized = normalize_distributed_message(message)
        if not normalized.metadata.target:
            raise ValueError("Distributed messages must have a target")
        return self._publish(
            self.message_stream(normalized.metadata.target),
            serialize_record(normalized),
            key=_conversation_key(normalized),
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
            except TypeError, ValueError:
                payload = repr(record)
        return self._publish(
            self.dead_letter_stream(target),
            payload,
            key=target,
            headers={
                "source_stream": source_stream,
                "group": group,
                "entry_id": entry_id,
                "attempts": str(attempts),
                "error": error,
                "recorded_at_ns": str(time.time_ns()),
            },
        )

    # ------------------------------------------------------------------
    # Reading a target with no group — the chat client's reply cursor
    # ------------------------------------------------------------------

    def last_message_id(self, target: str) -> str:
        """The entry at the end of *target*'s topic, or `BEGINNING` if it is empty.

        Iggy will not report a head without consuming, so this is one poll of a
        throwaway `last` consumer — which stores no offset and joins no group,
        so it takes nothing from anybody.

        It is deliberately not a timestamp. A message's timestamp is the
        *broker's*, taken when it lands, and a cursor from this process's clock
        is a few milliseconds of skew away from silently excluding the very
        reply it was opened to wait for.
        """
        topic = self._topic(self.message_stream(target))
        head = self._run(
            lambda: self._head_of(topic), timeout=_HEAD_TIMEOUT_SECONDS + _CALL_MARGIN_SECONDS
        )
        return BEGINNING if head is None else entry_id_of(*head)

    def read_messages(
        self,
        target: str,
        *,
        after_id: str = BEGINNING,
        block_ms: int = 1_000,
        count: int = 10,
    ) -> list[ConsumedRecord]:
        topic_name = self.message_stream(target)
        with self._lock:
            reader = self._readers.get(topic_name)
            stale = reader if reader is not None and reader.next_entry_id != after_id else None
        if reader is None or stale is not None:
            if stale is not None:
                self._shutdown(stale.consumer)
            fresh = TopicReader(
                consumer=self._reader_at(topic_name, after_id), next_entry_id=after_id
            )
            with self._lock:
                self._readers[topic_name] = fresh
            reader = fresh
        consumer = reader.consumer

        records = self._poll(consumer, topic_name, block_ms=block_ms, count=count)
        if records:
            with self._lock:
                current = self._readers.get(topic_name)
                if current is reader:
                    current.next_entry_id = records[-1].entry_id
        return records

    # ------------------------------------------------------------------
    # Reading a target as a group — the service's consume loop
    # ------------------------------------------------------------------

    def ensure_consumer_group(self, target: str, group: str) -> None:
        self._group(self.message_stream(target), group)

    def consume_target(
        self,
        target: str,
        *,
        group: str,
        consumer: str,
        block_ms: int = 1_000,
        count: int = 10,
    ) -> list[ConsumedRecord]:
        del consumer  # Iggy names the member by its group; the ledger is per process.
        topic_name = self.message_stream(target)
        joined = self._group(topic_name, group)
        records = self._poll(
            joined.consumer,
            topic_name,
            block_ms=block_ms,
            count=count,
            group=joined,
        )
        return records

    def autoclaim_pending(
        self,
        target: str,
        *,
        group: str,
        consumer: str,
        min_idle_ms: int = 5_000,
        count: int = 10,
    ) -> list[ConsumedRecord]:
        """Redeliver what was handed out and never acked.

        Redis stole those entries from a dead consumer's pending list. Here the
        ledger is this process's own: a message left for a retry comes back
        from it, and one lost with the process comes back from the group, which
        resumes at the stored watermark and replays everything above it.
        """
        del consumer
        topic_name = self.message_stream(target)
        joined = self._group(topic_name, group)
        now_ns = time.time_ns()
        min_idle_ns = min_idle_ms * 1_000_000
        records: list[ConsumedRecord] = []
        with self._lock:
            for partition, ledger in sorted(joined.ledgers.items()):
                remaining = count - len(records)
                if remaining <= 0:
                    break
                for offset, payload in ledger.idle(
                    min_idle_ns=min_idle_ns, now_ns=now_ns, limit=remaining
                ):
                    records.append(
                        ConsumedRecord(
                            stream=topic_name,
                            entry_id=entry_id_of(partition, offset),
                            record=_record_from(payload),
                        )
                    )
        return records

    def ack(self, stream: str, group: str, entry_id: str) -> int:
        """Mark one entry handled, and store the watermark if it moved."""
        try:
            partition, offset = parse_entry_id(entry_id)
        except ValueError:
            return 0
        with self._lock:
            joined = self._groups.get((stream, group))
            if joined is None:
                return 0
            ledger = joined.ledger(partition)
            if not ledger.ack(offset):
                return 0
            watermark = ledger.watermark()
            consumer = joined.consumer
        if watermark is not None:
            self._run(lambda: consumer.store_offset(watermark, partition=partition))
        return 1

    # ------------------------------------------------------------------
    # Tailing a topic into a fold — the registry
    # ------------------------------------------------------------------

    def tail(self, topic_name: str, *, sink: Callable[[str, str], None]) -> None:
        """Follow *topic_name* from its beginning, handing every payload to *sink*.

        Idempotent per topic. The task lives on the transport's loop and is
        cancelled by `close`, so a caller that folds a topic owns no thread.
        """
        with self._lock:
            if topic_name in self._tails or self._closed:
                return
            self._tails[topic_name] = TopicFold(task=None)
        task = self._loop.submit(lambda: self._follow(topic_name, sink))
        with self._lock:
            self._tails[topic_name].task = task

    def tail_settled(
        self,
        topic_name: str,
        *,
        through: str | None = None,
        timeout: float = 2.0,
    ) -> bool:
        """Wait until the fold has caught up, and read past *through* if given.

        False on timeout. The `through` argument is what makes a registry read
        its own write: the entry id a publish returned is folded before
        `live_agents` answers with it missing.
        """
        deadline = time.monotonic() + timeout
        wanted = parse_entry_id(through)[1] if through else None

        def settled() -> bool:
            tail = self._tails.get(topic_name)
            if tail is None or not tail.caught_up:
                return False
            if wanted is None:
                return True
            return tail.through is not None and parse_entry_id(tail.through)[1] >= wanted

        with self._tail_changed:
            while not settled():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._tail_changed.wait(timeout=remaining)
            return True

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            tails = list(self._tails.values())
            for tail in tails:
                tail.stopping = True
            consumers = [group.consumer for group in self._groups.values()]
            consumers.extend(reader.consumer for reader in self._readers.values())
            self._groups.clear()
            self._readers.clear()
        for tail in tails:
            # The entries stay in `_tails` until each fold has said it is done:
            # that is how it learns it should stop, and how it reports back.
            tail.finished.wait(timeout=2.0)
        with self._lock:
            self._tails.clear()
        for consumer in consumers:
            self._shutdown(consumer)
        self._loop.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run[T](self, work, *, timeout: float | None = None) -> T:
        """Everything this class asks of the loop, so a close reads as one."""
        try:
            return self._loop.run(work, timeout=timeout)
        except (CancelledError, LoopClosedError) as error:
            if self._closed:
                raise TransportClosedError("The Laser transport is closed.") from error
            raise

    def _publish(
        self,
        topic_name: str,
        payload: str,
        *,
        key: str,
        headers: dict[str, str] | None = None,
        partitions: int | None = None,
        expiry: str | None = None,
    ) -> str:
        producer = self._producer(topic_name, partitions=partitions, expiry=expiry)
        response = self._run(lambda: producer.send(payload, headers=headers, key=key))
        confirmations = getattr(response, "confirmations", None) or ()
        if not confirmations:
            return BEGINNING
        confirmed = confirmations[0]
        return entry_id_of(confirmed.partition_id, confirmed.base_offset)

    def _topic(self, topic_name: str, *, partitions: int | None = None) -> Topic:
        with self._lock:
            topic = self._topics.get(topic_name)
            if topic is not None:
                return topic
        resolved = self._run(lambda: self._open_topic(topic_name, partitions or self._partitions))
        with self._lock:
            return self._topics.setdefault(topic_name, resolved)

    async def _open_topic(self, topic_name: str, partitions: int) -> Topic:
        topic = self._laser.stream(self._prefix).topic(topic_name)
        await topic.ensure(partitions)
        return topic

    def _producer(
        self,
        topic_name: str,
        *,
        partitions: int | None = None,
        expiry: str | None = None,
    ) -> Producer:
        with self._lock:
            producer = self._producers.get(topic_name)
            if producer is not None:
                return producer
        topic = self._topic(topic_name, partitions=partitions)
        built = self._run(
            lambda: self._open_producer(
                topic,
                partitions or self._partitions,
                expiry or self._message_expiry,
            )
        )
        with self._lock:
            return self._producers.setdefault(topic_name, built)

    async def _open_producer(self, topic: Topic, partitions: int, expiry: str) -> Producer:
        producer = topic.producer(
            batch_length=1,
            linger_ms=0,
            create_stream=True,
            create_topic=True,
            partitions=partitions,
            message_expiry=expiry,
        )
        await producer.init()
        return producer

    def _group(self, topic_name: str, group: str) -> JoinedGroup:
        with self._lock:
            joined = self._groups.get((topic_name, group))
            if joined is not None:
                return joined
        topic = self._topic(topic_name)
        consumer = self._run(lambda: self._join_group(topic, group))
        with self._lock:
            existing = self._groups.get((topic_name, group))
            if existing is not None:
                self._shutdown(consumer)
                return existing
            created = JoinedGroup(consumer=consumer)
            self._groups[(topic_name, group)] = created
            return created

    async def _join_group(self, topic: Topic, group: str) -> Consumer:
        consumer = topic.consumer_group(
            group,
            batch_length=10,
            polling="next",
            auto_commit="disabled",
            create_group=True,
            auto_join_group=True,
            poll_interval_ms=100,
        )
        await consumer.init()
        return consumer

    def _reader_at(self, topic_name: str, after_id: str) -> Consumer:
        topic = self._topic(topic_name)
        return self._run(lambda: self._open_reader(topic, after_id))

    async def _head_of(self, topic: Topic) -> tuple[int, int] | None:
        """`(partition, offset)` of the last entry, or `None` for an empty topic."""
        consumer = topic.consumer(
            f"head-{uuid.uuid4().hex[:12]}",
            partition=0,
            batch_length=1,
            polling="last",
            auto_commit="disabled",
            poll_interval_ms=100,
            allow_replay=True,
        )
        await consumer.init()
        try:
            message = await asyncio.wait_for(consumer.next(), timeout=_HEAD_TIMEOUT_SECONDS)
        except TimeoutError:
            # Nothing has ever been published here, which is a head of its own.
            return None
        finally:
            with contextlib.suppress(Exception):
                await consumer.shutdown()
        return None if message is None else (message.partition_id, message.offset)

    async def _open_reader(self, topic: Topic, after_id: str) -> Consumer:
        _, offset = parse_entry_id(after_id)
        consumer = topic.consumer(
            f"reader-{uuid.uuid4().hex[:12]}",
            partition=0,
            batch_length=1,
            # `offset` polling is inclusive and the entry id is one-based, so
            # `offset + 1` is exactly the entry after the one the caller last
            # saw — and `"0-0"` lands on 0, the start of the topic.
            polling="offset",
            offset=offset + 1,
            auto_commit="disabled",
            poll_interval_ms=100,
            allow_replay=True,
        )
        await consumer.init()
        return consumer

    def _poll(
        self,
        consumer: Consumer,
        topic_name: str,
        *,
        block_ms: int,
        count: int,
        group: JoinedGroup | None = None,
    ) -> list[ConsumedRecord]:
        deadline = time.monotonic() + block_ms / 1000.0
        records: list[ConsumedRecord] = []
        while len(records) < count:
            # Only the first message is worth waiting for. Holding a handled one
            # back for a second message would cost it the whole block.
            remaining = (deadline - time.monotonic()) if not records else _DRAIN_GRACE_SECONDS
            if remaining <= 0:
                break
            message = self._next(consumer, timeout=remaining)
            if message is None:
                break
            payload = bytes(message.payload).decode("utf-8", errors="replace")
            entry_id = entry_id_of(message.partition_id, message.offset)
            if group is not None:
                with self._lock:
                    group.ledger(message.partition_id).deliver(
                        message.offset, payload, now_ns=time.time_ns()
                    )
            records.append(
                ConsumedRecord(stream=topic_name, entry_id=entry_id, record=_record_from(payload))
            )
        return records

    def _next(self, consumer: Consumer, *, timeout: float) -> Any:
        try:
            return self._run(
                lambda: asyncio.wait_for(consumer.next(), timeout=timeout),
                # The loop enforces the deadline; this only stops a caller
                # hanging forever if the loop itself has stopped answering.
                timeout=timeout + _CALL_MARGIN_SECONDS,
            )
        except TimeoutError, asyncio.CancelledError:
            return None

    async def _follow(self, topic_name: str, sink: Callable[[str, str], None]) -> None:
        topic = self._laser.stream(self._prefix).topic(topic_name)
        await topic.ensure(_REGISTRY_PARTITIONS)
        consumer = topic.consumer(
            f"fold-{uuid.uuid4().hex[:12]}",
            partition=0,
            batch_length=1,
            polling="offset",
            offset=0,
            auto_commit="disabled",
            poll_interval_ms=100,
            allow_replay=True,
        )
        await consumer.init()
        try:
            while not self._tail_stopping(topic_name):
                try:
                    message = await asyncio.wait_for(consumer.next(), timeout=0.25)
                except TimeoutError:
                    # Silence is the only "you are at the tail" signal there is.
                    self._mark_tail(topic_name, through=None, caught_up=True)
                    continue
                if message is None:
                    self._mark_tail(topic_name, through=None, caught_up=True)
                    return
                payload = bytes(message.payload).decode("utf-8", errors="replace")
                sink(entry_id_of(message.partition_id, message.offset), payload)
                self._mark_tail(
                    topic_name,
                    through=entry_id_of(message.partition_id, message.offset),
                    caught_up=False,
                )
        finally:
            with contextlib.suppress(Exception):
                await consumer.shutdown()
            self._finish_tail(topic_name)

    def _tail_stopping(self, topic_name: str) -> bool:
        with self._lock:
            tail = self._tails.get(topic_name)
            return tail is None or tail.stopping

    def _finish_tail(self, topic_name: str) -> None:
        with self._lock:
            tail = self._tails.get(topic_name)
        if tail is not None:
            tail.finished.set()

    def _mark_tail(self, topic_name: str, *, through: str | None, caught_up: bool) -> None:
        with self._tail_changed:
            tail = self._tails.get(topic_name)
            if tail is None:
                return
            if through is not None:
                tail.through = through
            if caught_up:
                tail.caught_up = True
            self._tail_changed.notify_all()

    def _shutdown(self, consumer: Consumer) -> None:
        if self._loop.closed:
            return
        try:
            self._run(lambda: consumer.shutdown(), timeout=2.0)
        # A close must not raise over a socket that is already gone.
        except Exception:
            pass


def _require_target(target: str, label: str) -> str:
    resolved = target.strip()
    if not resolved:
        raise ValueError(f"{label} must not be empty")
    return resolved


def _conversation_key(message: Message) -> str:
    """What keeps one conversation on one partition, the way a run does upstream."""
    metadata = message.metadata
    return (
        metadata.turn_id
        or metadata.correlation_id
        or metadata.session_id
        or metadata.target
        or "default"
    )


def _record_from(payload: str) -> object:
    try:
        return deserialize_record(payload)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return MalformedRecord(
            raw_payload=payload,
            error_type=type(error).__name__,
            error_message=str(error),
        )
