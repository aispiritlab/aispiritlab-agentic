from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, Protocol, Sequence

from agentic.workflow.event_store import EventStore, ReadStreamResult, StreamPosition
from agentic.workflow.messages import Message


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


class StartFrom(StrEnum):
    BEGINNING = "BEGINNING"
    END = "END"
    CURRENT = "CURRENT"


class CheckpointStore(Protocol):
    def read(self, processor_id: str) -> StreamPosition | None: ...
    def store(self, processor_id: str, position: StreamPosition) -> None: ...


class InMemoryCheckpointStore:
    def __init__(self) -> None:
        self._checkpoints: dict[str, StreamPosition] = {}

    def read(self, processor_id: str) -> StreamPosition | None:
        return self._checkpoints.get(processor_id)

    def store(self, processor_id: str, position: StreamPosition) -> None:
        self._checkpoints[processor_id] = position


# ---------------------------------------------------------------------------
# Message Processor
# ---------------------------------------------------------------------------


type BatchHandler = Callable[[Sequence[Message]], None]


@dataclass(frozen=True, slots=True)
class ProcessorConfig:
    processor_id: str
    start_from: StartFrom = StartFrom.BEGINNING
    batch_size: int = 100


@dataclass(slots=True)
class MessageProcessor:
    """Processes events from an event store stream with checkpointing.

    Reads events from a named stream, dispatches batches to the handler,
    and persists the checkpoint after each batch. On restart, resumes
    from the last checkpoint.
    """

    config: ProcessorConfig
    handler: BatchHandler
    checkpoint_store: CheckpointStore
    _active: bool = field(default=False, init=False)

    @property
    def processor_id(self) -> str:
        return self.config.processor_id

    @property
    def is_active(self) -> bool:
        return self._active

    def start(self, event_store: EventStore, stream_name: str) -> StreamPosition:
        """Resolve the starting position based on config and checkpoint."""
        self._active = True
        position = self._resolve_start_position(event_store, stream_name)
        if self.checkpoint_store.read(self.config.processor_id) is None:
            self.checkpoint_store.store(self.config.processor_id, position)
        return position

    def process(self, event_store: EventStore, stream_name: str) -> int:
        """Read a batch from the stream and process it. Returns count of events processed."""
        if not self._active:
            return 0

        from_position = self._resolve_start_position(event_store, stream_name)
        result = event_store.read_stream(
            stream_name,
            from_position=from_position,
            max_count=self.config.batch_size,
        )

        if not result.events:
            return 0

        self.handler(result.events)

        new_position = from_position + len(result.events)
        self.checkpoint_store.store(self.config.processor_id, new_position)

        return len(result.events)

    def run_to_end(self, event_store: EventStore, stream_name: str) -> int:
        """Process all available events in batches until caught up. Returns total processed."""
        total = 0
        while True:
            processed = self.process(event_store, stream_name)
            if processed == 0:
                break
            total += processed
        return total

    def close(self) -> None:
        self._active = False

    def _resolve_start_position(
        self,
        event_store: EventStore,
        stream_name: str,
    ) -> StreamPosition:
        checkpoint = self.checkpoint_store.read(self.config.processor_id)

        match self.config.start_from:
            case StartFrom.CURRENT:
                if checkpoint is not None:
                    return checkpoint
                result = event_store.read_stream(stream_name)
                return result.current_version
            case StartFrom.BEGINNING:
                return checkpoint if checkpoint is not None else 0
            case StartFrom.END:
                if checkpoint is not None:
                    return checkpoint
                result = event_store.read_stream(stream_name)
                return result.current_version
