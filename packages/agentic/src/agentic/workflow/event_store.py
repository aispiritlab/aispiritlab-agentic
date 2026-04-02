from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import reduce
from typing import Any, Callable, Generic, Literal, Protocol, Sequence, TypeVar

from agentic.workflow.errors import ConcurrencyConflictError
from agentic.workflow.messages import Message


S = TypeVar("S")
M = TypeVar("M", bound=Message)

logger = logging.getLogger(__name__)

type StreamPosition = int
type ExpectedVersion = StreamPosition | Literal["STREAM_EXISTS", "STREAM_DOES_NOT_EXIST", "NO_CONCURRENCY_CHECK"]


STREAM_EXISTS: ExpectedVersion = "STREAM_EXISTS"
STREAM_DOES_NOT_EXIST: ExpectedVersion = "STREAM_DOES_NOT_EXIST"
NO_CONCURRENCY_CHECK: ExpectedVersion = "NO_CONCURRENCY_CHECK"


# ---------------------------------------------------------------------------
# After-commit hooks
# ---------------------------------------------------------------------------

type AfterCommitHook = Callable[[str, Sequence[Message]], None]
"""Called after messages are durably stored: (stream_name, committed_messages) -> None."""


# ---------------------------------------------------------------------------
# Inline projections
# ---------------------------------------------------------------------------


class InlineProjection(Protocol):
    """Runs synchronously during append_to_stream, before after-commit hooks."""

    def project(self, stream_name: str, events: Sequence[Message]) -> None: ...


# ---------------------------------------------------------------------------
# Schema versioning (upcasting)
# ---------------------------------------------------------------------------

type Upcaster = Callable[[Message], Message]
"""Transforms an old message schema to the current schema during read."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReadStreamResult(Generic[M]):
    events: tuple[M, ...]
    current_version: StreamPosition
    stream_exists: bool


@dataclass(frozen=True, slots=True)
class AggregateStreamResult(Generic[S]):
    state: S
    current_version: StreamPosition
    stream_exists: bool


@dataclass(frozen=True, slots=True)
class AppendResult:
    next_version: StreamPosition


# ---------------------------------------------------------------------------
# EventStore protocol
# ---------------------------------------------------------------------------


class EventStore(Protocol):
    def read_stream(
        self,
        stream_name: str,
        *,
        from_position: StreamPosition = 0,
        max_count: int | None = None,
    ) -> ReadStreamResult[Message]: ...

    def aggregate_stream(
        self,
        stream_name: str,
        *,
        evolve: Callable[[Any, Message], Any],
        initial_state: Callable[[], Any],
        from_position: StreamPosition = 0,
    ) -> AggregateStreamResult[Any]: ...

    def append_to_stream(
        self,
        stream_name: str,
        events: Sequence[Message],
        *,
        expected_version: ExpectedVersion = NO_CONCURRENCY_CHECK,
    ) -> AppendResult: ...

    def stream_exists(self, stream_name: str) -> bool: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_expected_version(
    stream_name: str,
    current: StreamPosition,
    expected: ExpectedVersion,
) -> None:
    if expected == NO_CONCURRENCY_CHECK:
        return
    if expected == STREAM_DOES_NOT_EXIST:
        if current != 0:
            raise ConcurrencyConflictError(stream_name, expected, current)
        return
    if expected == STREAM_EXISTS:
        if current == 0:
            raise ConcurrencyConflictError(stream_name, expected, current)
        return
    if isinstance(expected, int) and current != expected:
        raise ConcurrencyConflictError(stream_name, expected, current)


def _apply_upcasters(messages: Sequence[Message], upcasters: Sequence[Upcaster]) -> tuple[Message, ...]:
    if not upcasters:
        return tuple(messages)
    result: list[Message] = []
    for message in messages:
        upcasted = message
        for upcaster in upcasters:
            upcasted = upcaster(upcasted)
        result.append(upcasted)
    return tuple(result)


# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------


class InMemoryEventStore:
    """In-memory event store with named streams, version tracking, and concurrency control.

    Supports:
    - After-commit hooks: side effects that run after events are stored
    - Inline projections: synchronous projections during append
    - Upcasters: schema migration on read
    """

    def __init__(
        self,
        *,
        after_commit_hooks: Sequence[AfterCommitHook] = (),
        inline_projections: Sequence[InlineProjection] = (),
        upcasters: Sequence[Upcaster] = (),
    ) -> None:
        self._streams: dict[str, list[Message]] = {}
        self._after_commit_hooks = list(after_commit_hooks)
        self._inline_projections = list(inline_projections)
        self._upcasters = list(upcasters)

    def add_after_commit_hook(self, hook: AfterCommitHook) -> None:
        self._after_commit_hooks.append(hook)

    def add_inline_projection(self, projection: InlineProjection) -> None:
        self._inline_projections.append(projection)

    def add_upcaster(self, upcaster: Upcaster) -> None:
        self._upcasters.append(upcaster)

    def read_stream(
        self,
        stream_name: str,
        *,
        from_position: StreamPosition = 0,
        max_count: int | None = None,
    ) -> ReadStreamResult[Message]:
        events = self._streams.get(stream_name, [])
        sliced = events[from_position:]
        if max_count is not None:
            sliced = sliced[:max_count]
        upcasted = _apply_upcasters(sliced, self._upcasters)
        return ReadStreamResult(
            events=upcasted,
            current_version=len(events),
            stream_exists=stream_name in self._streams,
        )

    def aggregate_stream(
        self,
        stream_name: str,
        *,
        evolve: Callable[[Any, Message], Any],
        initial_state: Callable[[], Any],
        from_position: StreamPosition = 0,
    ) -> AggregateStreamResult[Any]:
        result = self.read_stream(stream_name, from_position=from_position)
        state = reduce(evolve, result.events, initial_state())
        return AggregateStreamResult(
            state=state,
            current_version=result.current_version,
            stream_exists=result.stream_exists,
        )

    def append_to_stream(
        self,
        stream_name: str,
        events: Sequence[Message],
        *,
        expected_version: ExpectedVersion = NO_CONCURRENCY_CHECK,
    ) -> AppendResult:
        existed_before = stream_name in self._streams
        previous_events = list(self._streams.get(stream_name, []))
        current = len(previous_events)
        _check_expected_version(stream_name, current, expected_version)

        if stream_name not in self._streams:
            self._streams[stream_name] = []
        self._streams[stream_name].extend(events)

        try:
            # Inline projections run before hooks (synchronous, same "transaction")
            for projection in self._inline_projections:
                projection.project(stream_name, events)
        except Exception:
            if existed_before:
                self._streams[stream_name] = previous_events
            else:
                self._streams.pop(stream_name, None)
            raise

        # After-commit hooks (side effects, failures are logged but don't roll back)
        for hook in self._after_commit_hooks:
            try:
                hook(stream_name, events)
            except Exception:
                logger.exception("After-commit hook failed for stream '%s'", stream_name)

        return AppendResult(next_version=len(self._streams[stream_name]))

    def stream_exists(self, stream_name: str) -> bool:
        return stream_name in self._streams
