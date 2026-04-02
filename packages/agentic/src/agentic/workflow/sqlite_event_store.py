from __future__ import annotations

import logging
from pathlib import Path
import sqlite3
import time
from typing import Any, Callable, Sequence

from agentic.workflow.event_store import (
    NO_CONCURRENCY_CHECK,
    STREAM_DOES_NOT_EXIST,
    STREAM_EXISTS,
    AggregateStreamResult,
    AppendResult,
    ExpectedVersion,
    InlineProjection,
    ReadStreamResult,
    StreamPosition,
    Upcaster,
    _apply_upcasters,
    _check_expected_version,
)
from agentic.workflow.messages import Message
from agentic.workflow.processor import CheckpointStore
from agentic.workflow.serialization import deserialize_record, serialize_record

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS event_streams (
    stream_name TEXT PRIMARY KEY,
    current_version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS event_messages (
    global_position INTEGER PRIMARY KEY,
    stream_name TEXT NOT NULL,
    stream_position INTEGER NOT NULL,
    payload_json BLOB NOT NULL,
    created_at_ns INTEGER NOT NULL,
    UNIQUE(stream_name, stream_position)
);

CREATE INDEX IF NOT EXISTS idx_event_messages_stream_position
ON event_messages(stream_name, stream_position);

CREATE TABLE IF NOT EXISTS processor_checkpoints (
    processor_id TEXT PRIMARY KEY,
    position INTEGER NOT NULL
);
"""


def _default_path() -> Path:
    return Path(__file__).resolve().parents[4] / "data" / "workflow_event_store.sqlite3"


def _resolve_path(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    return _default_path()


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA synchronous=NORMAL;")
    connection.execute("PRAGMA busy_timeout=5000;")
    return connection


def _ensure_schema(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as connection:
        connection.executescript(_SCHEMA)
        connection.commit()


class SQLiteEventStore:
    """SQLite-backed event store for durable workflow and domain streams."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        after_commit_hooks: Sequence[Callable[[str, Sequence[Message]], None]] = (),
        inline_projections: Sequence[InlineProjection] = (),
        upcasters: Sequence[Upcaster] = (),
    ) -> None:
        self._path = _resolve_path(path)
        _ensure_schema(self._path)
        self._after_commit_hooks = list(after_commit_hooks)
        self._inline_projections = list(inline_projections)
        self._upcasters = list(upcasters)

    @property
    def path(self) -> Path:
        return self._path

    def add_after_commit_hook(self, hook: Callable[[str, Sequence[Message]], None]) -> None:
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
        with _connect(self._path) as connection:
            version = self._current_version(connection, stream_name)
            stream_exists = version > 0
            if max_count is None:
                rows = connection.execute(
                    """
                    SELECT stream_position, global_position, payload_json
                    FROM event_messages
                    WHERE stream_name = ? AND stream_position >= ?
                    ORDER BY stream_position
                    """,
                    (stream_name, from_position),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT stream_position, global_position, payload_json
                    FROM event_messages
                    WHERE stream_name = ? AND stream_position >= ?
                    ORDER BY stream_position
                    LIMIT ?
                    """,
                    (stream_name, from_position, max_count),
                ).fetchall()

        messages = tuple(
            self._bind_positions(
                message=self._deserialize_message(payload_json),
                stream_name=stream_name,
                stream_position=stream_position,
                global_position=global_position,
            )
            for stream_position, global_position, payload_json in rows
        )
        return ReadStreamResult(
            events=_apply_upcasters(messages, self._upcasters),
            current_version=version,
            stream_exists=stream_exists,
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
        state = initial_state()
        for message in result.events:
            state = evolve(state, message)
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
        if not events:
            with _connect(self._path) as connection:
                return AppendResult(next_version=self._current_version(connection, stream_name))

        serialized = tuple(serialize_record(message) for message in events)
        with _connect(self._path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current_version = self._current_version(connection, stream_name)
            _check_expected_version(stream_name, current_version, expected_version)

            try:
                for projection in self._inline_projections:
                    projection.project(stream_name, events)
            except Exception:
                connection.rollback()
                raise

            next_global_position = self._next_global_position(connection)
            created_at_ns = time.time_ns()
            connection.executemany(
                """
                INSERT INTO event_messages (
                    global_position,
                    stream_name,
                    stream_position,
                    payload_json,
                    created_at_ns
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        next_global_position + index,
                        stream_name,
                        current_version + index,
                        payload,
                        created_at_ns,
                    )
                    for index, payload in enumerate(serialized)
                ],
            )

            next_version = current_version + len(events)
            connection.execute(
                """
                INSERT INTO event_streams (stream_name, current_version)
                VALUES (?, ?)
                ON CONFLICT(stream_name) DO UPDATE SET current_version = excluded.current_version
                """,
                (stream_name, next_version),
            )
            connection.commit()

        for hook in self._after_commit_hooks:
            try:
                hook(stream_name, events)
            except Exception:
                logger.exception("After-commit hook failed for stream '%s'", stream_name)

        return AppendResult(next_version=next_version)

    def stream_exists(self, stream_name: str) -> bool:
        with _connect(self._path) as connection:
            return self._current_version(connection, stream_name) > 0

    @staticmethod
    def _current_version(connection: sqlite3.Connection, stream_name: str) -> int:
        row = connection.execute(
            "SELECT current_version FROM event_streams WHERE stream_name = ?",
            (stream_name,),
        ).fetchone()
        if row is None:
            return 0
        return int(row[0])

    @staticmethod
    def _next_global_position(connection: sqlite3.Connection) -> int:
        row = connection.execute(
            "SELECT COALESCE(MAX(global_position), -1) + 1 FROM event_messages"
        ).fetchone()
        return int(row[0]) if row is not None else 0

    @staticmethod
    def _deserialize_message(payload_json: bytes | str) -> Message:
        message = deserialize_record(payload_json)
        if not isinstance(message, Message):
            raise TypeError(f"Expected serialized workflow message, got {type(message)!r}")
        return message

    @staticmethod
    def _bind_positions(
        *,
        message: Message,
        stream_name: str,
        stream_position: int,
        global_position: int,
    ) -> Message:
        return message.with_metadata(
            stream_name=stream_name,
            stream_position=stream_position,
            global_position=global_position,
        )


class SQLiteCheckpointStore(CheckpointStore):
    """Persistent checkpoint store sharing the SQLite event-store file."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = _resolve_path(path)
        _ensure_schema(self._path)

    def read(self, processor_id: str) -> StreamPosition | None:
        with _connect(self._path) as connection:
            row = connection.execute(
                "SELECT position FROM processor_checkpoints WHERE processor_id = ?",
                (processor_id,),
            ).fetchone()
        if row is None:
            return None
        return int(row[0])

    def store(self, processor_id: str, position: StreamPosition) -> None:
        with _connect(self._path) as connection:
            connection.execute(
                """
                INSERT INTO processor_checkpoints (processor_id, position)
                VALUES (?, ?)
                ON CONFLICT(processor_id) DO UPDATE SET position = excluded.position
                """,
                (processor_id, int(position)),
            )
            connection.commit()
