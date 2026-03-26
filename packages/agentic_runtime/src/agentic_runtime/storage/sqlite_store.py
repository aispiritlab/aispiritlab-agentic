from __future__ import annotations

import atexit
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
import sqlite3
import threading
import time
from typing import Protocol

import orjson

from agentic_runtime.messaging.messages import Message
from agentic_runtime.storage.projections import (
    DEFAULT_PROJECTIONS,
    ConversationRecordRow,
    MessageRow,
    Projection,
    handle_projections,
    row_to_conversation_record,
)
from agentic_runtime.messaging.streaming import hash_text


class MessageStore(Protocol):
    def enqueue(self, message: Message) -> None:
        ...

    def close(self) -> None:
        ...


@dataclass
class ChunkAssembly:
    row: MessageRow
    logical_kind: str
    expected_chunks: int | None = None
    parts: dict[int, str] | None = None

    def __post_init__(self) -> None:
        if self.parts is None:
            self.parts = {}

    def add_chunk(self, row: MessageRow) -> None:
        if row.chunk_index is None or row.text is None:
            return
        self.parts[row.chunk_index] = row.text
        if self.expected_chunks is None and row.chunk_count is not None:
            self.expected_chunks = row.chunk_count

    def complete(self, row: MessageRow) -> ConversationRecordRow | None:
        payload: dict[str, object] = {}
        if row.payload_json is not None:
            loaded = orjson.loads(row.payload_json)
            if isinstance(loaded, dict):
                payload = loaded
        if self.expected_chunks is None:
            count = payload.get("chunk_count")
            if isinstance(count, int):
                self.expected_chunks = count
        if self.expected_chunks is None:
            self.expected_chunks = len(self.parts)
        if len(self.parts) != self.expected_chunks:
            return None

        text = "".join(self.parts[index] for index in sorted(self.parts))
        return ConversationRecordRow(
            message_id=row.message_id,
            runtime_id=row.runtime_id,
            turn_id=row.turn_id,
            reply_to_message_id=row.reply_to_message_id,
            kind=self.logical_kind,
            event_type="AssembledTransportMessage",
            role=row.role or self.row.role or "assistant",
            domain=row.domain,
            source=row.source,
            target=row.target,
            name=None,
            text=text,
            payload_json=None,
            sequence_no=row.sequence_no,
            tool_call_id=row.tool_call_id,
            agent_run_id=row.agent_run_id,
            prompt_name=row.prompt_name,
            prompt_hash=row.prompt_hash,
            status=row.status,
            content_sha256=row.content_sha256 or hash_text(text),
            trace_id=row.trace_id,
            created_at_ns=row.created_at_ns,
        )


class SQLiteMessageStore:
    _MESSAGE_STREAM_COLUMNS: dict[str, str] = {
        "event_id": "TEXT",
        "runtime_id": "TEXT NOT NULL",
        "turn_id": "TEXT NOT NULL",
        "message_id": "TEXT NOT NULL",
        "reply_to_message_id": "TEXT",
        "kind": "TEXT NOT NULL",
        "event_type": "TEXT NOT NULL",
        "role": "TEXT",
        "scope": "TEXT NOT NULL",
        "domain": "TEXT NOT NULL",
        "source": "TEXT NOT NULL",
        "target": "TEXT",
        "name": "TEXT",
        "text": "TEXT",
        "payload_json": "BLOB",
        "sequence_no": "INTEGER",
        "chunk_index": "INTEGER",
        "chunk_count": "INTEGER",
        "tool_call_id": "TEXT",
        "agent_run_id": "TEXT",
        "prompt_name": "TEXT",
        "prompt_hash": "TEXT",
        "status": "TEXT",
        "content_sha256": "TEXT",
        "trace_id": "TEXT",
        "created_at_ns": "INTEGER NOT NULL",
    }
    _CONVERSATION_RECORD_COLUMNS: dict[str, str] = {
        "message_id": "TEXT PRIMARY KEY",
        "runtime_id": "TEXT NOT NULL",
        "turn_id": "TEXT NOT NULL",
        "reply_to_message_id": "TEXT",
        "kind": "TEXT NOT NULL",
        "event_type": "TEXT NOT NULL",
        "role": "TEXT NOT NULL",
        "domain": "TEXT NOT NULL",
        "source": "TEXT NOT NULL",
        "target": "TEXT",
        "name": "TEXT",
        "text": "TEXT",
        "payload_json": "BLOB",
        "sequence_no": "INTEGER",
        "tool_call_id": "TEXT",
        "agent_run_id": "TEXT",
        "prompt_name": "TEXT",
        "prompt_hash": "TEXT",
        "status": "TEXT",
        "content_sha256": "TEXT",
        "trace_id": "TEXT",
        "created_at_ns": "INTEGER NOT NULL",
    }
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS message_stream (
        id INTEGER PRIMARY KEY,
        event_id TEXT,
        runtime_id TEXT NOT NULL,
        turn_id TEXT NOT NULL,
        message_id TEXT NOT NULL,
        reply_to_message_id TEXT,
        kind TEXT NOT NULL,
        event_type TEXT NOT NULL,
        role TEXT,
        scope TEXT NOT NULL,
        domain TEXT NOT NULL,
        source TEXT NOT NULL,
        target TEXT,
        name TEXT,
        text TEXT,
        payload_json BLOB,
        sequence_no INTEGER,
        chunk_index INTEGER,
        chunk_count INTEGER,
        tool_call_id TEXT,
        agent_run_id TEXT,
        prompt_name TEXT,
        prompt_hash TEXT,
        status TEXT,
        content_sha256 TEXT,
        trace_id TEXT,
        created_at_ns INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS conversation_records (
        message_id TEXT PRIMARY KEY,
        runtime_id TEXT NOT NULL,
        turn_id TEXT NOT NULL,
        reply_to_message_id TEXT,
        kind TEXT NOT NULL,
        event_type TEXT NOT NULL,
        role TEXT NOT NULL,
        domain TEXT NOT NULL,
        source TEXT NOT NULL,
        target TEXT,
        name TEXT,
        text TEXT,
        payload_json BLOB,
        sequence_no INTEGER,
        tool_call_id TEXT,
        agent_run_id TEXT,
        prompt_name TEXT,
        prompt_hash TEXT,
        status TEXT,
        content_sha256 TEXT,
        trace_id TEXT,
        created_at_ns INTEGER NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_message_stream_runtime
    ON message_stream(runtime_id);

    CREATE INDEX IF NOT EXISTS idx_message_stream_runtime_domain
    ON message_stream(runtime_id, domain);

    CREATE INDEX IF NOT EXISTS idx_message_stream_turn
    ON message_stream(turn_id);

    CREATE INDEX IF NOT EXISTS idx_message_stream_message
    ON message_stream(message_id);

    CREATE INDEX IF NOT EXISTS idx_conversation_records_runtime
    ON conversation_records(runtime_id);

    CREATE INDEX IF NOT EXISTS idx_conversation_records_turn
    ON conversation_records(turn_id);

    CREATE INDEX IF NOT EXISTS idx_message_stream_trace
    ON message_stream(trace_id);
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        batch_size: int = 64,
        flush_interval_seconds: float = 0.05,
        projections: list[Projection] | None = None,
    ) -> None:
        self._projections = projections or DEFAULT_PROJECTIONS
        self._path = self._resolve_path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._batch_size = max(1, batch_size)
        self._flush_interval_seconds = max(0.001, flush_interval_seconds)
        self._queue: Queue[Message | None] = Queue()
        self._closed = False
        self._writer_error: Exception | None = None
        self._close_lock = threading.Lock()
        self._chunk_assemblies: dict[str, ChunkAssembly] = {}

        self._ensure_schema()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="agentic-runtime-message-store",
            daemon=True,
        )
        self._writer_thread.start()
        atexit.register(self.close)

    @staticmethod
    def _resolve_path(path: str | Path | None) -> Path:
        if path is not None:
            return Path(path).expanduser()
        return Path(__file__).resolve().parents[4] / "data" / "message_stream.sqlite3"

    def enqueue(self, message: Message) -> None:
        if self._writer_error is not None:
            raise RuntimeError("SQLite message writer failed.") from self._writer_error
        if self._closed:
            raise RuntimeError("SQLite message writer is closed.")
        self._queue.put(message)

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(None)
            self._writer_thread.join(timeout=5)
            if self._writer_thread.is_alive():
                raise RuntimeError("SQLite message writer did not stop cleanly.")
            if self._writer_error is not None:
                raise RuntimeError("SQLite message writer failed.") from self._writer_error

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(self._SCHEMA)
            self._ensure_table_columns(connection, "message_stream", self._MESSAGE_STREAM_COLUMNS)
            self._ensure_table_columns(
                connection,
                "conversation_records",
                self._CONVERSATION_RECORD_COLUMNS,
            )
            connection.commit()

    @staticmethod
    def _ensure_table_columns(
        connection: sqlite3.Connection,
        table_name: str,
        columns: dict[str, str],
    ) -> None:
        existing_columns = {
            row[1]
            for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        for column_name, column_type in columns.items():
            if column_name in existing_columns:
                continue
            safe_column_type = (
                column_type
                .replace(" PRIMARY KEY", "")
                .replace(" NOT NULL", "")
            )
            connection.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {safe_column_type}"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        connection.execute("PRAGMA busy_timeout=5000;")
        connection.execute("PRAGMA temp_store=MEMORY;")
        return connection

    def _writer_loop(self) -> None:
        connection = self._connect()
        try:
            while True:
                message = self._queue.get()
                if message is None:
                    break

                batch = [message]
                deadline = time.monotonic() + self._flush_interval_seconds
                while len(batch) < self._batch_size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        queued = self._queue.get(timeout=remaining)
                    except Empty:
                        break
                    if queued is None:
                        self._flush(connection, batch)
                        connection.close()
                        return
                    batch.append(queued)

                self._flush(connection, batch)
        except Exception as error:  # pragma: no cover - exercised via enqueue/close guards
            self._writer_error = error
        finally:
            connection.close()

    def _conversation_rows_from_message(self, row: MessageRow) -> list[ConversationRecordRow]:
        direct_row = row_to_conversation_record(row)
        if direct_row is not None:
            return [direct_row]

        if row.kind == "message_started":
            payload: dict[str, object] = {}
            if row.payload_json is not None:
                loaded = orjson.loads(row.payload_json)
                if isinstance(loaded, dict):
                    payload = loaded
            chunk_count = payload.get("chunk_count")
            self._chunk_assemblies[row.message_id] = ChunkAssembly(
                row=row,
                logical_kind=str(payload.get("logical_kind", "assistant_message")),
                expected_chunks=chunk_count if isinstance(chunk_count, int) else None,
            )
            return []

        if row.kind == "message_chunk":
            assembly = self._chunk_assemblies.get(row.message_id)
            if assembly is not None:
                assembly.add_chunk(row)
            return []

        if row.kind == "message_completed":
            assembly = self._chunk_assemblies.pop(row.message_id, None)
            if assembly is None:
                return []
            completed = assembly.complete(row)
            return [completed] if completed is not None else []

        return []

    def _flush(self, connection: sqlite3.Connection, batch: Iterable[Message]) -> None:
        rows: list[MessageRow] = []
        for msg in batch:
            row = handle_projections(self._projections, msg)
            if row is not None:
                rows.append(row)
        if not rows:
            return

        connection.executemany(
            """
            INSERT INTO message_stream (
                event_id,
                runtime_id,
                turn_id,
                message_id,
                reply_to_message_id,
                kind,
                event_type,
                role,
                scope,
                domain,
                source,
                target,
                name,
                text,
                payload_json,
                sequence_no,
                chunk_index,
                chunk_count,
                tool_call_id,
                agent_run_id,
                prompt_name,
                prompt_hash,
                status,
                content_sha256,
                trace_id,
                created_at_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row.event_id,
                    row.runtime_id,
                    row.turn_id,
                    row.message_id,
                    row.reply_to_message_id,
                    row.kind,
                    row.event_type,
                    row.role,
                    row.scope,
                    row.domain,
                    row.source,
                    row.target,
                    row.name,
                    row.text,
                    row.payload_json,
                    row.sequence_no,
                    row.chunk_index,
                    row.chunk_count,
                    row.tool_call_id,
                    row.agent_run_id,
                    row.prompt_name,
                    row.prompt_hash,
                    row.status,
                    row.content_sha256,
                    row.trace_id,
                    row.created_at_ns,
                )
                for row in rows
            ],
        )

        conversation_rows: list[ConversationRecordRow] = []
        for row in rows:
            conversation_rows.extend(self._conversation_rows_from_message(row))

        if conversation_rows:
            connection.executemany(
                """
                INSERT OR IGNORE INTO conversation_records (
                    message_id,
                    runtime_id,
                    turn_id,
                    reply_to_message_id,
                    kind,
                    event_type,
                    role,
                    domain,
                    source,
                    target,
                    name,
                    text,
                    payload_json,
                    sequence_no,
                    tool_call_id,
                    agent_run_id,
                    prompt_name,
                    prompt_hash,
                    status,
                    content_sha256,
                    trace_id,
                    created_at_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row.message_id,
                        row.runtime_id,
                        row.turn_id,
                        row.reply_to_message_id,
                        row.kind,
                        row.event_type,
                        row.role,
                        row.domain,
                        row.source,
                        row.target,
                        row.name,
                        row.text,
                        row.payload_json,
                        row.sequence_no,
                        row.tool_call_id,
                        row.agent_run_id,
                        row.prompt_name,
                        row.prompt_hash,
                        row.status,
                        row.content_sha256,
                        row.trace_id,
                        row.created_at_ns,
                    )
                    for row in conversation_rows
                ],
            )

        connection.commit()
