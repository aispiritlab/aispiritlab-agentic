import sqlite3
from pathlib import Path

import orjson

from agentic_runtime.messaging.messages import CreatedNote, Event, NoteUpdated, UserCommand, UserMessage
from agentic_runtime.messaging.message_bus import InMemoryMessageBus
from agentic_runtime.storage.sqlite_store import SQLiteMessageStore


def test_sqlite_message_store_persists_runtime_stream(tmp_path: Path) -> None:
    store = SQLiteMessageStore(
        path=tmp_path / "message_stream.sqlite3",
        batch_size=2,
        flush_interval_seconds=0.01,
    )
    bus = InMemoryMessageBus(store=store)

    bus.publish(
        UserMessage(
            runtime_id="runtime-1",
            domain="general",
            source="user",
            text="hej",
        )
    )
    bus.publish(
        Event(
            runtime_id="runtime-1",
            domain="routing",
            source="router",
            target="manage_notes",
            name="workflow_selected",
            payload={"workflow": "manage_notes"},
        )
    )
    bus.publish(
        UserCommand(
            runtime_id="runtime-1",
            domain="manage_notes",
            source="runtime",
            name="reset",
        )
    )
    bus.publish(
        CreatedNote(
            runtime_id="runtime-1",
            source="manage_notes",
            note_name="Projekt",
            note_content="Plan sprintu",
        )
    )
    bus.publish(
        NoteUpdated(
            runtime_id="runtime-1",
            source="manage_notes",
            note_name="Projekt",
            note_path="/vault/Projekt.md",
        )
    )
    bus.close()

    with sqlite3.connect(tmp_path / "message_stream.sqlite3") as connection:
        rows = connection.execute(
            """
            SELECT kind, domain, source, target, name, text, payload_json
            FROM message_stream
            ORDER BY id
            """
        ).fetchall()

    assert rows[0] == ("conversation", "general", "user", None, None, "hej", None)
    assert rows[1][0:6] == (
        "event",
        "routing",
        "router",
        "manage_notes",
        "workflow_selected",
        None,
    )
    assert orjson.loads(rows[1][6]) == {"workflow": "manage_notes"}
    assert rows[2] == ("command", "manage_notes", "runtime", None, "reset", None, None)
    assert rows[3][0:6] == (
        "created_note",
        "manage_notes",
        "manage_notes",
        "organizer",
        "created_note",
        None,
    )
    assert orjson.loads(rows[3][6]) == {
        "note_name": "Projekt",
        "note_content": "Plan sprintu",
    }
    assert rows[4][0:6] == (
        "note_updated",
        "manage_notes",
        "manage_notes",
        "rag",
        "note_updated",
        None,
    )
    assert orjson.loads(rows[4][6]) == {
        "note_name": "Projekt",
        "note_path": "/vault/Projekt.md",
    }
