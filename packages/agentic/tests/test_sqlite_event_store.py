from __future__ import annotations

from pathlib import Path

from agentic.workflow import (
    STREAM_DOES_NOT_EXIST,
    STREAM_EXISTS,
    ConversationData,
    Event,
    RecordedMessageMetadata,
    SQLiteCheckpointStore,
    SQLiteEventStore,
    UserMessage,
)


def test_sqlite_event_store_round_trips_messages_with_positions(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "workflow.sqlite3")

    store.append_to_stream(
        "workflow:lab6:turn-1",
        (
            UserMessage(
                data=ConversationData(role="user", text="hello"),
                metadata=RecordedMessageMetadata(
                    runtime_id="runtime-1",
                    turn_id="turn-1",
                    source="chat",
                    target="planner",
                    message_id="msg-1",
                ),
            ),
            Event(
                type="planned",
                data={"query": "redis streams"},
                metadata=RecordedMessageMetadata(
                    runtime_id="runtime-1",
                    turn_id="turn-1",
                    source="planner",
                    target="search",
                    message_id="msg-2",
                    reply_to_message_id="msg-1",
                ),
            ),
        ),
        expected_version=STREAM_DOES_NOT_EXIST,
    )

    result = store.read_stream("workflow:lab6:turn-1")

    assert result.stream_exists is True
    assert result.current_version == 2
    assert isinstance(result.events[0], UserMessage)
    assert result.events[0].metadata.stream_name == "workflow:lab6:turn-1"
    assert result.events[0].metadata.stream_position == 0
    assert result.events[0].metadata.global_position == 0
    assert result.events[1].metadata.stream_position == 1
    assert result.events[1].metadata.global_position == 1
    assert result.events[1].metadata.reply_to_message_id == "msg-1"


def test_sqlite_event_store_enforces_expected_version(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "workflow.sqlite3")
    store.append_to_stream(
        "orders-1",
        (Event(type="created", data={"id": "orders-1"}),),
        expected_version=STREAM_DOES_NOT_EXIST,
    )

    result = store.append_to_stream(
        "orders-1",
        (Event(type="confirmed", data={"id": "orders-1"}),),
        expected_version=STREAM_EXISTS,
    )

    assert result.next_version == 2


def test_sqlite_checkpoint_store_persists_positions(tmp_path: Path) -> None:
    checkpoint_store = SQLiteCheckpointStore(tmp_path / "workflow.sqlite3")

    assert checkpoint_store.read("projection-1") is None

    checkpoint_store.store("projection-1", 3)
    checkpoint_store.store("projection-1", 7)

    assert checkpoint_store.read("projection-1") == 7
