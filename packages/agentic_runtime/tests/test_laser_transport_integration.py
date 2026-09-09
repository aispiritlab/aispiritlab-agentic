"""The Laser transport against a real Apache Iggy broker.

Skipped unless one is reachable — `make iggy` starts it, and
``AGENTIC_TEST_LASER_CONNECTION`` points these somewhere else. What is proved
here is what a fake could not: that a stored offset resumes where the ledger
said it would, and that an unacked message comes back.
"""

from __future__ import annotations

from collections.abc import Iterator
import os
import socket
from urllib.parse import urlsplit
import uuid

import pytest

from agentic_runtime.distributed.registry import LaserServiceRegistry
from agentic_runtime.distributed.transport import LaserTransport, parse_entry_id
from agentic_runtime.messaging.messages import (
    ConversationData,
    RecordedMessageMetadata,
    UserMessage,
)

CONNECTION = os.environ.get("AGENTIC_TEST_LASER_CONNECTION", "iggy:iggy@127.0.0.1:8090")


def _speaks_iggy(connection: str) -> bool:
    """Whether a broker is actually there — not merely whether the port is open.

    An open port is not enough: on a developer machine 8090 is as likely to be
    something else's forwarder, and a guard that accepts it turns "skipped"
    into twelve errors in everybody's test run.
    """
    host_port = urlsplit(f"//{connection.rpartition('@')[2]}")
    try:
        with socket.create_connection((host_port.hostname or "", host_port.port or 8090), 1.0):
            pass
    except OSError:
        return False
    try:
        LaserTransport(connection, prefix="probe", connect_timeout=5.0).close()
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _speaks_iggy(CONNECTION),
    reason=f"No Apache Iggy at {CONNECTION} — run `make iggy`",
)


@pytest.fixture()
def prefix() -> str:
    """A stream of its own, so one test never reads another's messages."""
    return f"t{uuid.uuid4().hex[:10]}"


@pytest.fixture()
def transport(prefix: str) -> Iterator[LaserTransport]:
    opened = LaserTransport(CONNECTION, prefix=prefix)
    yield opened
    opened.close()


def _question(text: str, *, target: str, source: str = "chat", turn: str = "") -> UserMessage:
    return UserMessage(
        data=ConversationData(role="user", text=text),
        metadata=RecordedMessageMetadata(
            runtime_id="runtime",
            turn_id=turn or uuid.uuid4().hex,
            domain="test",
            source=source,
            target=target,
        ),
    )


def test_a_published_message_comes_back_to_the_group_that_consumes_it(
    transport: LaserTransport,
) -> None:
    transport.publish_message(_question("hello", target="planner"))

    records = transport.consume_target("planner", group="planner", consumer="one", block_ms=5_000)

    assert len(records) == 1
    assert records[0].record.data.text == "hello"
    assert parse_entry_id(records[0].entry_id) == (0, 0)


def test_an_acked_message_is_not_delivered_to_the_group_again(
    prefix: str, transport: LaserTransport
) -> None:
    """The stored offset is what a restarted process resumes from."""
    transport.publish_message(_question("handled", target="planner"))
    records = transport.consume_target("planner", group="planner", consumer="one", block_ms=5_000)
    assert len(records) == 1
    assert transport.ack(records[0].stream, "planner", records[0].entry_id) == 1

    restarted = LaserTransport(CONNECTION, prefix=prefix)
    try:
        assert (
            restarted.consume_target("planner", group="planner", consumer="two", block_ms=1_000)
            == []
        )
    finally:
        restarted.close()


def test_a_message_that_was_never_acked_comes_back_after_a_restart(
    prefix: str, transport: LaserTransport
) -> None:
    transport.publish_message(_question("dropped", target="planner"))
    records = transport.consume_target("planner", group="planner", consumer="one", block_ms=5_000)
    assert len(records) == 1
    transport.close()

    restarted = LaserTransport(CONNECTION, prefix=prefix)
    try:
        again = restarted.consume_target(
            "planner", group="planner", consumer="two", block_ms=5_000
        )
        assert [record.record.data.text for record in again] == ["dropped"]
    finally:
        restarted.close()


def test_a_message_left_for_a_retry_holds_the_offset_of_the_one_after_it(
    prefix: str, transport: LaserTransport
) -> None:
    """The ledger's reason for existing, end to end.

    The first message is never acked and the second is. A transport that stored
    the offset it had just handled would commit past the first, and nothing
    would ever hand it back.
    """
    transport.publish_message(_question("first", target="planner"))
    transport.publish_message(_question("second", target="planner"))

    delivered = []
    while len(delivered) < 2:
        batch = transport.consume_target(
            "planner", group="planner", consumer="one", block_ms=5_000
        )
        assert batch, "the broker stopped delivering before both messages arrived"
        delivered.extend(batch)

    assert [record.record.data.text for record in delivered] == ["first", "second"]
    transport.ack(delivered[1].stream, "planner", delivered[1].entry_id)
    transport.close()

    restarted = LaserTransport(CONNECTION, prefix=prefix)
    try:
        again = restarted.consume_target(
            "planner", group="planner", consumer="two", block_ms=5_000
        )
        assert [record.record.data.text for record in again] == ["first", "second"]
    finally:
        restarted.close()


def test_an_unacked_message_is_redelivered_to_the_process_holding_it(
    transport: LaserTransport,
) -> None:
    transport.publish_message(_question("retry me", target="planner"))
    assert transport.consume_target("planner", group="planner", consumer="one", block_ms=5_000)

    reclaimed = transport.autoclaim_pending(
        "planner", group="planner", consumer="one", min_idle_ms=0
    )

    assert [record.record.data.text for record in reclaimed] == ["retry me"]


def test_an_acked_message_is_never_reclaimed(transport: LaserTransport) -> None:
    transport.publish_message(_question("done", target="planner"))
    records = transport.consume_target("planner", group="planner", consumer="one", block_ms=5_000)
    transport.ack(records[0].stream, "planner", records[0].entry_id)

    assert (
        transport.autoclaim_pending("planner", group="planner", consumer="one", min_idle_ms=0)
        == []
    )


def test_a_reply_cursor_reads_what_was_published_after_it_was_taken(
    transport: LaserTransport,
) -> None:
    transport.publish_message(_question("before", target="chat"))
    cursor = transport.last_message_id("chat")
    transport.publish_message(_question("after", target="chat"))

    records = transport.read_messages("chat", after_id=cursor, block_ms=5_000)

    assert [record.record.data.text for record in records] == ["after"]


def test_a_reply_cursor_resumes_from_the_entry_it_last_returned(
    transport: LaserTransport,
) -> None:
    cursor = transport.last_message_id("chat")
    transport.publish_message(_question("one", target="chat"))
    first = transport.read_messages("chat", after_id=cursor, block_ms=5_000)
    assert [record.record.data.text for record in first] == ["one"]

    transport.publish_message(_question("two", target="chat"))
    second = transport.read_messages("chat", after_id=first[-1].entry_id, block_ms=5_000)

    assert [record.record.data.text for record in second] == ["two"]


def test_a_dead_letter_is_published_without_disturbing_the_inbox(
    transport: LaserTransport,
) -> None:
    entry_id = transport.publish_dead_letter(
        target="planner",
        source_stream=transport.message_stream("planner"),
        group="planner",
        entry_id="0-1",
        record=_question("poison", target="planner"),
        error="RuntimeError: boom",
        attempts=3,
    )

    assert parse_entry_id(entry_id) == (0, 0)
    assert transport.consume_target("planner", group="planner", consumer="one", block_ms=500) == []


def test_a_registry_lists_the_agent_this_process_just_registered(
    transport: LaserTransport,
) -> None:
    """A fold is asynchronous; registering and not appearing would be a bug."""
    from agentic_runtime.distributed.contracts import AgentHeartbeat, AgentRegistration

    registry = LaserServiceRegistry(transport)
    registry.register(
        AgentRegistration(
            agent_name="planner",
            capabilities=("planning",),
            role="worker",
            consumer_group="planner",
        )
    )
    registry.heartbeat(AgentHeartbeat(agent_name="planner", status="ready"))

    live = registry.live_agents(max_age_seconds=30.0)

    assert [agent.agent_name for agent in live] == ["planner"]
    assert live[0].capabilities == ("planning",)
    assert live[0].status == "ready"
    assert registry.find_by_capability("planning", max_age_seconds=30.0) is not None


def test_a_registry_in_another_process_sees_the_same_agents(
    prefix: str, transport: LaserTransport
) -> None:
    from agentic_runtime.distributed.contracts import AgentHeartbeat, AgentRegistration

    LaserServiceRegistry(transport).register(
        AgentRegistration(agent_name="search", capabilities=("search",), consumer_group="search")
    )
    LaserServiceRegistry(transport).heartbeat(AgentHeartbeat(agent_name="search"))

    observer = LaserTransport(CONNECTION, prefix=prefix)
    try:
        agents = LaserServiceRegistry(observer).live_agents(max_age_seconds=30.0)
        assert [agent.agent_name for agent in agents] == ["search"]
    finally:
        observer.close()


def test_an_agent_whose_heartbeat_went_quiet_is_not_live(transport: LaserTransport) -> None:
    from agentic_runtime.distributed.contracts import AgentHeartbeat, AgentRegistration

    registry = LaserServiceRegistry(transport)
    registry.register(
        AgentRegistration(agent_name="summary", capabilities=(), consumer_group="summary")
    )
    registry.heartbeat(AgentHeartbeat(agent_name="summary"))

    assert registry.live_agents(max_age_seconds=30.0)
    assert registry.live_agents(max_age_seconds=0.0) == []
