from __future__ import annotations

from pathlib import Path

from agentic.workflow import SQLiteEventStore
from agentic_runtime.distributed.service import DistributedService
from agentic_runtime.messaging.messages import (
    AssistantMessage,
    ConversationData,
    Message,
    RecordedMessageMetadata,
    UserMessage,
)


class _FakeTransport:
    def __init__(self) -> None:
        self.published: list[Message] = []
        self.acks: list[tuple[str, str, str]] = []

    def publish_message(self, message: Message) -> str:
        self.published.append(message)
        return f"{len(self.published)}-0"

    def ack(self, stream: str, group: str, entry_id: str) -> int:
        self.acks.append((stream, group, entry_id))
        return 1


class _FakeRegistry:
    def register(self, registration) -> None:  # noqa: ANN001
        return None

    def heartbeat(self, heartbeat) -> None:  # noqa: ANN001
        return None


class _FakeDiscovery:
    def __init__(self, transport: _FakeTransport) -> None:
        self.transport = transport
        self.registry = _FakeRegistry()


def test_distributed_service_replays_recorded_outputs_on_duplicate_input(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(message: Message, discovery: _FakeDiscovery) -> tuple[Message, ...]:
        del discovery
        calls.append(getattr(message.metadata, "message_id", ""))
        return (
            AssistantMessage(
                data=ConversationData(role="assistant", text="done"),
                metadata=RecordedMessageMetadata(
                    runtime_id=message.metadata.runtime_id,
                    turn_id=message.metadata.turn_id,
                    domain=message.metadata.domain,
                    source="planner",
                    target="chat",
                ),
            ),
        )

    transport = _FakeTransport()
    discovery = _FakeDiscovery(transport)
    service = DistributedService(
        agent_name="planner",
        capabilities=("plan",),
        discovery=discovery,  # type: ignore[arg-type]
        handler=handler,  # type: ignore[arg-type]
        event_store=SQLiteEventStore(tmp_path / "workflow.sqlite3"),
    )
    message = UserMessage(
        data=ConversationData(role="user", text="hello"),
        metadata=RecordedMessageMetadata(
            runtime_id="runtime-1",
            turn_id="turn-1",
            domain="lab6",
            source="chat",
            target="planner",
            message_id="msg-1",
        ),
    )

    service._handle_record("test:messages:planner", "1-0", message)
    first_published = list(transport.published)

    assert len(calls) == 1
    assert len(first_published) == 1
    assert first_published[0].metadata.reply_to_message_id

    transport.published.clear()
    service._handle_record("test:messages:planner", "2-0", message)

    assert len(calls) == 1
    assert len(transport.published) == 1
    assert transport.published[0].data.text == "done"
    assert transport.published[0].metadata.message_id == first_published[0].metadata.message_id
    assert len(transport.acks) == 2
