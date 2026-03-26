from __future__ import annotations

from types import SimpleNamespace

from agentic_runtime.distributed.client import DistributedChatClient
from agentic_runtime.distributed.serialization import deserialize_record, serialize_record
from agentic_runtime.messaging.messages import AssistantMessage, UserMessage
from workshops.lab6.messages import SearchPlanned


class _FakeTransport:
    def __init__(self) -> None:
        self.published_messages: list[UserMessage] = []
        self._responses: list[list[SimpleNamespace]] = []

    def last_message_id(self, target: str) -> str:
        assert target == "chat"
        return "0-0"

    def publish_message(self, message: UserMessage) -> str:
        self.published_messages.append(message)
        self._responses.append(
            [
                SimpleNamespace(
                    entry_id="1-0",
                    record=AssistantMessage(
                        runtime_id=message.runtime_id,
                        turn_id=message.turn_id,
                        domain=message.domain,
                        source="summary",
                        target="chat",
                        text="final answer",
                    ),
                )
            ]
        )
        return "0-1"

    def read_messages(
        self,
        target: str,
        *,
        after_id: str = "0-0",
        block_ms: int = 1_000,
        count: int = 10,
    ) -> list[SimpleNamespace]:
        del after_id, block_ms, count
        assert target == "chat"
        if not self._responses:
            return []
        return self._responses.pop(0)

    def close(self) -> None:
        return None


def test_distributed_chat_client_publishes_user_message_and_returns_assistant_text() -> None:
    transport = _FakeTransport()
    client = DistributedChatClient(transport, entry_agent="planner", source="chat", timeout_seconds=1.0)

    reply = client.ask("Find fresh info about Redis")

    assert reply == "final answer"
    assert len(transport.published_messages) == 1
    message = transport.published_messages[0]
    assert message.target == "planner"
    assert message.source == "chat"
    assert message.text == "Find fresh info about Redis"


def test_search_planned_round_trips_through_serializer() -> None:
    message = SearchPlanned(
        runtime_id="runtime-1",
        turn_id="turn-1",
        domain="lab6",
        source="planner",
        target="search",
        question="What changed in Redis 8.6?",
        queries=("Redis 8.6 release notes",),
        reply_target="chat",
    )

    serialized = serialize_record(message)
    restored = deserialize_record(serialized)

    assert isinstance(restored, SearchPlanned)
    assert restored.question == "What changed in Redis 8.6?"
    assert restored.queries == ("Redis 8.6 release notes",)
