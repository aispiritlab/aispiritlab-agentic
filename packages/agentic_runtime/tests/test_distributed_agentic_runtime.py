from __future__ import annotations

from types import SimpleNamespace

from agentic_runtime.distributed.discovery import AgenticServiceDiscovery
from agentic_runtime.distributed.registry import AgentSnapshot
from agentic_runtime.distributed.runtime import DistributedAgenticRuntime
from agentic_runtime.messaging.messages import AssistantMessage, UserMessage


class _FakeTransport:
    """Minimal transport that returns a canned response for any published message."""

    def __init__(self) -> None:
        self.published_messages: list[UserMessage] = []
        self._responses: list[list[SimpleNamespace]] = []
        self.closed = False

    def last_message_id(self, target: str) -> str:
        del target
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
                        text="distributed answer",
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
        del target, after_id, block_ms, count
        if not self._responses:
            return []
        return self._responses.pop(0)

    def close(self) -> None:
        self.closed = True


class _FakeRegistry:
    def __init__(self, agents: list[AgentSnapshot] | None = None) -> None:
        self._agents = agents or []

    def find_by_capability(self, capability: str, *, max_age_seconds: float) -> AgentSnapshot | None:
        del max_age_seconds
        for agent in self._agents:
            if capability in agent.capabilities:
                return agent
        return None

    def live_agents(self, *, max_age_seconds: float) -> list[AgentSnapshot]:
        del max_age_seconds
        return list(self._agents)


_PLANNER_AGENT = AgentSnapshot(
    agent_name="planner",
    capabilities=("plan",),
    role="worker",
    consumer_group="planner",
    status="alive",
    last_seen_ns=0,
)


def _make_runtime() -> tuple[DistributedAgenticRuntime, _FakeTransport]:
    transport = _FakeTransport()
    registry = _FakeRegistry([_PLANNER_AGENT])
    discovery = AgenticServiceDiscovery(transport, registry, liveness_ttl_seconds=20.0)
    runtime = DistributedAgenticRuntime(
        discovery,
        entry_agent="planner",
        source="chat",
        timeout_seconds=1.0,
    )
    return runtime, transport


def test_run_delegates_to_client_and_returns_response() -> None:
    runtime, transport = _make_runtime()

    reply = runtime.run("What is Redis?")

    assert reply == "distributed answer"
    assert len(transport.published_messages) == 1
    assert transport.published_messages[0].target == "planner"
    assert transport.published_messages[0].text == "What is Redis?"


def test_start_returns_greeting() -> None:
    runtime, _ = _make_runtime()

    greeting = runtime.start()

    assert "rozproszony" in greeting


def test_run_chat_returns_error_string() -> None:
    runtime, _ = _make_runtime()

    result = runtime.run_chat("hello")

    assert isinstance(result, str)
    assert "nie jest dostępny" in result


def test_run_generate_image_returns_error_string() -> None:
    runtime, _ = _make_runtime()

    result = runtime.run_generate_image("a cat")

    assert isinstance(result, str)
    assert "nie jest dostępne" in result


def test_reset_chat_is_noop() -> None:
    runtime, _ = _make_runtime()
    runtime.reset_chat()  # should not raise


def test_clear_personalization_history_is_noop() -> None:
    runtime, _ = _make_runtime()
    runtime.clear_personalization_history()  # should not raise


def test_live_agents_delegates_to_discovery() -> None:
    runtime, _ = _make_runtime()

    agents = runtime.live_agents()

    assert len(agents) == 1
    assert agents[0].agent_name == "planner"


def test_stop_closes_transport() -> None:
    runtime, transport = _make_runtime()

    runtime.stop()

    assert transport.closed is True


def test_runtime_id_is_set() -> None:
    runtime, _ = _make_runtime()

    assert runtime.runtime_id
    assert isinstance(runtime.runtime_id, str)
