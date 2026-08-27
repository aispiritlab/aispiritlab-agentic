"""End-to-end resilience tests for the distributed agent pipeline.

Tests a 5-agent pipeline (coordinator -> researcher -> fact_checker -> editor -> summarizer)
that hits a real LLM. Agents are randomly crashed and restarted to verify they resume
from the last unacknowledged message in the stream.

Parameterized to run on both in-memory and Redis transports.

Usage:
    RUN_DISTRIBUTED_RESILIENCE=1 uv run pytest packages/agentic_runtime/tests/e2e/test_distributed_resilience.py -v
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import os
import random
import threading
import time
import uuid

import pytest

from agentic.llm_call import LLMCall
from agentic_runtime.distributed.client import DistributedChatClient
from agentic_runtime.distributed.discovery import AgenticServiceDiscovery
from agentic_runtime.distributed.in_memory_transport import (
    InMemoryServiceRegistry,
    InMemoryTransport,
)
from agentic_runtime.distributed.serialization import register_record_types
from agentic_runtime.distributed.service import DistributedService
from agentic_runtime.messaging.messages import (
    AssistantMessage,
    ConversationData,
    Event,
    Message,
    RecordedMessageMetadata,
    TurnCompleted,
    UserMessage,
)
from providers.api import OpenAIProvider

# ---------------------------------------------------------------------------
# Custom message types for the 5-agent pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class ResearchRequested(Event):
    kind: str = "research_requested"
    type: str = "research_requested"
    question: str = ""
    reply_target: str = "chat"

    def __post_init__(self) -> None:
        if not self.data:
            object.__setattr__(
                self, "data", {"question": self.question, "reply_target": self.reply_target}
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class FactCheckRequested(Event):
    kind: str = "fact_check_requested"
    type: str = "fact_check_requested"
    question: str = ""
    research_text: str = ""
    reply_target: str = "chat"

    def __post_init__(self) -> None:
        if not self.data:
            object.__setattr__(
                self,
                "data",
                {
                    "question": self.question,
                    "research_text": self.research_text,
                    "reply_target": self.reply_target,
                },
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class EditRequested(Event):
    kind: str = "edit_requested"
    type: str = "edit_requested"
    question: str = ""
    checked_text: str = ""
    reply_target: str = "chat"

    def __post_init__(self) -> None:
        if not self.data:
            object.__setattr__(
                self,
                "data",
                {
                    "question": self.question,
                    "checked_text": self.checked_text,
                    "reply_target": self.reply_target,
                },
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class SummarizeRequested(Event):
    kind: str = "summarize_requested"
    type: str = "summarize_requested"
    question: str = ""
    edited_text: str = ""
    reply_target: str = "chat"

    def __post_init__(self) -> None:
        if not self.data:
            object.__setattr__(
                self,
                "data",
                {
                    "question": self.question,
                    "edited_text": self.edited_text,
                    "reply_target": self.reply_target,
                },
            )


register_record_types(ResearchRequested, FactCheckRequested, EditRequested, SummarizeRequested)

# ---------------------------------------------------------------------------
# Agent handlers
# ---------------------------------------------------------------------------

_COORDINATOR_PROMPT = (
    "You are a coordinator agent. Given a user question, rephrase it concisely "
    "for a researcher. Reply with the rephrased question only, no commentary."
)
_RESEARCHER_PROMPT = (
    "You are a researcher agent. Given a question, write a short factual paragraph "
    "answering it. Keep it under 100 words."
)
_FACT_CHECKER_PROMPT = (
    "You are a fact-checker agent. Review the text for accuracy. "
    "Return the text with minor corrections if needed, under 100 words."
)
_EDITOR_PROMPT = (
    "You are an editor agent. Polish the text for clarity and grammar. "
    "Return the polished version, under 100 words."
)
_SUMMARIZER_PROMPT = (
    "You are a summarizer agent. Condense the text into 1-2 sentences. Return only the summary."
)


def _make_llm(system_prompt: str, model_name: str) -> LLMCall:
    return LLMCall(
        model_name=model_name,
        system_prompt=system_prompt,
        max_tokens=256,
        model_provider_type="openai",
    )


class CoordinatorHandler:
    def __init__(self, llm: LLMCall) -> None:
        self._llm = llm

    def __call__(self, message: Message, discovery: AgenticServiceDiscovery) -> Sequence[Message]:
        if not isinstance(message, UserMessage):
            return ()
        researcher = discovery.find("research")
        question = message.data.text or ""
        self._llm.reset()
        rephrased = self._llm.call(question).strip()
        return (
            ResearchRequested(
                question=rephrased or question,
                reply_target=message.metadata.source or "chat",
                metadata=RecordedMessageMetadata(
                    runtime_id=message.metadata.runtime_id,
                    turn_id=message.metadata.turn_id,
                    domain=message.metadata.domain or "resilience",
                    source="coordinator",
                    target=researcher.agent_name,
                ),
            ),
        )

    def close(self) -> None:
        self._llm.close()


class ResearcherHandler:
    def __init__(self, llm: LLMCall) -> None:
        self._llm = llm

    def __call__(self, message: Message, discovery: AgenticServiceDiscovery) -> Sequence[Message]:
        if not isinstance(message, ResearchRequested):
            return ()
        fact_checker = discovery.find("fact-check")
        self._llm.reset()
        research_text = self._llm.call(message.question).strip()
        return (
            FactCheckRequested(
                question=message.question,
                research_text=research_text,
                reply_target=message.reply_target,
                metadata=RecordedMessageMetadata(
                    runtime_id=message.metadata.runtime_id,
                    turn_id=message.metadata.turn_id,
                    domain=message.metadata.domain,
                    source="researcher",
                    target=fact_checker.agent_name,
                ),
            ),
        )

    def close(self) -> None:
        self._llm.close()


class FactCheckerHandler:
    def __init__(self, llm: LLMCall) -> None:
        self._llm = llm

    def __call__(self, message: Message, discovery: AgenticServiceDiscovery) -> Sequence[Message]:
        if not isinstance(message, FactCheckRequested):
            return ()
        editor = discovery.find("edit")
        self._llm.reset()
        checked_text = self._llm.call(message.research_text).strip()
        return (
            EditRequested(
                question=message.question,
                checked_text=checked_text,
                reply_target=message.reply_target,
                metadata=RecordedMessageMetadata(
                    runtime_id=message.metadata.runtime_id,
                    turn_id=message.metadata.turn_id,
                    domain=message.metadata.domain,
                    source="fact_checker",
                    target=editor.agent_name,
                ),
            ),
        )

    def close(self) -> None:
        self._llm.close()


class EditorHandler:
    def __init__(self, llm: LLMCall) -> None:
        self._llm = llm

    def __call__(self, message: Message, discovery: AgenticServiceDiscovery) -> Sequence[Message]:
        if not isinstance(message, EditRequested):
            return ()
        summarizer = discovery.find("summarize")
        self._llm.reset()
        edited_text = self._llm.call(message.checked_text).strip()
        return (
            SummarizeRequested(
                question=message.question,
                edited_text=edited_text,
                reply_target=message.reply_target,
                metadata=RecordedMessageMetadata(
                    runtime_id=message.metadata.runtime_id,
                    turn_id=message.metadata.turn_id,
                    domain=message.metadata.domain,
                    source="editor",
                    target=summarizer.agent_name,
                ),
            ),
        )

    def close(self) -> None:
        self._llm.close()


class SummarizerHandler:
    def __init__(self, llm: LLMCall) -> None:
        self._llm = llm

    def __call__(self, message: Message, discovery: AgenticServiceDiscovery) -> Sequence[Message]:
        del discovery
        if not isinstance(message, SummarizeRequested):
            return ()
        self._llm.reset()
        summary = self._llm.call(message.edited_text).strip()
        return (
            AssistantMessage(
                data=ConversationData(role="assistant", text=summary),
                metadata=RecordedMessageMetadata(
                    runtime_id=message.metadata.runtime_id,
                    turn_id=message.metadata.turn_id,
                    domain=message.metadata.domain,
                    source="summarizer",
                    target=message.reply_target,
                ),
            ),
            TurnCompleted(
                data={"workflow": "summarizer"},
                metadata=RecordedMessageMetadata(
                    runtime_id=message.metadata.runtime_id,
                    turn_id=message.metadata.turn_id,
                    domain=message.metadata.domain,
                    source="summarizer",
                    target=message.reply_target,
                    status="success",
                ),
            ),
        )

    def close(self) -> None:
        self._llm.close()


# ---------------------------------------------------------------------------
# Agent lifecycle management
# ---------------------------------------------------------------------------

_AGENT_SPECS: tuple[tuple[str, tuple[str, ...], str, type], ...] = (
    ("coordinator", ("coordinate",), _COORDINATOR_PROMPT, CoordinatorHandler),
    ("researcher", ("research",), _RESEARCHER_PROMPT, ResearcherHandler),
    ("fact_checker", ("fact-check",), _FACT_CHECKER_PROMPT, FactCheckerHandler),
    ("editor", ("edit",), _EDITOR_PROMPT, EditorHandler),
    ("summarizer", ("summarize",), _SUMMARIZER_PROMPT, SummarizerHandler),
)


class ManagedAgent:
    """Wraps a DistributedService with start/kill/restart lifecycle."""

    def __init__(
        self,
        name: str,
        service_factory: Callable[[], DistributedService],
    ) -> None:
        self.name = name
        self._factory = service_factory
        self._service: DistributedService | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._service = self._factory()
        self._thread = threading.Thread(
            target=self._service.run_forever,
            name=f"agent-{self.name}",
            daemon=True,
        )
        self._thread.start()

    def kill(self) -> None:
        if self._service is not None:
            self._service.close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._service = None
        self._thread = None

    def restart(self) -> None:
        self.kill()
        time.sleep(0.1)
        self.start()

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes"}


@pytest.fixture(autouse=True, scope="module")
def resilience_enabled():
    if _truthy_env("RUN_DISTRIBUTED_RESILIENCE"):
        return
    pytest.skip("Set RUN_DISTRIBUTED_RESILIENCE=1 to run distributed resilience tests.")


@pytest.fixture(scope="module")
def llm_available():
    """Skip if LLM server is not reachable."""
    import httpx

    try:
        resp = httpx.get("http://localhost:1234/v1/models", timeout=5.0)
        resp.raise_for_status()
    except Exception:
        pytest.skip("LLM server not available at localhost:1234")


def _redis_available() -> bool:
    try:
        from redis import Redis

        client = Redis.from_url("redis://localhost:6379/0")
        client.ping()
        client.close()
        return True
    except Exception:
        return False


@pytest.fixture(params=["in_memory", "redis"])
def transport_env(request, llm_available):
    """Provide (transport, registry, discovery) for each transport backend."""
    if request.param == "redis":
        if not _redis_available():
            pytest.skip("Redis not available at localhost:6379")

        from agentic_runtime.distributed.registry import RedisServiceRegistry
        from agentic_runtime.distributed.transport import RedisStreamsTransport

        prefix = f"resilience_{uuid.uuid4().hex[:8]}"
        transport = RedisStreamsTransport("redis://localhost:6379/0", prefix=prefix)
        registry = RedisServiceRegistry(transport)
        discovery = AgenticServiceDiscovery(transport, registry, liveness_ttl_seconds=30.0)
        yield discovery
        # Cleanup Redis keys
        client = transport.client
        for key in client.keys(f"{prefix}:*"):
            client.delete(key)
        transport.close()
    else:
        transport = InMemoryTransport(prefix="resilience")
        registry = InMemoryServiceRegistry()
        discovery = AgenticServiceDiscovery(transport, registry, liveness_ttl_seconds=30.0)
        yield discovery
        transport.close()


@pytest.fixture()
def model_name() -> str:
    from agentic_runtime.settings import settings

    return settings.orchestration_model_name


def _build_agents(
    discovery: AgenticServiceDiscovery,
    model_name: str,
    min_idle_ms: int = 100,
) -> dict[str, ManagedAgent]:
    OpenAIProvider.configure(
        base_url="http://localhost:1234",
        api_key=None,
        timeout=120.0,
    )
    agents: dict[str, ManagedAgent] = {}
    for name, capabilities, prompt, handler_cls in _AGENT_SPECS:

        def _factory(
            _name: str = name,
            _caps: tuple[str, ...] = capabilities,
            _prompt: str = prompt,
            _cls: type = handler_cls,
        ) -> DistributedService:
            llm = _make_llm(_prompt, model_name)
            handler = _cls(llm)
            return discovery.create_service(
                _name,
                capabilities=_caps,
                handler=handler,
                heartbeat_seconds=2.0,
                close_hook=handler.close,
                min_idle_ms=min_idle_ms,
            )

        agents[name] = ManagedAgent(name, _factory)
    return agents


def _start_all(agents: dict[str, ManagedAgent]) -> None:
    for agent in agents.values():
        agent.start()
    time.sleep(0.5)  # Let heartbeats register


def _kill_all(agents: dict[str, ManagedAgent]) -> None:
    for agent in agents.values():
        agent.kill()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.distributed_resilience
def test_pipeline_completes_without_crashes(transport_env, model_name) -> None:
    """Baseline: all 5 agents running, message flows through the entire pipeline."""
    discovery = transport_env
    agents = _build_agents(discovery, model_name)

    try:
        _start_all(agents)

        client = DistributedChatClient(
            discovery.transport,
            entry_agent="coordinator",
            source="chat",
            timeout_seconds=120.0,
            domain="resilience",
        )

        reply = client.ask("What is the capital of France?")
        assert isinstance(reply, str)
        # Pipeline completed — LLM may return empty text with thinking models
    finally:
        _kill_all(agents)


@pytest.mark.distributed_resilience
def test_pending_messages_are_recovered_after_restart(transport_env, model_name) -> None:
    """Start agents sequentially — each picks up the message left by the previous one."""
    discovery = transport_env
    agents = _build_agents(discovery, model_name, min_idle_ms=0)

    try:
        # Start only coordinator
        agents["coordinator"].start()
        time.sleep(0.5)

        # Send message — coordinator will process and emit ResearchRequested
        client = DistributedChatClient(
            discovery.transport,
            entry_agent="coordinator",
            source="chat",
            timeout_seconds=120.0,
            domain="resilience",
        )

        # Start client.ask in background since it blocks until final response
        result_holder: list[str] = []
        error_holder: list[Exception] = []

        def _ask() -> None:
            try:
                result_holder.append(client.ask("What is 2 + 2?"))
            except Exception as e:
                error_holder.append(e)

        ask_thread = threading.Thread(target=_ask, daemon=True)
        ask_thread.start()

        # Give coordinator time to process
        time.sleep(3.0)

        # Start remaining agents one by one
        for name in ("researcher", "fact_checker", "editor", "summarizer"):
            agents[name].start()
            time.sleep(2.0)

        # Wait for the full pipeline
        ask_thread.join(timeout=120.0)

        assert not error_holder, f"Client failed: {error_holder[0]}"
        assert len(result_holder) == 1
        assert isinstance(result_holder[0], str)
    finally:
        _kill_all(agents)


@pytest.mark.distributed_resilience
def test_pipeline_recovers_after_random_agent_kills(transport_env, model_name) -> None:
    """Kill random agents during pipeline execution, restart them, verify completion."""
    discovery = transport_env
    agents = _build_agents(discovery, model_name, min_idle_ms=100)

    try:
        _start_all(agents)

        client = DistributedChatClient(
            discovery.transport,
            entry_agent="coordinator",
            source="chat",
            timeout_seconds=180.0,
            domain="resilience",
        )

        result_holder: list[str] = []
        error_holder: list[Exception] = []

        def _ask() -> None:
            try:
                result_holder.append(client.ask("Explain gravity in one sentence"))
            except Exception as e:
                error_holder.append(e)

        ask_thread = threading.Thread(target=_ask, daemon=True)
        ask_thread.start()

        # Wait for some agents to be in the middle of processing
        time.sleep(3.0)

        # Kill 2 random agents (not coordinator — it already processed)
        rng = random.Random(42)
        killable = ["researcher", "fact_checker", "editor", "summarizer"]
        targets = rng.sample(killable, 2)
        for name in targets:
            agents[name].kill()

        # Wait a bit for the pending messages to age
        time.sleep(1.0)

        # Restart the killed agents
        for name in targets:
            agents[name].restart()

        # Wait for pipeline to complete
        ask_thread.join(timeout=180.0)

        assert not error_holder, f"Client failed: {error_holder[0]}"
        assert len(result_holder) == 1
        assert isinstance(result_holder[0], str)
    finally:
        _kill_all(agents)
