from __future__ import annotations

from agentic.integrations.search_provider import SearchResult
from agentic.workflow.messages import (
    ConversationData,
    RecordedMessageMetadata,
    normalize_recorded_message,
)
from agentic_runtime.distributed.registry import AgentSnapshot
from agentic_runtime.messaging.messages import AssistantMessage, TurnCompleted, UserMessage
from workshops.lab6.messages import (
    SearchPlanned,
    SearchRequested,
    SearchResultsFetched,
    SummaryRequested,
)
from workshops.lab6.runtime import (
    PlannerHandler,
    SearchHandler,
    SummaryHandler,
    _compact_results_for_summary,
)


class _FakeDiscovery:
    """Minimal fake satisfying the ``AgenticServiceDiscovery`` interface used by handlers."""

    def __init__(self, capability_targets: dict[str, str]) -> None:
        self._capability_targets = capability_targets

    def find(self, capability: str) -> AgentSnapshot:
        target = self._capability_targets.get(capability)
        if target is None:
            raise RuntimeError(f"No live agent with capability '{capability}' is registered.")
        return AgentSnapshot(
            agent_name=target,
            capabilities=(capability,),
            role="worker",
            consumer_group=target,
            status="alive",
            last_seen_ns=0,
        )

    def find_optional(self, capability: str) -> AgentSnapshot | None:
        target = self._capability_targets.get(capability)
        if target is None:
            return None
        return self.find(capability)


class _PlannerStub:
    def plan(self, question: str) -> tuple[str, ...]:
        return (f"{question} latest", f"{question} docs")

    def close(self) -> None:
        return None


class _SearchProviderStub:
    def search(self, query: str, *, count: int = 5) -> list[SearchResult]:
        del count
        return [
            SearchResult(title=f"Result for {query}", url="https://example.com/a", snippet="A"),
            SearchResult(
                title=f"Duplicate for {query}", url="https://example.com/a", snippet="A2"
            ),
            SearchResult(title=f"Second for {query}", url="https://example.com/b", snippet="B"),
        ]

    def close(self) -> None:
        return None


class _SummaryStub:
    def summarize(self, question: str, results) -> str:
        return f"summary:{question}:{len(results)}"

    def close(self) -> None:
        return None


def test_planner_handler_records_plan_and_emits_search_command() -> None:
    handler = PlannerHandler(planner=_PlannerStub())
    message = UserMessage(
        data=ConversationData(role="user", text="Redis 8.6"),
        metadata=RecordedMessageMetadata(
            runtime_id="runtime-1",
            turn_id="turn-1",
            domain="lab6",
            source="chat",
        ),
    )

    responses = handler(message, _FakeDiscovery({"web-search": "search"}))

    assert len(responses) == 2
    planned = responses[0]
    assert isinstance(planned, SearchPlanned)
    assert planned.metadata.target is None
    assert normalize_recorded_message(planned).metadata.contract_name == "lab6.search-planned"
    assert planned.reply_target == "chat"
    assert planned.queries == ("Redis 8.6 latest", "Redis 8.6 docs")
    requested = responses[1]
    assert isinstance(requested, SearchRequested)
    assert requested.metadata.target == "search"


def test_search_handler_emits_summary_request_with_deduplicated_results() -> None:
    handler = SearchHandler(search_provider=_SearchProviderStub(), results_per_query=5)
    message = SearchRequested(
        question="Redis 8.6",
        queries=("redis 8.6", "redis 8.6 release"),
        reply_target="chat",
        metadata=RecordedMessageMetadata(
            runtime_id="runtime-1",
            turn_id="turn-1",
            domain="lab6",
            source="planner",
            target="search",
        ),
    )

    responses = handler(message, _FakeDiscovery({"summarize": "summary"}))

    assert len(responses) == 2
    fetched = responses[0]
    assert isinstance(fetched, SearchResultsFetched)
    assert fetched.metadata.target is None
    summary_request = responses[1]
    assert isinstance(summary_request, SummaryRequested)
    assert summary_request.metadata.target == "summary"
    assert summary_request.reply_target == "chat"
    assert len(summary_request.results) == 2
    assert summary_request.results[0]["url"] == "https://example.com/a"
    assert summary_request.results[1]["url"] == "https://example.com/b"


def test_summary_handler_emits_final_assistant_message_and_turn_completed() -> None:
    handler = SummaryHandler(summary=_SummaryStub())
    message = SummaryRequested(
        question="Redis 8.6",
        queries=("redis 8.6",),
        results=({"title": "A", "url": "https://example.com/a", "snippet": "A"},),
        reply_target="chat",
        metadata=RecordedMessageMetadata(
            runtime_id="runtime-1",
            turn_id="turn-1",
            domain="lab6",
            source="search",
            target="summary",
        ),
    )

    responses = handler(message, _FakeDiscovery({}))

    assert len(responses) == 2
    assert isinstance(responses[0], AssistantMessage)
    assert responses[0].metadata.target == "chat"
    assert responses[0].data.text == "summary:Redis 8.6:1"
    assert isinstance(responses[1], TurnCompleted)
    assert responses[1].metadata.status == "success"


def test_compact_results_for_summary_limits_count_and_snippet_size() -> None:
    results = tuple(
        {
            "title": f"Title {index}",
            "url": f"https://example.com/{index}",
            "snippet": "x" * 500,
            "published_at": "2026-03-26",
        }
        for index in range(5)
    )

    compacted = _compact_results_for_summary(
        results,
        max_results=2,
        snippet_chars=120,
        total_chars=300,
    )

    assert len(compacted) == 2
    assert all(len(result["snippet"]) <= 120 for result in compacted)
    assert sum(sum(len(value) for value in result.values()) for result in compacted) <= 300
