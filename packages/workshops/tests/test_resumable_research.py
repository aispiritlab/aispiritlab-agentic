from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from agentic.integrations.search_provider import SearchResult
from agentic.workflow import SQLiteEventStore
from workshops.resumable_research import (
    ResearchFileRepository,
    ResearchRunStatus,
    ResumableResearchAgent,
)


@dataclass(slots=True)
class _PlannerStub:
    queries: tuple[str, ...]
    calls: int = 0

    def plan(self, question: str) -> tuple[str, ...]:
        assert question
        self.calls += 1
        return self.queries

    def close(self) -> None:
        return None


@dataclass(slots=True)
class _SearchStub:
    results: dict[str, list[SearchResult]]
    failures_remaining: int = 0
    calls: list[str] = field(default_factory=list)

    def search(self, query: str, *, count: int = 5) -> list[SearchResult]:
        assert count > 0
        self.calls.append(query)
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("temporary search outage")
        return self.results.get(query, [])

    def close(self) -> None:
        return None


@dataclass(slots=True)
class _SummaryStub:
    calls: int = 0
    received_urls: tuple[str, ...] = ()

    def summarize(self, question: str, results: list[dict[str, object]]) -> str:
        assert question
        self.calls += 1
        self.received_urls = tuple(str(result["url"]) for result in results)
        return f"Podsumowanie na podstawie {len(results)} źródeł."

    def close(self) -> None:
        return None


def _result(title: str, url: str) -> SearchResult:
    return SearchResult(title=title, url=url, snippet=f"Opis: {title}")


def _agent(
    tmp_path: Path,
    *,
    planner: _PlannerStub,
    search: _SearchStub,
    summary: _SummaryStub,
    after_file_stage=None,
) -> tuple[ResumableResearchAgent, SQLiteEventStore, ResearchFileRepository]:
    event_store = SQLiteEventStore(tmp_path / "events.sqlite3")
    files = ResearchFileRepository(tmp_path / "research")
    return (
        ResumableResearchAgent(
            event_store=event_store,
            files=files,
            planner=planner,
            search_provider=search,
            summarizer=summary,
            after_file_stage=after_file_stage,
        ),
        event_store,
        files,
    )


def test_research_agent_persists_process_and_material(tmp_path: Path) -> None:
    planner = _PlannerStub(("event sourcing agents", "durable AI workflows"))
    search = _SearchStub(
        {
            "event sourcing agents": [_result("Event sourcing", "https://example.com/es")],
            "durable AI workflows": [_result("Durable workflows", "https://example.com/wf")],
        }
    )
    summary = _SummaryStub()
    agent, event_store, files = _agent(
        tmp_path,
        planner=planner,
        search=search,
        summary=summary,
    )

    result = agent.start("Jak budować trwałe agentowe workflow?", research_id="research-1")

    assert result.status == ResearchRunStatus.COMPLETED
    assert result.completed_queries == 2
    assert result.total_queries == 2
    assert planner.calls == 1
    assert search.calls == ["event sourcing agents", "durable AI workflows"]
    assert summary.received_urls == ("https://example.com/es", "https://example.com/wf")

    document = files.load("research-1")
    assert document.summary == "Podsumowanie na podstawie 2 źródeł."
    assert not document.pending_operations
    assert len(document.committed_operation_ids) == 5

    event_types = [event.type for event in event_store.read_stream("research:research-1").events]
    assert event_types == [
        "research.started",
        "research.planning_started",
        "research.queries_planned",
        "research.query_started",
        "research.query_completed",
        "research.query_started",
        "research.query_completed",
        "research.summary_started",
        "research.summary_completed",
    ]


def test_new_process_resumes_without_repeating_completed_queries(tmp_path: Path) -> None:
    first_planner = _PlannerStub(("query one", "query two"))
    first_search = _SearchStub(
        {
            "query one": [_result("One", "https://example.com/one")],
            "query two": [_result("Two", "https://example.com/two")],
        }
    )
    first_agent, _, _ = _agent(
        tmp_path,
        planner=first_planner,
        search=first_search,
        summary=_SummaryStub(),
    )

    paused = first_agent.start("Question", research_id="resume-me", max_steps=2)

    assert paused.status == ResearchRunStatus.PAUSED
    assert paused.completed_queries == 1
    assert first_search.calls == ["query one"]

    second_planner = _PlannerStub(("must not be used",))
    second_search = _SearchStub({"query two": [_result("Two", "https://example.com/two")]})
    second_summary = _SummaryStub()
    second_agent, _, _ = _agent(
        tmp_path,
        planner=second_planner,
        search=second_search,
        summary=second_summary,
    )

    completed = second_agent.resume("resume-me")

    assert completed.status == ResearchRunStatus.COMPLETED
    assert second_planner.calls == 0
    assert second_search.calls == ["query two"]
    assert second_summary.calls == 1


def test_resume_reconciles_crash_after_file_commit_without_repeating_search(
    tmp_path: Path,
) -> None:
    crashed = False

    def crash_after_search_file_commit(operation) -> None:
        nonlocal crashed
        if operation.event_type == "research.query_completed" and not crashed:
            crashed = True
            raise RuntimeError("simulated process crash")

    first_search = _SearchStub(
        {"only query": [_result("Evidence", "https://example.com/evidence")]}
    )
    first_agent, _, files = _agent(
        tmp_path,
        planner=_PlannerStub(("only query",)),
        search=first_search,
        summary=_SummaryStub(),
        after_file_stage=crash_after_search_file_commit,
    )

    with pytest.raises(RuntimeError, match="simulated process crash"):
        first_agent.start("Question", research_id="crash-safe")

    staged = files.load("crash-safe")
    assert len(staged.pending_operations) == 1
    assert staged.pending_operations[0].event_type == "research.query_completed"
    assert first_search.calls == ["only query"]

    resumed_search = _SearchStub({})
    resumed_agent, _, files = _agent(
        tmp_path,
        planner=_PlannerStub(("must not run",)),
        search=resumed_search,
        summary=_SummaryStub(),
    )

    completed = resumed_agent.resume("crash-safe")

    assert completed.status == ResearchRunStatus.COMPLETED
    assert resumed_search.calls == []
    assert not files.load("crash-safe").pending_operations


def test_transient_search_failure_is_recorded_and_next_run_retries(tmp_path: Path) -> None:
    search = _SearchStub(
        {"retry query": [_result("Recovered", "https://example.com/recovered")]},
        failures_remaining=1,
    )
    agent, event_store, _ = _agent(
        tmp_path,
        planner=_PlannerStub(("retry query",)),
        search=search,
        summary=_SummaryStub(),
    )

    paused = agent.start("Question", research_id="retryable")

    assert paused.status == ResearchRunStatus.PAUSED
    assert paused.last_error == "temporary search outage"

    completed = agent.resume("retryable")

    assert completed.status == ResearchRunStatus.COMPLETED
    assert search.calls == ["retry query", "retry query"]
    events = event_store.read_stream("research:retryable").events
    assert any(event.type == "research.step_failed" for event in events)
