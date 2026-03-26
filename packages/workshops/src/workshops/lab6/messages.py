from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_runtime.distributed import register_record_types
from agentic_runtime.messaging.messages import Event


def _normalize_results(
    results: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    return tuple(dict(result) for result in results)


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchPlanned(Event):
    kind: str = "search_planned"
    name: str = "search_planned"
    target: str | None = "search"
    question: str = ""
    queries: tuple[str, ...] = ()
    reply_target: str = "chat"

    def __post_init__(self) -> None:
        normalized_queries = tuple(str(query) for query in self.queries)
        object.__setattr__(self, "queries", normalized_queries)
        if not self.payload:
            object.__setattr__(
                self,
                "payload",
                {
                    "question": self.question,
                    "queries": list(normalized_queries),
                    "reply_target": self.reply_target,
                },
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchResultsFetched(Event):
    kind: str = "search_results_fetched"
    name: str = "search_results_fetched"
    target: str | None = "summary"
    question: str = ""
    queries: tuple[str, ...] = ()
    results: tuple[dict[str, Any], ...] = ()
    reply_target: str = "chat"

    def __post_init__(self) -> None:
        normalized_queries = tuple(str(query) for query in self.queries)
        object.__setattr__(self, "queries", normalized_queries)
        normalized_results = _normalize_results(self.results)
        object.__setattr__(self, "results", normalized_results)
        if not self.payload:
            object.__setattr__(
                self,
                "payload",
                {
                    "question": self.question,
                    "queries": list(normalized_queries),
                    "results": list(normalized_results),
                    "reply_target": self.reply_target,
                },
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class SummaryRequested(Event):
    kind: str = "summary_requested"
    name: str = "summary_requested"
    target: str | None = "summary"
    question: str = ""
    queries: tuple[str, ...] = ()
    results: tuple[dict[str, Any], ...] = ()
    reply_target: str = "chat"

    def __post_init__(self) -> None:
        normalized_queries = tuple(str(query) for query in self.queries)
        object.__setattr__(self, "queries", normalized_queries)
        normalized_results = _normalize_results(self.results)
        object.__setattr__(self, "results", normalized_results)
        if not self.payload:
            object.__setattr__(
                self,
                "payload",
                {
                    "question": self.question,
                    "queries": list(normalized_queries),
                    "results": list(normalized_results),
                    "reply_target": self.reply_target,
                },
            )


register_record_types(SearchPlanned, SearchResultsFetched, SummaryRequested)
