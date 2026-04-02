from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from agentic.workflow.messages import Event, Message, RecordedMessageMetadata
from agentic_runtime.distributed import register_record_types


def _metadata_with_updates(metadata: RecordedMessageMetadata, **updates: Any) -> RecordedMessageMetadata:
    values = {field.name: getattr(metadata, field.name) for field in fields(RecordedMessageMetadata)}
    values.update(updates)
    return RecordedMessageMetadata(**values)


def _normalize_results(
    results: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    return tuple(dict(result) for result in results)


@dataclass(frozen=True, slots=True, kw_only=True, init=False)
class SearchPlanned(Event):
    def __init__(
        self,
        *,
        data: dict[str, Any] | None = None,
        question: str = "",
        queries: tuple[str, ...] = (),
        reply_target: str = "chat",
        metadata: RecordedMessageMetadata | None = None,
    ) -> None:
        normalized_queries = tuple(str(query) for query in queries)
        super().__init__(
            kind="search_planned",
            type="search_planned",
            data=data or {
                "question": question,
                "queries": list(normalized_queries),
                "reply_target": reply_target,
            },
            metadata=metadata or RecordedMessageMetadata(domain="lab6", target="search"),
        )

    @property
    def question(self) -> str:
        return str(self.data.get("question", ""))

    @property
    def queries(self) -> tuple[str, ...]:
        raw_queries = self.data.get("queries", [])
        if not isinstance(raw_queries, list | tuple):
            return ()
        return tuple(str(query) for query in raw_queries)

    @property
    def reply_target(self) -> str:
        return str(self.data.get("reply_target", "chat"))

    def with_metadata(self, **updates: Any) -> Message:
        return SearchPlanned(
            data=dict(self.data),
            metadata=_metadata_with_updates(self.metadata, **updates),
        )

    def with_data(self, **updates: Any) -> Message:
        return SearchPlanned(data={**self.data, **updates}, metadata=self.metadata)


@dataclass(frozen=True, slots=True, kw_only=True, init=False)
class SearchResultsFetched(Event):
    def __init__(
        self,
        *,
        data: dict[str, Any] | None = None,
        question: str = "",
        queries: tuple[str, ...] = (),
        results: tuple[dict[str, Any], ...] = (),
        reply_target: str = "chat",
        metadata: RecordedMessageMetadata | None = None,
    ) -> None:
        normalized_queries = tuple(str(query) for query in queries)
        normalized_results = _normalize_results(results)
        super().__init__(
            kind="search_results_fetched",
            type="search_results_fetched",
            data=data or {
                "question": question,
                "queries": list(normalized_queries),
                "results": list(normalized_results),
                "reply_target": reply_target,
            },
            metadata=metadata or RecordedMessageMetadata(domain="lab6", target="summary"),
        )

    @property
    def question(self) -> str:
        return str(self.data.get("question", ""))

    @property
    def queries(self) -> tuple[str, ...]:
        raw_queries = self.data.get("queries", [])
        if not isinstance(raw_queries, list | tuple):
            return ()
        return tuple(str(query) for query in raw_queries)

    @property
    def results(self) -> tuple[dict[str, Any], ...]:
        raw_results = self.data.get("results", [])
        if not isinstance(raw_results, list | tuple):
            return ()
        return _normalize_results(raw_results)

    @property
    def reply_target(self) -> str:
        return str(self.data.get("reply_target", "chat"))

    def with_metadata(self, **updates: Any) -> Message:
        return SearchResultsFetched(
            data=dict(self.data),
            metadata=_metadata_with_updates(self.metadata, **updates),
        )

    def with_data(self, **updates: Any) -> Message:
        return SearchResultsFetched(data={**self.data, **updates}, metadata=self.metadata)


@dataclass(frozen=True, slots=True, kw_only=True, init=False)
class SummaryRequested(Event):
    def __init__(
        self,
        *,
        data: dict[str, Any] | None = None,
        question: str = "",
        queries: tuple[str, ...] = (),
        results: tuple[dict[str, Any], ...] = (),
        reply_target: str = "chat",
        metadata: RecordedMessageMetadata | None = None,
    ) -> None:
        normalized_queries = tuple(str(query) for query in queries)
        normalized_results = _normalize_results(results)
        super().__init__(
            kind="summary_requested",
            type="summary_requested",
            data=data or {
                "question": question,
                "queries": list(normalized_queries),
                "results": list(normalized_results),
                "reply_target": reply_target,
            },
            metadata=metadata or RecordedMessageMetadata(domain="lab6", target="summary"),
        )

    @property
    def question(self) -> str:
        return str(self.data.get("question", ""))

    @property
    def queries(self) -> tuple[str, ...]:
        raw_queries = self.data.get("queries", [])
        if not isinstance(raw_queries, list | tuple):
            return ()
        return tuple(str(query) for query in raw_queries)

    @property
    def results(self) -> tuple[dict[str, Any], ...]:
        raw_results = self.data.get("results", [])
        if not isinstance(raw_results, list | tuple):
            return ()
        return _normalize_results(raw_results)

    @property
    def reply_target(self) -> str:
        return str(self.data.get("reply_target", "chat"))

    def with_metadata(self, **updates: Any) -> Message:
        return SummaryRequested(
            data=dict(self.data),
            metadata=_metadata_with_updates(self.metadata, **updates),
        )

    def with_data(self, **updates: Any) -> Message:
        return SummaryRequested(data={**self.data, **updates}, metadata=self.metadata)


register_record_types(SearchPlanned, SearchResultsFetched, SummaryRequested)
