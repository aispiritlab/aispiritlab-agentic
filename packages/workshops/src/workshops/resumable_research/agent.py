from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Protocol
import uuid

from agentic.integrations.search_provider import SearchProvider, SearchResult
from agentic.workflow import (
    ConcurrencyConflictError,
    Event,
    EventStore,
    RecordedMessageMetadata,
)

from .model import (
    PendingFileOperation,
    PlannedQuery,
    QueryEvidence,
    ResearchDocument,
    ResearchProcessState,
    ResearchSearchResult,
    normalize_queries,
    rebuild_research,
)
from .storage import ResearchFileIntegrityError, ResearchFileRepository


class ResearchPlanner(Protocol):
    def plan(self, question: str) -> Sequence[str]: ...

    def close(self) -> None: ...


class ResearchSummarizer(Protocol):
    def summarize(self, question: str, results: Sequence[dict[str, object]]) -> str: ...

    def close(self) -> None: ...


class ResearchRunStatus(StrEnum):
    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class ResearchRunResult:
    research_id: str
    status: ResearchRunStatus
    file_path: Path
    stream_version: int
    file_revision: int
    completed_queries: int
    total_queries: int
    summary: str | None
    last_error: str | None


AfterFileStageHook = Callable[[PendingFileOperation], None]


class ResumableResearchAgent:
    """Web research state machine that can resume after a process crash.

    The event stream owns process history. The research JSON file owns queries,
    evidence and the final summary. A pending-operation journal bridges the two
    durable stores and is reconciled before every step.
    """

    def __init__(
        self,
        *,
        event_store: EventStore,
        files: ResearchFileRepository,
        planner: ResearchPlanner,
        search_provider: SearchProvider,
        summarizer: ResearchSummarizer,
        results_per_query: int = 5,
        after_file_stage: AfterFileStageHook | None = None,
    ) -> None:
        if results_per_query < 1:
            raise ValueError("results_per_query must be positive")
        self._event_store = event_store
        self._files = files
        self._planner = planner
        self._search_provider = search_provider
        self._summarizer = summarizer
        self._results_per_query = results_per_query
        self._after_file_stage = after_file_stage

    def start(
        self,
        question: str,
        *,
        research_id: str | None = None,
        max_steps: int | None = None,
    ) -> ResearchRunResult:
        resolved_question = " ".join(question.split())
        if not resolved_question:
            raise ValueError("question must not be empty")
        resolved_id = research_id or uuid.uuid4().hex
        file_path = self._files.path_for(resolved_id)
        if self._files.exists(resolved_id):
            existing = self._files.load(resolved_id)
            if existing.question != resolved_question:
                raise ValueError(
                    f"research_id {resolved_id!r} already belongs to another question"
                )

        operation = self._files.stage_operation(
            resolved_id,
            operation_id="research:start:v1",
            event_type="research.started",
            event_data={
                "research_id": resolved_id,
                "question": resolved_question,
                "file_path": str(file_path),
            },
            mutate=lambda document: document,
            initial=lambda: ResearchDocument(
                research_id=resolved_id,
                question=resolved_question,
            ),
        )
        self._after_stage(operation)
        self._reconcile_pending(resolved_id)
        return self.resume(resolved_id, max_steps=max_steps)

    def resume(self, research_id: str, *, max_steps: int | None = None) -> ResearchRunResult:
        if max_steps is not None and max_steps < 0:
            raise ValueError("max_steps cannot be negative")

        external_steps = 0
        while True:
            self._reconcile_pending(research_id)
            state, stream_version = self._load_state(research_id)
            document = self._files.load(research_id)
            self._validate_sources(state, document)

            if document.summary is not None and state.summary_completed:
                return self._result(
                    document,
                    state,
                    stream_version,
                    ResearchRunStatus.COMPLETED,
                )
            if max_steps is not None and external_steps >= max_steps:
                return self._result(
                    document,
                    state,
                    stream_version,
                    ResearchRunStatus.PAUSED,
                )

            if not document.queries:
                succeeded = self._plan(research_id, document, state)
            else:
                pending_query = next(
                    (
                        query
                        for query in document.queries
                        if query.query_id not in document.completed_query_ids
                    ),
                    None,
                )
                if pending_query is not None:
                    succeeded = self._search(research_id, document, state, pending_query)
                else:
                    succeeded = self._summarize(research_id, document, state)

            external_steps += 1
            if not succeeded:
                self._reconcile_pending(research_id)
                failed_state, failed_version = self._load_state(research_id)
                failed_document = self._files.load(research_id)
                return self._result(
                    failed_document,
                    failed_state,
                    failed_version,
                    ResearchRunStatus.PAUSED,
                )

    def inspect(self, research_id: str) -> ResearchRunResult:
        self._reconcile_pending(research_id)
        state, stream_version = self._load_state(research_id)
        document = self._files.load(research_id)
        self._validate_sources(state, document)
        status = (
            ResearchRunStatus.COMPLETED
            if document.summary is not None and state.summary_completed
            else ResearchRunStatus.PAUSED
        )
        return self._result(document, state, stream_version, status)

    def close(self) -> None:
        closed: set[int] = set()
        for component in (self._planner, self._search_provider, self._summarizer):
            if id(component) in closed:
                continue
            closed.add(id(component))
            close = getattr(component, "close", None)
            if callable(close):
                close()

    def _plan(
        self,
        research_id: str,
        document: ResearchDocument,
        state: ResearchProcessState,
    ) -> bool:
        attempt = state.planning_attempts + 1
        self._append_runtime_event(
            research_id,
            event_type="research.planning_started",
            operation_id=f"research:plan:{attempt}:started",
            data={"attempt": attempt},
        )
        try:
            queries = normalize_queries(self._planner.plan(document.question))
            if not queries:
                raise ValueError("planner returned no usable queries")
        except Exception as error:
            self._record_failure(research_id, step="planning", attempt=attempt, error=error)
            return False

        operation = self._files.stage_operation(
            research_id,
            operation_id=f"research:plan:{attempt}:completed",
            event_type="research.queries_planned",
            event_data={
                "attempt": attempt,
                "queries": [query.to_dict() for query in queries],
            },
            mutate=lambda current: replace(current, queries=queries, summary=None),
        )
        self._after_stage(operation)
        self._reconcile_pending(research_id)
        return True

    def _search(
        self,
        research_id: str,
        document: ResearchDocument,
        state: ResearchProcessState,
        query: PlannedQuery,
    ) -> bool:
        attempt = state.query_attempts.get(query.query_id, 0) + 1
        self._append_runtime_event(
            research_id,
            event_type="research.query_started",
            operation_id=f"research:query:{query.query_id}:{attempt}:started",
            data={
                "query_id": query.query_id,
                "query": query.text,
                "attempt": attempt,
            },
        )
        try:
            raw_results = self._search_provider.search(
                query.text,
                count=self._results_per_query,
            )
        except Exception as error:
            self._record_failure(
                research_id,
                step=f"search:{query.query_id}",
                attempt=attempt,
                error=error,
            )
            return False

        results = self._normalize_search_results(raw_results)
        evidence = QueryEvidence(query_id=query.query_id, query=query.text, results=results)
        operation = self._files.stage_operation(
            research_id,
            operation_id=f"research:query:{query.query_id}:{attempt}:completed",
            event_type="research.query_completed",
            event_data={
                "query_id": query.query_id,
                "query": query.text,
                "attempt": attempt,
                "result_count": len(results),
            },
            mutate=lambda current: current.with_evidence(evidence),
        )
        self._after_stage(operation)
        self._reconcile_pending(research_id)
        return True

    def _summarize(
        self,
        research_id: str,
        document: ResearchDocument,
        state: ResearchProcessState,
    ) -> bool:
        attempt = state.summary_attempts + 1
        self._append_runtime_event(
            research_id,
            event_type="research.summary_started",
            operation_id=f"research:summary:{attempt}:started",
            data={"attempt": attempt, "result_count": len(document.all_results())},
        )
        try:
            summary = self._summarizer.summarize(
                document.question,
                tuple(dict(item) for item in document.all_results()),
            ).strip()
            if not summary:
                raise ValueError("summarizer returned an empty response")
        except Exception as error:
            self._record_failure(research_id, step="summary", attempt=attempt, error=error)
            return False

        operation = self._files.stage_operation(
            research_id,
            operation_id=f"research:summary:{attempt}:completed",
            event_type="research.summary_completed",
            event_data={"attempt": attempt, "result_count": len(document.all_results())},
            mutate=lambda current: replace(current, summary=summary),
        )
        self._after_stage(operation)
        self._reconcile_pending(research_id)
        return True

    def _record_failure(
        self,
        research_id: str,
        *,
        step: str,
        attempt: int,
        error: Exception,
    ) -> None:
        self._append_runtime_event(
            research_id,
            event_type="research.step_failed",
            operation_id=f"research:{step}:{attempt}:failed",
            data={
                "step": step,
                "attempt": attempt,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )

    def _reconcile_pending(self, research_id: str) -> None:
        for _ in range(32):
            document = self._files.load(research_id)
            if not document.pending_operations:
                return
            operation = document.pending_operations[0]
            state, stream_version = self._load_state(research_id)
            if operation.operation_id not in state.seen_operation_ids:
                try:
                    self._event_store.append_to_stream(
                        self._stream_name(research_id),
                        (self._event(research_id, operation.event_type, operation.data),),
                        expected_version=stream_version,
                    )
                except ConcurrencyConflictError:
                    continue
            self._files.mark_committed(research_id, operation.operation_id)
        raise ConcurrencyConflictError(self._stream_name(research_id), "stable version", "busy")

    def _append_runtime_event(
        self,
        research_id: str,
        *,
        event_type: str,
        operation_id: str,
        data: Mapping[str, object],
    ) -> None:
        for _ in range(32):
            state, stream_version = self._load_state(research_id)
            if operation_id in state.seen_operation_ids:
                return
            event_data = {**data, "operation_id": operation_id}
            try:
                self._event_store.append_to_stream(
                    self._stream_name(research_id),
                    (self._event(research_id, event_type, event_data),),
                    expected_version=stream_version,
                )
                return
            except ConcurrencyConflictError:
                continue
        raise ConcurrencyConflictError(self._stream_name(research_id), "stable version", "busy")

    def _load_state(self, research_id: str) -> tuple[ResearchProcessState, int]:
        result = self._event_store.read_stream(self._stream_name(research_id))
        return rebuild_research(result.events), result.current_version

    def _validate_sources(
        self,
        state: ResearchProcessState,
        document: ResearchDocument,
    ) -> None:
        if not state.started:
            raise ResearchFileIntegrityError("Research event stream has no start event")
        if state.research_id != document.research_id or state.question != document.question:
            raise ResearchFileIntegrityError(
                "Research event stream and file identify different work"
            )
        if Path(state.file_path).resolve() != self._files.path_for(document.research_id):
            raise ResearchFileIntegrityError("Research event stream points to a different file")
        if state.last_file_revision != document.revision:
            raise ResearchFileIntegrityError(
                "Research file revision is not synchronized with the event stream"
            )
        if state.last_content_sha256 != document.content_sha256:
            raise ResearchFileIntegrityError(
                "Research file content is not synchronized with the event stream"
            )

    def _after_stage(self, operation: PendingFileOperation | None) -> None:
        if operation is not None and self._after_file_stage is not None:
            self._after_file_stage(operation)

    @staticmethod
    def _normalize_search_results(
        results: Sequence[SearchResult],
    ) -> tuple[ResearchSearchResult, ...]:
        deduplicated: dict[str, ResearchSearchResult] = {}
        for result in results:
            url = result.url.strip()
            key = url or f"missing-url:{len(deduplicated)}"
            deduplicated.setdefault(
                key,
                ResearchSearchResult(
                    title=result.title.strip(),
                    url=url,
                    snippet=result.snippet.strip(),
                    published_at=result.published_at,
                ),
            )
        return tuple(deduplicated.values())

    @staticmethod
    def _stream_name(research_id: str) -> str:
        return f"research:{research_id}"

    @staticmethod
    def _event(
        research_id: str,
        event_type: str,
        data: Mapping[str, object],
    ) -> Event:
        operation_id = str(data["operation_id"])
        event_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL, f"resumable-research:event:{research_id}:{operation_id}"
            )
        )
        message_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"resumable-research:message:{research_id}:{operation_id}",
            )
        )
        return Event(
            type=event_type,
            data=dict(data),
            metadata=RecordedMessageMetadata(
                runtime_id=research_id,
                session_id=research_id,
                turn_id=research_id,
                domain="research",
                source="resumable-research-agent",
                event_id=event_id,
                message_id=message_id,
            ),
        )

    def _result(
        self,
        document: ResearchDocument,
        state: ResearchProcessState,
        stream_version: int,
        status: ResearchRunStatus,
    ) -> ResearchRunResult:
        return ResearchRunResult(
            research_id=document.research_id,
            status=status,
            file_path=self._files.path_for(document.research_id),
            stream_version=stream_version,
            file_revision=document.revision,
            completed_queries=len(document.completed_query_ids),
            total_queries=len(document.queries),
            summary=document.summary,
            last_error=state.last_error,
        )
