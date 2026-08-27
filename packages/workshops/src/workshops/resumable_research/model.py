from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import Any

from agentic.workflow import Message

RESEARCH_SCHEMA_VERSION = 1


def _sequence(value: object, field_name: str) -> list[object] | tuple[object, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"Research file field {field_name!r} must be an array")
    return value


def query_id(query: str) -> str:
    normalized = " ".join(query.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def normalize_queries(queries: Sequence[str], *, limit: int = 8) -> tuple[PlannedQuery, ...]:
    unique: dict[str, PlannedQuery] = {}
    for raw_query in queries:
        text = " ".join(str(raw_query).split())
        if not text:
            continue
        identifier = query_id(text)
        unique.setdefault(identifier, PlannedQuery(query_id=identifier, text=text))
        if len(unique) >= limit:
            break
    return tuple(unique.values())


@dataclass(frozen=True, slots=True)
class PlannedQuery:
    query_id: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"query_id": self.query_id, "text": self.text}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PlannedQuery:
        return cls(query_id=str(value["query_id"]), text=str(value["text"]))


@dataclass(frozen=True, slots=True)
class ResearchSearchResult:
    title: str
    url: str
    snippet: str
    published_at: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "published_at": self.published_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ResearchSearchResult:
        published_at = value.get("published_at")
        return cls(
            title=str(value.get("title", "")),
            url=str(value.get("url", "")),
            snippet=str(value.get("snippet", "")),
            published_at=None if published_at is None else str(published_at),
        )


@dataclass(frozen=True, slots=True)
class QueryEvidence:
    query_id: str
    query: str
    results: tuple[ResearchSearchResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "results": [result.to_dict() for result in self.results],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> QueryEvidence:
        raw_results = _sequence(value.get("results", []), "results")
        results = tuple(
            ResearchSearchResult.from_dict(item)
            for item in raw_results
            if isinstance(item, Mapping)
        )
        return cls(
            query_id=str(value["query_id"]),
            query=str(value["query"]),
            results=results,
        )


@dataclass(frozen=True, slots=True)
class PendingFileOperation:
    operation_id: str
    event_type: str
    data: Mapping[str, Any]

    def to_dict(self) -> dict[str, object]:
        return {
            "operation_id": self.operation_id,
            "event_type": self.event_type,
            "data": dict(self.data),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PendingFileOperation:
        raw_data = value.get("data", {})
        if not isinstance(raw_data, Mapping):
            raise ValueError("Pending operation data must be an object")
        return cls(
            operation_id=str(value["operation_id"]),
            event_type=str(value["event_type"]),
            data=dict(raw_data),
        )


@dataclass(frozen=True, slots=True)
class ResearchDocument:
    research_id: str
    question: str
    revision: int = 0
    queries: tuple[PlannedQuery, ...] = ()
    evidence: tuple[QueryEvidence, ...] = ()
    summary: str | None = None
    pending_operations: tuple[PendingFileOperation, ...] = ()
    committed_operation_ids: tuple[str, ...] = ()

    def content_dict(self) -> dict[str, object]:
        return {
            "research_id": self.research_id,
            "question": self.question,
            "queries": [query.to_dict() for query in self.queries],
            "evidence": [evidence.to_dict() for evidence in self.evidence],
            "summary": self.summary,
        }

    @property
    def content_sha256(self) -> str:
        serialized = json.dumps(
            self.content_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @property
    def completed_query_ids(self) -> frozenset[str]:
        return frozenset(item.query_id for item in self.evidence)

    def all_results(self) -> tuple[dict[str, str | None], ...]:
        deduplicated: dict[str, dict[str, str | None]] = {}
        for evidence in self.evidence:
            for result in evidence.results:
                key = result.url.strip() or f"{evidence.query_id}:{len(deduplicated)}"
                deduplicated.setdefault(key, result.to_dict())
        return tuple(deduplicated.values())

    def with_evidence(self, item: QueryEvidence) -> ResearchDocument:
        without_previous = tuple(
            evidence for evidence in self.evidence if evidence.query_id != item.query_id
        )
        return replace(self, evidence=(*without_previous, item))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RESEARCH_SCHEMA_VERSION,
            "revision": self.revision,
            "content_sha256": self.content_sha256,
            "content": self.content_dict(),
            "pending_operations": [item.to_dict() for item in self.pending_operations],
            "committed_operation_ids": list(self.committed_operation_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ResearchDocument:
        if value.get("schema_version") != RESEARCH_SCHEMA_VERSION:
            raise ValueError(f"Unsupported research file schema: {value.get('schema_version')}")
        content = value.get("content")
        if not isinstance(content, Mapping):
            raise ValueError("Research file content must be an object")

        raw_queries = _sequence(content.get("queries", []), "content.queries")
        raw_evidence = _sequence(content.get("evidence", []), "content.evidence")
        raw_pending = _sequence(value.get("pending_operations", []), "pending_operations")
        raw_committed = _sequence(
            value.get("committed_operation_ids", []),
            "committed_operation_ids",
        )
        revision = value.get("revision", 0)
        if not isinstance(revision, int) or isinstance(revision, bool):
            raise ValueError("Research file revision must be an integer")
        document = cls(
            research_id=str(content["research_id"]),
            question=str(content["question"]),
            revision=revision,
            queries=tuple(
                PlannedQuery.from_dict(item) for item in raw_queries if isinstance(item, Mapping)
            ),
            evidence=tuple(
                QueryEvidence.from_dict(item) for item in raw_evidence if isinstance(item, Mapping)
            ),
            summary=None if content.get("summary") is None else str(content["summary"]),
            pending_operations=tuple(
                PendingFileOperation.from_dict(item)
                for item in raw_pending
                if isinstance(item, Mapping)
            ),
            committed_operation_ids=tuple(str(item) for item in raw_committed),
        )
        recorded_hash = str(value.get("content_sha256", ""))
        if recorded_hash != document.content_sha256:
            raise ValueError("Research file content hash does not match its payload")
        return document


@dataclass(frozen=True, slots=True)
class ResearchProcessState:
    started: bool = False
    research_id: str = ""
    question: str = ""
    file_path: str = ""
    queries: tuple[PlannedQuery, ...] = ()
    completed_query_ids: frozenset[str] = frozenset()
    planning_attempts: int = 0
    query_attempts: Mapping[str, int] = field(default_factory=dict)
    summary_attempts: int = 0
    summary_completed: bool = False
    seen_operation_ids: frozenset[str] = frozenset()
    last_error: str | None = None
    last_file_revision: int = 0
    last_content_sha256: str = ""


def evolve_research(state: ResearchProcessState, event: Message) -> ResearchProcessState:
    data = event.data if isinstance(event.data, dict) else {}
    operation_id = str(data.get("operation_id", ""))
    seen = state.seen_operation_ids
    if operation_id:
        seen = frozenset((*seen, operation_id))

    common: dict[str, Any] = {"seen_operation_ids": seen}
    if "file_revision" in data:
        common["last_file_revision"] = int(data["file_revision"])
        common["last_content_sha256"] = str(data.get("content_sha256", ""))

    if event.type == "research.started":
        return replace(
            state,
            started=True,
            research_id=str(data["research_id"]),
            question=str(data["question"]),
            file_path=str(data["file_path"]),
            last_error=None,
            **common,
        )
    if event.type == "research.planning_started":
        return replace(
            state,
            planning_attempts=max(state.planning_attempts, int(data["attempt"])),
            **common,
        )
    if event.type == "research.queries_planned":
        raw_queries = data.get("queries", [])
        queries = tuple(
            PlannedQuery.from_dict(item) for item in raw_queries if isinstance(item, Mapping)
        )
        return replace(state, queries=queries, last_error=None, **common)
    if event.type == "research.query_started":
        attempts = dict(state.query_attempts)
        identifier = str(data["query_id"])
        attempts[identifier] = max(attempts.get(identifier, 0), int(data["attempt"]))
        return replace(state, query_attempts=attempts, **common)
    if event.type == "research.query_completed":
        completed = frozenset((*state.completed_query_ids, str(data["query_id"])))
        return replace(state, completed_query_ids=completed, last_error=None, **common)
    if event.type == "research.summary_started":
        return replace(
            state,
            summary_attempts=max(state.summary_attempts, int(data["attempt"])),
            **common,
        )
    if event.type == "research.summary_completed":
        return replace(state, summary_completed=True, last_error=None, **common)
    if event.type == "research.step_failed":
        return replace(state, last_error=str(data.get("error", "unknown error")), **common)
    return replace(state, **common)


def rebuild_research(events: Sequence[Message]) -> ResearchProcessState:
    state = ResearchProcessState()
    for event in events:
        state = evolve_research(state, event)
    return state
