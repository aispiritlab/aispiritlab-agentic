"""AgentGraph-specific workflow events."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from agentic.workflow.messages import Event, Message, RecordedMessageMetadata
from agentic_runtime.distributed.serialization import register_record_types


def _metadata_with_updates(
    metadata: RecordedMessageMetadata,
    **updates: Any,
) -> RecordedMessageMetadata:
    values = {
        field.name: getattr(metadata, field.name) for field in fields(RecordedMessageMetadata)
    }
    values.update(updates)
    return RecordedMessageMetadata(**values)


@dataclass(frozen=True, slots=True, kw_only=True, init=False)
class GraphDispatchEvent(Event):
    def __init__(
        self,
        *,
        kind: str = "graph_dispatch",
        type: str = "graph_dispatch",
        data: dict[str, Any] | None = None,
        source_node_id: str = "",
        source_alias: str = "",
        target_node_id: str = "",
        target_alias: str = "",
        text: str = "",
        metadata: RecordedMessageMetadata | None = None,
    ) -> None:
        super().__init__(
            kind=kind,
            type=type,
            data=data
            or {
                "source_node_id": source_node_id,
                "source_alias": source_alias,
                "target_node_id": target_node_id,
                "target_alias": target_alias,
                "text": text,
            },
            metadata=metadata or RecordedMessageMetadata(),
        )

    @property
    def source_node_id(self) -> str:
        return str(self.data.get("source_node_id", ""))

    @property
    def source_alias(self) -> str:
        return str(self.data.get("source_alias", ""))

    @property
    def target_node_id(self) -> str:
        return str(self.data.get("target_node_id", ""))

    @property
    def target_alias(self) -> str:
        return str(self.data.get("target_alias", ""))

    @property
    def text(self) -> str:
        return str(self.data.get("text", ""))

    def with_metadata(self, **updates: Any) -> Message:
        return GraphDispatchEvent(
            data=dict(self.data),
            metadata=_metadata_with_updates(self.metadata, **updates),
        )

    def with_data(self, **updates: Any) -> Message:
        return GraphDispatchEvent(data={**self.data, **updates}, metadata=self.metadata)


@dataclass(frozen=True, slots=True, kw_only=True, init=False)
class GraphCompletionEvent(Event):
    def __init__(
        self,
        *,
        kind: str = "graph_completion",
        type: str = "graph_completion",
        data: dict[str, Any] | None = None,
        source_node_id: str = "",
        source_alias: str = "",
        text: str = "",
        summarizer_node_ids: tuple[str, ...] = (),
        payload: dict[str, str] | None = None,
        metadata: RecordedMessageMetadata | None = None,
    ) -> None:
        super().__init__(
            kind=kind,
            type=type,
            data=data
            or {
                "source_node_id": source_node_id,
                "source_alias": source_alias,
                "text": text,
                "summarizer_node_ids": list(summarizer_node_ids),
                "payload": payload or {},
            },
            metadata=metadata or RecordedMessageMetadata(),
        )

    @property
    def source_node_id(self) -> str:
        return str(self.data.get("source_node_id", ""))

    @property
    def source_alias(self) -> str:
        return str(self.data.get("source_alias", ""))

    @property
    def text(self) -> str:
        return str(self.data.get("text", ""))

    @property
    def summarizer_node_ids(self) -> tuple[str, ...]:
        return tuple(str(node_id) for node_id in self.data.get("summarizer_node_ids", ()))

    @property
    def payload(self) -> dict[str, str]:
        raw = self.data.get("payload", {})
        return dict(raw) if isinstance(raw, dict) else {}

    def with_metadata(self, **updates: Any) -> Message:
        return GraphCompletionEvent(
            data=dict(self.data),
            metadata=_metadata_with_updates(self.metadata, **updates),
        )

    def with_data(self, **updates: Any) -> Message:
        return GraphCompletionEvent(data={**self.data, **updates}, metadata=self.metadata)


@dataclass(frozen=True, slots=True, kw_only=True, init=False)
class GraphOutputReadyEvent(Event):
    def __init__(
        self,
        *,
        kind: str = "graph_output_ready",
        type: str = "graph_output_ready",
        data: dict[str, Any] | None = None,
        source_node_id: str = "",
        source_alias: str = "",
        output_node_id: str = "",
        output_name: str = "",
        text: str = "",
        metadata: RecordedMessageMetadata | None = None,
    ) -> None:
        super().__init__(
            kind=kind,
            type=type,
            data=data
            or {
                "source_node_id": source_node_id,
                "source_alias": source_alias,
                "output_node_id": output_node_id,
                "output_name": output_name,
                "text": text,
            },
            metadata=metadata or RecordedMessageMetadata(),
        )

    @property
    def source_node_id(self) -> str:
        return str(self.data.get("source_node_id", ""))

    @property
    def source_alias(self) -> str:
        return str(self.data.get("source_alias", ""))

    @property
    def output_node_id(self) -> str:
        return str(self.data.get("output_node_id", ""))

    @property
    def output_name(self) -> str:
        return str(self.data.get("output_name", ""))

    @property
    def text(self) -> str:
        return str(self.data.get("text", ""))

    def with_metadata(self, **updates: Any) -> Message:
        return GraphOutputReadyEvent(
            data=dict(self.data),
            metadata=_metadata_with_updates(self.metadata, **updates),
        )

    def with_data(self, **updates: Any) -> Message:
        return GraphOutputReadyEvent(data={**self.data, **updates}, metadata=self.metadata)


register_record_types(GraphDispatchEvent, GraphCompletionEvent, GraphOutputReadyEvent)
