from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from agentic.workflow.messages import Event, Message, RecordedMessageMetadata


def _metadata_with_updates(
    metadata: RecordedMessageMetadata, **updates: Any
) -> RecordedMessageMetadata:
    values = {
        field.name: getattr(metadata, field.name) for field in fields(RecordedMessageMetadata)
    }
    values.update(updates)
    return RecordedMessageMetadata(**values)


@dataclass(frozen=True, slots=True, kw_only=True, init=False)
class TaskDelegated(Event):
    """Emitted when a PlannerAgent delegates a task to another agent."""

    def __init__(
        self,
        *,
        data: dict[str, Any] | None = None,
        target_agent: str = "",
        task_description: str = "",
        metadata: RecordedMessageMetadata | None = None,
    ) -> None:
        super().__init__(
            kind="task_delegated",
            type="task_delegated",
            data=data
            or {
                "target_agent": target_agent,
                "task_description": task_description,
            },
            metadata=metadata or RecordedMessageMetadata(),
        )

    @property
    def target_agent(self) -> str:
        return str(self.data.get("target_agent", ""))

    @property
    def task_description(self) -> str:
        return str(self.data.get("task_description", ""))

    def with_metadata(self, **updates: Any) -> Message:
        return TaskDelegated(
            data=dict(self.data), metadata=_metadata_with_updates(self.metadata, **updates)
        )

    def with_data(self, **updates: Any) -> Message:
        return TaskDelegated(data={**self.data, **updates}, metadata=self.metadata)


@dataclass(frozen=True, slots=True, kw_only=True, init=False)
class TaskCompleted(Event):
    """Emitted when a delegated task has been completed by an agent."""

    def __init__(
        self,
        *,
        data: dict[str, Any] | None = None,
        target_agent: str = "",
        task_description: str = "",
        result: str = "",
        metadata: RecordedMessageMetadata | None = None,
    ) -> None:
        super().__init__(
            kind="task_completed",
            type="task_completed",
            data=data
            or {
                "target_agent": target_agent,
                "task_description": task_description,
                "result": result,
            },
            metadata=metadata or RecordedMessageMetadata(),
        )

    @property
    def target_agent(self) -> str:
        return str(self.data.get("target_agent", ""))

    @property
    def task_description(self) -> str:
        return str(self.data.get("task_description", ""))

    @property
    def result(self) -> str:
        return str(self.data.get("result", ""))

    def with_metadata(self, **updates: Any) -> Message:
        return TaskCompleted(
            data=dict(self.data), metadata=_metadata_with_updates(self.metadata, **updates)
        )

    def with_data(self, **updates: Any) -> Message:
        return TaskCompleted(data={**self.data, **updates}, metadata=self.metadata)
