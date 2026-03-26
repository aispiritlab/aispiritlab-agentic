from __future__ import annotations

from typing import Protocol, Sequence

from agentic.metadata import Description
from agentic.workflow.execution import WorkflowExecution
from agentic.workflow.messages import Message


class AgenticWorkflow(Protocol):
    description: Description
    inputs: Sequence[str]

    def handle(self, message: Message) -> WorkflowExecution | str:
        ...

    def close(self) -> None:
        ...
