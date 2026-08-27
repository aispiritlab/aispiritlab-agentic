from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from agentic.metadata import Description
from agentic.workflow.execution import WorkflowExecution
from agentic.workflow.messages import Message


@runtime_checkable
class AgenticWorkflow(Protocol):
    description: Description
    inputs: Sequence[str]

    def handle(self, message: Message) -> WorkflowExecution | str: ...

    def close(self) -> None: ...
