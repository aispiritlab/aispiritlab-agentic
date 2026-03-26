"""WorkshopRuntime — thin wrapper over the shared workflow runtime kernel."""
from __future__ import annotations

from agentic.workflow import WorkflowRuntime


class WorkshopRuntime(WorkflowRuntime):
    """Workshop-facing name for the shared runtime kernel."""

    def run(self, text: str, workflow_name: str) -> str:
        return self.run_text(text, workflow_name)

    def stop(self) -> None:
        self.close()
