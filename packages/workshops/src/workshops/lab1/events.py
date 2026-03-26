from __future__ import annotations

from dataclasses import dataclass

from agentic.workflow.messages import Event


@dataclass(frozen=True, slots=True, kw_only=True)
class WriterCompleted(Event):
    """Emitted when the Writer agent finishes generating text."""

    kind: str = "writer_completed"
    name: str = "writer_completed"
    writer_output: str = ""
