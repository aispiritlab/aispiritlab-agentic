from __future__ import annotations

from dataclasses import dataclass

from agentic.workflow.messages import Event, UserMessage


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageMessage(UserMessage):
    """Carries both user text and image path through the workflow."""

    kind: str = "image_message"
    image_path: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageDescribed(Event):
    """Emitted when the VLM agent finishes describing an image."""

    kind: str = "image_described"
    name: str = "image_described"
    image_path: str = ""
    description: str = ""
