"""Deciders — pure routing logic for Image Summary and Art Writer workflows.

Each decider takes a Message and returns commands/events to append to the stream.
Deciders decide WHAT happens, not HOW.
"""
from __future__ import annotations

from typing import Callable, Sequence

from agentic.workflow.messages import Message, UserMessage
from agentic.workflow.reactor import Decider, LLMResponse

from .messages import ImageDescribed, ImageMessage


def make_image_decider(image_path: str = "") -> Decider:
    """Create an image decider that remembers the image_path for the ImageDescribed event.

    ImageMessage -> [ImageMessage]  (route to VLM reactor)
    UserMessage  -> [UserMessage]   (route to VLM reactor, no image)
    LLMResponse  -> [ImageDescribed]  (emit domain event for Art Writer)
    """
    captured_path = image_path

    def image_decider(msg: Message) -> Sequence[Message]:
        nonlocal captured_path
        if isinstance(msg, ImageMessage):
            captured_path = msg.image_path
            return [msg]
        if isinstance(msg, UserMessage):
            return [msg]
        if isinstance(msg, LLMResponse):
            return [
                ImageDescribed(
                    runtime_id=msg.runtime_id,
                    turn_id=msg.turn_id,
                    source="image_summary",
                    image_path=captured_path,
                    description=msg.text or "",
                )
            ]
        return []

    return image_decider


def art_writer_decider(msg: Message) -> Sequence[Message]:
    """Art Writer: pass input to LLM reactor, terminate on response.

    UserMessage -> [UserMessage]  (route to LLM reactor)
    LLMResponse -> []  (terminate, no further events)
    """
    if isinstance(msg, UserMessage):
        return [msg]
    return []
