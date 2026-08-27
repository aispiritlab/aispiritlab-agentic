"""Message routers for Image Summary and Art Writer workflows.

Each router takes a Message and returns commands/events to append to the stream.
Routers decide WHAT happens, not HOW.
"""

from __future__ import annotations

from collections.abc import Sequence

from agentic.workflow.conversation import message_text, reply_metadata
from agentic.workflow.messages import Message, UserMessage
from agentic.workflow.reactor import LLMResponse, MessageRouter

from .messages import ImageDescribed, ImageMessage


def make_image_decider(image_path: str = "") -> MessageRouter:
    """Create an image router that remembers the image_path for the ImageDescribed event.

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
                    metadata=reply_metadata(msg, source="image_summary"),
                    image_path=captured_path,
                    description=message_text(msg),
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
