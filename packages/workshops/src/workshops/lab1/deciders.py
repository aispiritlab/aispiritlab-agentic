"""Deciders — pure routing logic for Writer and Critic workflows.

Each decider takes a Message and returns commands/events to append to the stream.
Deciders decide WHAT happens, not HOW.
"""
from __future__ import annotations

from typing import Sequence

from agentic.workflow.messages import Message, UserMessage
from agentic.workflow.reactor import LLMResponse

from .events import WriterCompleted


def writer_decider(msg: Message) -> Sequence[Message]:
    """Writer: pass user input to LLM, then emit WriterCompleted event.

    UserMessage -> [UserMessage]  (route to LLM reactor)
    LLMResponse -> [WriterCompleted]  (emit domain event for Critic)
    """
    if isinstance(msg, UserMessage):
        return [msg]
    if isinstance(msg, LLMResponse):
        return [
            WriterCompleted(
                runtime_id=msg.runtime_id,
                turn_id=msg.turn_id,
                source="writer",
                writer_output=msg.text or "",
            )
        ]
    return []


def critic_decider(msg: Message) -> Sequence[Message]:
    """Critic: pass user input to LLM, terminate on response.

    UserMessage -> [UserMessage]  (route to LLM reactor)
    LLMResponse -> []  (terminate, no further events)
    """
    if isinstance(msg, UserMessage):
        return [msg]
    return []
