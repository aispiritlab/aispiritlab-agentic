"""Deciders for lab2 worker agents.

Worker agents use a simple passthrough decider: UserMessage passes to LLM,
everything else terminates the stream.
"""
from __future__ import annotations

from typing import Sequence

from agentic.workflow.messages import Message, UserMessage


def worker_decider(msg: Message) -> Sequence[Message]:
    """Passthrough decider for worker agents (researcher, writer)."""
    if isinstance(msg, UserMessage):
        return [msg]
    return []
