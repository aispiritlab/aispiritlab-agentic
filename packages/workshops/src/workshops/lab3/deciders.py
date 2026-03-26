"""Deciders for lab3 worker agents."""
from __future__ import annotations

from typing import Sequence

from agentic.workflow.messages import Message, UserMessage


def worker_decider(msg: Message) -> Sequence[Message]:
    """Passthrough decider for worker agents."""
    if isinstance(msg, UserMessage):
        return [msg]
    return []
