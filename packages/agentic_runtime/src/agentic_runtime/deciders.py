"""Deciders — generic workflow archetype functions.

Each decider takes a Message from the stream and returns commands/events to append.
Deciders are pure routing logic: they decide WHAT happens, not HOW.
"""

from __future__ import annotations

from typing import Sequence

from agentic_runtime.messaging.messages import Message, UserMessage


def passthrough_decider(msg: Message) -> Sequence[Message]:
    """Simplest decider: UserMessage passes through to LLM, everything else terminates."""
    if isinstance(msg, UserMessage):
        return [msg]
    return []
