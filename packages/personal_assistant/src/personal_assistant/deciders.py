"""Deciders — Workflow archetype functions.

Each decider takes a Message from the stream and returns commands/events to append.
Deciders are pure routing logic: they decide WHAT happens, not HOW.
"""
from __future__ import annotations

from typing import Callable, Sequence

from structlog import get_logger

from agentic.prompts import PromptTemplate
from agentic.tools import Toolsets

from agentic.workflow.messages import Message, UserMessage
from agentic_runtime.reactor import Decider, LLMResponse

from personal_assistant.messaging.events import CreatedNote, NoteUpdated

logger = get_logger(__name__)


def passthrough_decider(msg: Message) -> Sequence[Message]:
    """Simplest decider: UserMessage passes through to LLM, everything else terminates.

    Used by: Personalize, DiscoveryNotes.
    """
    if isinstance(msg, UserMessage):
        return [msg]
    return []


def make_manage_notes_decider(
    toolsets: Toolsets,
    resolve_note_path: Callable[[str], str],
    agent_name: str = "manage_notes",
) -> Decider:
    """Decider for ManageNotes: extracts domain events from LLM tool calls.

    UserMessage → [UserMessage] (pass to LLM)
    LLMResponse with tool_calls → [CreatedNote, NoteUpdated, ...] (domain events)
    """
    from personal_assistant.agents.manage_notes.commands import AddNoteCommand, EditNoteCommand

    def decider(msg: Message) -> Sequence[Message]:
        if isinstance(msg, UserMessage):
            return [msg]

        if isinstance(msg, LLMResponse) and msg.has_tool_calls:
            tool_call = msg.tool_calls[0]
            command = toolsets.parse_tool(tool_call)
            if command is None:
                return []

            match command:
                case AddNoteCommand(note_name=name, note=content):
                    logger.info("created_note", note_name=name)
                    return [
                        CreatedNote(
                            runtime_id=msg.runtime_id,
                            turn_id=msg.turn_id,
                            source=agent_name,
                            note_name=name,
                            note_content=content,
                        ),
                        NoteUpdated(
                            runtime_id=msg.runtime_id,
                            turn_id=msg.turn_id,
                            source=agent_name,
                            note_name=name,
                            note_path=resolve_note_path(name),
                        ),
                    ]
                case EditNoteCommand(note_name=name):
                    logger.info("updated_note", note_name=name)
                    return [
                        NoteUpdated(
                            runtime_id=msg.runtime_id,
                            turn_id=msg.turn_id,
                            source=agent_name,
                            note_name=name,
                            note_path=resolve_note_path(name),
                        ),
                    ]
                case _:
                    return []

        return []

    return decider


def make_organizer_decider() -> Decider:
    """Decider for Organizer: handles CreatedNote and UserMessage.

    CreatedNote → [UserMessage with formatted payload]
    UserMessage → [UserMessage] (pass through)
    """

    def decider(msg: Message) -> Sequence[Message]:
        if isinstance(msg, CreatedNote):
            payload = PromptTemplate(
                template=(
                    "Nazwa notatki: {note_name}\n"
                    "Treść notatki:\n"
                    "{note_content}\n"
                ),
                context_variables=["note_name", "note_content"],
            ).format(
                note_name=msg.note_name,
                note_content=msg.note_content,
            )
            return [
                UserMessage(
                    text=payload,
                    domain=msg.domain,
                    runtime_id=msg.runtime_id,
                    turn_id=msg.turn_id,
                    source=msg.source,
                )
            ]

        if isinstance(msg, UserMessage):
            return [msg]

        return []

    return decider


def sage_decider(msg: Message) -> Sequence[Message]:
    """Decider for Sage: passes UserMessage to MultiTurnLLMReactor.

    Multi-turn tool cycle is handled by MultiTurnLLMReactor internally.
    """
    if isinstance(msg, UserMessage):
        return [msg]
    return []
