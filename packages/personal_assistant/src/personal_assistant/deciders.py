"""Message routers for workflow orchestration.

Each router takes a Message from the stream and returns commands/events to append.
Routers are pure workflow logic: they decide WHAT happens, not HOW.
"""
from __future__ import annotations

from typing import Callable, Sequence

from structlog import get_logger

from agentic.prompts import PromptTemplate
from agentic.tools import Command, Toolsets

from agentic.workflow.messages import ConversationData, Message, RecordedMessageMetadata, UserMessage
from agentic_runtime.reactor import LLMResponse, MessageRouter

from personal_assistant.messaging.events import CreatedNote, NoteUpdated

logger = get_logger(__name__)


def build_note_events(
    command: Command,
    *,
    resolve_note_path: Callable[[str], str],
    agent_name: str,
    metadata: RecordedMessageMetadata,
) -> tuple[Message, ...]:
    """Build domain events from a parsed manage_notes tool command."""
    from personal_assistant.agents.manage_notes.commands import AddNoteCommand, EditNoteCommand

    match command:
        case AddNoteCommand(note_name=name, note=content):
            logger.info("created_note", note_name=name)
            return (
                CreatedNote(
                    note_name=name,
                    note_content=content,
                    metadata=RecordedMessageMetadata(
                        runtime_id=metadata.runtime_id,
                        session_id=metadata.session_id,
                        turn_id=metadata.turn_id,
                        source=agent_name,
                        domain="manage_notes",
                        target="organizer",
                        trace=metadata.trace,
                    ),
                ),
                NoteUpdated(
                    note_name=name,
                    note_path=resolve_note_path(name),
                    metadata=RecordedMessageMetadata(
                        runtime_id=metadata.runtime_id,
                        session_id=metadata.session_id,
                        turn_id=metadata.turn_id,
                        source=agent_name,
                        domain="manage_notes",
                        target="rag",
                        trace=metadata.trace,
                    ),
                ),
            )
        case EditNoteCommand(note_name=name):
            logger.info("updated_note", note_name=name)
            return (
                NoteUpdated(
                    note_name=name,
                    note_path=resolve_note_path(name),
                    metadata=RecordedMessageMetadata(
                        runtime_id=metadata.runtime_id,
                        session_id=metadata.session_id,
                        turn_id=metadata.turn_id,
                        source=agent_name,
                        domain="manage_notes",
                        target="rag",
                        trace=metadata.trace,
                    ),
                ),
            )
        case _:
            return ()


def passthrough_decider(msg: Message) -> Sequence[Message]:
    """Simplest message router: UserMessage passes through to LLM, everything else terminates.

    Used by: Personalize, DiscoveryNotes.
    """
    if isinstance(msg, UserMessage):
        return [msg]
    return []


def make_manage_notes_decider(
    toolsets: Toolsets,
    resolve_note_path: Callable[[str], str],
    agent_name: str = "manage_notes",
) -> MessageRouter:
    """Message router for ManageNotes: extracts domain events from LLM tool calls.

    UserMessage → [UserMessage] (pass to LLM)
    LLMResponse with tool_calls → [CreatedNote, NoteUpdated, ...] (domain events)
    """

    def decider(msg: Message) -> Sequence[Message]:
        if isinstance(msg, UserMessage):
            return [msg]

        if isinstance(msg, LLMResponse) and msg.has_tool_calls:
            tool_call = msg.tool_calls[0]
            command = toolsets.parse_tool(tool_call)
            if command is None:
                logger.warning("failed_to_parse_tool_call", tool_call=tool_call)
                return []

            return list(build_note_events(
                command,
                resolve_note_path=resolve_note_path,
                agent_name=agent_name,
                metadata=msg.metadata,
            ))

        return []

    return decider


def make_organizer_decider() -> MessageRouter:
    """Message router for Organizer: handles CreatedNote and UserMessage.

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
                    data=ConversationData(role="user", text=payload),
                    metadata=RecordedMessageMetadata(
                        runtime_id=msg.metadata.runtime_id,
                        session_id=msg.metadata.session_id,
                        turn_id=msg.metadata.turn_id,
                        domain=msg.metadata.domain,
                        source=msg.metadata.source,
                        trace=msg.metadata.trace,
                    ),
                )
            ]

        if isinstance(msg, UserMessage):
            return [msg]

        return []

    return decider


def sage_decider(msg: Message) -> Sequence[Message]:
    """Message router for Sage: passes UserMessage to MultiTurnLLMReactor.

    Multi-turn tool cycle is handled by MultiTurnLLMReactor internally.
    """
    if isinstance(msg, UserMessage):
        return [msg]
    return []
