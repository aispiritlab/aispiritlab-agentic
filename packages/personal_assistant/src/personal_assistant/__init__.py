"""Personal Assistant — multi-agent system for note management, search, and decision support."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentic.models import ModelProvider
from agentic.workflow.messages import (
    AssistantMessage,
    Command,
    Conversation,
    Event,
    Message,
    MessageChunk,
    MessageCompleted,
    MessageStarted,
    PromptSnapshot,
    ToolCallEvent,
    ToolResultMessage,
    TurnCompleted,
    TurnStarted,
    UserCommand,
    UserMessage,
)
from registry import Prompts, get_prompt

from personal_assistant.agents.personalize.personalize_agent import PersonalizeAgent as Agent
from personal_assistant.messaging.events import CreatedNote, NoteDeleted, NoteUpdated

if TYPE_CHECKING:
    from personal_assistant.runtime import PARuntime

_runtime: PARuntime | None = None


def _close_knowledge_base() -> None:
    try:
        from knowledge_base import close_knowledge_base
    except ModuleNotFoundError:
        return
    close_knowledge_base()


def get_runtime() -> PARuntime:
    global _runtime
    if _runtime is None:
        from personal_assistant.settings import settings

        if settings.agentic_transport == "redis_streams":
            from agentic_runtime.distributed.runtime import DistributedAgenticRuntime

            _runtime = DistributedAgenticRuntime.from_settings()  # type: ignore[assignment]
        else:
            from personal_assistant.runtime import PARuntime

            _runtime = PARuntime()
    return _runtime  # type: ignore[return-value]


def _reset_runtime() -> None:
    global _runtime
    if _runtime is not None:
        _runtime.stop()
    _runtime = None


def shutdown_application(*resources: object) -> None:
    global _runtime

    if _runtime is not None:
        try:
            _runtime.stop()
        finally:
            _runtime = None

    for resource in resources:
        close = getattr(resource, "close", None)
        if callable(close):
            close()

    ModelProvider.shutdown_all()
    _close_knowledge_base()


def personalize_agent(message: str) -> str:
    runtime = get_runtime()
    return runtime.handle(
        UserMessage(
            runtime_id=runtime.runtime_id,
            domain="personalize",
            source="user",
            target="personalize",
            text=message,
        )
    )


def sage_agent(message: str) -> str:
    runtime = get_runtime()
    return runtime.handle(
        UserMessage(
            runtime_id=runtime.runtime_id,
            domain="sage",
            source="user",
            target="sage",
            text=message,
        )
    )


def ai_spirit_agent(message: str) -> str:
    runtime = get_runtime()
    return runtime.run(message)


def chat_agent(message: str) -> str:
    runtime = get_runtime()
    return runtime.run_chat(message)


def generate_image_agent(message: str, images: str | list[str] | None = None):  # noqa: ANN201
    runtime = get_runtime()
    return runtime.run_generate_image(message, images=images)


def clear_chat_history() -> None:
    get_runtime().reset_chat()


def clear_personalization_history() -> None:
    get_runtime().clear_personalization_history()


def get_initial_greeting() -> str:
    return get_runtime().start()


def main() -> None:
    """Entry point for personal-assistant CLI."""
    from personal_assistant.ui.app import launch_app

    launch_app()


__all__ = [
    "Agent",
    "AssistantMessage",
    "Command",
    "Conversation",
    "CreatedNote",
    "Event",
    "Message",
    "MessageChunk",
    "MessageCompleted",
    "MessageStarted",
    "NoteDeleted",
    "NoteUpdated",
    "Prompts",
    "PromptSnapshot",
    "ToolCallEvent",
    "ToolResultMessage",
    "TurnCompleted",
    "TurnStarted",
    "UserCommand",
    "UserMessage",
    "ai_spirit_agent",
    "chat_agent",
    "clear_chat_history",
    "clear_personalization_history",
    "generate_image_agent",
    "get_initial_greeting",
    "get_prompt",
    "get_runtime",
    "personalize_agent",
    "sage_agent",
    "shutdown_application",
]
