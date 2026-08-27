"""Personal Assistant — multi-agent system for note management, search, and decision support."""

from __future__ import annotations

import contextvars
import threading
from typing import TYPE_CHECKING

from agentic.workflow.messages import (
    AssistantMessage,
    Command,
    Conversation,
    ConversationData,
    Event,
    Message,
    MessageChunk,
    MessageCompleted,
    MessageStarted,
    PromptSnapshot,
    RecordedMessageMetadata,
    ToolCallEvent,
    ToolResultMessage,
    TurnCompleted,
    TurnStarted,
    UserCommand,
    UserMessage,
)
from agentic_runtime.workspaces import build_session_id, get_active_workspace, parse_session_id
from personal_assistant.agents.personalize.personalize_agent import PersonalizeAgent as Agent
from personal_assistant.messaging.events import CreatedNote, NoteDeleted, NoteUpdated
from providers.orchestrator import ModelProvider
from registry import Prompts, get_prompt

if TYPE_CHECKING:
    from personal_assistant.runtime import PARuntime

# ---------------------------------------------------------------------------
# Per-request user context (thread-safe via contextvars)
# ---------------------------------------------------------------------------

_active_user: contextvars.ContextVar[str] = contextvars.ContextVar(
    "active_user", default="default"
)
_runtimes: dict[str, PARuntime] = {}
_runtimes_lock = threading.Lock()


def get_active_user() -> str:
    return _active_user.get()


def set_active_user(name: str) -> contextvars.Token[str]:
    return _active_user.set(name)


def _resolve_user(user: str | None) -> str:
    return user or _active_user.get()


def _resolve_workspace(workspace: str | None) -> str:
    return workspace or get_active_workspace()


def _runtime_cache_key(user: str, workspace: str) -> str:
    return build_session_id(user, workspace)


def _close_knowledge_base() -> None:
    try:
        from knowledge_base import close_knowledge_base
    except ModuleNotFoundError:
        return
    close_knowledge_base()


def get_runtime(user: str | None = None, workspace: str | None = None) -> PARuntime:
    resolved_user = _resolve_user(user)
    resolved_workspace = _resolve_workspace(workspace)
    cache_key = _runtime_cache_key(resolved_user, resolved_workspace)
    with _runtimes_lock:
        if cache_key not in _runtimes:
            from personal_assistant.settings import settings

            if settings.agentic_transport == "redis_streams":
                from agentic_runtime.distributed.runtime import DistributedAgenticRuntime

                runtime = DistributedAgenticRuntime.from_settings()
                runtime.user_slug = resolved_user
                runtime.workspace_slug = resolved_workspace
                runtime.session_id = cache_key
                _runtimes[cache_key] = runtime  # type: ignore[assignment]
            else:
                from personal_assistant.runtime import PARuntime

                _runtimes[cache_key] = PARuntime(
                    user_slug=resolved_user,
                    workspace_slug=resolved_workspace,
                )
    return _runtimes[cache_key]


def switch_user(name: str, workspace: str | None = None) -> str:
    """Switch active user and return the greeting from their runtime."""
    _active_user.set(name)
    return get_runtime(name, workspace=workspace).start()


def drop_runtime_sessions(*, user: str | None = None, workspace: str | None = None) -> None:
    runtimes_to_stop: list[PARuntime] = []
    with _runtimes_lock:
        removable_keys = [
            key
            for key in _runtimes
            if (
                (user is None or parse_session_id(key)[0] == user)
                and (workspace is None or parse_session_id(key)[1] == workspace)
            )
        ]
        for key in removable_keys:
            runtimes_to_stop.append(_runtimes.pop(key))

    for runtime in runtimes_to_stop:
        try:
            runtime.stop()
        except Exception:
            pass


def _reset_runtime() -> None:
    drop_runtime_sessions(
        user=_active_user.get(),
        workspace=get_active_workspace(),
    )


def shutdown_application(*resources: object) -> None:
    with _runtimes_lock:
        runtimes_to_stop = list(_runtimes.values())
        _runtimes.clear()

    for runtime in runtimes_to_stop:
        try:
            runtime.stop()
        except Exception:
            pass

    for resource in resources:
        close = getattr(resource, "close", None)
        if callable(close):
            close()

    ModelProvider.shutdown_all()
    _close_knowledge_base()


def personalize_agent(
    message: str,
    user: str | None = None,
    workspace: str | None = None,
) -> str:
    resolved = _resolve_user(user)
    resolved_workspace = _resolve_workspace(workspace)
    runtime = get_runtime(resolved, workspace=resolved_workspace)
    return runtime.handle(
        UserMessage(
            data=ConversationData(role="user", text=message),
            metadata=RecordedMessageMetadata(
                runtime_id=runtime.runtime_id,
                session_id=runtime.session_id,
                domain="personalize",
                source="user",
                target="personalize",
            ),
        )
    )


def sage_agent(
    message: str,
    user: str | None = None,
    workspace: str | None = None,
) -> str:
    resolved = _resolve_user(user)
    resolved_workspace = _resolve_workspace(workspace)
    runtime = get_runtime(resolved, workspace=resolved_workspace)
    return runtime.handle(
        UserMessage(
            data=ConversationData(role="user", text=message),
            metadata=RecordedMessageMetadata(
                runtime_id=runtime.runtime_id,
                session_id=runtime.session_id,
                domain="sage",
                source="user",
                target="sage",
            ),
        )
    )


def ai_spirit_agent(
    message: str,
    user: str | None = None,
    workspace: str | None = None,
) -> str:
    resolved = _resolve_user(user)
    resolved_workspace = _resolve_workspace(workspace)
    return get_runtime(resolved, workspace=resolved_workspace).run(message)


def chat_agent(
    message: str,
    user: str | None = None,
    workspace: str | None = None,
) -> str:
    resolved = _resolve_user(user)
    resolved_workspace = _resolve_workspace(workspace)
    return get_runtime(resolved, workspace=resolved_workspace).run_chat(message)


def generate_image_agent(
    message: str,
    images: str | list[str] | None = None,
    user: str | None = None,
    workspace: str | None = None,
):
    resolved = _resolve_user(user)
    resolved_workspace = _resolve_workspace(workspace)
    return get_runtime(resolved, workspace=resolved_workspace).run_generate_image(
        message,
        images=images,
    )


def clear_chat_history(user: str | None = None, workspace: str | None = None) -> None:
    resolved = _resolve_user(user)
    resolved_workspace = _resolve_workspace(workspace)
    get_runtime(resolved, workspace=resolved_workspace).reset_chat()


def clear_personalization_history(
    user: str | None = None,
    workspace: str | None = None,
) -> None:
    resolved = _resolve_user(user)
    resolved_workspace = _resolve_workspace(workspace)
    get_runtime(resolved, workspace=resolved_workspace).clear_personalization_history()


def get_initial_greeting(user: str | None = None, workspace: str | None = None) -> str:
    resolved = _resolve_user(user)
    resolved_workspace = _resolve_workspace(workspace)
    return get_runtime(resolved, workspace=resolved_workspace).start()


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
    "PromptSnapshot",
    "Prompts",
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
    "drop_runtime_sessions",
    "generate_image_agent",
    "get_active_user",
    "get_initial_greeting",
    "get_prompt",
    "get_runtime",
    "personalize_agent",
    "sage_agent",
    "set_active_user",
    "shutdown_application",
    "switch_user",
]
