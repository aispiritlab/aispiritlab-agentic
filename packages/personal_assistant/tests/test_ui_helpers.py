"""Client-supplied identity resolution, streaming and error surfacing in the UI.

``active_user`` and ``active_workspace`` arrive from the browser on every
request. They are display names, and they end up selecting a filesystem-backed
profile — so the UI must map them through the registry rather than trust them.
"""

from __future__ import annotations

import pytest

from agentic_runtime.users import UserProfile
from agentic_runtime.workspaces import WorkspacePreset
from personal_assistant.ui import app
from providers.api.http_client import ModelConnectionError


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        app,
        "list_users",
        lambda: [
            UserProfile(name="Jan Kowalski", slug="jan-kowalski", created_at=""),
            UserProfile(name="Default", slug="default", created_at=""),
        ],
    )
    monkeypatch.setattr(
        app,
        "list_workspaces",
        lambda: [
            WorkspacePreset(slug="default", name="Default", description="", created_at=""),
            WorkspacePreset(slug="badania", name="Badania", description="", created_at=""),
        ],
    )


# ---------------------------------------------------------------------------
# Identity resolution
# ---------------------------------------------------------------------------


def test_resolve_user_slug_maps_a_display_name(registry: None) -> None:
    assert app._resolve_user_slug("Jan Kowalski") == "jan-kowalski"


def test_resolve_user_slug_accepts_a_registered_slug(registry: None) -> None:
    # The client sometimes round-trips the slug rather than the display name.
    assert app._resolve_user_slug("jan-kowalski") == "jan-kowalski"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "Nieznany",
        "jan kowalski",  # right person, wrong casing/format
        "JAN-KOWALSKI",
        "../../etc/passwd",
        "..",
        "/etc/passwd",
        "default/../jan-kowalski",
    ],
)
def test_resolve_user_slug_returns_none_for_anything_unregistered(
    registry: None, value: str
) -> None:
    # The regression this guards: an earlier version fell back to
    # `slug_map.get(name, name)`, echoing the client value straight into a path.
    assert app._resolve_user_slug(value) is None


def test_resolve_workspace_slug_maps_a_display_name(registry: None) -> None:
    assert app._resolve_workspace_slug("Badania") == "badania"


def test_resolve_workspace_slug_accepts_a_registered_slug(registry: None) -> None:
    assert app._resolve_workspace_slug("badania") == "badania"


@pytest.mark.parametrize("value", ["", "Nieznany", "../../etc", "..", "/etc/passwd"])
def test_resolve_workspace_slug_returns_none_for_anything_unregistered(
    registry: None, value: str
) -> None:
    assert app._resolve_workspace_slug(value) is None


def test_resolvers_return_none_when_the_registry_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app, "list_users", list)
    monkeypatch.setattr(app, "list_workspaces", list)

    assert app._resolve_user_slug("Default") is None
    assert app._resolve_workspace_slug("Default") is None


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_typing_chunks_yields_a_single_empty_chunk_for_empty_text() -> None:
    assert list(app._typing_chunks("")) == [""]


def test_typing_chunks_yields_the_whole_text_when_shorter_than_a_chunk() -> None:
    assert list(app._typing_chunks("krótko", chunk_size=24)) == ["krótko"]


def test_typing_chunks_yields_growing_prefixes() -> None:
    text = "abcdefghij"

    assert list(app._typing_chunks(text, chunk_size=3)) == [
        "abc",
        "abcdef",
        "abcdefghi",
        "abcdefghij",
    ]


def test_typing_chunks_does_not_repeat_the_final_prefix() -> None:
    # A text that is an exact multiple of the chunk size must not yield the
    # full string twice — that would re-render the chat for no reason.
    assert list(app._typing_chunks("abcdef", chunk_size=3)) == ["abc", "abcdef"]


@pytest.mark.parametrize("length", [1, 5, 23, 24, 25, 47, 48, 200])
@pytest.mark.parametrize("chunk_size", [1, 3, 24])
def test_typing_chunks_always_ends_with_the_complete_text(length: int, chunk_size: int) -> None:
    text = "x" * length
    chunks = list(app._typing_chunks(text, chunk_size=chunk_size))

    assert chunks[-1] == text
    assert all(text.startswith(chunk) for chunk in chunks)
    assert chunks == sorted(chunks, key=len)
    assert len(set(chunks)) == len(chunks)


def test_default_chunk_size_is_used_when_not_given() -> None:
    text = "y" * (app._STREAM_CHUNK_CHARS * 2)

    assert list(app._typing_chunks(text)) == [text[: app._STREAM_CHUNK_CHARS], text]


# ---------------------------------------------------------------------------
# Error surfacing
# ---------------------------------------------------------------------------


def test_connection_errors_get_an_actionable_message() -> None:
    message = app._user_facing_error(
        ModelConnectionError("connect to http://10.0.0.4:8080 failed")
    )

    assert "serwer LLM" in message


def test_value_errors_get_a_rephrase_hint() -> None:
    assert "sformułować" in app._user_facing_error(ValueError("bad input"))


def test_unknown_errors_fall_back_to_a_generic_message() -> None:
    assert app._user_facing_error(RuntimeError("boom")) == app._DEFAULT_ERROR_MESSAGE


@pytest.mark.parametrize(
    "error",
    [
        ModelConnectionError("POST http://192.168.1.50:11434/v1/chat failed: refused"),
        ValueError("/Users/mkubaszek/.aispiritagent/users/jan/personalization.json missing"),
        RuntimeError("Bearer sk-proj-abcdef123456"),
        KeyError("/etc/passwd"),
    ],
)
def test_error_messages_never_leak_the_exception_text(error: Exception) -> None:
    # Exception text carries endpoints, local paths and occasionally tokens.
    # It belongs in the log, not in the chat window.
    message = app._user_facing_error(error)
    leaky = ["http://", "https://", "/Users/", "/etc/", "sk-", "Bearer", "11434"]

    assert not any(fragment in message for fragment in leaky), message
    assert str(error) not in message


def test_subclasses_inherit_their_parent_mapping() -> None:
    class SpecificConnectionError(ModelConnectionError):
        pass

    assert (
        app._user_facing_error(SpecificConnectionError("x"))
        == app._ERROR_MESSAGES[ModelConnectionError]
    )
