"""Prompt registry: catalogue integrity and the MLflow boundary.

The prompts themselves are the product here — a missing constant or a renamed
enum member breaks agent construction at runtime rather than at import. MLflow
is stubbed throughout: these tests must not need a running tracking server.
"""

from __future__ import annotations

from typing import Any

import pytest

from registry import Prompts
from registry import prompts as prompts_module
from registry.main import init_registry_prompt
from registry.prompts import (
    DECISION_PROMPT,
    DISCOVERY_NOTES_PROMPT,
    GREETING_PROMPT,
    MANAGE_NOTES_PROMPT,
    ORGANIZER_PROMPT,
    SAGE_PROMPT,
    get_prompt,
)
from registry.register_prompt import RegisterPrompt, register_prompt

ALL_PROMPT_TEXTS = {
    "greeting": GREETING_PROMPT,
    "manage_notes": MANAGE_NOTES_PROMPT,
    "discovery_notes": DISCOVERY_NOTES_PROMPT,
    "organizer": ORGANIZER_PROMPT,
    "sage": SAGE_PROMPT,
    "decision": DECISION_PROMPT,
}


class FakeGenai:
    def __init__(self) -> None:
        self.registered: list[dict[str, Any]] = []
        self.loaded: list[str] = []
        self.template = "TEMPLATE FROM MLFLOW"

    def register_prompt(self, **kwargs: Any) -> None:
        self.registered.append(kwargs)

    def load_prompt(self, name: str) -> Any:
        self.loaded.append(name)
        return type("Prompt", (), {"template": self.template})()


class FakeMlflow:
    def __init__(self) -> None:
        self.genai = FakeGenai()
        self.registry_uris: list[str] = []

    def set_registry_uri(self, uri: str) -> None:
        self.registry_uris.append(uri)


@pytest.fixture
def mlflow_stub(monkeypatch: pytest.MonkeyPatch) -> FakeMlflow:
    from registry import register_prompt as register_module

    stub = FakeMlflow()
    monkeypatch.setattr(prompts_module, "mlflow", stub)
    monkeypatch.setattr(register_module, "mlflow", stub)
    return stub


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


def test_every_enum_member_has_a_string_value_equal_to_its_name() -> None:
    for member in Prompts:
        assert member.value == member.name.lower()


def test_prompts_is_a_str_enum_usable_as_a_plain_name() -> None:
    assert Prompts.SAGE == "sage"
    assert f"{Prompts.SAGE}" == "sage"


@pytest.mark.parametrize(("name", "text"), sorted(ALL_PROMPT_TEXTS.items()))
def test_each_registered_prompt_has_non_empty_content(name: str, text: str) -> None:
    assert name in {member.value for member in Prompts}
    assert text.strip(), f"{name} prompt is empty"


@pytest.mark.parametrize(("name", "text"), sorted(ALL_PROMPT_TEXTS.items()))
def test_prompt_text_carries_no_unresolved_placeholder(name: str, text: str) -> None:
    del name
    assert "TODO" not in text
    assert "{{" not in text


def test_prompt_texts_are_distinct() -> None:
    # A copy-paste slip here silently gives two agents the same instructions.
    assert len(set(ALL_PROMPT_TEXTS.values())) == len(ALL_PROMPT_TEXTS)


def test_the_chat_prompt_is_declared_even_though_it_is_not_registered() -> None:
    # CHAT is served from code, not MLflow; keep the member so callers can
    # reference it uniformly.
    assert Prompts.CHAT.value == "chat"
    assert Prompts.CHAT.value not in ALL_PROMPT_TEXTS


# ---------------------------------------------------------------------------
# RegisterPrompt
# ---------------------------------------------------------------------------


def test_register_prompt_model_defaults() -> None:
    entry = RegisterPrompt(name="greeting", prompt="text")

    assert entry.commit_message == "Initial commit"
    assert entry.tags == {"author": "John Doe"}


def test_register_prompt_default_tags_are_not_shared_between_instances() -> None:
    first = RegisterPrompt(name="a", prompt="x")
    second = RegisterPrompt(name="b", prompt="y")

    first.tags["author"] = "Mateusz"

    assert second.tags == {"author": "John Doe"}


def test_register_prompt_requires_a_name_and_a_prompt() -> None:
    from pydantic import ValidationError

    # Validated from a payload rather than a call, which is how the field would
    # actually go missing (a hand-written entry in the registration script).
    with pytest.raises(ValidationError):
        RegisterPrompt.model_validate({"name": "only-name"})


# ---------------------------------------------------------------------------
# MLflow boundary
# ---------------------------------------------------------------------------


def test_register_prompt_points_mlflow_at_the_configured_registry(
    mlflow_stub: FakeMlflow,
) -> None:
    from core import settings

    register_prompt(RegisterPrompt(name="greeting", prompt="text"))

    assert mlflow_stub.registry_uris == [settings.mlflow_registry_uri]


def test_register_prompt_forwards_every_field(mlflow_stub: FakeMlflow) -> None:
    register_prompt(
        RegisterPrompt(
            name="greeting",
            prompt="text",
            commit_message="update",
            tags={"author": "Mateusz"},
        )
    )

    assert mlflow_stub.genai.registered == [
        {
            "name": "greeting",
            "template": "text",
            "commit_message": "update",
            "tags": {"author": "Mateusz"},
        }
    ]


def test_get_prompt_returns_the_template_from_the_registry(mlflow_stub: FakeMlflow) -> None:
    assert get_prompt("greeting") == "TEMPLATE FROM MLFLOW"
    assert mlflow_stub.genai.loaded == ["greeting"]


def test_get_prompt_sets_the_registry_uri_before_loading(mlflow_stub: FakeMlflow) -> None:
    from core import settings

    get_prompt("sage")

    assert mlflow_stub.registry_uris == [settings.mlflow_registry_uri]


def test_init_registry_prompt_registers_the_whole_catalogue(mlflow_stub: FakeMlflow) -> None:
    init_registry_prompt()

    registered = {entry["name"]: entry["template"] for entry in mlflow_stub.genai.registered}

    assert registered == ALL_PROMPT_TEXTS
