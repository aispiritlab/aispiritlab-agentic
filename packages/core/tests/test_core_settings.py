"""Core settings and the shared agent protocol.

Every test constructs ``Settings`` with ``_env_file=None``: the class points at
``.env`` by default, and a test that reads the developer's real ``.env`` would
pass or fail depending on their local secrets.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.agent import Agent
from core.settings import Settings


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_defaults_point_at_the_local_mlflow_server() -> None:
    settings = _settings()

    assert settings.mlflow_tracking_uri == "http://127.0.0.1:5001"
    assert settings.mlflow_registry_uri == "http://127.0.0.1:5001"


def test_tracking_and_registry_share_one_server_by_default() -> None:
    # `make mlflow-ui` starts a single server; splitting them is opt-in.
    settings = _settings()

    assert settings.mlflow_tracking_uri == settings.mlflow_registry_uri


def test_experiment_names_have_defaults() -> None:
    settings = _settings()

    assert settings.mlflow_experiment_name == "AI Spirit"
    assert settings.mlflow_evaluation_experiment_name == "AI Spirit/evaluation"


def test_debug_is_off_by_default() -> None:
    assert _settings().debug is False


# ---------------------------------------------------------------------------
# Environment binding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env_name", "field"),
    [
        ("MLFLOW_TRACKING_URI", "mlflow_tracking_uri"),
        ("MLFLOW_REGISTRY_URI", "mlflow_registry_uri"),
        ("MLFLOW_EXPERIMENT_NAME", "mlflow_experiment_name"),
        ("MLFLOW_EVALUATION_EXPERIMENT_NAME", "mlflow_evaluation_experiment_name"),
    ],
)
def test_fields_are_populated_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, env_name: str, field: str
) -> None:
    monkeypatch.setenv(env_name, "from-env")

    assert getattr(_settings(), field) == "from-env"


def test_environment_names_are_matched_case_insensitively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("mlflow_tracking_uri", "http://lowercase:5001")

    assert _settings().mlflow_tracking_uri == "http://lowercase:5001"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1", True), ("true", True), ("True", True), ("0", False), ("false", False)],
)
def test_debug_is_parsed_from_a_string(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool
) -> None:
    monkeypatch.setenv("DEBUG", raw)

    assert _settings().debug is expected


def test_unknown_environment_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    # The repo's .env carries keys for every package; core must not choke on them.
    monkeypatch.setenv("SOME_UNRELATED_PACKAGE_KEY", "value")

    assert _settings().mlflow_tracking_uri == "http://127.0.0.1:5001"


def test_explicit_arguments_win_over_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://from-env:5001")

    assert _settings(mlflow_tracking_uri="http://explicit:5001").mlflow_tracking_uri == (
        "http://explicit:5001"
    )


def test_the_module_level_settings_object_is_a_settings_instance() -> None:
    from core import settings as module_settings

    assert isinstance(module_settings, Settings)


# ---------------------------------------------------------------------------
# Agent protocol
# ---------------------------------------------------------------------------


def test_anything_with_a_handler_satisfies_the_agent_protocol() -> None:
    class Handler:
        def handler(self, context: Any) -> Any:
            return context

    # Structural, so low-level packages can type against an agent without
    # importing the SDK.
    assert isinstance(Handler(), Agent)


def test_an_object_without_a_handler_does_not_satisfy_the_protocol() -> None:
    class NotAnAgent:
        def run(self, context: Any) -> Any:
            return context

    assert not isinstance(NotAnAgent(), Agent)


def test_the_protocol_needs_no_inheritance() -> None:
    class Duck:
        def handler(self, context: Any) -> Any:
            return context

    assert Agent not in Duck.__mro__
    assert isinstance(Duck(), Agent)
