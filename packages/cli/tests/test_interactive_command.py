from __future__ import annotations

import importlib

import click
from click.testing import CliRunner
import pytest

from cli.main import cli

cli_main = importlib.import_module("cli.main")


def test_cli_without_command_launches_textual_chat(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(cli_main, "launch_textual_chat", lambda: calls.append("launched"))

    result = CliRunner().invoke(cli, [])

    assert result.exit_code == 0
    assert calls == ["launched"]


def test_runtime_interactive_launches_textual_chat(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(cli_main, "launch_textual_chat", lambda: calls.append("launched"))

    result = CliRunner().invoke(cli, ["runtime-interactive"])

    assert result.exit_code == 0
    assert calls == ["launched"]


def test_launch_textual_chat_reports_missing_textual_dependency(monkeypatch) -> None:
    def fake_loader() -> None:
        error = ModuleNotFoundError("No module named 'textual'")
        error.name = "textual"
        raise error

    monkeypatch.setattr(cli_main, "_load_textual_chat_runner", fake_loader)

    with pytest.raises(click.ClickException, match="Interactive chat requires `textual` and `rich`"):
        cli_main.launch_textual_chat()


def test_chat_command_keeps_single_query_behavior(monkeypatch) -> None:
    monkeypatch.setattr(cli_main, "ai_spirit_agent", lambda query: f"response:{query}")

    result = CliRunner().invoke(cli, ["chat", "hello", "world"])

    assert result.exit_code == 0
    assert "response:hello world" in result.output
