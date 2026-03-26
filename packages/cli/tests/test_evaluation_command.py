from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

import evaluation
from cli.main import cli


def test_evaluation_optimize_prompt_command_uses_definition(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeOptimization:
        def __init__(self, **kwargs) -> None:
            calls.update(kwargs)

        def run(self) -> Path:
            return Path("/tmp/optimized.txt")

    monkeypatch.setattr(evaluation, "AgentPromptOptimization", FakeOptimization)
    monkeypatch.setattr(
        evaluation,
        "load_evaluation_definition",
        lambda spec: {"definition": spec},
    )

    result = CliRunner().invoke(
        cli,
        [
            "evaluation",
            "optimize-prompt",
            "--definition",
            "agentic_runtime.manage_notes.evaluation:NOTES_EVALUATION",
            "--runtime-option",
            "vault_path=/tmp/vault",
        ],
    )

    assert result.exit_code == 0
    assert "Zapisano zoptymalizowany prompt: /tmp/optimized.txt" in result.output
    assert calls["definition"] == {
        "definition": "agentic_runtime.manage_notes.evaluation:NOTES_EVALUATION"
    }
    assert calls["runtime_options"] == {"vault_path": "/tmp/vault"}
