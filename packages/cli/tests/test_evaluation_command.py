from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from cli.main import cli
import evaluation


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


def test_sync_mlflow_dataset_command(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluation,
        "sync_dataset_from_definition",
        lambda **kwargs: evaluation.DatasetSyncResult(
            dataset_id="d-1",
            dataset_name=kwargs["dataset_name"],
            record_count=12,
            sources=("synthetic", "traces"),
        ),
    )

    result = CliRunner().invoke(
        cli,
        [
            "evaluation",
            "sync-mlflow-dataset",
            "--dataset-name",
            "notes-hybrid",
        ],
    )

    assert result.exit_code == 0
    assert "MLflow dataset synced: notes-hybrid (d-1), records=12" in result.output


def test_evaluate_mlflow_traces_command(monkeypatch, tmp_path: Path) -> None:
    summary = evaluation.MLflowEvaluationSummary(
        row_count=5,
        metric_keys=("retry_burden", "stream_trace_alignment"),
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        evaluation,
        "evaluate_mlflow_traces",
        lambda **kwargs: (object(), summary),
    )

    def _write_summary(payload, output_path):
        calls["summary"] = payload
        calls["output_path"] = output_path
        return tmp_path / "summary.json"

    monkeypatch.setattr(evaluation, "write_evaluation_summary", _write_summary)

    result = CliRunner().invoke(
        cli,
        [
            "evaluation",
            "evaluate-mlflow-traces",
            "--summary-output",
            str(tmp_path / "summary.json"),
        ],
    )

    assert result.exit_code == 0
    assert "MLflow trace evaluation rows=5" in result.output
    assert calls["summary"] == summary


def test_evaluate_mlflow_conversations_command(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluation,
        "evaluate_mlflow_sessions",
        lambda **kwargs: (
            object(),
            evaluation.MLflowEvaluationSummary(
                row_count=3,
                metric_keys=("session_turn_count",),
            ),
        ),
    )

    result = CliRunner().invoke(
        cli,
        [
            "evaluation",
            "evaluate-mlflow-conversations",
        ],
    )

    assert result.exit_code == 0
    assert "MLflow conversation evaluation rows=3" in result.output


def test_sync_mlflow_dataset_command_shows_error_on_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluation,
        "sync_dataset_from_definition",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("bad config")),
    )

    result = CliRunner().invoke(
        cli,
        ["evaluation", "sync-mlflow-dataset", "--dataset-name", "test"],
    )

    assert result.exit_code == 1
    assert "bad config" in result.output


def test_sync_mlflow_dataset_command_requires_store_path_for_trace_source(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluation,
        "sync_dataset_from_definition",
        lambda **kwargs: (_ for _ in ()).throw(
            ValueError("Trace datasets require --store-path (or store_path) to be set.")
        ),
    )

    result = CliRunner().invoke(
        cli,
        [
            "evaluation",
            "sync-mlflow-dataset",
            "--dataset-name",
            "notes-traces",
            "--source",
            "traces",
        ],
    )

    assert result.exit_code == 1
    assert "--store-path" in result.output


def test_evaluate_mlflow_traces_command_shows_error_on_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluation,
        "evaluate_mlflow_traces",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("mlflow down")),
    )

    result = CliRunner().invoke(
        cli,
        ["evaluation", "evaluate-mlflow-traces"],
    )

    assert result.exit_code == 1
    assert "mlflow down" in result.output


def test_evaluate_mlflow_conversations_command_shows_error_on_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluation,
        "evaluate_mlflow_sessions",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("mlflow down")),
    )

    result = CliRunner().invoke(
        cli,
        ["evaluation", "evaluate-mlflow-conversations"],
    )

    assert result.exit_code == 1
    assert "mlflow down" in result.output
