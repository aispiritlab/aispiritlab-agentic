from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import Session, Trace
from mlflow.genai.scorers import scorer
import pandas as pd

from agentic_runtime.fine_tuning import export_agent_fine_tuning_rows
from agentic_runtime.trace import get_experiment_id
from evaluation.contracts import EvaluationDefinition
from evaluation.definition_loader import load_evaluation_definition
from evaluation.eval_dataset import build_conversation_examples


def _message_content(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    return str(content) if content is not None else ""


def _last_assistant_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message
    return None


def build_trace_dataset_records(
    store_path: str | Path,
    *,
    runtime_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    rows = export_agent_fine_tuning_rows(store_path, runtime_id=runtime_id)
    if limit is not None:
        rows = rows[: max(limit, 0)]

    records: list[dict[str, Any]] = []
    for row in rows:
        messages = list(row.get("messages", []))
        last_assistant = _last_assistant_message(messages)
        records.append(
            {
                "inputs": {
                    "messages": messages[:-1] if last_assistant is not None else messages,
                    "tools": list(row.get("tools", [])),
                },
                "outputs": _message_content(last_assistant) if last_assistant is not None else "",
                "expectations": {
                    "final_assistant": last_assistant,
                    "tool_count": len(row.get("tools", [])),
                },
                "metadata": {
                    **dict(row.get("metadata", {})),
                    "dataset_source": "traces",
                },
            }
        )
    return records


def build_conversation_dataset_records(
    definition: EvaluationDefinition,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    examples = build_conversation_examples(definition)
    if limit is not None:
        examples = examples[: max(limit, 0)]

    records: list[dict[str, Any]] = []
    for example in examples:
        messages = list(example.get("messages", []))
        last_assistant = _last_assistant_message(messages)
        records.append(
            {
                "inputs": {
                    "messages": messages[:-1] if last_assistant is not None else messages,
                    "conversation_name": example.get("name"),
                    "steps": list(example.get("steps", [])),
                },
                "outputs": _message_content(last_assistant) if last_assistant is not None else "",
                "expectations": {
                    "conversation_name": example.get("name"),
                    "steps": list(example.get("steps", [])),
                    "full_conversation": messages,
                },
                "metadata": {
                    "evaluation_definition": definition.name,
                    "dataset_source": "synthetic",
                    "conversation_name": example.get("name"),
                },
            }
        )
    return records


@dataclass(frozen=True, slots=True)
class DatasetSyncResult:
    dataset_id: str
    dataset_name: str
    record_count: int
    sources: tuple[str, ...]


def sync_mlflow_dataset(
    *,
    dataset_name: str,
    records: list[dict[str, Any]],
    experiment_id: str | list[str] | None = None,
    tags: dict[str, Any] | None = None,
) -> DatasetSyncResult:
    experiment_ref = experiment_id or get_experiment_id(evaluation=True) or get_experiment_id()
    dataset = mlflow.genai.create_dataset(
        name=dataset_name,
        experiment_id=experiment_ref,
        tags=tags or {},
    )
    if records:
        dataset.merge_records(records)
    sources = tuple(
        sorted(
            {
                str(record.get("metadata", {}).get("dataset_source", "unknown"))
                for record in records
            }
        )
    )
    return DatasetSyncResult(
        dataset_id=dataset.dataset_id,
        dataset_name=dataset.name,
        record_count=len(records),
        sources=sources,
    )


def sync_dataset_from_definition(
    *,
    dataset_name: str,
    definition_spec: str,
    store_path: str | Path | None = None,
    runtime_id: str | None = None,
    source: str = "hybrid",
    limit: int | None = None,
) -> DatasetSyncResult:
    definition = load_evaluation_definition(definition_spec)
    normalized_source = source.strip().lower()
    allowed_sources = {"hybrid", "synthetic", "traces"}
    if normalized_source not in allowed_sources:
        raise ValueError(
            f"Unsupported dataset source '{source}'. Expected one of: {', '.join(sorted(allowed_sources))}."
        )
    if normalized_source == "traces" and store_path is None:
        raise ValueError("Trace datasets require --store-path (or store_path) to be set.")
    records: list[dict[str, Any]] = []

    if normalized_source in {"hybrid", "synthetic"}:
        records.extend(build_conversation_dataset_records(definition, limit=limit))
    if normalized_source in {"hybrid", "traces"} and store_path is not None:
        records.extend(build_trace_dataset_records(store_path, runtime_id=runtime_id, limit=limit))

    return sync_mlflow_dataset(
        dataset_name=dataset_name,
        records=records,
        tags={
            "definition": definition.name,
            "source": normalized_source,
        },
    )


def _count_retry_attempts(trace: Trace) -> int:
    retries = 0
    for span in trace.search_spans():
        attempt_no = span.get_attribute("attempt_no")
        if isinstance(attempt_no, int) and attempt_no > 1:
            retries += attempt_no - 1
    return retries


def _count_loop_iterations(trace: Trace) -> int:
    loop_iterations = 0
    for span in trace.search_spans():
        iteration = span.get_attribute("loop_iteration")
        if isinstance(iteration, int):
            loop_iterations = max(loop_iterations, iteration)
    return loop_iterations


@scorer  # type: ignore[misc]
def retry_burden(trace: Trace) -> int:
    return _count_retry_attempts(trace)


@scorer  # type: ignore[misc]
def loop_iteration_burden(trace: Trace) -> int:
    return _count_loop_iterations(trace)


@scorer  # type: ignore[misc]
def stream_trace_alignment(trace: Trace, expectations: dict[str, Any] | None = None) -> bool:
    if expectations is None:
        return True
    count = expectations.get("stream_record_count")
    if isinstance(count, int):
        return count > 0
    return True


@scorer  # type: ignore[misc]
def session_turn_count(session: Session) -> int:
    return len(session)


@scorer  # type: ignore[misc]
def session_retry_burden(session: Session) -> int:
    return sum(_count_retry_attempts(trace) for trace in session.traces)


@scorer  # type: ignore[misc]
def session_loop_iteration_burden(session: Session) -> int:
    return sum(_count_loop_iterations(trace) for trace in session.traces)


def _stream_record_counts(
    store_path: str | Path | None,
) -> dict[str, int]:
    if store_path is None:
        return {}

    import sqlite3

    with sqlite3.connect(Path(store_path).expanduser()) as connection:
        rows = connection.execute(
            """
            SELECT trace_id, COUNT(*)
            FROM message_stream
            WHERE trace_id IS NOT NULL AND trace_id != ''
            GROUP BY trace_id
            """
        ).fetchall()
    return {str(trace_id): int(count) for trace_id, count in rows if trace_id}


def _trace_evaluation_rows(
    traces: list[Trace],
    *,
    store_path: str | Path | None = None,
) -> pd.DataFrame:
    stream_counts = _stream_record_counts(store_path)
    rows = [
        {
            "trace": trace,
            "inputs": {
                "trace_id": trace.info.trace_id,
                "request_preview": trace.info.request_preview,
            },
            "outputs": trace.info.response_preview,
            "expectations": {
                "stream_record_count": stream_counts.get(trace.info.trace_id, 0),
            },
        }
        for trace in traces
    ]
    return pd.DataFrame(rows)


def _session_evaluation_rows(sessions: list[Session]) -> pd.DataFrame:
    rows = []
    for session in sessions:
        last_trace = session.traces[-1] if session.traces else None
        rows.append(
            {
                "session": session,
                "inputs": {
                    "session_id": session.id,
                    "turn_count": len(session),
                },
                "outputs": {
                    "final_trace_id": last_trace.info.trace_id if last_trace is not None else None,
                    "final_response_preview": (
                        last_trace.info.response_preview if last_trace is not None else None
                    ),
                },
                "expectations": {
                    "turn_count": len(session),
                },
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True, slots=True)
class MLflowEvaluationSummary:
    row_count: int
    metric_keys: tuple[str, ...]


def evaluate_mlflow_traces(
    *,
    experiment_id: str | None = None,
    max_results: int = 100,
    filter_string: str | None = None,
    store_path: str | Path | None = None,
) -> tuple[Any, MLflowEvaluationSummary]:
    experiment_ref = [experiment_id] if experiment_id else None
    traces = mlflow.search_traces(
        experiment_ids=experiment_ref,
        filter_string=filter_string,
        max_results=max_results,
        return_type="list",
    )
    frame = _trace_evaluation_rows(traces, store_path=store_path)
    result = mlflow.genai.evaluate(
        data=frame,
        scorers=[retry_burden, loop_iteration_burden, stream_trace_alignment],
    )
    metrics = tuple(sorted(getattr(result, "metrics", {}).keys()))
    return result, MLflowEvaluationSummary(row_count=len(frame.index), metric_keys=metrics)


def evaluate_mlflow_sessions(
    *,
    max_results: int = 100,
) -> tuple[Any, MLflowEvaluationSummary]:
    sessions = mlflow.search_sessions(max_results=max_results)
    frame = _session_evaluation_rows(sessions)
    result = mlflow.genai.evaluate(
        data=frame,
        scorers=[session_turn_count, session_retry_burden, session_loop_iteration_burden],
    )
    metrics = tuple(sorted(getattr(result, "metrics", {}).keys()))
    return result, MLflowEvaluationSummary(row_count=len(frame.index), metric_keys=metrics)


def write_evaluation_summary(
    summary: MLflowEvaluationSummary,
    output_path: str | Path,
) -> Path:
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return destination
