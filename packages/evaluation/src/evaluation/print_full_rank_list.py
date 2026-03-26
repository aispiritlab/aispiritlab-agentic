from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_REPORT_PATH = Path(
    "packages/evaluation/src/evaluation/notes_openrouter_benchmark_report.json"
)


PREFERRED_COLUMN_ORDER = (
    "rank",
    "model_id",
    "status",
    "composite_score",
    "exact_tool_match_rate",
    "avg_exact_turn_match_rate",
    "avg_g_eval",
    "avg_tool_correctness",
    "avg_step_efficiency",
    "avg_task_completion",
    "avg_latency_ms",
    "scenarios_completed",
    "scenarios_total",
    "conversations_completed",
    "conversations_total",
    "cases_completed",
    "cases_total",
    "requested_model_id",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print `full_rank_list` from a benchmark JSON report as a table.",
    )
    parser.add_argument(
        "report",
        nargs="?",
        default=str(DEFAULT_REPORT_PATH),
        help=(
            "Path to JSON report. Defaults to notes benchmark report "
            f"({DEFAULT_REPORT_PATH})."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Show only top N rows.",
    )
    return parser.parse_args(argv)


def _load_report(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise SystemExit(f"Report file not found: {path}") from error
    except OSError as error:
        raise SystemExit(f"Cannot read report file {path}: {error}") from error

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid JSON in {path}: {error}") from error

    if not isinstance(payload, dict):
        raise SystemExit(f"Unexpected JSON root in {path}: expected object.")
    return payload


def _get_full_rank_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("full_rank_list")
    if rows is None:
        raise SystemExit("Missing `full_rank_list` in report.")
    if not isinstance(rows, list):
        raise SystemExit("`full_rank_list` must be a list.")

    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise SystemExit(f"`full_rank_list[{index - 1}]` must be an object.")
        normalized.append(row)
    return normalized


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _select_columns(rows: list[dict[str, Any]]) -> list[str]:
    present = {key for row in rows for key in row.keys()}
    ordered = [key for key in PREFERRED_COLUMN_ORDER if key in present]
    remaining = sorted(present - set(ordered))
    return ordered + remaining


def _render_table(rows: list[dict[str, Any]], *, top: int | None = None) -> str:
    if top is not None and top >= 0:
        rows = rows[:top]

    if not rows:
        return "No rows in full_rank_list."

    columns = _select_columns(rows)
    rendered_rows = [
        {column: _format_value(row.get(column)) for column in columns}
        for row in rows
    ]

    widths = {
        column: max(len(column), *(len(row[column]) for row in rendered_rows))
        for column in columns
    }

    def render_separator() -> str:
        return "+-" + "-+-".join("-" * widths[column] for column in columns) + "-+"

    def render_row(values: dict[str, str]) -> str:
        cells = []
        for column in columns:
            text = values[column]
            if _looks_numeric(text):
                cells.append(text.rjust(widths[column]))
            else:
                cells.append(text.ljust(widths[column]))
        return "| " + " | ".join(cells) + " |"

    header_values = {column: column for column in columns}
    parts = [render_separator(), render_row(header_values), render_separator()]
    parts.extend(render_row(row) for row in rendered_rows)
    parts.append(render_separator())
    return "\n".join(parts)


def _looks_numeric(text: str) -> bool:
    if not text:
        return False
    try:
        float(text)
    except ValueError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report_path = Path(args.report).expanduser()
    payload = _load_report(report_path)
    rows = _get_full_rank_list(payload)
    print(_render_table(rows, top=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

