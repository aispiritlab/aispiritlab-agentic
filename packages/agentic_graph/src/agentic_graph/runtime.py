"""Runtime preview execution for AgentGraph."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Literal

from agentic_graph.builder import AgenticGraphBuilder
from agentic_graph.compiler import build_compiled_graph_system
from agentic_graph.models import AgentGraph

RuntimeStatus = Literal["ok", "error"]


@dataclass(frozen=True, slots=True)
class RuntimeOutput:
    output_node_id: str
    output_name: str
    path: str


@dataclass(frozen=True, slots=True)
class RuntimeEventRecord:
    index: int
    type_name: str
    kind: str
    source: str
    target: str
    status: str
    text: str
    turn_id: str = ""
    session_id: str = ""
    runtime_id: str = ""
    detail_markdown: str = ""


@dataclass(frozen=True, slots=True)
class RuntimeExecutionResult:
    status: RuntimeStatus
    entry_agent: str
    response: str
    steps: tuple[str, ...] = ()
    outputs: tuple[RuntimeOutput, ...] = ()
    events: tuple[RuntimeEventRecord, ...] = ()


def _normalize_text(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _truncate(value: str, max_len: int = 60) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def _format_event_detail(message: object) -> str:
    kind = getattr(message, "kind", "?")
    msg_type = type(message).__name__
    metadata = getattr(message, "metadata", None)

    lines = [
        f"### {msg_type}",
        f"**Kind:** `{kind}`",
    ]
    if metadata is not None:
        lines.append(f"**Source:** `{getattr(metadata, 'source', '')}`")
        lines.append(f"**Target:** `{getattr(metadata, 'target', '')}`")
        turn_id = getattr(metadata, "turn_id", "")
        if turn_id:
            lines.append(f"**Turn:** `{turn_id}`")
        session_id = getattr(metadata, "session_id", "")
        if session_id:
            lines.append(f"**Session:** `{session_id}`")
        status = getattr(metadata, "status", "")
        if status:
            lines.append(f"**Status:** `{status}`")
        runtime_id = getattr(metadata, "runtime_id", "")
        if runtime_id:
            lines.append(f"**Runtime:** `{runtime_id}`")

    data = getattr(message, "data", None)
    if data:
        if hasattr(data, "text") and data.text:
            lines.extend(["", "**Text:**", f"```\n{data.text}\n```"])
        elif isinstance(data, dict) and data:
            lines.extend(
                ["", "**Data:**", f"```json\n{json.dumps(data, indent=2, ensure_ascii=False)}\n```"]
            )

    return "\n".join(lines)


def _build_event_record(index: int, message: object) -> RuntimeEventRecord:
    kind = getattr(message, "kind", "?")
    msg_type = type(message).__name__
    metadata = getattr(message, "metadata", None)
    data = getattr(message, "data", None)

    if hasattr(data, "text") and data.text:
        text = _truncate(data.text)
    elif isinstance(data, dict):
        text = _truncate(json.dumps(data, ensure_ascii=False))
    else:
        text = ""

    return RuntimeEventRecord(
        index=index,
        type_name=msg_type,
        kind=_normalize_text(kind),
        source=_normalize_text(getattr(metadata, "source", "")),
        target=_normalize_text(getattr(metadata, "target", "")),
        status=_normalize_text(getattr(metadata, "status", "")),
        text=text,
        turn_id=_normalize_text(getattr(metadata, "turn_id", "")),
        session_id=_normalize_text(getattr(metadata, "session_id", "")),
        runtime_id=_normalize_text(getattr(metadata, "runtime_id", "")),
        detail_markdown=_format_event_detail(message),
    )


class GraphRuntime:
    """Execute an AgentGraph directly for local preview."""

    def __init__(
        self,
        graph: AgentGraph,
        *,
        runtime_secrets: dict[str, str] | None = None,
    ) -> None:
        self._graph = graph
        self._runtime_secrets = {
            node_id: _normalize_text(value).strip()
            for node_id, value in (runtime_secrets or {}).items()
            if _normalize_text(value).strip()
        }
        self._builder = AgenticGraphBuilder(graph, runtime_secrets=self._runtime_secrets)
        self._system = None

    def run(self, message: str | None) -> RuntimeExecutionResult:
        text = _normalize_text(message).strip()
        if not text:
            raise ValueError("Runtime preview requires a non-empty input message.")

        issues = self._builder.validate()
        errors = [issue for issue in issues if issue.level == "error"]
        if errors:
            raise ValueError("\n".join(issue.message for issue in errors))

        entry_alias = self._builder.entry_alias()
        if entry_alias is None:
            raise ValueError("The graph requires an entry agent before runtime preview.")

        self._system = build_compiled_graph_system(
            self._graph,
            runtime_secrets=self._runtime_secrets,
        )
        response = self._system.run(text, entry_alias=entry_alias)
        return RuntimeExecutionResult(
            status="ok",
            entry_agent=entry_alias,
            response=_normalize_text(response),
            steps=tuple(self._system.steps),
            outputs=tuple(
                RuntimeOutput(
                    output_node_id=output.output_node_id,
                    output_name=output.output_name,
                    path=output.path,
                )
                for output in self._system.outputs
            ),
            events=tuple(
                _build_event_record(index, message)
                for index, message in enumerate(self._system.runtime.message_log, 1)
            ),
        )

    def close(self) -> None:
        if self._system is not None:
            self._system.close()
            self._system.runtime.close()


def run_graph_runtime(
    graph: AgentGraph,
    message: str | None,
    *,
    runtime_secrets: dict[str, str] | None = None,
) -> RuntimeExecutionResult:
    runtime = GraphRuntime(graph, runtime_secrets=runtime_secrets)
    try:
        return runtime.run(message)
    finally:
        runtime.close()
