"""Runtime preview execution for AgentGraph."""

from __future__ import annotations

from dataclasses import dataclass
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
class RuntimeExecutionResult:
    status: RuntimeStatus
    entry_agent: str
    response: str
    steps: tuple[str, ...] = ()
    outputs: tuple[RuntimeOutput, ...] = ()


def _normalize_text(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


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
