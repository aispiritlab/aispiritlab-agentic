"""Validation, summaries, and code generation for AgentGraph."""

from __future__ import annotations

from dataclasses import dataclass
from pprint import pformat
from typing import Literal

from agentic_graph.compiler import _normalize_text, _sanitize_alias as _slugify, compile_graph
from agentic_graph.models import AgentGraph, AgentNode, Connection
from agentic_graph.registry import BlockSpec, get_block_by_name
from agentic_graph.serialization import graph_to_dict

IssueLevel = Literal["error", "warning"]

_SEARCH_PROVIDER_NAMES = {"tavily_search", "valyu_search"}
_LLM_AGENT_NAMES = {"llm_chat", "api_router", "planner", "summarizer", "searcher"}


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    level: IssueLevel
    message: str


def _redact_config(config: dict[str, str]) -> dict[str, str]:
    redacted = dict(config)
    if redacted.get("api_key"):
        redacted["api_key"] = "***"
    return redacted


class AgenticGraphBuilder:
    def __init__(
        self,
        graph: AgentGraph,
        *,
        runtime_secrets: dict[str, str] | None = None,
    ) -> None:
        self._graph = graph
        self._runtime_secrets = {
            node_id: str(value).strip()
            for node_id, value in (runtime_secrets or {}).items()
            if value is not None and str(value).strip()
        }
        self._node_map: dict[str, AgentNode] = {node.node_id: node for node in graph.nodes}
        self._aliases = self._build_aliases()

    def _build_aliases(self) -> dict[str, str]:
        aliases: dict[str, str] = {}
        used: set[str] = set()
        for node in self._graph.nodes:
            base = _slugify(node.display_name or node.agent_name or node.node_id)
            alias = base
            counter = 2
            while alias in used:
                alias = f"{base}_{counter}"
                counter += 1
            used.add(alias)
            aliases[node.node_id] = alias
        return aliases

    def aliases(self) -> dict[str, str]:
        return dict(self._aliases)

    def entry_alias(self) -> str | None:
        if self._graph.entry_node_id is None:
            return None
        return self._aliases.get(self._graph.entry_node_id)

    def _node(self, node_id: str) -> AgentNode | None:
        return self._node_map.get(node_id)

    def _block(self, node: AgentNode) -> BlockSpec | None:
        return get_block_by_name(node.agent_name)

    def _incoming_connections(self, node_id: str) -> tuple[Connection, ...]:
        return tuple(conn for conn in self._graph.connections if conn.target_node_id == node_id)

    def _outgoing_connections(self, node_id: str) -> tuple[Connection, ...]:
        return tuple(conn for conn in self._graph.connections if conn.source_node_id == node_id)

    def _connected_nodes(self, node_id: str) -> tuple[AgentNode, ...]:
        neighbors: list[AgentNode] = []
        for conn in self._graph.connections:
            other_id: str | None = None
            if conn.source_node_id == node_id:
                other_id = conn.target_node_id
            elif conn.target_node_id == node_id:
                other_id = conn.source_node_id
            if other_id is None:
                continue
            other = self._node(other_id)
            if other is not None:
                neighbors.append(other)
        return tuple(neighbors)

    def _runtime_secret(self, node_id: str) -> str:
        return self._runtime_secrets.get(node_id, "")

    def _integration_api_key_env(self, node: AgentNode) -> str:
        config_map = dict(node.config)
        raw_value = config_map.get("api_key_env", "")
        return "" if raw_value is None else str(raw_value).strip()

    def validate(self) -> tuple[ValidationIssue, ...]:
        issues: list[ValidationIssue] = []
        seen_node_ids: set[str] = set()
        seen_connection_ids: set[str] = set()

        if not self._graph.nodes:
            issues.append(ValidationIssue("warning", "The graph has no blocks yet."))

        for node in self._graph.nodes:
            if node.node_id in seen_node_ids:
                issues.append(
                    ValidationIssue("error", f"Duplicate node_id detected: {node.node_id}")
                )
            seen_node_ids.add(node.node_id)

            block = self._block(node)
            if block is None:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"Node '{node.display_name}' uses an unknown block '{node.agent_name}'.",
                    )
                )
                continue
            if node.node_type != block.block_kind:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"Node '{node.display_name}' has node_type '{node.node_type}' but "
                        f"block '{node.agent_name}' expects '{block.block_kind}'.",
                    )
                )

        if self._graph.entry_node_id is not None:
            entry_node = self._node(self._graph.entry_node_id)
            if entry_node is None:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"entry_node_id '{self._graph.entry_node_id}' does not point to a node.",
                    )
                )
            elif entry_node.node_type != "agent":
                issues.append(ValidationIssue("error", "The entry block must be an agent block."))

        for conn in self._graph.connections:
            if conn.connection_id in seen_connection_ids:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"Duplicate connection_id detected: {conn.connection_id}",
                    )
                )
            seen_connection_ids.add(conn.connection_id)

            source = self._node(conn.source_node_id)
            target = self._node(conn.target_node_id)
            if source is None or target is None:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"Connection '{conn.connection_id}' points to a missing node.",
                    )
                )
                continue
            if source.node_id == target.node_id:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"Connection '{conn.connection_id}' cannot point a node to itself.",
                    )
                )
                continue

            if source.node_type == "agent" and target.node_type == "agent":
                if source.agent_name == target.agent_name:
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"Cannot connect two blocks of the same agent type: "
                            f"{source.display_name} -> {target.display_name}.",
                        )
                    )
                continue
            if source.node_type == "agent" and target.node_type == "structural_output":
                continue
            if {source.node_type, target.node_type} == {"integration", "agent"}:
                agent_node = source if source.node_type == "agent" else target
                if agent_node.agent_name != "searcher":
                    issues.append(
                        ValidationIssue(
                            "error",
                            "Integration blocks may connect only to searcher blocks: "
                            f"{source.display_name} -> {target.display_name}.",
                        )
                    )
                continue
            if {source.node_type, target.node_type} == {"provider", "agent"}:
                agent_node = source if source.node_type == "agent" else target
                if agent_node.agent_name not in _LLM_AGENT_NAMES:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "Provider blocks may connect only to workflow LLM agent blocks: "
                            f"{source.display_name} -> {target.display_name}.",
                        )
                    )
                continue

            issues.append(
                ValidationIssue(
                    "error",
                    f"Unsupported connection: {source.display_name} -> {target.display_name}.",
                )
            )

        for node in self._graph.nodes:
            neighbors = self._connected_nodes(node.node_id)
            config_map = dict(node.config)

            if node.agent_name in _LLM_AGENT_NAMES:
                providers = [neighbor for neighbor in neighbors if neighbor.node_type == "provider"]
                if len(providers) == 0:
                    issues.append(
                        ValidationIssue(
                            "warning",
                            f"Agent '{node.display_name}' has no connected provider block.",
                        )
                    )
                elif len(providers) > 1:
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"Agent '{node.display_name}' has multiple connected provider blocks. "
                            "Keep exactly one.",
                        )
                    )

            if node.agent_name == "searcher":
                search_integrations = [
                    neighbor for neighbor in neighbors if neighbor.agent_name in _SEARCH_PROVIDER_NAMES
                ]
                knowledge_bases = [
                    neighbor for neighbor in neighbors if neighbor.agent_name == "knowledge_base"
                ]
                if len(search_integrations) == 0:
                    issues.append(
                        ValidationIssue(
                            "warning",
                            f"Searcher '{node.display_name}' is not wired yet. Connect exactly "
                            "one search integration block before generating code.",
                        )
                    )
                elif len(search_integrations) > 1:
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"Searcher '{node.display_name}' has multiple connected search "
                            "integration blocks. Keep exactly one.",
                        )
                    )
                if len(knowledge_bases) > 1:
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"Searcher '{node.display_name}' may have at most one connected "
                            "knowledge base block.",
                        )
                    )

            if node.node_type == "provider":
                config_provider = _normalize_text(config_map.get("provider_type", "")).strip()
                config_model = _normalize_text(config_map.get("model_id", "")).strip()
                connected_agents = [neighbor for neighbor in neighbors if neighbor.node_type == "agent"]
                if not config_provider:
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"Provider '{node.display_name}' is missing provider_type.",
                        )
                    )
                if not config_model:
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"Provider '{node.display_name}' is missing model_id.",
                        )
                    )
                if not connected_agents:
                    issues.append(
                        ValidationIssue(
                            "warning",
                            f"Provider '{node.display_name}' has no connected agent blocks.",
                        )
                    )

            if node.agent_name in _SEARCH_PROVIDER_NAMES:
                connected_searchers = [neighbor for neighbor in neighbors if neighbor.agent_name == "searcher"]
                if connected_searchers and not (
                    self._integration_api_key_env(node) or self._runtime_secret(node.node_id)
                ):
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"Search integration '{node.display_name}' has no API token configured. "
                            "Set an API token env var or a session token before generating code.",
                        )
                    )

            if node.node_type == "structural_output":
                incoming_agents = [
                    self._node(conn.source_node_id)
                    for conn in self._incoming_connections(node.node_id)
                ]
                if not any(
                    source is not None and source.node_type == "agent"
                    for source in incoming_agents
                ):
                    issues.append(
                        ValidationIssue(
                            "warning",
                            f"Structural output '{node.display_name}' has no connected agent blocks.",
                        )
                    )

            if node.node_type == "agent":
                outgoing_agents = [
                    self._node(conn.target_node_id)
                    for conn in self._outgoing_connections(node.node_id)
                    if self._node(conn.target_node_id) is not None
                    and self._node(conn.target_node_id).node_type == "agent"
                ]
                dispatch_mode = _normalize_text(config_map.get("dispatch_mode", "broadcast")).strip() or "broadcast"
                if dispatch_mode not in {"broadcast", "route_one"}:
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"Agent '{node.display_name}' uses unsupported dispatch_mode "
                            f"'{dispatch_mode}'. Use 'broadcast' or 'route_one'.",
                        )
                    )
                if node.agent_name == "summarizer" and not self._incoming_connections(node.node_id):
                    issues.append(
                        ValidationIssue(
                            "warning",
                            f"Summarizer '{node.display_name}' has no incoming agent connections.",
                        )
                    )
                if dispatch_mode == "route_one" and len(outgoing_agents) <= 1:
                    issues.append(
                        ValidationIssue(
                            "warning",
                            f"Agent '{node.display_name}' uses route_one but has {len(outgoing_agents)} "
                            "downstream agent target(s).",
                        )
                    )

        return tuple(issues)

    def generate_summary(self) -> str:
        issues = self.validate()
        lines = [f"# {self._graph.name}", "", "## Blocks"]
        for kind in ("agent", "provider", "integration", "structural_output"):
            group = [node for node in self._graph.nodes if node.node_type == kind]
            if not group:
                continue
            lines.append("")
            lines.append(f"### {kind.replace('_', ' ').title()} Blocks")
            for node in group:
                config = _redact_config(dict(node.config))
                config_text = ", ".join(f"{key}={value}" for key, value in config.items())
                alias = self._aliases[node.node_id]
                detail = f"- `{alias}`: {node.display_name} (`{node.agent_name}`)"
                if config_text:
                    detail += f" [{config_text}]"
                lines.append(detail)

        if self._graph.connections:
            lines.extend(["", "## Connections"])
            for conn in self._graph.connections:
                source = self._node(conn.source_node_id)
                target = self._node(conn.target_node_id)
                if source is None or target is None:
                    continue
                lines.append(
                    f"- `{self._aliases[source.node_id]}` -> `{self._aliases[target.node_id]}` "
                    f"({conn.message_type or 'UserMessage'})"
                )

        if self._graph.entry_node_id:
            entry = self._node(self._graph.entry_node_id)
            if entry is not None:
                lines.extend(
                    [
                        "",
                        "## Entry Block",
                        f"- `{self._aliases[entry.node_id]}`: {entry.display_name}",
                    ]
                )

        lines.extend(["", "## Validation", render_validation_report(issues)])
        return "\n".join(lines).strip()

    def generate_python(self) -> str:
        issues = self.validate()
        errors = [issue for issue in issues if issue.level == "error"]
        if errors:
            raise ValueError(render_validation_report(errors))

        for node in self._graph.nodes:
            if node.agent_name in _SEARCH_PROVIDER_NAMES:
                connected_searchers = [
                    neighbor for neighbor in self._connected_nodes(node.node_id) if neighbor.agent_name == "searcher"
                ]
                if connected_searchers and not self._integration_api_key_env(node):
                    raise ValueError(
                        render_validation_report(
                            [
                                ValidationIssue(
                                    "error",
                                    f"Search integration '{node.display_name}' requires an API token "
                                    "env var before code generation.",
                                )
                            ]
                        )
                    )

        compiled = compile_graph(self._graph)
        graph_data = pformat(graph_to_dict(self._graph), width=100)
        lines = [
            '"""Auto-generated builder code from AgentGraph."""',
            "",
            "from __future__ import annotations",
            "",
            "from agentic.workflow import WorkflowBuilder, WorkflowRuntime",
            "from agentic_graph.compiler import build_compiled_graph_system, compile_graph",
            "from agentic_graph.events import GraphCompletionEvent, GraphDispatchEvent, GraphOutputReadyEvent",
            "from agentic_graph.serialization import graph_from_dict",
            "from agentic_runtime.settings import Settings",
            "",
            f"GRAPH_DATA = {graph_data}",
            "",
            "",
            "def build_system() -> dict[str, object]:",
            f"    \"\"\"Build the '{self._graph.name}' system from the saved graph.\"\"\"",
            "    settings = Settings()",
            "    graph = graph_from_dict(GRAPH_DATA)",
            "    compiled = compile_graph(graph)",
            "    system = build_compiled_graph_system(graph, settings=settings)",
            "    return {",
            f"        'graph_id': {self._graph.graph_id!r},",
            f"        'graph_name': {self._graph.name!r},",
            f"        'entry_agent': {compiled.entry_alias!r},",
            "        'compiled': compiled,",
            "        'runtime': system.runtime,",
            "        'system': system,",
            "        'workflow_api': (WorkflowBuilder, WorkflowRuntime),",
            "        'event_types': (GraphDispatchEvent, GraphCompletionEvent, GraphOutputReadyEvent),",
            "    }",
            "",
        ]
        return "\n".join(lines)


def render_validation_report(issues: tuple[ValidationIssue, ...] | list[ValidationIssue]) -> str:
    issue_list = list(issues)
    if not issue_list:
        return "- No validation issues."
    return "\n".join(
        f"- {issue.level.upper()}: {issue.message}" for issue in issue_list
    )


def validate_graph(graph: AgentGraph) -> tuple[ValidationIssue, ...]:
    return AgenticGraphBuilder(graph).validate()


def generate_python_code(graph: AgentGraph) -> str:
    return AgenticGraphBuilder(graph).generate_python()


def generate_graph_summary(graph: AgentGraph) -> str:
    return AgenticGraphBuilder(graph).generate_summary()
