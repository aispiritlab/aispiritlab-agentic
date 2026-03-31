"""Validation, summaries, and code generation for AgentGraph."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Literal

from agentic_graph.models import AgentGraph, AgentNode, Connection
from agentic_graph.registry import (
    BlockSpec,
    get_block_by_name,
    is_agent_block,
    is_integration_block,
    is_structural_output_block,
)

IssueLevel = Literal["error", "warning"]

_PA_WORKFLOW_IMPORTS: dict[str, str] = {
    "router": "from personal_assistant.agents.router.router_agent import RouterAgent as PARouterAgent",
    "personalize": "from personal_assistant.agents.personalize.personlize_workflow import PersonalizeWorkflow",
    "manage_notes": "from personal_assistant.agents.manage_notes.manage_notes_workflow import ManageNotesWorkflow",
    "discovery_notes": "from personal_assistant.agents.discovery_notes.detective_workflow import DiscoveryNotesWorkflow",
    "sage": "from personal_assistant.agents.sage.sage_workflow import SageWorkflow",
    "organizer": "from personal_assistant.agents.organizer.organizer_workflow import OrganizerWorkflow",
}

_PA_WORKFLOW_CONSTRUCTORS: dict[str, str] = {
    "personalize": 'PersonalizeWorkflow(inputs=["UserMessage", "UserCommand"], tracer=tracer, context=context)',
    "manage_notes": 'ManageNotesWorkflow(inputs=["UserMessage", "UserCommand"], tracer=tracer, context=context)',
    "discovery_notes": 'DiscoveryNotesWorkflow(inputs=["UserMessage", "UserCommand"], tracer=tracer, context=context)',
    "sage": 'SageWorkflow(inputs=["UserMessage", "UserCommand"], tracer=tracer, context=context)',
    "organizer": 'OrganizerWorkflow(inputs=["CreatedNote", "UserCommand", "UserMessage"], tracer=tracer, context=context)',
}

_SEARCH_PROVIDER_NAMES = {"tavily_search", "valyu_search"}


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    level: IssueLevel
    message: str


def _slugify(value: str) -> str:
    normalized = "" if value is None else str(value)
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", normalized.strip().lower())
    slug = slug.strip("_")
    if not slug:
        return "node"
    if slug[0].isdigit():
        slug = f"node_{slug}"
    return slug


def _repr_config(config: tuple[tuple[str, str], ...]) -> str:
    if not config:
        return "{}"
    parts = ", ".join(f"{key!r}: {value!r}" for key, value in config)
    return "{" + parts + "}"


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

    def _node(self, node_id: str) -> AgentNode | None:
        return self._node_map.get(node_id)

    def _block(self, node: AgentNode) -> BlockSpec | None:
        return get_block_by_name(node.agent_name)

    def _incoming_connections(self, node_id: str) -> tuple[Connection, ...]:
        return tuple(
            conn for conn in self._graph.connections if conn.target_node_id == node_id
        )

    def _outgoing_connections(self, node_id: str) -> tuple[Connection, ...]:
        return tuple(
            conn for conn in self._graph.connections if conn.source_node_id == node_id
        )

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

    def _connected_search_integrations(self, node_id: str) -> tuple[AgentNode, ...]:
        return tuple(
            source
            for source in self._connected_nodes(node_id)
            if source.agent_name in _SEARCH_PROVIDER_NAMES
        )

    def _connected_searchers(self, node_id: str) -> tuple[AgentNode, ...]:
        return tuple(
            source
            for source in self._connected_nodes(node_id)
            if source.agent_name == "searcher"
        )

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
                issues.append(
                    ValidationIssue(
                        "error",
                        "The entry block must be an agent block.",
                    )
                )

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
                if source.agent_name != "planner":
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"Only planner blocks may connect to agent blocks: "
                            f"{source.display_name} -> {target.display_name}.",
                        )
                    )
                elif target.agent_name == "planner":
                    issues.append(
                        ValidationIssue(
                            "error",
                            "Planner blocks cannot delegate to another planner block in v1.",
                        )
                    )
                continue

            if {source.node_type, target.node_type} == {"integration", "agent"}:
                agent_node = source if source.node_type == "agent" else target
                integration_node = source if source.node_type == "integration" else target
                if agent_node.agent_name != "searcher":
                    issues.append(
                        ValidationIssue(
                            "error",
                            "Integration blocks may connect only to searcher blocks in v1: "
                            f"{integration_node.display_name} <-> {agent_node.display_name}.",
                        )
                    )
                continue

            if source.node_type == "agent" and target.node_type == "structural_output":
                continue

            issues.append(
                ValidationIssue(
                    "error",
                    f"Unsupported connection: {source.display_name} -> {target.display_name}.",
                )
            )

        for node in self._graph.nodes:
            if node.agent_name == "planner":
                targets = [
                    self._node(conn.target_node_id)
                    for conn in self._outgoing_connections(node.node_id)
                ]
                agent_targets = [target for target in targets if target is not None and target.node_type == "agent"]
                if not agent_targets:
                    issues.append(
                        ValidationIssue(
                            "warning",
                            f"Planner '{node.display_name}' has no connected agent targets.",
                        )
                    )

            if node.agent_name == "searcher":
                sources = list(self._connected_nodes(node.node_id))
                search_integrations = list(self._connected_search_integrations(node.node_id))
                knowledge_bases = [
                    source for source in sources if source.agent_name == "knowledge_base"
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

            if node.agent_name in _SEARCH_PROVIDER_NAMES:
                connected_searchers = self._connected_searchers(node.node_id)
                if connected_searchers and not (
                    self._integration_api_key_env(node) or self._runtime_secret(node.node_id)
                ):
                    issues.append(
                        ValidationIssue(
                            "warning",
                            f"Search integration '{node.display_name}' has no API token configured. "
                            "Set an API token env var or a session token before runtime preview.",
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

        return tuple(issues)

    def generate_summary(self) -> str:
        issues = self.validate()
        lines = [f"# {self._graph.name}", "", "## Blocks"]
        for kind in ("agent", "integration", "structural_output"):
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
                    f"({conn.message_type})"
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
            if node.agent_name != "searcher":
                continue
            search_integrations = list(self._connected_search_integrations(node.node_id))
            if len(search_integrations) != 1:
                raise ValueError(
                    render_validation_report(
                        [
                            ValidationIssue(
                                "error",
                                f"Searcher '{node.display_name}' requires exactly one connected "
                                "search integration block before code generation.",
                            )
                        ]
                    )
                )
            integration = search_integrations[0]
            if not self._integration_api_key_env(integration):
                raise ValueError(
                    render_validation_report(
                        [
                            ValidationIssue(
                                "error",
                                f"Search integration '{integration.display_name}' requires an "
                                "API token env var before code generation.",
                            )
                        ]
                    )
                )

        node_vars = {
            node.node_id: f"{self._aliases[node.node_id]}_block"
            for node in self._graph.nodes
        }

        imports = {
            "from __future__ import annotations",
            "import os",
            "from pathlib import Path",
            "from agentic.providers.api import OpenAIProvider",
            "from agentic_runtime.settings import Settings",
        }
        lines = ['"""Auto-generated builder code from AgentGraph."""', ""]

        if any(node.agent_name == "llm_chat" for node in self._graph.nodes):
            imports.add("from agentic.llm_call import LLMCall")
        if any(node.agent_name == "planner" for node in self._graph.nodes):
            imports.add("from agentic.specialized_agents.planner_agent import PlannerAgent")
        if any(node.agent_name == "summarizer" for node in self._graph.nodes):
            imports.add(
                "from agentic.specialized_agents.summarization_agent import SummarizationAgent"
            )
        if any(node.agent_name == "searcher" for node in self._graph.nodes):
            imports.add("from agentic.specialized_agents.search_agent import SearchAgent")
        if any(node.agent_name == "api_router" for node in self._graph.nodes):
            imports.add(
                "from agentic.specialized_agents.router_agent import RouterAgent as GenericRouter"
            )
        if any(node.agent_name == "tavily_search" for node in self._graph.nodes):
            imports.add("from agentic.integrations import TavilySearchProvider")
        if any(node.agent_name == "valyu_search" for node in self._graph.nodes):
            imports.add("from agentic.integrations import ValyuSearchProvider")
        if any(node.agent_name == "knowledge_base" for node in self._graph.nodes):
            imports.add("from knowledge_base.store import open_knowledge_base")
        if any(node.agent_name in _PA_WORKFLOW_IMPORTS for node in self._graph.nodes):
            imports.update(
                {
                    "from agentic.workflow import WorkflowRuntime",
                    "from agentic_runtime.messaging.message_bus import InMemoryMessageBus",
                    "from agentic_runtime.storage.sqlite_store import SQLiteMessageStore",
                    "from agentic_runtime.trace import create_tracer",
                    "from personal_assistant.runtime import WorkflowContext",
                }
            )
            for node in self._graph.nodes:
                import_line = _PA_WORKFLOW_IMPORTS.get(node.agent_name)
                if import_line is not None:
                    imports.add(import_line)

        lines.extend(sorted(imports))
        lines.extend(
            [
                "",
                "",
                "def write_markdown_output(text: str, *, path: str) -> str:",
                "    target = Path(path)",
                "    target.parent.mkdir(parents=True, exist_ok=True)",
                "    target.write_text(text, encoding=\"utf-8\")",
                "    return str(target)",
                "",
                "",
                "def build_system() -> dict[str, object]:",
                f"    \"\"\"Build the '{self._graph.name}' system from the saved graph.\"\"\"",
                "    settings = Settings()",
                "    OpenAIProvider.configure(",
                "        base_url=settings.api_base_url,",
                "        api_key=settings.api_key,",
                "        timeout=settings.api_timeout,",
                "    )",
                "    agents: dict[str, object] = {}",
                "    integrations: dict[str, object] = {}",
                "    outputs: dict[str, object] = {}",
                "    delegation_map: dict[str, list[str]] = {}",
                "    output_map: dict[str, list[str]] = {}",
            ]
        )

        if any(node.agent_name in _PA_WORKFLOW_CONSTRUCTORS for node in self._graph.nodes):
            lines.extend(
                [
                    "    tracer = create_tracer(enabled=True)",
                    "    store = SQLiteMessageStore(\":memory:\")",
                    "    bus = InMemoryMessageBus(store=store)",
                    "    context = WorkflowContext(bus)",
                    "    runtime = WorkflowRuntime(bus=bus, tracer=tracer)",
                ]
            )

        for node in self._graph.nodes:
            config_map = dict(node.config)
            var_name = node_vars[node.node_id]
            alias = self._aliases[node.node_id]

            if node.agent_name == "tavily_search":
                api_key_env = self._integration_api_key_env(node)
                lines.append(
                    f"    {var_name} = TavilySearchProvider(api_key=os.getenv({api_key_env!r}))"
                )
                lines.append(f"    integrations[{alias!r}] = {var_name}")
                continue

            if node.agent_name == "valyu_search":
                api_key_env = self._integration_api_key_env(node)
                lines.append(
                    f"    {var_name} = ValyuSearchProvider(api_key=os.getenv({api_key_env!r}))"
                )
                lines.append(f"    integrations[{alias!r}] = {var_name}")
                continue

            if node.agent_name == "knowledge_base":
                lines.append(
                    f"    {var_name} = open_knowledge_base(Path({config_map.get('path', 'data/knowledge_base')!r}))"
                )
                lines.append(f"    integrations[{alias!r}] = {var_name}")
                continue

            if node.agent_name == "markdown_output":
                lines.append(
                    f"    {var_name} = {{'kind': 'markdown_output', 'path': {config_map.get('path', 'outputs/agentic_graph.md')!r}}}"
                )
                lines.append(f"    outputs[{alias!r}] = {var_name}")
                continue

            if node.agent_name == "llm_chat":
                lines.append(
                    f"    {var_name} = LLMCall(model_name=settings.model_name)"
                )
            elif node.agent_name == "api_router":
                lines.append(
                    f"    {var_name} = GenericRouter(model_id=settings.orchestration_model_name, model_provider_type='openai')"
                )
            elif node.agent_name == "planner":
                delegated_aliases = [
                    self._aliases[conn.target_node_id]
                    for conn in self._outgoing_connections(node.node_id)
                    if self._node(conn.target_node_id) is not None
                    and self._node(conn.target_node_id).node_type == "agent"
                ]
                lines.append(
                    f"    {var_name} = PlannerAgent(model_id=settings.model_name, "
                    f"agent_names={delegated_aliases!r}, model_provider_type='openai')"
                )
                lines.append(f"    delegation_map[{alias!r}] = {delegated_aliases!r}")
            elif node.agent_name == "summarizer":
                lines.append(
                    f"    {var_name} = SummarizationAgent(model_id=settings.model_name, model_provider_type='openai')"
                )
            elif node.agent_name == "searcher":
                search_provider_var = None
                knowledge_base_var = "None"
                for source in self._connected_nodes(node.node_id):
                    if source.agent_name in _SEARCH_PROVIDER_NAMES:
                        search_provider_var = node_vars[source.node_id]
                    if source.agent_name == "knowledge_base":
                        knowledge_base_var = node_vars[source.node_id]
                lines.append(
                    f"    {var_name} = SearchAgent(model_id=settings.model_name, "
                    f"search_provider={search_provider_var}, knowledge_base={knowledge_base_var}, "
                    "model_provider_type='openai')"
                )
            elif node.agent_name == "router":
                lines.append(f"    {var_name} = PARouterAgent()")
            elif node.agent_name in _PA_WORKFLOW_CONSTRUCTORS:
                lines.append(
                    f"    {var_name} = {_PA_WORKFLOW_CONSTRUCTORS[node.agent_name]}"
                )
                lines.append(f"    runtime.register_workflow({alias!r}, {var_name})")
            else:
                lines.append(f"    {var_name} = None")

            lines.append(f"    agents[{alias!r}] = {var_name}")

        for node in self._graph.nodes:
            if not is_agent_block(node.agent_name):
                continue
            alias = self._aliases[node.node_id]
            connected_outputs = [
                self._aliases[conn.target_node_id]
                for conn in self._outgoing_connections(node.node_id)
                if self._node(conn.target_node_id) is not None
                and self._node(conn.target_node_id).node_type == "structural_output"
            ]
            if connected_outputs:
                lines.append(f"    output_map[{alias!r}] = {connected_outputs!r}")

        entry_alias = None
        if self._graph.entry_node_id is not None:
            entry_alias = self._aliases.get(self._graph.entry_node_id)

        lines.extend(
            [
                "    return {",
                f"        'graph_id': {self._graph.graph_id!r},",
                f"        'graph_name': {self._graph.name!r},",
                f"        'entry_agent': {entry_alias!r},",
                "        'agents': agents,",
                "        'integrations': integrations,",
                "        'outputs': outputs,",
                "        'delegation_map': delegation_map,",
                "        'output_map': output_map,",
            ]
        )
        if any(node.agent_name in _PA_WORKFLOW_CONSTRUCTORS for node in self._graph.nodes):
            lines.append("        'runtime': runtime,")
        lines.extend(["    }", ""])

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
