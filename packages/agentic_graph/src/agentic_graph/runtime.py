"""Runtime preview execution for AgentGraph."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Literal

from agentic.integrations import TavilySearchProvider, ValyuSearchProvider
from agentic.llm_call import LLMCall
from agentic.providers.api import OpenAIProvider
from agentic.specialized_agents.events import TaskDelegated
from agentic.specialized_agents.planner_agent import PlannerAgent
from agentic.specialized_agents.router_agent import RouterAgent as GenericRouter
from agentic.specialized_agents.search_agent import SearchAgent
from agentic.specialized_agents.summarization_agent import SummarizationAgent
from agentic_runtime.settings import Settings
from knowledge_base.store import open_knowledge_base

from agentic_graph.builder import AgenticGraphBuilder
from agentic_graph.models import AgentGraph, AgentNode

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


def _write_markdown_output(text: str, *, path: str) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return str(target)


def _normalize_text(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _sanitize_alias(value: str | None) -> str:
    alias = re.sub(r"[^a-zA-Z0-9_]+", "_", _normalize_text(value).strip().lower()).strip("_")
    if not alias:
        return "node"
    if alias[0].isdigit():
        return f"node_{alias}"
    return alias


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
        self._node_map = {node.node_id: node for node in graph.nodes}
        self._aliases = self._builder.aliases()
        self._settings = Settings()
        self._agents: dict[str, object] = {}
        self._integrations: dict[str, object] = {}
        self._steps: list[str] = []
        self._outputs: list[RuntimeOutput] = []

    def run(self, message: str | None) -> RuntimeExecutionResult:
        text = _normalize_text(message).strip()
        if not text:
            raise ValueError("Runtime preview requires a non-empty input message.")

        issues = self._builder.validate()
        errors = [issue for issue in issues if issue.level == "error"]
        if errors:
            raise ValueError("\n".join(issue.message for issue in errors))

        entry_node = self._entry_node()
        if entry_node is None:
            raise ValueError("The graph requires an entry agent before runtime preview.")

        OpenAIProvider.configure(
            base_url=self._settings.api_base_url,
            api_key=self._settings.api_key,
            timeout=self._settings.api_timeout,
        )

        self._instantiate_integrations()
        self._instantiate_agents()
        response = self._execute_agent(entry_node, text)
        self._write_connected_outputs(entry_node, response)
        return RuntimeExecutionResult(
            status="ok",
            entry_agent=self._aliases[entry_node.node_id],
            response=_normalize_text(response),
            steps=tuple(self._steps),
            outputs=tuple(self._outputs),
        )

    def close(self) -> None:
        for agent in self._agents.values():
            close = getattr(agent, "close", None)
            if callable(close):
                close()
        for integration in self._integrations.values():
            close = getattr(integration, "close", None)
            if callable(close):
                close()

    def _entry_node(self) -> AgentNode | None:
        if self._graph.entry_node_id is None:
            return None
        return self._node_map.get(self._graph.entry_node_id)

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
            node = self._node_map.get(other_id)
            if node is not None:
                neighbors.append(node)
        return tuple(neighbors)

    def _outgoing_output_nodes(self, node_id: str) -> tuple[AgentNode, ...]:
        output_nodes: list[AgentNode] = []
        for conn in self._graph.connections:
            if conn.source_node_id != node_id:
                continue
            node = self._node_map.get(conn.target_node_id)
            if node is not None and node.node_type == "structural_output":
                output_nodes.append(node)
        return tuple(output_nodes)

    def _search_provider_node(self, node_id: str) -> AgentNode:
        providers = [
            node
            for node in self._connected_nodes(node_id)
            if node.agent_name in {"tavily_search", "valyu_search"}
        ]
        if len(providers) != 1:
            raise ValueError(
                f"Searcher '{self._aliases[node_id]}' requires exactly one connected search integration."
            )
        return providers[0]

    def _knowledge_base_node(self, node_id: str) -> AgentNode | None:
        matches = [
            node for node in self._connected_nodes(node_id) if node.agent_name == "knowledge_base"
        ]
        if len(matches) > 1:
            raise ValueError(
                f"Searcher '{self._aliases[node_id]}' may have at most one connected knowledge base."
            )
        return matches[0] if matches else None

    def _integration_api_key(self, node: AgentNode) -> str:
        secret = self._runtime_secrets.get(node.node_id, "")
        if secret:
            return secret
        config_map = dict(node.config)
        env_name = _normalize_text(config_map.get("api_key_env", "")).strip()
        if env_name:
            return _normalize_text(os.getenv(env_name, "")).strip()
        return ""

    def _instantiate_integrations(self) -> None:
        for node in self._graph.nodes:
            config_map = dict(node.config)
            if node.agent_name == "tavily_search":
                api_key = self._integration_api_key(node)
                if not api_key:
                    raise ValueError(
                        f"Search integration '{node.display_name}' requires an API token "
                        "env var or a session token before runtime preview."
                    )
                self._integrations[node.node_id] = TavilySearchProvider(api_key=api_key)
            elif node.agent_name == "valyu_search":
                api_key = self._integration_api_key(node)
                if not api_key:
                    raise ValueError(
                        f"Search integration '{node.display_name}' requires an API token "
                        "env var or a session token before runtime preview."
                    )
                self._integrations[node.node_id] = ValyuSearchProvider(api_key=api_key)
            elif node.agent_name == "knowledge_base":
                path_value = _normalize_text(config_map.get("path", "data/knowledge_base")) or "data/knowledge_base"
                self._integrations[node.node_id] = open_knowledge_base(Path(path_value))

    def _instantiate_agents(self) -> None:
        for node in self._graph.nodes:
            if node.node_type != "agent":
                continue
            if node.agent_name == "llm_chat":
                self._agents[node.node_id] = LLMCall(model_name=self._settings.model_name)
            elif node.agent_name == "api_router":
                self._agents[node.node_id] = GenericRouter(
                    model_id=self._settings.orchestration_model_name,
                    model_provider_type="openai",
                )
            elif node.agent_name == "planner":
                delegated_aliases = [
                    self._aliases[conn.target_node_id]
                    for conn in self._graph.connections
                    if conn.source_node_id == node.node_id
                    and conn.target_node_id in self._node_map
                    and self._node_map[conn.target_node_id].node_type == "agent"
                ]
                self._agents[node.node_id] = PlannerAgent(
                    model_id=self._settings.model_name,
                    agent_names=delegated_aliases,
                    model_provider_type="openai",
                )
            elif node.agent_name == "summarizer":
                self._agents[node.node_id] = SummarizationAgent(
                    model_id=self._settings.model_name,
                    model_provider_type="openai",
                )
            elif node.agent_name == "searcher":
                provider_node = self._search_provider_node(node.node_id)
                knowledge_base_node = self._knowledge_base_node(node.node_id)
                self._agents[node.node_id] = SearchAgent(
                    model_id=self._settings.model_name,
                    search_provider=self._integrations[provider_node.node_id],
                    knowledge_base=(
                        self._integrations[knowledge_base_node.node_id]
                        if knowledge_base_node is not None
                        else None
                    ),
                    model_provider_type="openai",
                )
            else:
                raise ValueError(
                    f"Runtime preview does not support the agent block '{node.agent_name}' yet."
                )

    def _available_agent_descriptions(self, router_node: AgentNode) -> str:
        lines: list[str] = []
        for conn in self._graph.connections:
            if conn.source_node_id != router_node.node_id:
                continue
            node = self._node_map.get(conn.target_node_id)
            if node is None or node.node_type != "agent":
                continue
            alias = self._aliases[node.node_id]
            lines.append(f"{alias}: {node.description}")
        return "\n".join(lines)

    def _resolve_alias_to_node(self, alias: str) -> AgentNode | None:
        normalized = _sanitize_alias(alias)
        for node_id, node_alias in self._aliases.items():
            if node_alias == normalized:
                return self._node_map[node_id]
        return None

    def _execute_agent(self, node: AgentNode, message: str) -> str:
        agent = self._agents[node.node_id]
        alias = self._aliases[node.node_id]

        if node.agent_name == "planner":
            planner = agent
            tasks = planner.plan(message)
            if not tasks:
                response = planner.summarize("No delegated tasks were created.")
                self._steps.append(f"{alias}: no delegated tasks created")
                return response

            completed: list[str] = []
            for task in tasks:
                if not isinstance(task, TaskDelegated):
                    continue
                target_node = self._resolve_alias_to_node(task.target_agent)
                if target_node is None:
                    raise ValueError(
                        f"Planner delegated to unknown agent alias '{task.target_agent}'."
                    )
                worker_result = self._execute_agent(target_node, task.task_description)
                self._write_connected_outputs(target_node, worker_result)
                completed.append(
                    "\n".join(
                        [
                            f"Agent: {self._aliases[target_node.node_id]}",
                            f"Task: {task.task_description}",
                            "Result:",
                            worker_result,
                        ]
                    )
                )
                self._steps.append(
                    f"{alias} -> {self._aliases[target_node.node_id]}: {task.task_description}"
                )
            return planner.summarize("\n\n".join(completed))

        if node.agent_name == "searcher":
            searcher = agent
            self._steps.append(f"{alias}: search")
            return searcher.search(message)

        if node.agent_name == "summarizer":
            summarizer = agent
            self._steps.append(f"{alias}: summarize")
            return summarizer.summarize(message)

        if node.agent_name == "llm_chat":
            chat = agent
            self._steps.append(f"{alias}: chat")
            return chat.call(message)

        if node.agent_name == "api_router":
            router = agent
            available_agents = self._available_agent_descriptions(node)
            if not available_agents:
                raise ValueError(
                    f"Router '{node.display_name}' has no connected agent targets for runtime preview."
                )
            selected_alias = _normalize_text(router.route(message, available_agents))
            target_node = self._resolve_alias_to_node(selected_alias)
            if target_node is None:
                raise ValueError(
                    f"Router '{node.display_name}' selected unknown agent '{selected_alias}'."
                )
            self._steps.append(f"{alias} -> {self._aliases[target_node.node_id]}: routed")
            worker_result = self._execute_agent(target_node, message)
            self._write_connected_outputs(target_node, worker_result)
            return worker_result

        raise ValueError(f"Runtime preview does not support the agent block '{node.agent_name}'.")

    def _write_connected_outputs(self, node: AgentNode, response: str) -> None:
        for output_node in self._outgoing_output_nodes(node.node_id):
            config_map = dict(output_node.config)
            path_value = _normalize_text(config_map.get("path", "outputs/agentic_graph.md")) or "outputs/agentic_graph.md"
            written_path = _write_markdown_output(_normalize_text(response), path=path_value)
            self._outputs.append(
                RuntimeOutput(
                    output_node_id=output_node.node_id,
                    output_name=output_node.display_name,
                    path=written_path,
                )
            )


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
