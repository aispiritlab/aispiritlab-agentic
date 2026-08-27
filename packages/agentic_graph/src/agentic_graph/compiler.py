"""Compilation helpers that turn AgentGraph definitions into workflows."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any

from agentic.integrations import TavilySearchProvider, ValyuSearchProvider
from agentic.llm_call import LLMCall
from agentic.specialized_agents.planner_agent import PlannerAgent
from agentic.specialized_agents.router_agent import RouterAgent as GenericRouter
from agentic.specialized_agents.search_agent import SearchAgent
from agentic.specialized_agents.summarization_agent import SummarizationAgent
from agentic.workflow import (
    InMemoryMessageBus,
    UserMessage,
    WorkflowBuilder,
    WorkflowExecution,
    WorkflowRuntime,
    workflow_output_handler,
)
from agentic.workflow.messages import ConversationData, RecordedMessageMetadata
from agentic_graph.events import GraphCompletionEvent, GraphDispatchEvent, GraphOutputReadyEvent
from agentic_graph.models import AgentGraph, AgentNode, Connection
from agentic_runtime.settings import Settings
from knowledge_base.store import open_knowledge_base
from providers.api import OpenAIProvider

_SEARCH_PROVIDER_NAMES = {"tavily_search", "valyu_search"}
_LLM_AGENT_NAMES = {"llm_chat", "api_router", "planner", "summarizer", "searcher"}


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


def _slugify(value: str) -> str:
    return _sanitize_alias(value)


@dataclass(frozen=True, slots=True)
class CompiledProviderBinding:
    node_id: str
    alias: str
    provider_type: str
    model_id: str


@dataclass(frozen=True, slots=True)
class CompiledOutputNode:
    node_id: str
    alias: str
    display_name: str
    path: str


@dataclass(frozen=True, slots=True)
class CompiledAgentNode:
    node_id: str
    alias: str
    agent_name: str
    display_name: str
    description: str
    dispatch_mode: str
    provider: CompiledProviderBinding | None
    dispatch_targets: tuple[str, ...]
    summarizer_targets: tuple[str, ...]
    output_targets: tuple[str, ...]
    search_provider_node_id: str | None = None
    knowledge_base_node_id: str | None = None


@dataclass(frozen=True, slots=True)
class CompiledGraph:
    graph: AgentGraph
    aliases: dict[str, str]
    agents: dict[str, CompiledAgentNode]
    providers: dict[str, CompiledProviderBinding]
    outputs: dict[str, CompiledOutputNode]
    entry_alias: str | None


@dataclass(frozen=True, slots=True)
class GraphOutputRecord:
    output_node_id: str
    output_name: str
    path: str


def compile_graph(graph: AgentGraph) -> CompiledGraph:
    aliases = _build_aliases(graph)
    node_map = {node.node_id: node for node in graph.nodes}
    providers = _compile_providers(graph, aliases)
    outputs = _compile_outputs(graph, aliases)
    agents: dict[str, CompiledAgentNode] = {}

    for node in graph.nodes:
        if node.node_type != "agent":
            continue
        neighbors = _connected_nodes(graph, node_map, node.node_id)
        provider = _resolve_provider(neighbors, providers)
        search_provider = _resolve_search_provider(neighbors)
        knowledge_base = _resolve_knowledge_base(neighbors)
        dispatch_targets: list[str] = []
        summarizer_targets: list[str] = []
        output_targets: list[str] = []

        for conn in _outgoing_connections(graph, node.node_id):
            target = node_map.get(conn.target_node_id)
            if target is None:
                continue
            if target.node_type == "agent":
                target_alias = aliases[target.node_id]
                if target.agent_name == "summarizer":
                    summarizer_targets.append(target_alias)
                else:
                    dispatch_targets.append(target_alias)
            elif target.node_type == "structural_output":
                output_targets.append(aliases[target.node_id])

        config_map = dict(node.config)
        dispatch_mode = _normalize_text(config_map.get("dispatch_mode", "")).strip() or "broadcast"
        agents[node.node_id] = CompiledAgentNode(
            node_id=node.node_id,
            alias=aliases[node.node_id],
            agent_name=node.agent_name,
            display_name=node.display_name,
            description=node.description,
            dispatch_mode=dispatch_mode,
            provider=provider,
            dispatch_targets=tuple(dispatch_targets),
            summarizer_targets=tuple(summarizer_targets),
            output_targets=tuple(output_targets),
            search_provider_node_id=search_provider,
            knowledge_base_node_id=knowledge_base,
        )

    entry_alias = aliases.get(graph.entry_node_id) if graph.entry_node_id else None
    return CompiledGraph(
        graph=graph,
        aliases=aliases,
        agents=agents,
        providers=providers,
        outputs=outputs,
        entry_alias=entry_alias,
    )


class CompiledGraphSystem:
    """Registered workflows, handlers, and resources for a compiled graph."""

    def __init__(
        self,
        *,
        compiled: CompiledGraph,
        runtime: WorkflowRuntime,
        settings: Settings,
        runtime_secrets: dict[str, str] | None = None,
    ) -> None:
        self.compiled = compiled
        self.runtime = runtime
        self.settings = settings
        self.runtime_secrets = {
            node_id: _normalize_text(value).strip()
            for node_id, value in (runtime_secrets or {}).items()
            if _normalize_text(value).strip()
        }
        self._node_map = {node.node_id: node for node in compiled.graph.nodes}
        self._agents: dict[str, object] = {}
        self._integrations: dict[str, object] = {}
        self._expected_completion_counts: dict[tuple[str, str], int] = defaultdict(int)
        self._completion_buckets: dict[tuple[str, str], list[GraphCompletionEvent]] = defaultdict(
            list
        )
        self._completed_summaries: set[tuple[str, str]] = set()
        self._reserved_completion_sources: set[tuple[str, str]] = set()
        self._last_inputs: dict[tuple[str, str], str] = {}
        self.steps: list[str] = []
        self.outputs: list[GraphOutputRecord] = []
        self.final_responses: dict[str, str] = {}

    def register(self) -> CompiledGraphSystem:
        OpenAIProvider.configure(
            base_url=self.settings.api_base_url,
            api_key=self.settings.api_key,
            timeout=self.settings.api_timeout,
        )
        self._instantiate_integrations()
        self._instantiate_agents()
        self._register_workflows()
        self._register_output_handlers()
        return self

    def close(self) -> None:
        for agent in self._agents.values():
            close = getattr(agent, "close", None)
            if callable(close):
                close()
        for integration in self._integrations.values():
            close = getattr(integration, "close", None)
            if callable(close):
                close()

    def run(self, message: str, *, entry_alias: str | None = None) -> str:
        alias = entry_alias or self.compiled.entry_alias
        if alias is None:
            raise ValueError("The graph requires an entry agent before runtime preview.")
        runtime_id = os.urandom(8).hex()
        turn_id = os.urandom(8).hex()
        incoming = UserMessage(
            data=ConversationData(role="user", text=message),
            metadata=RecordedMessageMetadata(
                runtime_id=runtime_id,
                turn_id=turn_id,
                domain=alias,
                source="user",
                target=alias,
            ),
        )
        self.runtime.execute_workflow(alias, incoming)
        self.runtime.flush_output_handlers()
        return self.final_responses.get(turn_id, "")

    def _instantiate_integrations(self) -> None:
        for node in self.compiled.graph.nodes:
            config_map = dict(node.config)
            if node.agent_name == "tavily_search":
                api_key = self._integration_api_key(node)
                self._integrations[node.node_id] = TavilySearchProvider(api_key=api_key)
            elif node.agent_name == "valyu_search":
                api_key = self._integration_api_key(node)
                self._integrations[node.node_id] = ValyuSearchProvider(api_key=api_key)
            elif node.agent_name == "knowledge_base":
                path_value = (
                    _normalize_text(config_map.get("path", "data/knowledge_base"))
                    or "data/knowledge_base"
                )
                self._integrations[node.node_id] = open_knowledge_base(Path(path_value))

    def _instantiate_agents(self) -> None:
        for node_id, plan in self.compiled.agents.items():
            provider_type = plan.provider.provider_type if plan.provider is not None else "openai"
            model_id = (
                plan.provider.model_id if plan.provider is not None else self.settings.model_name
            )
            if plan.agent_name == "llm_chat":
                self._agents[node_id] = LLMCall(
                    model_name=model_id,
                    model_provider_type=provider_type,
                )
            elif plan.agent_name == "api_router":
                self._agents[node_id] = GenericRouter(
                    model_id=model_id,
                    model_provider_type=provider_type,
                )
            elif plan.agent_name == "planner":
                self._agents[node_id] = PlannerAgent(
                    model_id=model_id,
                    agent_names=list(plan.dispatch_targets),
                    model_provider_type=provider_type,
                )
            elif plan.agent_name == "summarizer":
                self._agents[node_id] = SummarizationAgent(
                    model_id=model_id,
                    model_provider_type=provider_type,
                )
            elif plan.agent_name == "searcher":
                if plan.search_provider_node_id is None:
                    raise ValueError(
                        f"Searcher '{plan.display_name}' requires a connected search integration."
                    )
                self._agents[node_id] = SearchAgent(
                    model_id=model_id,
                    search_provider=self._integrations[plan.search_provider_node_id],
                    knowledge_base=(
                        self._integrations[plan.knowledge_base_node_id]
                        if plan.knowledge_base_node_id is not None
                        else None
                    ),
                    model_provider_type=provider_type,
                )
            else:
                raise ValueError(
                    f"Graph workflows do not support the agent block '{plan.agent_name}' yet."
                )

    def _register_workflows(self) -> None:
        for node_id, plan in self.compiled.agents.items():
            agent = self._agents[node_id]
            if plan.agent_name == "searcher":
                workflow = (
                    WorkflowBuilder(plan.alias)
                    .agent(agent)
                    .reactor("multiturn_llm")
                    .map_input(self._make_passthrough_input_mapper(plan))
                    .emit_events(self._make_response_emitter(plan))
                    .build()
                )
                self.runtime.register_workflow(plan.alias, workflow)
                continue
            if plan.agent_name == "llm_chat":
                workflow = (
                    WorkflowBuilder(plan.alias)
                    .agent(agent)
                    .map_input(self._make_passthrough_input_mapper(plan))
                    .emit_events(self._make_response_emitter(plan))
                    .build()
                )
                self.runtime.register_workflow(plan.alias, workflow)
                continue
            if plan.agent_name == "summarizer":
                workflow = self._make_summarizer_workflow(plan, agent)
                self.runtime.register_workflow(plan.alias, workflow)
                continue
            if plan.agent_name == "planner":
                self.runtime.register_workflow(
                    plan.alias, self._make_planner_workflow(plan, agent)
                )
                continue
            if plan.agent_name == "api_router":
                self.runtime.register_workflow(plan.alias, self._make_router_workflow(plan, agent))
                continue
            raise ValueError(f"Unsupported workflow agent: {plan.agent_name}")

    def _register_output_handlers(self) -> None:
        self.runtime.register_output_handler(
            workflow_output_handler(
                can_handle=(GraphDispatchEvent,),
                each_message=self._handle_dispatch_event,
                name="graph_dispatch",
            )
        )
        self.runtime.register_output_handler(
            workflow_output_handler(
                can_handle=(GraphCompletionEvent,),
                each_message=self._handle_completion_event,
                name="graph_completion",
            )
        )
        self.runtime.register_output_handler(
            workflow_output_handler(
                can_handle=(GraphOutputReadyEvent,),
                each_message=self._handle_output_ready_event,
                name="graph_output_writer",
            )
        )

    def _make_passthrough_input_mapper(self, plan: CompiledAgentNode):
        def _mapper(message: Any) -> UserMessage | None:
            if not isinstance(message, UserMessage):
                return None
            self._record_expected_completions(plan, message.metadata.turn_id)
            self._last_inputs[(message.metadata.turn_id, plan.node_id)] = message.data.text or ""
            self.steps.append(f"{plan.alias}: {self._step_label(plan)}")
            return message

        return _mapper

    def _make_response_emitter(self, plan: CompiledAgentNode):
        def _emit(response: Any) -> tuple[object, ...]:
            data = getattr(response, "data", None)
            text = _normalize_text(getattr(data, "text", "") if data is not None else "")
            metadata = getattr(response, "metadata", None)
            turn_id = getattr(metadata, "turn_id", "") if metadata is not None else ""
            return tuple(self._build_result_events(plan, text, turn_id))

        return _emit

    def _make_summarizer_workflow(self, plan: CompiledAgentNode, agent: SummarizationAgent):
        def _handle(message: UserMessage) -> WorkflowExecution:
            self._record_expected_completions(plan, message.metadata.turn_id)
            self._last_inputs[(message.metadata.turn_id, plan.node_id)] = message.data.text or ""
            self.steps.append(f"{plan.alias}: summarize")
            text = agent.summarize(message.data.text or "")
            self.final_responses[message.metadata.turn_id] = text
            return WorkflowExecution(
                text=text,
                emitted_events=tuple(
                    self._build_result_events(plan, text, message.metadata.turn_id)
                ),
            )

        return _handle

    def _make_planner_workflow(self, plan: CompiledAgentNode, agent: PlannerAgent):
        def _handle(message: UserMessage) -> WorkflowExecution:
            self._record_expected_completions(plan, message.metadata.turn_id)
            self._last_inputs[(message.metadata.turn_id, plan.node_id)] = message.data.text or ""
            tasks = agent.plan(message.data.text or "")
            emitted_events: list[object] = []
            for task in tasks:
                target_alias = _normalize_text(task.target_agent)
                if target_alias not in plan.dispatch_targets:
                    raise ValueError(
                        f"Planner '{plan.display_name}' selected an unconnected target '{target_alias}'."
                    )
                emitted_events.append(
                    GraphDispatchEvent(
                        source_node_id=plan.node_id,
                        source_alias=plan.alias,
                        target_node_id=self._alias_to_node_id(target_alias),
                        target_alias=target_alias,
                        text=task.task_description,
                        metadata=RecordedMessageMetadata(source=plan.alias),
                    )
                )
                self.steps.append(f"{plan.alias} -> {target_alias}: {task.task_description}")
            if tasks:
                text = agent.summarize(
                    "\n".join(task.task_description for task in tasks if task.task_description)
                )
            else:
                text = "No delegated tasks were created."
            emitted_events.extend(self._completion_and_output_events(plan, text))
            self._remember_final_response(plan, message.metadata.turn_id, text)
            return WorkflowExecution(text=text, emitted_events=tuple(emitted_events))

        return _handle

    def _make_router_workflow(self, plan: CompiledAgentNode, agent: GenericRouter):
        def _handle(message: UserMessage) -> WorkflowExecution:
            self._record_expected_completions(plan, message.metadata.turn_id)
            self._last_inputs[(message.metadata.turn_id, plan.node_id)] = message.data.text or ""
            available_agents = self._available_target_descriptions(plan.dispatch_targets)
            if not available_agents:
                text = "No connected downstream agents are available."
                self._remember_final_response(plan, message.metadata.turn_id, text)
                return WorkflowExecution(
                    text=text,
                    emitted_events=tuple(self._completion_and_output_events(plan, text)),
                )
            selected_alias = _normalize_text(
                agent.route(message.data.text or "", available_agents)
            )
            if selected_alias not in plan.dispatch_targets:
                raise ValueError(
                    f"Router '{plan.display_name}' selected an unconnected target '{selected_alias}'."
                )
            self.steps.append(f"{plan.alias} -> {selected_alias}: routed")
            text = f"Routed to {selected_alias}"
            emitted_events = [
                GraphDispatchEvent(
                    source_node_id=plan.node_id,
                    source_alias=plan.alias,
                    target_node_id=self._alias_to_node_id(selected_alias),
                    target_alias=selected_alias,
                    text=message.data.text or "",
                    metadata=RecordedMessageMetadata(source=plan.alias),
                ),
                *self._completion_and_output_events(plan, text),
            ]
            self._remember_final_response(plan, message.metadata.turn_id, text)
            return WorkflowExecution(text=text, emitted_events=tuple(emitted_events))

        return _handle

    def _build_result_events(
        self,
        plan: CompiledAgentNode,
        text: str,
        turn_id: str,
    ) -> list[object]:
        emitted: list[object] = []
        emitted.extend(self._completion_and_output_events(plan, text))
        if plan.agent_name in {"planner", "api_router"}:
            self._remember_final_response(plan, turn_id, text)
            return emitted
        dispatch_text = self._last_inputs.get((turn_id, plan.node_id), text)
        selected_targets = self._select_dispatch_targets(plan, dispatch_text)
        for target_alias in selected_targets:
            self._reserve_expected_completions_for_target(target_alias, turn_id)
            self.steps.append(f"{plan.alias} -> {target_alias}: dispatched")
            emitted.append(
                GraphDispatchEvent(
                    source_node_id=plan.node_id,
                    source_alias=plan.alias,
                    target_node_id=self._alias_to_node_id(target_alias),
                    target_alias=target_alias,
                    text=dispatch_text,
                    metadata=RecordedMessageMetadata(source=plan.alias),
                )
            )
        self._remember_final_response(plan, turn_id, text)
        return emitted

    def _completion_and_output_events(
        self,
        plan: CompiledAgentNode,
        text: str,
    ) -> list[object]:
        emitted: list[object] = [
            GraphCompletionEvent(
                source_node_id=plan.node_id,
                source_alias=plan.alias,
                text=text,
                summarizer_node_ids=tuple(
                    self._alias_to_node_id(alias) for alias in plan.summarizer_targets
                ),
                payload={"agent_name": plan.agent_name},
                metadata=RecordedMessageMetadata(source=plan.alias),
            )
        ]
        for output_alias in plan.output_targets:
            output = self._output_by_alias(output_alias)
            emitted.append(
                GraphOutputReadyEvent(
                    source_node_id=plan.node_id,
                    source_alias=plan.alias,
                    output_node_id=output.node_id,
                    output_name=output.display_name,
                    text=text,
                    metadata=RecordedMessageMetadata(source=plan.alias),
                )
            )
        return emitted

    def _select_dispatch_targets(self, plan: CompiledAgentNode, text: str) -> tuple[str, ...]:
        if not plan.dispatch_targets:
            return ()
        if len(plan.dispatch_targets) == 1:
            return plan.dispatch_targets
        if plan.dispatch_mode == "route_one":
            router = GenericRouter(
                model_id=plan.provider.model_id
                if plan.provider is not None
                else self.settings.orchestration_model_name,
                model_provider_type=(
                    plan.provider.provider_type if plan.provider is not None else "openai"
                ),
            )
            try:
                selected_alias = _normalize_text(
                    router.route(text, self._available_target_descriptions(plan.dispatch_targets))
                )
            finally:
                router.close()
            if selected_alias not in plan.dispatch_targets:
                raise ValueError(
                    f"Agent '{plan.display_name}' routed to an unconnected target '{selected_alias}'."
                )
            return (selected_alias,)
        return plan.dispatch_targets

    def _handle_dispatch_event(self, message: Any) -> str | None:
        if not isinstance(message, GraphDispatchEvent):
            return None
        incoming = UserMessage(
            data=ConversationData(role="user", text=message.text),
            metadata=RecordedMessageMetadata(
                runtime_id=message.metadata.runtime_id,
                turn_id=message.metadata.turn_id,
                domain=message.target_alias,
                source=message.source_alias or "graph",
                target=message.target_alias,
            ),
        )
        return self.runtime.execute_workflow(message.target_alias, incoming)

    def _handle_completion_event(self, message: Any) -> str | None:
        if not isinstance(message, GraphCompletionEvent):
            return None
        latest_result: str | None = None
        for summarizer_node_id in message.summarizer_node_ids:
            key = (message.metadata.turn_id, summarizer_node_id)
            bucket = self._completion_buckets[key]
            if not any(
                existing.source_node_id == message.source_node_id and existing.text == message.text
                for existing in bucket
            ):
                bucket.append(message)
            expected = self._expected_completion_counts.get(key, 0)
            if expected == 0 or len(bucket) < expected or key in self._completed_summaries:
                continue
            summarizer_alias = self.compiled.aliases[summarizer_node_id]
            self._completed_summaries.add(key)
            self.steps.append(f"{summarizer_alias}: summarizing {len(bucket)} event(s)")
            incoming = UserMessage(
                data=ConversationData(role="user", text=self._build_summarizer_input(bucket)),
                metadata=RecordedMessageMetadata(
                    runtime_id=message.metadata.runtime_id,
                    turn_id=message.metadata.turn_id,
                    domain=summarizer_alias,
                    source="graph",
                    target=summarizer_alias,
                ),
            )
            latest_result = self.runtime.execute_workflow(summarizer_alias, incoming)
            self.final_responses[message.metadata.turn_id] = latest_result
        return latest_result

    def _handle_output_ready_event(self, message: Any) -> str | None:
        if not isinstance(message, GraphOutputReadyEvent):
            return None
        output = self.compiled.outputs[message.output_node_id]
        target = Path(output.path).resolve()
        allowed_base = Path.cwd().resolve()
        if not str(target).startswith(str(allowed_base)):
            raise ValueError(
                f"Output path '{output.path}' resolves outside the working directory."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(message.text, encoding="utf-8")
        self.outputs.append(
            GraphOutputRecord(
                output_node_id=output.node_id,
                output_name=output.display_name,
                path=str(target),
            )
        )
        return str(target)

    def _record_expected_completions(self, plan: CompiledAgentNode, turn_id: str) -> None:
        source_key = (turn_id, plan.node_id)
        if source_key in self._reserved_completion_sources:
            return
        self._reserved_completion_sources.add(source_key)
        for summarizer_alias in plan.summarizer_targets:
            key = (turn_id, self._alias_to_node_id(summarizer_alias))
            self._expected_completion_counts[key] += 1

    def _reserve_expected_completions_for_target(self, target_alias: str, turn_id: str) -> None:
        target_plan = self.compiled.agents[self._alias_to_node_id(target_alias)]
        self._record_expected_completions(target_plan, turn_id)

    def _remember_final_response(self, plan: CompiledAgentNode, turn_id: str, text: str) -> None:
        if plan.agent_name == "summarizer":
            self.final_responses[turn_id] = text
            return
        if not plan.dispatch_targets and not plan.summarizer_targets:
            self.final_responses[turn_id] = text
            return
        self.final_responses.setdefault(turn_id, text)

    def _available_target_descriptions(self, aliases: tuple[str, ...]) -> str:
        lines: list[str] = []
        for alias in aliases:
            node_id = self._alias_to_node_id(alias)
            plan = self.compiled.agents[node_id]
            lines.append(f"{alias}: {plan.description}")
        return "\n".join(lines)

    def _alias_to_node_id(self, alias: str) -> str:
        for node_id, candidate in self.compiled.aliases.items():
            if candidate == alias:
                return node_id
        raise KeyError(alias)

    def _output_by_alias(self, alias: str) -> CompiledOutputNode:
        for output in self.compiled.outputs.values():
            if output.alias == alias:
                return output
        raise KeyError(alias)

    @staticmethod
    def _build_summarizer_input(events: list[GraphCompletionEvent]) -> str:
        parts: list[str] = []
        for event in events:
            parts.append(
                "\n".join(
                    [
                        f"Agent: {event.source_alias}",
                        "Result:",
                        event.text,
                    ]
                )
            )
        return "\n\n".join(parts)

    @staticmethod
    def _step_label(plan: CompiledAgentNode) -> str:
        return {
            "llm_chat": "chat",
            "searcher": "search",
            "summarizer": "summarize",
        }.get(plan.agent_name, plan.agent_name)

    def _integration_api_key(self, node: AgentNode) -> str:
        secret = self.runtime_secrets.get(node.node_id, "")
        if secret:
            return secret
        config_map = dict(node.config)
        env_name = _normalize_text(config_map.get("api_key_env", "")).strip()
        if env_name:
            return _normalize_text(os.getenv(env_name, "")).strip()
        return ""


def register_compiled_graph(
    *,
    runtime: WorkflowRuntime,
    compiled: CompiledGraph,
    settings: Settings,
    runtime_secrets: dict[str, str] | None = None,
) -> CompiledGraphSystem:
    return CompiledGraphSystem(
        compiled=compiled,
        runtime=runtime,
        settings=settings,
        runtime_secrets=runtime_secrets,
    ).register()


def build_compiled_graph_system(
    graph: AgentGraph,
    *,
    settings: Settings | None = None,
    runtime_secrets: dict[str, str] | None = None,
    runtime: WorkflowRuntime | None = None,
) -> CompiledGraphSystem:
    resolved_settings = settings or Settings()
    resolved_runtime = runtime or WorkflowRuntime(bus=InMemoryMessageBus())
    compiled = compile_graph(graph)
    return register_compiled_graph(
        runtime=resolved_runtime,
        compiled=compiled,
        settings=resolved_settings,
        runtime_secrets=runtime_secrets,
    )


def _build_aliases(graph: AgentGraph) -> dict[str, str]:
    aliases: dict[str, str] = {}
    used: set[str] = set()
    for node in graph.nodes:
        base = _slugify(node.display_name or node.agent_name or node.node_id)
        alias = base
        counter = 2
        while alias in used:
            alias = f"{base}_{counter}"
            counter += 1
        used.add(alias)
        aliases[node.node_id] = alias
    return aliases


def _compile_providers(
    graph: AgentGraph,
    aliases: dict[str, str],
) -> dict[str, CompiledProviderBinding]:
    providers: dict[str, CompiledProviderBinding] = {}
    for node in graph.nodes:
        if node.node_type != "provider":
            continue
        config_map = dict(node.config)
        providers[node.node_id] = CompiledProviderBinding(
            node_id=node.node_id,
            alias=aliases[node.node_id],
            provider_type=_normalize_text(config_map.get("provider_type", "openai")) or "openai",
            model_id=_normalize_text(config_map.get("model_id", "")) or "qwen3.5-4b",
        )
    return providers


def _compile_outputs(
    graph: AgentGraph,
    aliases: dict[str, str],
) -> dict[str, CompiledOutputNode]:
    outputs: dict[str, CompiledOutputNode] = {}
    for node in graph.nodes:
        if node.node_type != "structural_output":
            continue
        config_map = dict(node.config)
        outputs[node.node_id] = CompiledOutputNode(
            node_id=node.node_id,
            alias=aliases[node.node_id],
            display_name=node.display_name,
            path=_normalize_text(config_map.get("path", "outputs/agentic_graph.md"))
            or "outputs/agentic_graph.md",
        )
    return outputs


def _resolve_provider(
    neighbors: tuple[AgentNode, ...],
    providers: dict[str, CompiledProviderBinding],
) -> CompiledProviderBinding | None:
    for neighbor in neighbors:
        if neighbor.node_type == "provider":
            return providers.get(neighbor.node_id)
    return None


def _resolve_search_provider(neighbors: tuple[AgentNode, ...]) -> str | None:
    for neighbor in neighbors:
        if neighbor.agent_name in _SEARCH_PROVIDER_NAMES:
            return neighbor.node_id
    return None


def _resolve_knowledge_base(neighbors: tuple[AgentNode, ...]) -> str | None:
    for neighbor in neighbors:
        if neighbor.agent_name == "knowledge_base":
            return neighbor.node_id
    return None


def _connected_nodes(
    graph: AgentGraph,
    node_map: dict[str, AgentNode],
    node_id: str,
) -> tuple[AgentNode, ...]:
    neighbors: list[AgentNode] = []
    for conn in graph.connections:
        other_id: str | None = None
        if conn.source_node_id == node_id:
            other_id = conn.target_node_id
        elif conn.target_node_id == node_id:
            other_id = conn.source_node_id
        if other_id is None:
            continue
        other = node_map.get(other_id)
        if other is not None:
            neighbors.append(other)
    return tuple(neighbors)


def _outgoing_connections(graph: AgentGraph, node_id: str) -> tuple[Connection, ...]:
    return tuple(conn for conn in graph.connections if conn.source_node_id == node_id)


__all__ = [
    "CompiledAgentNode",
    "CompiledGraph",
    "CompiledGraphSystem",
    "CompiledOutputNode",
    "CompiledProviderBinding",
    "GraphOutputRecord",
    "build_compiled_graph_system",
    "compile_graph",
    "register_compiled_graph",
]
