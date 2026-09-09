"""Compilation helpers that turn AgentGraph definitions into workflows."""

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import re
import threading
import time
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
from agentic_graph.join import (
    AnswerDiscarded,
    Completion,
    JoinLedger,
    JoinTimedOut,
    MemoryJoinLedger,
    OnDeadline,
)
from agentic_graph.models import AgentGraph, AgentNode, Connection
from agentic_runtime.settings import Settings
from agentic_runtime.trace import create_tracer
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
        join: JoinLedger | None = None,
        on_deadline: OnDeadline = OnDeadline.SUMMARIZE,
        deadline_seconds: float | None = None,
        renew_claim_every: float | None = None,
        clock: Callable[[], float] = time.time,
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
        # How many arrivals each join waits for, which have arrived, and
        # whether the summarizer has been fired — three questions that used to
        # be three dictionaries on this object, and therefore died with the
        # process. In memory by default, because a preview that runs once needs
        # nothing else; over an event stream when a worker's join has to
        # outlive it.
        self.join: JoinLedger = join or MemoryJoinLedger()
        # What a join does when it has waited long enough, and how long that is.
        # `None` schedules nothing, which is the honest default: a deadline
        # needs something that wakes up and looks, and a preview running in one
        # process has nothing that does — writing one down there would be a
        # fact with no reader.
        self._on_deadline = on_deadline
        self._deadline_seconds = deadline_seconds
        # Whether a running summarizer keeps its claim alive, and how often.
        # `None` does not, which is today's behaviour and sometimes the right
        # one: a fan-out re-run beside itself is a wasted call rather than a
        # wrong answer, and a heartbeat costs a thread and an append per beat.
        # Set it well under the ledger's lease — a third of it is the usual
        # shape — or it renews after the window it was meant to hold open.
        self._renew_claim_every = renew_claim_every
        self._clock = clock
        self._last_inputs: dict[tuple[str, str], str] = {}
        self.steps: list[str] = []
        #: Every join whose deadline arrived before every source did, and what
        #: was done about it. A partial answer that says nothing about being
        #: partial is the failure this list exists to prevent.
        self.timed_out_joins: list[JoinTimedOut] = []
        #: Summaries that ran and whose answers were thrown away, because the
        #: claim stopped being this system's while they were running. Empty is
        #: the ordinary state; a row here is work that was done twice, which is
        #: worth seeing even though the answer was not written.
        self.discarded_answers: list[AnswerDiscarded] = []
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
        turn_id = message.metadata.turn_id
        for summarizer_node_id in message.summarizer_node_ids:
            self.join.record_completion(
                turn_id,
                summarizer_node_id,
                Completion(
                    source_node_id=message.source_node_id,
                    source_alias=message.source_alias,
                    text=message.text,
                ),
            )
            expected = self.join.get_expected(turn_id, summarizer_node_id)
            bucket = self.join.get_completions(turn_id, summarizer_node_id)
            if expected == 0 or len(bucket) < expected:
                continue
            # Claimed rather than checked-then-set. Two workers that both saw
            # the last arrival would both find no claim and both fire, and a
            # graph that answers twice is worse than one that answers late.
            if not self.join.claim_summary(turn_id, summarizer_node_id):
                continue
            latest_result = self._fire_summary(
                turn_id,
                summarizer_node_id,
                bucket,
                runtime_id=message.metadata.runtime_id,
            )
        return latest_result

    def sweep_due_joins(self, turn_id: str, now: float | None = None) -> tuple[str, ...]:
        """Fire every join in this turn whose deadline has passed.

        The last inch of a deadline. `schedule_deadline` writes one down and
        :meth:`handle_join_deadline` knows what to do when it arrives; between
        the two, something has to wake up and look. This is what a wake-up
        calls, and it asks the *ledger* what is due rather than the wake-up —
        so a timer that fires early, fires twice, or fires for a join that was
        answered in the meantime costs a fold and changes nothing.

        With `AiwatcherEventStore` and `JoinTimers` the wake-up is the engine
        handing the scheduling message back at the time it was asked to. That
        is the whole of what `agentic_graph` has nothing of its own to do: a
        process holds no clock that survives it, and a deadline nobody wakes for
        is a fact with no reader.

        Answers which summarizers this call actually fired — not which were due,
        because a claim somebody else holds is a join this call left alone.
        """
        moment = self._clock() if now is None else now
        fired: list[str] = []
        for summarizer_node_id in self.join.due_joins(turn_id, moment):
            # `timed_out_joins` grows exactly when the claim was taken, which is
            # the one thing `handle_join_deadline`'s `None` cannot distinguish:
            # it answers `None` both for a claim it was refused and for a join
            # it decided not to answer.
            decided = len(self.timed_out_joins)
            self.handle_join_deadline(turn_id, summarizer_node_id)
            if len(self.timed_out_joins) > decided:
                fired.append(summarizer_node_id)
        return tuple(fired)

    def handle_join_deadline(self, turn_id: str, summarizer_node_id: str) -> str | None:
        """What a join does when it has waited long enough.

        The remaining silence, turned into a decision: a node that never
        completes used to leave the fan-in waiting for ever, with nothing in the
        stream to distinguish it from one still thinking.

        It claims first, and everything else follows from that. A completion
        that landed while this was on its way finds the claim taken and does not
        fire beside it; a summary that already finished refuses the claim
        outright. Which is why this may run late, twice, or against a join that
        turned out to be complete — the last of those summarizes normally,
        because a deadline that arrives after every source did has nothing to be
        partial about, and firing it is the recovery for a completion that was
        lost on the way.
        """
        if not self.join.claim_summary(turn_id, summarizer_node_id):
            return None
        arrived = self.join.get_completions(turn_id, summarizer_node_id)
        answered = {one.source_node_id for one in arrived}
        missing = tuple(
            source_node_id
            for source_node_id in self.join.get_reserved_sources(turn_id, summarizer_node_id)
            if source_node_id not in answered
        )
        summarizer_alias = self.compiled.aliases[summarizer_node_id]
        # Nothing arrived, so there is nothing to be partial about. Recorded as
        # missed whatever the graph asked for: an answer composed out of no
        # results is worse than a turn that says it never got any.
        gave_up = bool(missing) and (self._on_deadline is OnDeadline.FAIL or not arrived)
        self.timed_out_joins.append(
            JoinTimedOut(
                turn_id=turn_id,
                summarizer_node_id=summarizer_node_id,
                arrived=tuple(one.source_node_id for one in arrived),
                missing=missing,
                outcome=OnDeadline.FAIL if gave_up else OnDeadline.SUMMARIZE,
            )
        )
        if gave_up:
            self.steps.append(f"{summarizer_alias}: gave up waiting for {', '.join(missing)}")
            # Final, so that nothing takes the claim over and asks again: this
            # join has been decided, and the decision was not to answer.
            self.join.complete_summary(turn_id, summarizer_node_id)
            return None
        if missing:
            self.steps.append(f"{summarizer_alias}: answering without {', '.join(missing)}")
        return self._fire_summary(turn_id, summarizer_node_id, arrived)

    def _fire_summary(
        self,
        turn_id: str,
        summarizer_node_id: str,
        bucket: Sequence[Completion],
        *,
        runtime_id: str = "",
    ) -> str | None:
        """Run the summarizer on what the join holds. The caller owns the claim.

        `None` when the claim stopped being this system's while it ran, which is
        an answer that is thrown away rather than written beside the one that
        replaced it. The work is spent either way; what is dropped is the
        writing down.
        """
        summarizer_alias = self.compiled.aliases[summarizer_node_id]
        self.steps.append(f"{summarizer_alias}: summarizing {len(bucket)} event(s)")
        incoming = UserMessage(
            data=ConversationData(role="user", text=self._build_summarizer_input(bucket)),
            metadata=RecordedMessageMetadata(
                runtime_id=runtime_id,
                turn_id=turn_id,
                domain=summarizer_alias,
                source="graph",
                target=summarizer_alias,
            ),
        )
        # `_make_summarizer_workflow` writes `final_responses` on its own way
        # out — it is reachable without a join at all — so discarding is putting
        # back what was there rather than declining to write. This dict is this
        # system's own, so what is restored is this system's previous answer and
        # never the replacement's, which lives in the process that took over.
        previous_answer = self.final_responses.get(turn_id)
        with self._renewing(turn_id, summarizer_node_id):
            result = self.runtime.execute_workflow(summarizer_alias, incoming)
        lost_to = self._lost_the_claim(turn_id, summarizer_node_id)
        if lost_to is not None:
            if previous_answer is None:
                self.final_responses.pop(turn_id, None)
            else:
                self.final_responses[turn_id] = previous_answer
            self.discarded_answers.append(
                AnswerDiscarded(
                    turn_id=turn_id,
                    summarizer_node_id=summarizer_node_id,
                    taken_over_by=lost_to,
                )
            )
            self.steps.append(
                f"{summarizer_alias}: discarded its answer, {lost_to} owns this join"
            )
            return None
        # After it returned, never before: a claim is released by finishing, and
        # one whose summarizer raised is left to expire so that whatever comes
        # next takes it over rather than finding a join that is permanently
        # somebody else's.
        self.join.complete_summary(turn_id, summarizer_node_id)
        self.final_responses[turn_id] = result
        return result

    def _lost_the_claim(self, turn_id: str, summarizer_node_id: str) -> str | None:
        """Who owns this join now, when it is no longer this system.

        Read once, after the summarizer returned, and it holds whether or not
        anything was renewing: a heartbeat stops a takeover from happening and
        this stops one that happened from being written down twice.
        """
        state = self.join.get_claim(turn_id, summarizer_node_id)
        if state.completed:
            return "a summary that already completed"
        if state.held is None:
            return "nobody, the claim is gone"
        if state.held.holder != self.join.holder:
            return state.held.holder
        return None

    @contextmanager
    def _renewing(self, turn_id: str, summarizer_node_id: str) -> Generator[None]:
        """Keep the claim alive while the summarizer runs, when asked to.

        A daemon thread, which is the shape the distributed runtime already uses
        for its own heartbeat. It stops on its own the moment a renewal is
        refused: past that point the claim is somebody else's and pushing at it
        would be taking it back from whoever is now running the summary.
        """
        if self._renew_claim_every is None:
            yield
            return
        done = threading.Event()

        def beat() -> None:
            while not done.wait(self._renew_claim_every):
                if not self.join.renew_claim(turn_id, summarizer_node_id):
                    return

        heart = threading.Thread(target=beat, name="join-claim-renewal", daemon=True)
        heart.start()
        try:
            yield
        finally:
            done.set()
            heart.join(timeout=self._renew_claim_every)

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
        summarizer_node_ids = tuple(
            self._alias_to_node_id(summarizer_alias)
            for summarizer_alias in plan.summarizer_targets
        )
        self.join.reserve_fan_in(turn_id, plan.node_id, summarizer_node_ids)
        if self._deadline_seconds is None:
            return
        due_at = self._clock() + self._deadline_seconds
        for summarizer_node_id in summarizer_node_ids:
            # The ledger keeps the first: a fan-in of three reserves three
            # times, and a deadline that moved with each reservation would be
            # pushed out by the dispatcher slowness it is there to bound.
            self.join.schedule_deadline(turn_id, summarizer_node_id, due_at)

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
    def _build_summarizer_input(events: Sequence[Completion]) -> str:
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
    join: JoinLedger | None = None,
    on_deadline: OnDeadline = OnDeadline.SUMMARIZE,
    deadline_seconds: float | None = None,
    renew_claim_every: float | None = None,
) -> CompiledGraphSystem:
    return CompiledGraphSystem(
        compiled=compiled,
        runtime=runtime,
        settings=settings,
        runtime_secrets=runtime_secrets,
        join=join,
        on_deadline=on_deadline,
        deadline_seconds=deadline_seconds,
        renew_claim_every=renew_claim_every,
    ).register()


def build_compiled_graph_system(
    graph: AgentGraph,
    *,
    settings: Settings | None = None,
    runtime_secrets: dict[str, str] | None = None,
    runtime: WorkflowRuntime | None = None,
    join: JoinLedger | None = None,
    on_deadline: OnDeadline = OnDeadline.SUMMARIZE,
    deadline_seconds: float | None = None,
    renew_claim_every: float | None = None,
) -> CompiledGraphSystem:
    resolved_settings = settings or Settings()
    # A tracer, because the alternative is a graph that emits nothing while the
    # personal assistant emits everything — `BaseRuntime` has called
    # `create_tracer` all along, and this is the one path that never did. The
    # runtime already took the argument and defaulted it to a no-op, so the
    # preview was silent by omission rather than by decision.
    #
    # `create_tracer` is itself opt-in and fail-soft: without `AIWATCHER_URL`,
    # or without the SDK installed, it returns exactly the MLflow tracer it
    # always did.
    resolved_runtime = runtime or WorkflowRuntime(bus=InMemoryMessageBus(), tracer=create_tracer())
    compiled = compile_graph(graph)
    return register_compiled_graph(
        runtime=resolved_runtime,
        compiled=compiled,
        settings=resolved_settings,
        runtime_secrets=runtime_secrets,
        join=join,
        on_deadline=on_deadline,
        deadline_seconds=deadline_seconds,
        renew_claim_every=renew_claim_every,
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
