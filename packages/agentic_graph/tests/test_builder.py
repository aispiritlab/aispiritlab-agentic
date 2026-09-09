from __future__ import annotations

from pathlib import Path
import time
import tomllib

import orjson
import pytest

from agentic.workflow import InMemoryEventStore
from agentic.workflow.messages import RecordedMessageMetadata
from agentic_graph import AgenticGraphBuilder
from agentic_graph.compiler import build_compiled_graph_system, compile_graph
from agentic_graph.events import GraphCompletionEvent
from agentic_graph.join import Completion, OnDeadline, StreamJoinLedger
from agentic_graph.models import AgentGraph, AgentNode, Connection, NodePosition
from agentic_graph.registry import get_block_by_name
from agentic_graph.runtime import run_graph_runtime
from agentic_graph.serialization import graph_from_json, graph_to_json
from agentic_graph.tab import _sanitize_graph_and_secrets
from agentic_runtime.distributed.serialization import deserialize_record, serialize_record
from agentic_runtime.storage.projections import GenericProjection


def _node(
    node_id: str,
    agent_name: str,
    *,
    display_name: str,
    node_type: str,
    config: tuple[tuple[str, str], ...] = (),
) -> AgentNode:
    return AgentNode(
        node_id=node_id,
        agent_name=agent_name,
        display_name=display_name,
        description=f"{display_name} description",
        capabilities=(agent_name,),
        position=NodePosition(0, 0),
        node_type=node_type,
        config=config,
    )


def _shared_provider() -> AgentNode:
    return _node(
        "provider-1",
        "model_provider",
        display_name="Shared Provider",
        node_type="provider",
        config=(
            ("provider_type", "openai"),
            ("model_id", "shared-model"),
        ),
    )


def test_model_provider_defaults_match_local_runtime_model_id() -> None:
    block = get_block_by_name("model_provider")

    assert block is not None
    assert ("model_id", "qwen3.5-4b") in block.config_defaults


def test_compile_graph_falls_back_to_local_runtime_model_id() -> None:
    graph = AgentGraph(
        graph_id="graph-provider-default",
        name="Provider Default",
        nodes=(
            _node("entry-1", "llm_chat", display_name="Entry", node_type="agent"),
            _node("provider-1", "model_provider", display_name="Provider", node_type="provider"),
        ),
        connections=(Connection("conn-1", "provider-1", "entry-1"),),
        entry_node_id="entry-1",
    )

    compiled = compile_graph(graph)

    assert compiled.providers["provider-1"].model_id == "qwen3.5-4b"


def _three_search_graph(output_path: str) -> AgentGraph:
    return AgentGraph(
        graph_id="graph-search-v2",
        name="Search Workflow V2",
        nodes=(
            _node(
                "entry-1",
                "llm_chat",
                display_name="Entry",
                node_type="agent",
                config=(("dispatch_mode", "broadcast"),),
            ),
            _node("searcher-1", "searcher", display_name="Search Tavily", node_type="agent"),
            _node("searcher-2", "searcher", display_name="Search Valyu", node_type="agent"),
            _node("searcher-3", "searcher", display_name="Search Notes", node_type="agent"),
            _node("summarizer-1", "summarizer", display_name="Summarizer", node_type="agent"),
            _shared_provider(),
            _node(
                "integration-1",
                "tavily_search",
                display_name="Tavily",
                node_type="integration",
                config=(("api_key_env", "TAVILY_API_KEY"),),
            ),
            _node(
                "integration-2",
                "valyu_search",
                display_name="Valyu",
                node_type="integration",
                config=(("api_key_env", "VALYU_API_KEY"),),
            ),
            _node(
                "integration-3",
                "tavily_search",
                display_name="Notes Search",
                node_type="integration",
                config=(("api_key_env", "NOTES_TAVILY_API_KEY"),),
            ),
            _node(
                "output-1",
                "markdown_output",
                display_name="Markdown",
                node_type="structural_output",
                config=(("path", output_path),),
            ),
        ),
        connections=(
            Connection("conn-1", "entry-1", "searcher-1"),
            Connection("conn-2", "entry-1", "searcher-2"),
            Connection("conn-3", "entry-1", "searcher-3"),
            Connection("conn-4", "provider-1", "entry-1"),
            Connection("conn-5", "provider-1", "searcher-1"),
            Connection("conn-6", "provider-1", "searcher-2"),
            Connection("conn-7", "provider-1", "searcher-3"),
            Connection("conn-8", "provider-1", "summarizer-1"),
            Connection("conn-9", "searcher-1", "integration-1"),
            Connection("conn-10", "searcher-2", "integration-2"),
            Connection("conn-11", "searcher-3", "integration-3"),
            Connection("conn-12", "searcher-1", "summarizer-1"),
            Connection("conn-13", "searcher-2", "summarizer-1"),
            Connection("conn-14", "searcher-3", "summarizer-1"),
            Connection("conn-15", "summarizer-1", "output-1"),
        ),
        entry_node_id="entry-1",
    )


def test_builder_accepts_three_searchers_with_shared_provider(tmp_path: Path) -> None:
    builder = AgenticGraphBuilder(_three_search_graph(str(tmp_path / "search.md")))
    issues = builder.validate()

    assert not [issue for issue in issues if issue.level == "error"]
    summary = builder.generate_summary()
    assert "Provider Blocks" in summary
    assert "Shared Provider" in summary
    assert "Summarizer" in summary


def test_builder_warns_for_unwired_searcher_and_provider() -> None:
    graph = AgentGraph(
        graph_id="graph-draft",
        name="Draft Workflow",
        nodes=(
            _node("entry-1", "llm_chat", display_name="Entry", node_type="agent"),
            _node("searcher-1", "searcher", display_name="Searcher", node_type="agent"),
            _shared_provider(),
        ),
        connections=(Connection("conn-1", "provider-1", "entry-1"),),
        entry_node_id="entry-1",
    )

    issues = AgenticGraphBuilder(graph).validate()

    assert not [issue for issue in issues if issue.level == "error"]
    messages = [issue.message for issue in issues if issue.level == "warning"]
    assert any("Searcher 'Searcher' is not wired yet" in message for message in messages)
    assert any(
        "Agent 'Searcher' has no connected provider block" in message for message in messages
    )


def test_generate_python_code_uses_workflow_runtime_and_graph_events(tmp_path: Path) -> None:
    code = AgenticGraphBuilder(_three_search_graph(str(tmp_path / "search.md"))).generate_python()

    assert "WorkflowBuilder" in code
    assert "WorkflowRuntime" in code
    assert "GraphDispatchEvent" in code
    assert "GraphCompletionEvent" in code
    assert "build_compiled_graph_system" in code
    assert "graph_from_dict" in code


def test_graph_completion_event_round_trips_through_serializer() -> None:
    message = GraphCompletionEvent(
        source_node_id="searcher-1",
        source_alias="search_tavily",
        text="Found fresh sources.",
        summarizer_node_ids=("summarizer-1",),
        payload={"agent_name": "searcher"},
        metadata=RecordedMessageMetadata(
            runtime_id="runtime-1",
            turn_id="turn-1",
            source="graph",
        ),
    )

    restored = deserialize_record(serialize_record(message))

    assert isinstance(restored, GraphCompletionEvent)
    assert restored.type == "graph_completion"
    assert restored.source_node_id == "searcher-1"
    assert restored.source_alias == "search_tavily"
    assert restored.text == "Found fresh sources."
    assert restored.summarizer_node_ids == ("summarizer-1",)
    assert restored.payload == {"agent_name": "searcher"}


def test_graph_completion_event_projection_preserves_text_and_payload() -> None:
    message = GraphCompletionEvent(
        source_node_id="searcher-1",
        source_alias="search_tavily",
        text="Found fresh sources.",
        summarizer_node_ids=("summarizer-1",),
        payload={"agent_name": "searcher"},
        metadata=RecordedMessageMetadata(
            runtime_id="runtime-1",
            turn_id="turn-1",
            source="graph",
        ),
    )

    row = GenericProjection().handle(message)

    assert row.name == "graph_completion"
    assert row.text == "Found fresh sources."
    assert orjson.loads(row.payload_json or b"{}") == {
        "source_node_id": "searcher-1",
        "source_alias": "search_tavily",
        "text": "Found fresh sources.",
        "summarizer_node_ids": ["summarizer-1"],
        "payload": {"agent_name": "searcher"},
    }


def test_generate_python_code_requires_env_var_for_connected_search_integrations(
    tmp_path: Path,
) -> None:
    graph = AgentGraph(
        graph_id="graph-env-var",
        name="Env Var Workflow",
        nodes=(
            _node("searcher-1", "searcher", display_name="Searcher", node_type="agent"),
            _shared_provider(),
            _node(
                "integration-1",
                "tavily_search",
                display_name="Broken Search",
                node_type="integration",
            ),
        ),
        connections=(
            Connection("conn-1", "provider-1", "searcher-1"),
            Connection("conn-2", "searcher-1", "integration-1"),
        ),
        entry_node_id="searcher-1",
    )

    with pytest.raises(ValueError, match="API token"):
        AgenticGraphBuilder(graph).generate_python()


def test_tab_sanitizer_moves_legacy_api_key_to_session_secrets() -> None:
    graph = AgentGraph(
        graph_id="graph-7",
        name="Legacy Token Workflow",
        nodes=(
            _node("searcher-1", "searcher", display_name="Searcher", node_type="agent"),
            _node(
                "integration-1",
                "tavily_search",
                display_name="Tavily",
                node_type="integration",
                config=(("api_key", "secret-token"), ("api_key_env", "TAVILY_API_KEY")),
            ),
        ),
        connections=(Connection("conn-1", "searcher-1", "integration-1"),),
        entry_node_id="searcher-1",
    )

    sanitized_graph, secret_state = _sanitize_graph_and_secrets(graph)
    integration_config = dict(sanitized_graph.nodes[1].config)

    assert "api_key" not in integration_config
    assert integration_config["api_key_env"] == "TAVILY_API_KEY"
    assert secret_state == {"integration-1": "secret-token"}


class _StubProvider:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def close(self) -> None:
        return None


class _StubAgentResult:
    def __init__(self) -> None:
        self.tool_calls = ()
        self.run_id = "run-1"
        self.prompt_snapshot = None
        self.trace = None
        self.attempt_no = None
        self.loop_iteration = None
        self.request_usage = type(
            "_Usage",
            (),
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "latency_ms": 0.0,
                "model": "",
                "finish_reason": "",
            },
        )()


class _StubResponse:
    def __init__(self, output: str) -> None:
        self.output = output
        self.result = _StubAgentResult()
        self.tool_results = ()


class _StubLLMCall:
    def __init__(
        self, model_name: str, *, model_provider_type: str = "openai", **_: object
    ) -> None:
        self.model_name = model_name
        self.model_provider_type = model_provider_type

    def respond(self, message: str) -> _StubResponse:
        return _StubResponse(f"entry:{message}:{self.model_name}:{self.model_provider_type}")

    def close(self) -> None:
        return None


class _StubSearchAgent:
    def __init__(
        self,
        model_id: str,
        search_provider,
        *,
        knowledge_base=None,
        model_provider_type: str = "openai",
    ) -> None:
        del knowledge_base
        self.model_id = model_id
        self.search_provider = search_provider
        self.model_provider_type = model_provider_type

    def respond(self, message: str) -> _StubResponse:
        return _StubResponse(
            f"search:{message}:{self.search_provider.api_key}:{self.model_id}:{self.model_provider_type}"
        )

    def close(self) -> None:
        return None


class _StubSummarizer:
    def __init__(self, model_id: str, *, model_provider_type: str = "openai", **_: object) -> None:
        self.model_id = model_id
        self.model_provider_type = model_provider_type

    def summarize(self, text: str) -> str:
        return f"summary:{self.model_id}:{self.model_provider_type}:{text}"

    def close(self) -> None:
        return None


class _StubRouteOneRouter:
    def __init__(self, model_id: str, *, model_provider_type: str = "openai", **_: object) -> None:
        self.model_id = model_id
        self.model_provider_type = model_provider_type

    def route(self, message: str, available_agents: str) -> str:
        del message, available_agents
        return "search_valyu"

    def close(self) -> None:
        return None


def test_run_graph_runtime_broadcasts_to_three_searchers_and_summarizes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    graph = _three_search_graph(str(tmp_path / "runtime.md"))

    monkeypatch.setattr("agentic_graph.compiler.TavilySearchProvider", _StubProvider)
    monkeypatch.setattr("agentic_graph.compiler.ValyuSearchProvider", _StubProvider)
    monkeypatch.setattr("agentic_graph.compiler.LLMCall", _StubLLMCall)
    monkeypatch.setattr("agentic_graph.compiler.SearchAgent", _StubSearchAgent)
    monkeypatch.setattr("agentic_graph.compiler.SummarizationAgent", _StubSummarizer)
    monkeypatch.setenv("TAVILY_API_KEY", "env-tavily")
    monkeypatch.setenv("VALYU_API_KEY", "env-valyu")
    monkeypatch.setenv("NOTES_TAVILY_API_KEY", "env-notes")

    result = run_graph_runtime(graph, "latest updates")

    output_path = tmp_path / "runtime.md"
    assert result.status == "ok"
    assert result.entry_agent == "entry"
    assert "summary:shared-model:openai:" in result.response
    assert "search:latest updates:env-tavily:shared-model:openai" in result.response
    assert "search:latest updates:env-valyu:shared-model:openai" in result.response
    assert "search:latest updates:env-notes:shared-model:openai" in result.response
    assert output_path.read_text(encoding="utf-8") == result.response
    assert any("entry -> search_tavily" in step for step in result.steps)
    assert any("summarizer: summarizing 3 event(s)" in step for step in result.steps)
    assert result.events
    assert any(event.type_name == "UserMessage" for event in result.events)
    assert any(event.source == "user" for event in result.events)


def test_a_fan_in_of_three_survives_a_worker_restart_and_summarizes_once(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Phase 13's own exit, on a real compiled graph.

    The join used to be dictionaries on the system object, so a worker that went
    away between the second and third completion took the fan-in with it. It is
    a stream now: what restarts here is the *system* — one takes two arrivals
    and is thrown away, and a second one, holding nothing it learnt, takes the
    third and fires the summarizer. Once.
    """
    monkeypatch.chdir(tmp_path)
    graph = _three_search_graph(str(tmp_path / "restart.md"))

    monkeypatch.setattr("agentic_graph.compiler.TavilySearchProvider", _StubProvider)
    monkeypatch.setattr("agentic_graph.compiler.ValyuSearchProvider", _StubProvider)
    monkeypatch.setattr("agentic_graph.compiler.LLMCall", _StubLLMCall)
    monkeypatch.setattr("agentic_graph.compiler.SearchAgent", _StubSearchAgent)
    monkeypatch.setattr("agentic_graph.compiler.SummarizationAgent", _StubSummarizer)
    monkeypatch.setenv("TAVILY_API_KEY", "env-tavily")
    monkeypatch.setenv("VALYU_API_KEY", "env-valyu")
    monkeypatch.setenv("NOTES_TAVILY_API_KEY", "env-notes")

    store = InMemoryEventStore()
    stream = "graph:search-workflow-v2"
    searchers = ("search-tavily-1", "search-valyu-1", "search-notes-1")

    def system() -> object:
        return build_compiled_graph_system(graph, join=StreamJoinLedger(store, stream))

    def completion(source_node_id: str, text: str) -> GraphCompletionEvent:
        return GraphCompletionEvent(
            source_node_id=source_node_id,
            source_alias=source_node_id,
            text=text,
            summarizer_node_ids=("summarizer-1",),
            metadata=RecordedMessageMetadata(turn_id="turn-restart"),
        )

    before = system()
    for node_id in searchers:
        before.join.reserve_fan_in("turn-restart", node_id, ("summarizer-1",))
    assert before._handle_completion_event(completion(searchers[0], "a")) is None
    assert before._handle_completion_event(completion(searchers[1], "b")) is None
    assert not any("summarizing" in step for step in before.steps)

    # The worker is gone. Everything it knew is in the stream.
    del before
    after = system()
    summary = after._handle_completion_event(completion(searchers[2], "c"))

    assert summary is not None, "the join completed after the restart"
    assert any("summarizer: summarizing 3 event(s)" in step for step in after.steps)
    # Once: a redelivery of the last arrival, and a third system entirely, both
    # find the summary already claimed.
    assert after._handle_completion_event(completion(searchers[2], "c")) is None
    third = system()
    assert third._handle_completion_event(completion(searchers[2], "c")) is None


def test_run_graph_runtime_route_one_dispatches_single_target(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    graph = _three_search_graph(str(tmp_path / "runtime-route-one.md"))
    nodes = []
    for node in graph.nodes:
        if node.node_id == "entry-1":
            nodes.append(
                _node(
                    "entry-1",
                    "llm_chat",
                    display_name="Entry",
                    node_type="agent",
                    config=(("dispatch_mode", "route_one"),),
                )
            )
        else:
            nodes.append(node)
    graph = AgentGraph(
        graph_id=graph.graph_id,
        name=graph.name,
        nodes=tuple(nodes),
        connections=graph.connections,
        entry_node_id=graph.entry_node_id,
    )

    monkeypatch.setattr("agentic_graph.compiler.TavilySearchProvider", _StubProvider)
    monkeypatch.setattr("agentic_graph.compiler.ValyuSearchProvider", _StubProvider)
    monkeypatch.setattr("agentic_graph.compiler.LLMCall", _StubLLMCall)
    monkeypatch.setattr("agentic_graph.compiler.SearchAgent", _StubSearchAgent)
    monkeypatch.setattr("agentic_graph.compiler.SummarizationAgent", _StubSummarizer)
    monkeypatch.setattr("agentic_graph.compiler.GenericRouter", _StubRouteOneRouter)
    monkeypatch.setenv("TAVILY_API_KEY", "env-tavily")
    monkeypatch.setenv("VALYU_API_KEY", "env-valyu")
    monkeypatch.setenv("NOTES_TAVILY_API_KEY", "env-notes")

    result = run_graph_runtime(graph, "latest updates")

    assert result.status == "ok"
    assert "search:latest updates:env-valyu:shared-model:openai" in result.response
    assert "env-tavily" not in result.response
    assert "env-notes" not in result.response
    assert any("entry -> search_valyu: dispatched" in step for step in result.steps)
    assert result.events


def test_serialization_normalizes_legacy_node_types() -> None:
    raw = """
    {
      "graph_id": "graph-4",
      "name": "Legacy Graph",
      "nodes": [
        {
          "node_id": "planner-1",
          "agent_name": "planner",
          "display_name": "Planner",
          "description": "Planner",
          "capabilities": ["planner"],
          "position": {"x": 0, "y": 0},
          "node_type": "entry_point",
          "config": []
        }
      ],
      "connections": [],
      "entry_node_id": "planner-1"
    }
    """

    graph = graph_from_json(raw)

    assert graph.nodes[0].node_type == "agent"
    assert "entry_point" not in graph_to_json(graph)


def test_agentic_graph_package_does_not_depend_on_personal_assistant() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    dependencies = data["project"]["dependencies"]
    assert "personal-assistant" not in dependencies


def test_public_imports_still_work() -> None:
    import agentic_graph
    import personal_assistant.ui.app

    assert agentic_graph.AgenticGraphBuilder is not None
    assert personal_assistant.ui.app is not None


def test_run_graph_runtime_rejects_none_message_with_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty input message"):
        run_graph_runtime(_three_search_graph(str(tmp_path / "noop.md")), None)


class _MovableClock:
    """A clock a test moves, so a lease is crossed without sleeping through it."""

    def __init__(self, now: float = 1_700_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def after(self, seconds: float) -> None:
        self.now += seconds


def _deadline_system(
    monkeypatch,
    tmp_path: Path,
    store: InMemoryEventStore,
    *,
    on_deadline: OnDeadline,
):
    """A three-searcher graph whose join has already waited long enough."""
    graph = _three_search_graph(str(tmp_path / "deadline.md"))
    monkeypatch.setattr("agentic_graph.compiler.TavilySearchProvider", _StubProvider)
    monkeypatch.setattr("agentic_graph.compiler.ValyuSearchProvider", _StubProvider)
    monkeypatch.setattr("agentic_graph.compiler.LLMCall", _StubLLMCall)
    monkeypatch.setattr("agentic_graph.compiler.SearchAgent", _StubSearchAgent)
    monkeypatch.setattr("agentic_graph.compiler.SummarizationAgent", _StubSummarizer)
    monkeypatch.setenv("TAVILY_API_KEY", "env-tavily")
    monkeypatch.setenv("VALYU_API_KEY", "env-valyu")
    monkeypatch.setenv("NOTES_TAVILY_API_KEY", "env-notes")
    return build_compiled_graph_system(
        graph,
        join=StreamJoinLedger(store, "graph:deadline"),
        on_deadline=on_deadline,
        # Already due, so a tick a second later finds it. The clock is the
        # scheduler's; what a deadline *does* reads no clock at all.
        deadline_seconds=0.0,
    )


def _reserve_three(system: object, turn_id: str) -> None:
    for node_id in ("searcher-1", "searcher-2", "searcher-3"):
        system._record_expected_completions(system.compiled.agents[node_id], turn_id)


def _arrival(source_node_id: str, text: str, turn_id: str) -> GraphCompletionEvent:
    return GraphCompletionEvent(
        source_node_id=source_node_id,
        source_alias=source_node_id,
        text=text,
        summarizer_node_ids=("summarizer-1",),
        metadata=RecordedMessageMetadata(turn_id=turn_id),
    )


def _tick(system: object, turn_id: str) -> tuple[str, ...]:
    """What a worker does when it wakes up, and the whole of what it does.

    One public call, and it asks the *ledger* what is due rather than being told
    — the stall this closes is one nobody is watching, so the test must not be
    the thing watching either. A worker over `AiwatcherEventStore` makes exactly
    this call when the engine hands its scheduling message back.
    """
    return system.sweep_due_joins(turn_id, time.time() + 1)


def test_a_fan_in_whose_third_node_never_answers_summarizes_what_arrived(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Phase 14's exit, on the permissive setting.

    Two searchers came back and the third never did. Left alone the join waits
    for ever and says nothing; the deadline turns that silence into an answer
    that records what it did not have.
    """
    monkeypatch.chdir(tmp_path)
    store = InMemoryEventStore()
    system = _deadline_system(monkeypatch, tmp_path, store, on_deadline=OnDeadline.SUMMARIZE)
    turn = "turn-partial"

    _reserve_three(system, turn)
    assert system._handle_completion_event(_arrival("searcher-1", "a", turn)) is None
    assert system._handle_completion_event(_arrival("searcher-2", "b", turn)) is None
    assert not any("summarizing" in step for step in system.steps), "still waiting"

    _tick(system, turn)

    assert any("summarizer: summarizing 2 event(s)" in step for step in system.steps)
    assert system.final_responses[turn]
    (recorded,) = system.timed_out_joins
    assert recorded.outcome is OnDeadline.SUMMARIZE
    assert recorded.missing == ("searcher-3",), "the answer says which node it did not have"
    assert recorded.arrived == ("searcher-1", "searcher-2")

    # And once. The node that was late comes back, a second tick runs, and
    # neither answers beside the first: a late answer is visible and a double
    # answer is two results nobody can tell apart afterwards.
    assert system._handle_completion_event(_arrival("searcher-3", "c", turn)) is None
    _tick(system, turn)
    assert len([step for step in system.steps if "summarizing" in step]) == 1
    assert len(system.timed_out_joins) == 1


def test_a_fan_in_whose_third_node_never_answers_can_fail_the_turn_instead(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The same silence, on a graph that said a partial set is not an answer.

    Both are reachable per graph, because one instance runs both kinds.
    """
    monkeypatch.chdir(tmp_path)
    store = InMemoryEventStore()
    system = _deadline_system(monkeypatch, tmp_path, store, on_deadline=OnDeadline.FAIL)
    turn = "turn-failed"

    _reserve_three(system, turn)
    system._handle_completion_event(_arrival("searcher-1", "a", turn))
    system._handle_completion_event(_arrival("searcher-2", "b", turn))

    _tick(system, turn)

    assert not any("summarizing" in step for step in system.steps), "it did not answer"
    assert turn not in system.final_responses, "a turn that failed has no answer"
    (recorded,) = system.timed_out_joins
    assert recorded.outcome is OnDeadline.FAIL
    assert recorded.missing == ("searcher-3",), "and it says which node never came back"
    assert any("gave up waiting for searcher-3" in step for step in system.steps)

    # Decided, so nothing takes the claim over and asks again.
    assert system._handle_completion_event(_arrival("searcher-3", "c", turn)) is None
    _tick(system, turn)
    assert len(system.timed_out_joins) == 1


def test_a_fan_in_where_nothing_arrived_is_never_summarized_out_of_nothing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # Not a second policy: a fan-in where no node came back has nothing to be
    # partial about, and an answer composed out of no results is worse than a
    # turn that says it never got any.
    monkeypatch.chdir(tmp_path)
    store = InMemoryEventStore()
    system = _deadline_system(monkeypatch, tmp_path, store, on_deadline=OnDeadline.SUMMARIZE)
    turn = "turn-silent"

    _reserve_three(system, turn)
    _tick(system, turn)

    (recorded,) = system.timed_out_joins
    assert recorded.outcome is OnDeadline.FAIL
    assert recorded.arrived == ()
    assert not any("summarizing" in step for step in system.steps)


def test_a_deadline_that_arrives_after_every_source_did_answers_normally(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # The recovery for a completion that was lost on the way: the ledger holds
    # three arrivals and nobody fired, so the deadline is what notices.
    monkeypatch.chdir(tmp_path)
    store = InMemoryEventStore()
    system = _deadline_system(monkeypatch, tmp_path, store, on_deadline=OnDeadline.FAIL)
    turn = "turn-complete"

    _reserve_three(system, turn)
    for source, text in (("searcher-1", "a"), ("searcher-2", "b")):
        system._handle_completion_event(_arrival(source, text, turn))
    # The third arrival reaches the ledger and never reaches the handler.
    system.join.record_completion(
        turn, "summarizer-1", Completion("searcher-3", "searcher-3", "c")
    )

    _tick(system, turn)

    assert any("summarizer: summarizing 3 event(s)" in step for step in system.steps)
    (recorded,) = system.timed_out_joins
    assert recorded.missing == (), "nothing was missing, so there was nothing to fail on"
    assert recorded.outcome is OnDeadline.SUMMARIZE


def test_a_deadline_does_not_move_when_a_fan_in_reserves_again(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # Three sources reserve one fan-in, and a deadline that moved with each
    # would be pushed out by the very dispatcher slowness it bounds.
    monkeypatch.chdir(tmp_path)
    store = InMemoryEventStore()
    system = _deadline_system(monkeypatch, tmp_path, store, on_deadline=OnDeadline.SUMMARIZE)
    turn = "turn-once"

    _reserve_three(system, turn)
    first = system.join.get_deadline(turn, "summarizer-1")
    _reserve_three(system, turn)

    assert system.join.get_deadline(turn, "summarizer-1") == first


class _TakenOverSummarizer:
    """A summarizer that loses its claim to somebody else while it is running.

    The takeover is done from inside `summarize`, so there is no thread and no
    sleep in the test: the interleaving that matters is forced rather than
    waited for.
    """

    rival: object = None
    clock: object = None

    def __init__(self, model_id: str, **_: object) -> None:
        self.model_id = model_id

    def summarize(self, text: str) -> str:
        _TakenOverSummarizer.clock.after(600.0)
        _TakenOverSummarizer.rival.claim_summary("turn-lost", "summarizer-1")
        return f"summary:{text}"

    def close(self) -> None:
        return None


def test_a_summarizer_that_lost_its_claim_throws_its_answer_away(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The check that holds whether or not anything is renewing.

    A worker slower than the lease is taken over, and by the time it returns the
    join is somebody else's. Writing its answer beside the replacement's is two
    results nobody can tell apart afterwards, which is the one outcome none of
    this is worth — so the work is spent and the writing down is dropped.
    """
    monkeypatch.chdir(tmp_path)
    clock = _MovableClock()
    store = InMemoryEventStore()
    rival = StreamJoinLedger(
        store, "graph:lost", holder="worker-b", lease_seconds=300.0, clock=clock
    )
    _TakenOverSummarizer.rival = rival
    _TakenOverSummarizer.clock = clock

    graph = _three_search_graph(str(tmp_path / "lost.md"))
    monkeypatch.setattr("agentic_graph.compiler.TavilySearchProvider", _StubProvider)
    monkeypatch.setattr("agentic_graph.compiler.ValyuSearchProvider", _StubProvider)
    monkeypatch.setattr("agentic_graph.compiler.LLMCall", _StubLLMCall)
    monkeypatch.setattr("agentic_graph.compiler.SearchAgent", _StubSearchAgent)
    monkeypatch.setattr("agentic_graph.compiler.SummarizationAgent", _TakenOverSummarizer)
    for name in ("TAVILY_API_KEY", "VALYU_API_KEY", "NOTES_TAVILY_API_KEY"):
        monkeypatch.setenv(name, "env")
    system = build_compiled_graph_system(
        graph,
        join=StreamJoinLedger(
            store, "graph:lost", holder="worker-a", lease_seconds=300.0, clock=clock
        ),
    )

    turn = "turn-lost"
    for node_id in ("searcher-1", "searcher-2"):
        system._record_expected_completions(system.compiled.agents[node_id], turn)
    system._handle_completion_event(_arrival("searcher-1", "a", turn))
    assert system._handle_completion_event(_arrival("searcher-2", "b", turn)) is None, (
        "the answer was not written"
    )

    assert turn not in system.final_responses
    (discarded,) = system.discarded_answers
    assert discarded.taken_over_by == "worker-b"
    assert any("discarded its answer" in step for step in system.steps)


class _SlowSummarizer:
    """A summarizer that takes long enough for a heartbeat to beat."""

    def __init__(self, model_id: str, **_: object) -> None:
        self.model_id = model_id

    def summarize(self, text: str) -> str:
        time.sleep(0.2)
        return f"summary:{text}"

    def close(self) -> None:
        return None


def test_a_running_summarizer_keeps_its_claim_alive_when_asked_to(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # The other dial: renewing is opt-in, because it costs a thread and an
    # append per beat and being taken over is sometimes acceptable.
    monkeypatch.chdir(tmp_path)
    store = InMemoryEventStore()
    ledger = StreamJoinLedger(store, "graph:renewing", holder="worker-a", lease_seconds=300.0)

    graph = _three_search_graph(str(tmp_path / "renewing.md"))
    monkeypatch.setattr("agentic_graph.compiler.TavilySearchProvider", _StubProvider)
    monkeypatch.setattr("agentic_graph.compiler.ValyuSearchProvider", _StubProvider)
    monkeypatch.setattr("agentic_graph.compiler.LLMCall", _StubLLMCall)
    monkeypatch.setattr("agentic_graph.compiler.SearchAgent", _StubSearchAgent)
    monkeypatch.setattr("agentic_graph.compiler.SummarizationAgent", _SlowSummarizer)
    for name in ("TAVILY_API_KEY", "VALYU_API_KEY", "NOTES_TAVILY_API_KEY"):
        monkeypatch.setenv(name, "env")
    system = build_compiled_graph_system(graph, join=ledger, renew_claim_every=0.02)

    turn = "turn-renewing"
    system._record_expected_completions(system.compiled.agents["searcher-1"], turn)
    before = time.time()
    assert system._handle_completion_event(_arrival("searcher-1", "a", turn)) is not None

    # The claim was refreshed while the summarizer ran, so its instant is later
    # than the moment it was taken.
    held = ledger.get_claim(turn, "summarizer-1").held
    assert held is not None
    assert held.claimed_at > before, "nothing renewed"
    assert system.discarded_answers == []
