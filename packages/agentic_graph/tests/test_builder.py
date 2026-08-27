from __future__ import annotations

from pathlib import Path
import tomllib

import orjson
import pytest

from agentic.workflow.messages import RecordedMessageMetadata
from agentic_graph import AgenticGraphBuilder
from agentic_graph.compiler import compile_graph
from agentic_graph.events import GraphCompletionEvent
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
