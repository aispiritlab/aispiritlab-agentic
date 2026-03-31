from __future__ import annotations

from pathlib import Path
import tomllib

from agentic.specialized_agents.events import TaskDelegated
from agentic_graph import AgenticGraphBuilder
from agentic_graph.models import AgentGraph, AgentNode, Connection, NodePosition
from agentic_graph.runtime import run_graph_runtime
from agentic_graph.serialization import graph_from_json, graph_to_json
from agentic_graph.tab import _sanitize_graph_and_secrets
import pytest


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


def _valid_graph() -> AgentGraph:
    return AgentGraph(
        graph_id="graph-1",
        name="Search Workflow",
        nodes=(
            _node("planner-1", "planner", display_name="Planner", node_type="agent"),
            _node("searcher-1", "searcher", display_name="Search Tavily", node_type="agent"),
            _node(
                "integration-1",
                "tavily_search",
                display_name="Tavily",
                node_type="integration",
                config=(("api_key_env", "TAVILY_API_KEY"),),
            ),
            _node(
                "output-1",
                "markdown_output",
                display_name="Markdown",
                node_type="structural_output",
                config=(("path", "outputs/search.md"),),
            ),
        ),
        connections=(
            Connection("conn-1", "planner-1", "searcher-1"),
            Connection("conn-2", "integration-1", "searcher-1"),
            Connection("conn-3", "searcher-1", "output-1"),
        ),
        entry_node_id="planner-1",
    )


def test_builder_accepts_planner_search_markdown_graph() -> None:
    builder = AgenticGraphBuilder(_valid_graph())
    issues = builder.validate()

    assert not [issue for issue in issues if issue.level == "error"]
    summary = builder.generate_summary()
    assert "Search Workflow" in summary
    assert "Connections" in summary
    assert "Markdown" in summary


def test_builder_accepts_searcher_to_integration_attachment() -> None:
    graph = AgentGraph(
        graph_id="graph-reverse",
        name="Reverse Integration Edge",
        nodes=(
            _node("planner-1", "planner", display_name="Planner", node_type="agent"),
            _node("searcher-1", "searcher", display_name="Searcher", node_type="agent"),
            _node(
                "integration-1",
                "tavily_search",
                display_name="Tavily",
                node_type="integration",
                config=(("api_key_env", "TAVILY_API_KEY"),),
            ),
        ),
        connections=(
            Connection("conn-1", "planner-1", "searcher-1"),
            Connection("conn-2", "searcher-1", "integration-1"),
        ),
        entry_node_id="planner-1",
    )

    issues = AgenticGraphBuilder(graph).validate()

    assert not [issue for issue in issues if issue.level == "error"]
    code = AgenticGraphBuilder(graph).generate_python()
    assert "SearchAgent" in code
    assert "TavilySearchProvider" in code


def test_builder_rejects_invalid_output_to_agent_edge() -> None:
    graph = AgentGraph(
        graph_id="graph-2",
        name="Invalid Workflow",
        nodes=(
            _node(
                "output-1",
                "markdown_output",
                display_name="Markdown",
                node_type="structural_output",
                config=(("path", "outputs/search.md"),),
            ),
            _node("searcher-1", "searcher", display_name="Searcher", node_type="agent"),
            _node(
                "integration-1",
                "tavily_search",
                display_name="Tavily",
                node_type="integration",
                config=(("api_key_env", "TAVILY_API_KEY"),),
            ),
        ),
        connections=(
            Connection("conn-1", "output-1", "searcher-1"),
            Connection("conn-2", "integration-1", "searcher-1"),
        ),
        entry_node_id="searcher-1",
    )

    issues = AgenticGraphBuilder(graph).validate()

    assert any("Unsupported connection" in issue.message for issue in issues if issue.level == "error")


def test_builder_treats_unwired_searcher_as_warning_during_editing() -> None:
    graph = AgentGraph(
        graph_id="graph-draft",
        name="Draft Workflow",
        nodes=(
            _node("planner-1", "planner", display_name="Planner", node_type="agent"),
            _node("searcher-1", "searcher", display_name="Searcher", node_type="agent"),
        ),
        connections=(Connection("conn-1", "planner-1", "searcher-1"),),
        entry_node_id="planner-1",
    )

    issues = AgenticGraphBuilder(graph).validate()

    assert not [issue for issue in issues if issue.level == "error"]
    assert any("not wired yet" in issue.message for issue in issues if issue.level == "warning")
    try:
        AgenticGraphBuilder(graph).generate_python()
    except ValueError as error:
        assert "requires exactly one connected search integration block" in str(error)
    else:
        raise AssertionError("generate_python() should fail for an unwired searcher")


def test_generate_python_code_uses_typed_wiring_and_aliases() -> None:
    graph = AgentGraph(
        graph_id="graph-3",
        name="Dual Search Workflow",
        nodes=(
            _node("planner-1", "planner", display_name="Planner", node_type="agent"),
            _node("searcher-1", "searcher", display_name="Search Tavily", node_type="agent"),
            _node("searcher-2", "searcher", display_name="Search Valyu", node_type="agent"),
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
                "output-1",
                "markdown_output",
                display_name="Markdown",
                node_type="structural_output",
                config=(("path", "outputs/search.md"),),
            ),
        ),
        connections=(
            Connection("conn-1", "planner-1", "searcher-1"),
            Connection("conn-2", "planner-1", "searcher-2"),
            Connection("conn-3", "integration-1", "searcher-1"),
            Connection("conn-4", "integration-2", "searcher-2"),
            Connection("conn-5", "searcher-1", "output-1"),
            Connection("conn-6", "searcher-2", "output-1"),
        ),
        entry_node_id="planner-1",
    )

    code = AgenticGraphBuilder(graph).generate_python()

    assert "PlannerAgent" in code
    assert "SearchAgent" in code
    assert "TavilySearchProvider" in code
    assert "ValyuSearchProvider" in code
    assert "delegation_map['planner'] = ['search_tavily', 'search_valyu']" in code
    assert "output_map['search_tavily'] = ['markdown']" in code


def test_generate_python_code_uses_env_var_only_and_redacts_legacy_secret() -> None:
    graph = AgentGraph(
        graph_id="graph-5",
        name="Token Workflow",
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

    builder = AgenticGraphBuilder(graph)
    code = builder.generate_python()
    summary = builder.generate_summary()

    assert "os.getenv('TAVILY_API_KEY')" in code
    assert "secret-token" not in code
    assert "api_key=***" in summary
    assert "secret-token" not in summary


def test_validation_accepts_runtime_secret_for_preview_readiness() -> None:
    graph = AgentGraph(
        graph_id="graph-6",
        name="Preview Workflow",
        nodes=(
            _node("searcher-1", "searcher", display_name="Searcher", node_type="agent"),
            _node(
                "integration-1",
                "tavily_search",
                display_name="Tavily",
                node_type="integration",
            ),
        ),
        connections=(Connection("conn-1", "searcher-1", "integration-1"),),
        entry_node_id="searcher-1",
    )

    issues_without_secret = AgenticGraphBuilder(graph).validate()
    issues_with_secret = AgenticGraphBuilder(
        graph,
        runtime_secrets={"integration-1": "session-token"},
    ).validate()

    assert any(
        "has no API token configured" in issue.message
        for issue in issues_without_secret
        if issue.level == "warning"
    )
    assert not any(
        "has no API token configured" in issue.message
        for issue in issues_with_secret
        if issue.level == "warning"
    )


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


def test_run_graph_runtime_executes_searcher_and_writes_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    graph = AgentGraph(
        graph_id="graph-runtime-1",
        name="Runtime Search Workflow",
        nodes=(
            _node("searcher-1", "searcher", display_name="Searcher", node_type="agent"),
            _node(
                "integration-1",
                "tavily_search",
                display_name="Tavily",
                node_type="integration",
            ),
            _node(
                "output-1",
                "markdown_output",
                display_name="Markdown",
                node_type="structural_output",
                config=(("path", str(tmp_path / "runtime.md")),),
            ),
        ),
        connections=(
            Connection("conn-1", "searcher-1", "integration-1"),
            Connection("conn-2", "searcher-1", "output-1"),
        ),
        entry_node_id="searcher-1",
    )

    class FakeProvider:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def close(self) -> None:
            return None

    class FakeSearchAgent:
        def __init__(
            self,
            model_id: str,
            search_provider,
            *,
            knowledge_base=None,
            model_provider_type: str = "openai",
        ) -> None:
            del model_id, knowledge_base, model_provider_type
            self.search_provider = search_provider

        def search(self, message: str) -> str:
            return f"search:{message}:{self.search_provider.api_key}"

        def close(self) -> None:
            return None

    monkeypatch.setattr("agentic_graph.runtime.OpenAIProvider.configure", lambda **_: None)
    monkeypatch.setattr("agentic_graph.runtime.TavilySearchProvider", FakeProvider)
    monkeypatch.setattr("agentic_graph.runtime.SearchAgent", FakeSearchAgent)

    result = run_graph_runtime(
        graph,
        "latest updates",
        runtime_secrets={"integration-1": "session-token"},
    )

    output_path = tmp_path / "runtime.md"

    assert result.status == "ok"
    assert result.entry_agent == "searcher"
    assert result.response == "search:latest updates:session-token"
    assert output_path.read_text(encoding="utf-8") == result.response
    assert result.outputs[0].path == str(output_path)


def test_run_graph_runtime_executes_planner_delegation(monkeypatch) -> None:
    graph = AgentGraph(
        graph_id="graph-runtime-2",
        name="Runtime Planner Workflow",
        nodes=(
            _node("planner-1", "planner", display_name="Planner", node_type="agent"),
            _node("searcher-1", "searcher", display_name="Searcher", node_type="agent"),
            _node(
                "integration-1",
                "tavily_search",
                display_name="Tavily",
                node_type="integration",
                config=(("api_key_env", "TAVILY_API_KEY"),),
            ),
        ),
        connections=(
            Connection("conn-1", "planner-1", "searcher-1"),
            Connection("conn-2", "searcher-1", "integration-1"),
        ),
        entry_node_id="planner-1",
    )

    class FakeProvider:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def close(self) -> None:
            return None

    class FakePlannerAgent:
        def __init__(
            self,
            model_id: str,
            agent_names: list[str],
            *,
            model_provider_type: str = "openai",
        ) -> None:
            del model_id, model_provider_type
            self.agent_names = agent_names

        def plan(self, message: str) -> list[TaskDelegated]:
            return [
                TaskDelegated(
                    source="planner",
                    target_agent=self.agent_names[0],
                    task_description=f"task:{message}",
                )
            ]

        def summarize(self, completed_tasks: str) -> str:
            return f"summary:{completed_tasks}"

        def close(self) -> None:
            return None

    class FakeSearchAgent:
        def __init__(
            self,
            model_id: str,
            search_provider,
            *,
            knowledge_base=None,
            model_provider_type: str = "openai",
        ) -> None:
            del model_id, knowledge_base, model_provider_type
            self.search_provider = search_provider

        def search(self, message: str) -> str:
            return f"worker:{message}:{self.search_provider.api_key}"

        def close(self) -> None:
            return None

    monkeypatch.setattr("agentic_graph.runtime.OpenAIProvider.configure", lambda **_: None)
    monkeypatch.setattr("agentic_graph.runtime.TavilySearchProvider", FakeProvider)
    monkeypatch.setattr("agentic_graph.runtime.PlannerAgent", FakePlannerAgent)
    monkeypatch.setattr("agentic_graph.runtime.SearchAgent", FakeSearchAgent)
    monkeypatch.setenv("TAVILY_API_KEY", "env-token")

    result = run_graph_runtime(graph, "research topic")

    assert result.status == "ok"
    assert result.entry_agent == "planner"
    assert "summary:" in result.response
    assert "worker:task:research topic:env-token" in result.response
    assert any("planner -> searcher" in step for step in result.steps)


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


def test_run_graph_runtime_rejects_none_message_with_clear_error() -> None:
    with pytest.raises(ValueError, match="non-empty input message"):
        run_graph_runtime(_valid_graph(), None)
