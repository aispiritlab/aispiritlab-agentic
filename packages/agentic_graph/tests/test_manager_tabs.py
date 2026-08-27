from __future__ import annotations

from agentic_graph.events_tab import _clear_events, _filter_event_records, _select_event_detail
from agentic_graph.models import AgentGraph, AgentNode, NodePosition
from agentic_graph.runtime import RuntimeExecutionResult
from agentic_graph.serialization import graph_to_json
from agentic_graph.tab_actions import run_runtime


def _minimal_graph() -> AgentGraph:
    return AgentGraph(
        graph_id="graph-1",
        name="Minimal",
        nodes=(
            AgentNode(
                node_id="entry-1",
                agent_name="llm_chat",
                display_name="Entry",
                description="Entry node",
                capabilities=("chat",),
                position=NodePosition(0, 0),
                node_type="agent",
                config=(),
            ),
        ),
        connections=(),
        entry_node_id="entry-1",
    )


def test_event_filtering_keeps_detail_mapping_after_filter() -> None:
    records = [
        {
            "index": 1,
            "type_name": "UserMessage",
            "kind": "conversation",
            "source": "user",
            "target": "entry",
            "status": "",
            "text": "hello",
            "detail_markdown": "detail-user",
        },
        {
            "index": 2,
            "type_name": "AssistantMessage",
            "kind": "conversation",
            "source": "entry",
            "target": "user",
            "status": "success",
            "text": "world",
            "detail_markdown": "detail-entry",
        },
    ]

    filtered, rows, count_text = _filter_event_records(records, "entry", "AssistantMessage")

    assert count_text == "Showing 1 of 2 events"
    assert rows == [["2", "AssistantMessage", "entry → user", "success", "world"]]
    assert _select_event_detail(0, filtered) == "detail-entry"


def test_clear_events_resets_preview_state() -> None:
    cleared = _clear_events()

    assert cleared[0] == []
    assert cleared[3] == []
    assert cleared[4] == "Cleared"


def test_run_runtime_returns_ok_status(monkeypatch) -> None:
    monkeypatch.setattr(
        "agentic_graph.tab_actions.run_graph_runtime",
        lambda graph, runtime_message, runtime_secrets=None: RuntimeExecutionResult(
            status="ok",
            entry_agent="entry",
            response=f"reply:{runtime_message}",
        ),
    )

    validation, output, status = run_runtime(graph_to_json(_minimal_graph()), {}, "hello")

    assert "Validation" in validation
    assert "reply:hello" in output
    assert status == "ok"


def test_run_runtime_returns_error_status_when_runtime_fails(monkeypatch) -> None:
    def _raise_runtime_error(graph, runtime_message, runtime_secrets=None):
        del graph, runtime_message, runtime_secrets
        raise RuntimeError("boom")

    monkeypatch.setattr("agentic_graph.tab_actions.run_graph_runtime", _raise_runtime_error)

    validation, output, status = run_runtime(graph_to_json(_minimal_graph()), {}, "hello")

    assert "Validation" in validation
    assert "Runtime error" in output
    assert status == "error"
