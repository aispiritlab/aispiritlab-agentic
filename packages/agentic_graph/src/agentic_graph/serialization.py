"""JSON serialization and compatibility helpers for AgentGraph."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from agentic_graph.models import AgentGraph, AgentNode, Connection, NodePosition
from agentic_graph.registry import get_block_by_name


def graph_to_json(graph: AgentGraph) -> str:
    return json.dumps(asdict(graph), indent=2, ensure_ascii=False)


def graph_from_json(raw: str) -> AgentGraph:
    data: dict[str, Any] = json.loads(raw)
    return _graph_from_dict(data)


def graph_to_dict(graph: AgentGraph) -> dict[str, Any]:
    return asdict(graph)


def graph_from_dict(data: dict[str, Any]) -> AgentGraph:
    return _graph_from_dict(data)


def _normalize_node_type(agent_name: str, raw_type: str | None) -> str:
    if raw_type in {"agent", "integration", "structural_output"}:
        return raw_type
    if raw_type in {"router", "entry_point"}:
        return "agent"
    block = get_block_by_name(agent_name)
    if block is not None:
        return block.block_kind
    return "agent"


def _graph_from_dict(data: dict[str, Any]) -> AgentGraph:
    nodes = tuple(
        AgentNode(
            node_id=n["node_id"],
            agent_name=n["agent_name"],
            display_name=n["display_name"],
            description=n["description"],
            capabilities=tuple(n["capabilities"]),
            position=NodePosition(x=n["position"]["x"], y=n["position"]["y"]),
            node_type=_normalize_node_type(
                str(n["agent_name"]),
                n.get("node_type"),
            ),
            config=tuple(tuple(pair) for pair in n.get("config", ())),
        )
        for n in data.get("nodes", ())
    )
    connections = tuple(
        Connection(
            connection_id=c["connection_id"],
            source_node_id=c["source_node_id"],
            target_node_id=c["target_node_id"],
            message_type=c.get("message_type", "UserMessage"),
            label=c.get("label", ""),
        )
        for c in data.get("connections", ())
    )
    return AgentGraph(
        graph_id=data["graph_id"],
        name=data["name"],
        nodes=nodes,
        connections=connections,
        entry_node_id=data.get("entry_node_id"),
    )
