"""Agent graph data models used by the builder UI and code generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

NodeType = Literal["agent", "integration", "structural_output"]


@dataclass(frozen=True, slots=True)
class NodePosition:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class AgentNode:
    node_id: str
    agent_name: str
    display_name: str
    description: str
    capabilities: tuple[str, ...]
    position: NodePosition
    node_type: NodeType = "agent"
    config: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class Connection:
    connection_id: str
    source_node_id: str
    target_node_id: str
    message_type: str = "UserMessage"
    label: str = ""


@dataclass(frozen=True, slots=True)
class AgentGraph:
    graph_id: str
    name: str
    nodes: tuple[AgentNode, ...]
    connections: tuple[Connection, ...]
    entry_node_id: str | None = None
