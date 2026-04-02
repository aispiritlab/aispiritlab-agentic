"""Shared graph state helpers for the AgentGraph tab."""

from __future__ import annotations

import uuid

from agentic_graph.builder import AgenticGraphBuilder, render_validation_report
from agentic_graph.models import AgentGraph, AgentNode
from agentic_graph.registry import BlockKind, get_blocks_by_kind

SecretState = dict[str, str]


def normalize_text(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def initial_graph() -> AgentGraph:
    return AgentGraph(
        graph_id=uuid.uuid4().hex[:12],
        name="New Agent System",
        nodes=(),
        connections=(),
        entry_node_id=None,
    )


def block_choices(kind: BlockKind) -> list[str]:
    return [block.agent_name for block in get_blocks_by_kind(kind)]


def kind_options() -> tuple[BlockKind, ...]:
    return ("agent", "provider", "integration", "structural_output")


def sanitize_graph_and_secrets(
    graph: AgentGraph,
    existing_secrets: SecretState | None = None,
) -> tuple[AgentGraph, SecretState]:
    carried_secrets = {
        node_id: normalize_text(value).strip()
        for node_id, value in (existing_secrets or {}).items()
        if normalize_text(value).strip()
    }
    sanitized_nodes: list[AgentNode] = []
    next_secrets: SecretState = {}

    for node in graph.nodes:
        legacy_secret = ""
        sanitized_config: list[tuple[str, str]] = []
        for key, value in node.config:
            if key == "api_key":
                legacy_secret = normalize_text(value).strip()
                continue
            sanitized_config.append((key, value))

        secret_value = legacy_secret or carried_secrets.get(node.node_id, "")
        if secret_value:
            next_secrets[node.node_id] = secret_value

        sanitized_nodes.append(
            AgentNode(
                node_id=node.node_id,
                agent_name=node.agent_name,
                display_name=node.display_name,
                description=node.description,
                capabilities=node.capabilities,
                position=node.position,
                node_type=node.node_type,
                config=tuple(sanitized_config),
            )
        )

    sanitized_graph = AgentGraph(
        graph_id=graph.graph_id,
        name=graph.name,
        nodes=tuple(sanitized_nodes),
        connections=graph.connections,
        entry_node_id=graph.entry_node_id,
    )
    return sanitized_graph, next_secrets


def format_validation_report(
    graph: AgentGraph,
    runtime_secrets: SecretState | None = None,
) -> str:
    issues = AgenticGraphBuilder(graph, runtime_secrets=runtime_secrets).validate()
    return "### Validation\n" + render_validation_report(issues)


def parse_config(raw: str) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for line in normalize_text(raw).strip().splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        normalized_key = normalize_text(key).strip()
        if normalized_key == "api_key":
            continue
        pairs.append((normalized_key, normalize_text(value).strip()))
    return tuple(pairs)


def build_config(
    *,
    node: AgentNode,
    api_key_env: str,
    path_value: str,
    extra_config_raw: str,
    provider_type: str = "",
    model_id: str = "",
) -> tuple[tuple[str, str], ...]:
    existing = {
        key: value
        for key, value in node.config
        if key not in {"api_key", "api_key_env", "path", "provider_type", "model_id"}
    }
    for key, value in parse_config(extra_config_raw):
        existing[key] = value

    config_items: list[tuple[str, str]] = []
    if api_key_env:
        config_items.append(("api_key_env", api_key_env))
    if path_value:
        config_items.append(("path", path_value))
    if provider_type:
        config_items.append(("provider_type", provider_type))
    if model_id:
        config_items.append(("model_id", model_id))
    for key, value in existing.items():
        if value:
            config_items.append((key, value))
    return tuple(config_items)
