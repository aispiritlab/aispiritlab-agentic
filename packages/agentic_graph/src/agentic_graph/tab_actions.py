"""Backend actions for the AgentGraph tab."""

from __future__ import annotations

import json
import uuid

import gradio as gr

from agentic_graph.builder import AgenticGraphBuilder
from agentic_graph.models import AgentGraph, AgentNode, NodePosition
from agentic_graph.registry import get_block_by_name
from agentic_graph.runtime import RuntimeExecutionResult, run_graph_runtime
from agentic_graph.serialization import graph_from_json, graph_to_json
from agentic_graph.tab_state import (
    SecretState,
    build_config,
    format_validation_report,
    initial_graph,
    normalize_text,
    sanitize_graph_and_secrets,
)


def format_runtime_result(result: RuntimeExecutionResult) -> str:
    status_icon = {"ok": "✅", "error": "❌"}.get(result.status, "⚪")
    lines = [
        f"# {status_icon} Runtime Result",
        f"**Status:** {normalize_text(result.status)}",
        f"**Entry agent:** {normalize_text(result.entry_agent)}",
        "",
        "## Response",
        normalize_text(result.response),
    ]
    if result.steps:
        lines.extend(["", "## Steps"])
        lines.extend(f"- {normalize_text(step)}" for step in result.steps)
    if result.outputs:
        lines.extend(["", "## Outputs"])
        lines.extend(
            f"- {normalize_text(output.output_name)}: {normalize_text(output.path)}"
            for output in result.outputs
        )
    return "\n".join(lines)


def add_block(block_name: str, graph_json: str) -> str:
    block = get_block_by_name(block_name)
    if block is None:
        return graph_json
    graph = graph_from_json(graph_json)
    node_id = f"node_{uuid.uuid4().hex[:8]}"
    offset = len(graph.nodes)
    new_node = AgentNode(
        node_id=node_id,
        agent_name=block.agent_name,
        display_name=block.display_name,
        description=block.description,
        capabilities=block.capabilities,
        position=NodePosition(
            x=100 + (offset % 4) * 220,
            y=80 + (offset // 4) * 120,
        ),
        node_type=block.block_kind,
        config=block.config_defaults,
    )
    entry_node_id = graph.entry_node_id
    if entry_node_id is None and block.block_kind == "agent":
        entry_node_id = node_id
    updated = AgentGraph(
        graph_id=graph.graph_id,
        name=graph.name,
        nodes=(*graph.nodes, new_node),
        connections=graph.connections,
        entry_node_id=entry_node_id,
    )
    return graph_to_json(updated)


def on_graph_bridge_change(
    bridge_json: str,
    secret_state: SecretState,
) -> tuple[str, str, SecretState]:
    if not bridge_json:
        graph = initial_graph()
        graph_json = graph_to_json(graph)
        return graph_json, format_validation_report(graph), {}
    try:
        graph = graph_from_json(bridge_json)
    except Exception as error:
        return bridge_json, f"### Validation\n- ERROR: {error}", secret_state
    sanitized_graph, sanitized_secrets = sanitize_graph_and_secrets(graph, secret_state)
    sanitized_json = graph_to_json(sanitized_graph)
    return (
        sanitized_json,
        format_validation_report(sanitized_graph, sanitized_secrets),
        sanitized_secrets,
    )


def on_node_selected(
    node_id: str,
    graph_json: str,
    secret_state: SecretState,
) -> tuple[str, str, str, str, str, str, str, str, str, str, str, str]:
    if not node_id or not graph_json:
        return "", "", "", "", "", "", "", "", "", "", "", ""
    try:
        graph = graph_from_json(graph_json)
    except (json.JSONDecodeError, KeyError):
        return "", "", "", "", "", "", "", "", "", "", "", ""
    for node in graph.nodes:
        if node.node_id == node_id:
            is_entry = "Yes" if graph.entry_node_id == node_id else "No"
            config_map = dict(node.config)
            extra_config = "\n".join(
                f"{key}={value}"
                for key, value in node.config
                if key not in {"api_key", "api_key_env", "path", "provider_type", "model_id"}
            )
            return (
                node.display_name,
                node.agent_name,
                node.node_type,
                node.description,
                ", ".join(node.capabilities),
                is_entry,
                secret_state.get(node.node_id, ""),
                config_map.get("api_key_env", ""),
                config_map.get("path", ""),
                extra_config,
                config_map.get("provider_type", ""),
                config_map.get("model_id", ""),
            )
    return "", "", "", "", "", "", "", "", "", "", "", ""


def delete_selected(node_id: str, graph_json: str) -> str:
    if not node_id or not graph_json:
        return graph_json
    graph = graph_from_json(graph_json)
    new_nodes = tuple(node for node in graph.nodes if node.node_id != node_id)
    new_connections = tuple(
        conn
        for conn in graph.connections
        if conn.source_node_id != node_id and conn.target_node_id != node_id
    )
    entry_node_id = graph.entry_node_id if graph.entry_node_id != node_id else None
    updated = AgentGraph(
        graph_id=graph.graph_id,
        name=graph.name,
        nodes=new_nodes,
        connections=new_connections,
        entry_node_id=entry_node_id,
    )
    return graph_to_json(updated)


def set_entry(node_id: str, graph_json: str) -> str:
    if not node_id or not graph_json:
        return graph_json
    graph = graph_from_json(graph_json)
    node = next((item for item in graph.nodes if item.node_id == node_id), None)
    if node is None:
        return graph_json
    if node.node_type != "agent":
        gr.Warning("Only agent blocks can be entry blocks.")
        return graph_json
    updated = AgentGraph(
        graph_id=graph.graph_id,
        name=graph.name,
        nodes=graph.nodes,
        connections=graph.connections,
        entry_node_id=node_id,
    )
    return graph_to_json(updated)


def update_graph_name(name: str, graph_json: str) -> str:
    if not graph_json:
        return graph_json
    graph = graph_from_json(graph_json)
    updated = AgentGraph(
        graph_id=graph.graph_id,
        name=name,
        nodes=graph.nodes,
        connections=graph.connections,
        entry_node_id=graph.entry_node_id,
    )
    return graph_to_json(updated)


def update_node_property(
    node_id: str,
    graph_json: str,
    display_name: str,
    description: str,
    api_key_env: str,
    path_value: str,
    config_raw: str,
    provider_type: str = "",
    model_id: str = "",
) -> str:
    if not node_id or not graph_json:
        return graph_json
    try:
        graph = graph_from_json(graph_json)
    except Exception:
        return graph_json
    updated_nodes: list[AgentNode] = []
    for node in graph.nodes:
        if node.node_id != node_id:
            updated_nodes.append(node)
            continue
        block = get_block_by_name(node.agent_name)
        updated_nodes.append(
            AgentNode(
                node_id=node.node_id,
                agent_name=node.agent_name,
                display_name=display_name or node.display_name,
                description=description or node.description,
                capabilities=node.capabilities,
                position=node.position,
                node_type=block.block_kind if block is not None else node.node_type,
                config=build_config(
                    node=node,
                    api_key_env=api_key_env,
                    path_value=path_value,
                    extra_config_raw=config_raw,
                    provider_type=provider_type,
                    model_id=model_id,
                ),
            )
        )
    updated = AgentGraph(
        graph_id=graph.graph_id,
        name=graph.name,
        nodes=tuple(updated_nodes),
        connections=graph.connections,
        entry_node_id=graph.entry_node_id,
    )
    return graph_to_json(updated)


def update_session_secret(
    node_id: str,
    secret_value: str,
    secret_state: SecretState,
) -> SecretState:
    next_state = dict(secret_state)
    if not node_id:
        return next_state
    trimmed_secret = normalize_text(secret_value).strip()
    if trimmed_secret:
        next_state[node_id] = trimmed_secret
    else:
        next_state.pop(node_id, None)
    return next_state


def export_json(graph_json: str, secret_state: SecretState) -> tuple[str, str]:
    try:
        graph = graph_from_json(graph_json)
    except Exception as error:
        return f"### Validation\n- ERROR: {error}", graph_json
    sanitized_graph, sanitized_secrets = sanitize_graph_and_secrets(graph, secret_state)
    return (
        format_validation_report(sanitized_graph, sanitized_secrets),
        graph_to_json(sanitized_graph),
    )


def validate_graph_artifact(
    graph_json: str,
    secret_state: SecretState,
) -> tuple[str, str]:
    try:
        graph = graph_from_json(graph_json)
    except Exception as error:
        return f"### Validation\n- ERROR: {error}", ""
    sanitized_graph, sanitized_secrets = sanitize_graph_and_secrets(graph, secret_state)
    return format_validation_report(sanitized_graph, sanitized_secrets), ""


def generate_summary(graph_json: str, secret_state: SecretState) -> tuple[str, str]:
    try:
        graph = graph_from_json(graph_json)
    except Exception as error:
        return f"### Validation\n- ERROR: {error}", ""
    sanitized_graph, sanitized_secrets = sanitize_graph_and_secrets(graph, secret_state)
    builder = AgenticGraphBuilder(sanitized_graph, runtime_secrets=sanitized_secrets)
    return format_validation_report(sanitized_graph, sanitized_secrets), builder.generate_summary()


def generate_code(graph_json: str, secret_state: SecretState) -> tuple[str, str]:
    try:
        graph = graph_from_json(graph_json)
    except Exception as error:
        return f"### Validation\n- ERROR: {error}", ""
    sanitized_graph, sanitized_secrets = sanitize_graph_and_secrets(graph, secret_state)
    builder = AgenticGraphBuilder(
        sanitized_graph,
        runtime_secrets=sanitized_secrets,
    )
    try:
        code = builder.generate_python()
    except ValueError as error:
        return (
            format_validation_report(sanitized_graph, sanitized_secrets),
            f"# Code generation error\n{error}",
        )
    return format_validation_report(sanitized_graph, sanitized_secrets), code


def run_runtime(
    graph_json: str,
    secret_state: SecretState,
    runtime_message: str,
) -> tuple[str, str, str]:
    try:
        graph = graph_from_json(graph_json)
    except Exception as error:
        return f"### Validation\n- ERROR: {error}", "", "error"
    sanitized_graph, sanitized_secrets = sanitize_graph_and_secrets(graph, secret_state)

    try:
        result = run_graph_runtime(
            sanitized_graph,
            runtime_message,
            runtime_secrets=sanitized_secrets,
        )
    except Exception as error:
        return (
            format_validation_report(sanitized_graph, sanitized_secrets),
            f"# Runtime error\n- ERROR: {error}",
            "error",
        )

    return (
        format_validation_report(sanitized_graph, sanitized_secrets),
        format_runtime_result(result),
        result.status,
    )


def import_json(raw: str) -> tuple[str, str]:
    try:
        graph = graph_from_json(raw)
    except Exception as error:
        return "", f"### Validation\n- ERROR: {error}"
    sanitized_graph, _ = sanitize_graph_and_secrets(graph)
    return graph_to_json(sanitized_graph), ""


def save_as_workspace(
    graph_json: str,
    secret_state: SecretState,
    workspace_name: str,
) -> str:
    """Save the current graph as a workspace preset."""
    from agentic_runtime.workspaces import create_workspace

    resolved_name = workspace_name.strip()
    if not resolved_name:
        return "Enter a workspace name."

    try:
        graph = graph_from_json(graph_json)
    except Exception as error:
        return f"Error: invalid graph — {error}"

    sanitized_graph, _ = sanitize_graph_and_secrets(graph, secret_state)
    builder = AgenticGraphBuilder(sanitized_graph)
    issues = builder.validate()
    errors = [i for i in issues if i.level == "error"]
    if errors:
        return "Graph has validation errors:\n" + "\n".join(f"- {e.message}" for e in errors)

    sanitized_json = graph_to_json(sanitized_graph)
    preset = create_workspace(
        name=resolved_name,
        graph_json=sanitized_json,
        description=f"Agent graph: {graph.name}",
    )
    return f"Workspace **{preset.name}** saved (`{preset.slug}`)."


def load_from_workspace(workspace_name: str) -> tuple[str, str]:
    """Load a graph from a saved workspace preset."""
    from agentic_runtime.workspaces import list_workspaces, load_workspace_graph

    if not workspace_name:
        return "", "Select a workspace to load."

    slug_map = {w.name: w.slug for w in list_workspaces()}
    slug = slug_map.get(workspace_name, workspace_name)
    graph_json = load_workspace_graph(slug)
    if not graph_json or graph_json == "{}":
        return "", f"Workspace **{workspace_name}** has no saved graph."

    try:
        graph_from_json(graph_json)
    except Exception as error:
        return "", f"Error loading workspace graph: {error}"

    return graph_json, f"Loaded graph from **{workspace_name}**."


def delete_workspace_action(workspace_name: str) -> tuple[object, str]:
    """Delete a workspace preset."""
    from agentic_runtime.workspaces import delete_workspace, list_workspaces
    import gradio as gr

    if not workspace_name:
        return gr.skip(), "Select a workspace to delete."

    slug_map = {w.name: w.slug for w in list_workspaces()}
    slug = slug_map.get(workspace_name, workspace_name)
    if not slug:
        return gr.skip(), "Workspace not found."
    try:
        delete_workspace(slug)
    except ValueError as e:
        return gr.skip(), str(e)
    try:
        from personal_assistant import drop_runtime_sessions

        drop_runtime_sessions(workspace=slug)
    except Exception:
        pass

    choices = [w.name for w in list_workspaces()]
    return (
        gr.Dropdown(choices=choices, value=choices[0] if choices else None),
        f"Deleted **{workspace_name}**.",
    )
