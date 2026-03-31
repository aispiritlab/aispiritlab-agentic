"""Gradio tab construction for the Agent Builder."""

from __future__ import annotations

import json
import uuid

import gradio as gr

from agentic_graph.builder import AgenticGraphBuilder, render_validation_report
from agentic_graph.canvas import CANVAS_GET_GRAPH_JS, CANVAS_LOAD_JS, build_canvas_html
from agentic_graph.models import AgentGraph, AgentNode, NodePosition
from agentic_graph.registry import BlockKind, get_block_by_name, get_blocks_by_kind
from agentic_graph.runtime import RuntimeExecutionResult, run_graph_runtime
from agentic_graph.serialization import graph_from_json, graph_to_json

SecretState = dict[str, str]


def _normalize_text(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _initial_graph() -> AgentGraph:
    return AgentGraph(
        graph_id=uuid.uuid4().hex[:12],
        name="New Agent System",
        nodes=(),
        connections=(),
        entry_node_id=None,
    )


def _block_choices(kind: BlockKind) -> list[str]:
    return [block.agent_name for block in get_blocks_by_kind(kind)]


def _kind_options() -> tuple[BlockKind, ...]:
    return ("agent", "integration", "structural_output")


def _sanitize_graph_and_secrets(
    graph: AgentGraph,
    existing_secrets: SecretState | None = None,
) -> tuple[AgentGraph, SecretState]:
    carried_secrets = {
        node_id: _normalize_text(value).strip()
        for node_id, value in (existing_secrets or {}).items()
        if _normalize_text(value).strip()
    }
    sanitized_nodes: list[AgentNode] = []
    next_secrets: SecretState = {}

    for node in graph.nodes:
        legacy_secret = ""
        sanitized_config: list[tuple[str, str]] = []
        for key, value in node.config:
            if key == "api_key":
                legacy_secret = _normalize_text(value).strip()
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


def _format_validation_report(
    graph: AgentGraph,
    runtime_secrets: SecretState | None = None,
) -> str:
    issues = AgenticGraphBuilder(graph, runtime_secrets=runtime_secrets).validate()
    return "### Validation\n" + render_validation_report(issues)


def _parse_config(raw: str) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for line in _normalize_text(raw).strip().splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        normalized_key = _normalize_text(key).strip()
        if normalized_key == "api_key":
            continue
        pairs.append((normalized_key, _normalize_text(value).strip()))
    return tuple(pairs)


def _build_config(
    *,
    node: AgentNode,
    api_key_env: str,
    path_value: str,
    extra_config_raw: str,
) -> tuple[tuple[str, str], ...]:
    existing = {
        key: value
        for key, value in node.config
        if key not in {"api_key", "api_key_env", "path"}
    }
    for key, value in _parse_config(extra_config_raw):
        existing[key] = value

    config_items: list[tuple[str, str]] = []
    if api_key_env:
        config_items.append(("api_key_env", api_key_env))
    if path_value:
        config_items.append(("path", path_value))
    for key, value in existing.items():
        if value:
            config_items.append((key, value))
    return tuple(config_items)


def _format_runtime_result(result: RuntimeExecutionResult) -> str:
    lines = [
        "# Runtime Result",
        f"Status: {_normalize_text(result.status)}",
        f"Entry agent: {_normalize_text(result.entry_agent)}",
        "",
        "## Response",
        _normalize_text(result.response),
    ]
    if result.steps:
        lines.extend(["", "## Steps"])
        lines.extend(f"- {_normalize_text(step)}" for step in result.steps)
    if result.outputs:
        lines.extend(["", "## Outputs"])
        lines.extend(
            f"- {_normalize_text(output.output_name)}: {_normalize_text(output.path)}"
            for output in result.outputs
        )
    return "\n".join(lines)


def build_agent_builder_tab() -> None:
    """Build the Agent Builder UI inside the current gr.Blocks context."""
    initial = _initial_graph()
    initial_json = graph_to_json(initial)

    gr.HTML(
        """
<style>
  .agent-graph-bridge {
    position: absolute !important;
    width: 1px !important;
    height: 1px !important;
    overflow: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
  }
</style>
"""
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=220):
            gr.Markdown("### Block Palette")
            block_kind_selector = gr.Radio(
                choices=list(_kind_options()),
                value="agent",
                label="Block kind",
            )
            block_selector = gr.Dropdown(
                choices=_block_choices("agent"),
                value=_block_choices("agent")[0] if _block_choices("agent") else None,
                label="Prepared block",
            )
            add_block_btn = gr.Button("Add Block", variant="primary", size="sm")
            gr.Markdown("---")
            gr.Markdown("### Graph")
            graph_name_input = gr.Textbox(label="Graph name", value=initial.name)
            set_entry_btn = gr.Button("Set Selected as Entry", size="sm")

        with gr.Column(scale=3, min_width=420):
            gr.HTML(value=build_canvas_html())
            graph_json_bridge = gr.Textbox(
                value=initial_json,
                elem_id="graph-json-bridge",
                elem_classes=["agent-graph-bridge"],
            )
            selected_node_bridge = gr.Textbox(
                value="",
                elem_id="selected-node-bridge",
                elem_classes=["agent-graph-bridge"],
            )

        with gr.Column(scale=1, min_width=240):
            gr.Markdown("### Block Properties")
            prop_name = gr.Textbox(label="Display name", interactive=True)
            prop_agent = gr.Textbox(label="Block id", interactive=False)
            prop_type = gr.Textbox(label="Block kind", interactive=False)
            prop_desc = gr.Textbox(label="Description", interactive=True, lines=3)
            prop_caps = gr.Textbox(label="Capabilities", interactive=False)
            prop_is_entry = gr.Textbox(label="Entry point", interactive=False)
            prop_api_key = gr.Textbox(
                label="API token",
                interactive=True,
                type="password",
                placeholder="Optional inline secret",
            )
            prop_api_key_env = gr.Textbox(
                label="API token env var",
                interactive=True,
                placeholder="TAVILY_API_KEY",
            )
            prop_path = gr.Textbox(
                label="Path",
                interactive=True,
                placeholder="outputs/agentic_graph.md",
            )
            prop_config = gr.Textbox(
                label="Advanced config (non-secret key=value)",
                interactive=True,
                lines=4,
                placeholder="timeout=20\nmodel=custom",
            )
            delete_node_btn = gr.Button("Delete Selected Block", variant="stop", size="sm")

    with gr.Row():
        export_btn = gr.Button("Export JSON", size="sm")
        validate_btn = gr.Button("Validate Graph", size="sm")
        summary_btn = gr.Button("Generate Summary", size="sm")
        generate_btn = gr.Button("Generate Builder Code", variant="primary", size="sm")

    with gr.Row():
        runtime_input = gr.Textbox(
            label="Runtime Input",
            lines=3,
            placeholder="Ask the runtime something...",
            scale=4,
        )
        run_runtime_btn = gr.Button("Run Runtime", variant="primary", size="sm")

    with gr.Row():
        import_input = gr.Textbox(
            label="Import JSON",
            lines=3,
            placeholder="Paste graph JSON here...",
            scale=4,
        )
        import_btn = gr.Button("Import", size="sm")

    validation_output = gr.Markdown(value=_format_validation_report(initial))
    artifact_output = gr.Textbox(
        label="Artifact Output",
        lines=18,
        interactive=False,
    )
    runtime_output = gr.Textbox(
        label="Runtime Output",
        lines=18,
        interactive=False,
    )

    graph_state = gr.State(value=initial_json)
    integration_secret_state = gr.State(value={})

    def update_block_selector(kind: str) -> dict[str, object]:
        choices = _block_choices(kind) if kind in _kind_options() else []
        value = choices[0] if choices else None
        return gr.update(choices=choices, value=value)

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
            graph = _initial_graph()
            graph_json = graph_to_json(graph)
            return graph_json, _format_validation_report(graph), {}
        try:
            graph = graph_from_json(bridge_json)
        except Exception as error:
            return bridge_json, f"### Validation\n- ERROR: {error}", secret_state
        sanitized_graph, sanitized_secrets = _sanitize_graph_and_secrets(graph, secret_state)
        sanitized_json = graph_to_json(sanitized_graph)
        return (
            sanitized_json,
            _format_validation_report(sanitized_graph, sanitized_secrets),
            sanitized_secrets,
        )

    def on_node_selected(
        node_id: str,
        graph_json: str,
        secret_state: SecretState,
    ) -> tuple[str, str, str, str, str, str, str, str, str, str]:
        if not node_id or not graph_json:
            return "", "", "", "", "", "", "", "", "", ""
        try:
            graph = graph_from_json(graph_json)
        except (json.JSONDecodeError, KeyError):
            return "", "", "", "", "", "", "", "", "", ""
        for node in graph.nodes:
            if node.node_id == node_id:
                is_entry = "Yes" if graph.entry_node_id == node_id else "No"
                config_map = dict(node.config)
                extra_config = "\n".join(
                    f"{key}={value}"
                    for key, value in node.config
                    if key not in {"api_key", "api_key_env", "path"}
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
                )
        return "", "", "", "", "", "", "", "", "", ""

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
                    config=_build_config(
                        node=node,
                        api_key_env=api_key_env,
                        path_value=path_value,
                        extra_config_raw=config_raw,
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
        trimmed_secret = _normalize_text(secret_value).strip()
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
        sanitized_graph, sanitized_secrets = _sanitize_graph_and_secrets(graph, secret_state)
        return (
            _format_validation_report(sanitized_graph, sanitized_secrets),
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
        sanitized_graph, sanitized_secrets = _sanitize_graph_and_secrets(graph, secret_state)
        return _format_validation_report(sanitized_graph, sanitized_secrets), ""

    def generate_summary(graph_json: str, secret_state: SecretState) -> tuple[str, str]:
        try:
            graph = graph_from_json(graph_json)
        except Exception as error:
            return f"### Validation\n- ERROR: {error}", ""
        sanitized_graph, sanitized_secrets = _sanitize_graph_and_secrets(graph, secret_state)
        builder = AgenticGraphBuilder(sanitized_graph, runtime_secrets=sanitized_secrets)
        return _format_validation_report(sanitized_graph, sanitized_secrets), builder.generate_summary()

    def generate_code(graph_json: str, secret_state: SecretState) -> tuple[str, str]:
        try:
            graph = graph_from_json(graph_json)
        except Exception as error:
            return f"### Validation\n- ERROR: {error}", ""
        sanitized_graph, sanitized_secrets = _sanitize_graph_and_secrets(graph, secret_state)
        builder = AgenticGraphBuilder(
            sanitized_graph,
            runtime_secrets=sanitized_secrets,
        )
        try:
            code = builder.generate_python()
        except ValueError as error:
            return (
                _format_validation_report(sanitized_graph, sanitized_secrets),
                f"# Code generation error\n{error}",
            )
        return _format_validation_report(sanitized_graph, sanitized_secrets), code

    def run_runtime(
        graph_json: str,
        secret_state: SecretState,
        runtime_message: str,
    ) -> tuple[str, str]:
        try:
            graph = graph_from_json(graph_json)
        except Exception as error:
            return f"### Validation\n- ERROR: {error}", ""
        sanitized_graph, sanitized_secrets = _sanitize_graph_and_secrets(graph, secret_state)
        try:
            result = run_graph_runtime(
                sanitized_graph,
                runtime_message,
                runtime_secrets=sanitized_secrets,
            )
        except Exception as error:
            return (
                _format_validation_report(sanitized_graph, sanitized_secrets),
                f"# Runtime error\n- ERROR: {error}",
            )
        return (
            _format_validation_report(sanitized_graph, sanitized_secrets),
            _format_runtime_result(result),
        )

    def import_json(raw: str) -> tuple[str, str]:
        try:
            graph = graph_from_json(raw)
        except Exception as error:
            return "", f"### Validation\n- ERROR: {error}"
        sanitized_graph, _ = _sanitize_graph_and_secrets(graph)
        return graph_to_json(sanitized_graph), ""

    block_kind_selector.change(
        update_block_selector,
        inputs=[block_kind_selector],
        outputs=[block_selector],
    )

    add_block_btn.click(
        add_block,
        inputs=[block_selector, graph_json_bridge],
        outputs=[graph_json_bridge],
    ).then(
        None,
        inputs=[graph_json_bridge],
        js=CANVAS_LOAD_JS,
    )

    graph_json_bridge.change(
        on_graph_bridge_change,
        inputs=[graph_json_bridge, integration_secret_state],
        outputs=[graph_state, validation_output, integration_secret_state],
    )

    selected_node_bridge.change(
        on_node_selected,
        inputs=[selected_node_bridge, graph_json_bridge, integration_secret_state],
        outputs=[
            prop_name,
            prop_agent,
            prop_type,
            prop_desc,
            prop_caps,
            prop_is_entry,
            prop_api_key,
            prop_api_key_env,
            prop_path,
            prop_config,
        ],
    )

    delete_node_btn.click(
        delete_selected,
        inputs=[selected_node_bridge, graph_json_bridge],
        outputs=[graph_json_bridge],
    ).then(
        None,
        inputs=[graph_json_bridge],
        js=CANVAS_LOAD_JS,
    )

    set_entry_btn.click(
        set_entry,
        inputs=[selected_node_bridge, graph_json_bridge],
        outputs=[graph_json_bridge],
    ).then(
        None,
        inputs=[graph_json_bridge],
        js=CANVAS_LOAD_JS,
    )

    prop_api_key.change(
        update_session_secret,
        inputs=[selected_node_bridge, prop_api_key, integration_secret_state],
        outputs=[integration_secret_state],
    )

    for component in (prop_name, prop_desc, prop_api_key_env, prop_path, prop_config):
        component.change(
            update_node_property,
            inputs=[
                selected_node_bridge,
                graph_json_bridge,
                prop_name,
                prop_desc,
                prop_api_key_env,
                prop_path,
                prop_config,
            ],
            outputs=[graph_json_bridge],
        ).then(
            None,
            inputs=[graph_json_bridge],
            js=CANVAS_LOAD_JS,
        )

    graph_name_input.change(
        update_graph_name,
        inputs=[graph_name_input, graph_json_bridge],
        outputs=[graph_json_bridge],
    )

    export_btn.click(
        export_json,
        inputs=[graph_json_bridge, integration_secret_state],
        outputs=[validation_output, artifact_output],
        js=CANVAS_GET_GRAPH_JS,
    )
    validate_btn.click(
        validate_graph_artifact,
        inputs=[graph_json_bridge, integration_secret_state],
        outputs=[validation_output, artifact_output],
        js=CANVAS_GET_GRAPH_JS,
    )
    summary_btn.click(
        generate_summary,
        inputs=[graph_json_bridge, integration_secret_state],
        outputs=[validation_output, artifact_output],
        js=CANVAS_GET_GRAPH_JS,
    )
    generate_btn.click(
        generate_code,
        inputs=[graph_json_bridge, integration_secret_state],
        outputs=[validation_output, artifact_output],
        js=CANVAS_GET_GRAPH_JS,
    )
    run_runtime_btn.click(
        run_runtime,
        inputs=[graph_json_bridge, integration_secret_state, runtime_input],
        outputs=[validation_output, runtime_output],
        js=CANVAS_GET_GRAPH_JS,
    )

    import_btn.click(
        import_json,
        inputs=[import_input],
        outputs=[graph_json_bridge, validation_output],
    ).then(
        None,
        inputs=[graph_json_bridge],
        js=CANVAS_LOAD_JS,
    )
