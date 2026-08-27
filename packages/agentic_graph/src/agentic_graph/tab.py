"""Gradio tab construction for the Agent Builder (Editor tab)."""

from __future__ import annotations

import gradio as gr

from agentic_graph.canvas import (
    CANVAS_GET_GRAPH_JS,
    CANVAS_LOAD_JS,
    CANVAS_RUNTIME_DONE_JS,
    CANVAS_RUNTIME_START_JS,
    build_canvas_html,
)
from agentic_graph.serialization import graph_to_json
from agentic_graph.tab_actions import (
    add_block,
    delete_selected,
    delete_workspace_action,
    export_json,
    generate_code,
    generate_summary,
    import_json,
    load_from_workspace,
    on_graph_bridge_change,
    on_node_selected,
    run_runtime,
    save_as_workspace,
    set_entry,
    update_graph_name,
    update_node_property,
    update_session_secret,
    validate_graph_artifact,
)
from agentic_graph.tab_state import (
    SecretState,
    block_choices,
    format_validation_report,
    initial_graph,
    kind_options,
    sanitize_graph_and_secrets,
)


def _get_workspace_names() -> list[str]:
    try:
        from agentic_runtime.workspaces import list_workspaces

        return [w.name for w in list_workspaces()]
    except Exception:
        return []


def _get_workspace_slug(name: str) -> str:
    try:
        from agentic_runtime.workspaces import list_workspaces

        for workspace in list_workspaces():
            if workspace.name == name:
                return workspace.slug
    except Exception:
        pass
    return name or "default"


_PANEL_CSS = """
<style>
  .agent-graph-bridge {
    position: absolute !important;
    width: 1px !important;
    height: 1px !important;
    overflow: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
  }
  .settings-section {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    padding: 10px 12px !important;
    margin: 4px 0 !important;
  }
  .section-label {
    font-size: 10px !important;
    font-weight: 700 !important;
    color: #10b981 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    margin-bottom: 4px !important;
  }
  .section-label-amber { color: #f59e0b !important; }
  .section-label-cyan { color: #06b6d4 !important; }
  .section-label-violet { color: #8b5cf6 !important; }
</style>
"""


def _on_node_selected_with_visibility(
    node_id: str,
    graph_json: str,
    secret_state: SecretState,
) -> tuple:
    """Wrap on_node_selected to add type-specific group visibility."""
    vals = on_node_selected(node_id, graph_json, secret_state)
    node_type = vals[2] if len(vals) > 2 else ""
    has_node = bool(node_type)

    return (
        *vals,
        gr.update(visible=has_node and node_type == "agent"),
        gr.update(visible=has_node and node_type == "provider"),
        gr.update(visible=has_node and node_type == "integration"),
        gr.update(visible=has_node and node_type == "structural_output"),
        gr.update(visible=has_node),
    )


def build_agent_builder_tab(*, active_workspace_state: gr.State | None = None) -> None:
    """Build the Editor tab UI."""
    initial = initial_graph()
    initial_json = graph_to_json(initial)

    gr.HTML(_PANEL_CSS)

    with gr.Row():
        # ══════════════════════════════════════
        # LEFT SIDEBAR
        # ══════════════════════════════════════
        with gr.Column(scale=1, min_width=200):
            # Block Palette
            with gr.Group(elem_classes=["settings-section"]):
                gr.Markdown("Palette", elem_classes=["section-label"])
                block_kind_selector = gr.Radio(
                    choices=list(kind_options()),
                    value="agent",
                    label="Kind",
                )
                block_selector = gr.Dropdown(
                    choices=block_choices("agent"),
                    value=block_choices("agent")[0] if block_choices("agent") else None,
                    label="Block",
                )
                add_block_btn = gr.Button("Add Block", variant="primary", size="sm")

            # Graph
            with gr.Group(elem_classes=["settings-section"]):
                gr.Markdown("Graph", elem_classes=["section-label"])
                graph_name_input = gr.Textbox(label="Name", value=initial.name)
                set_entry_btn = gr.Button("Set Entry", size="sm")

            # Presets
            with gr.Group(elem_classes=["settings-section"]):
                gr.Markdown("Presets", elem_classes=["section-label", "section-label-cyan"])
                workspace_load_selector = gr.Dropdown(
                    label="Saved presets",
                    choices=_get_workspace_names(),
                    interactive=True,
                )
                with gr.Row():
                    load_workspace_btn = gr.Button("Load", size="sm")
                    delete_workspace_btn = gr.Button("Del", variant="stop", size="sm")
                workspace_name_input = gr.Textbox(
                    label="Save as",
                    placeholder="my-preset",
                )
                save_workspace_btn = gr.Button("Save Preset", variant="primary", size="sm")
                workspace_save_status = gr.Markdown()

        # ══════════════════════════════════════
        # CENTER CANVAS
        # ══════════════════════════════════════
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

        # ══════════════════════════════════════
        # RIGHT SIDEBAR — Properties (collapsible)
        # ══════════════════════════════════════
        with gr.Accordion("Properties", open=True):
            prop_name = gr.Textbox(label="Display name", interactive=True)
            prop_agent = gr.Textbox(label="Block id", interactive=False)
            prop_type = gr.Textbox(label="Kind", interactive=False, visible=False)
            prop_desc = gr.Textbox(label="Description", interactive=True, lines=2)

            with gr.Group(visible=False, elem_classes=["settings-section"]) as agent_group:
                gr.Markdown("Agent", elem_classes=["section-label"])
                prop_caps = gr.Textbox(label="Capabilities", interactive=False)
                prop_is_entry = gr.Textbox(label="Entry point", interactive=False)

            with gr.Group(visible=False, elem_classes=["settings-section"]) as provider_group:
                gr.Markdown("Model Provider", elem_classes=["section-label", "section-label-cyan"])
                prop_provider_type = gr.Textbox(
                    label="Provider type",
                    interactive=True,
                    placeholder="openai",
                    info="openai, mlx, vllm, transformers",
                )
                prop_model_id = gr.Textbox(
                    label="Model ID",
                    interactive=True,
                    placeholder="qwen3.5-4b",
                )

            with gr.Group(visible=False, elem_classes=["settings-section"]) as integration_group:
                gr.Markdown("Integration", elem_classes=["section-label", "section-label-amber"])
                prop_api_key = gr.Textbox(
                    label="API token",
                    interactive=True,
                    type="password",
                    placeholder="Session-only secret",
                )
                prop_api_key_env = gr.Textbox(
                    label="API token env var",
                    interactive=True,
                    placeholder="TAVILY_API_KEY",
                )

            with gr.Group(visible=False, elem_classes=["settings-section"]) as output_group:
                gr.Markdown("Output", elem_classes=["section-label", "section-label-violet"])
                prop_path = gr.Textbox(
                    label="Output path",
                    interactive=True,
                    placeholder="outputs/agentic_graph.md",
                )

            with gr.Group(visible=False) as common_bottom:
                prop_config = gr.Textbox(
                    label="Advanced config",
                    interactive=True,
                    lines=3,
                    placeholder="key=value (one per line)",
                )
                delete_node_btn = gr.Button("Delete Block", variant="stop", size="sm")

    # ══════════════════════════════════════
    # TOOLBAR
    # ══════════════════════════════════════
    with gr.Row():
        validate_btn = gr.Button("Validate", size="sm")
        export_btn = gr.Button("Export", size="sm")
        import_btn = gr.Button("Import", size="sm")
        summary_btn = gr.Button("Summary", size="sm")
        generate_btn = gr.Button("Code", variant="primary", size="sm")

    # ══════════════════════════════════════
    # OUTPUTS (collapsible)
    # ══════════════════════════════════════
    with gr.Accordion("Validation", open=False):
        validation_output = gr.Markdown(value=format_validation_report(initial))

    with gr.Accordion("Generated Output", open=False):
        artifact_output = gr.Textbox(label="Output", lines=14, interactive=False)

    with gr.Accordion("Import", open=False):
        import_input = gr.Textbox(
            label="Paste graph JSON",
            lines=4,
            placeholder='{"graph_id": "...", "nodes": [...]}',
        )
        import_apply_btn = gr.Button("Apply Import", size="sm")

    with gr.Accordion("Runtime", open=False):
        with gr.Row():
            runtime_input = gr.Textbox(
                label="Input",
                lines=2,
                placeholder="Ask the runtime something...",
                scale=4,
            )
            run_runtime_btn = gr.Button("Run", variant="primary", size="sm")
        runtime_output = gr.Textbox(label="Output", lines=14, interactive=False)

    graph_state = gr.State(value=initial_json)
    integration_secret_state = gr.State(value={})
    runtime_status_state = gr.State(value="idle")

    # ══════════════════════════════════════
    # EVENT WIRING
    # ══════════════════════════════════════

    def update_block_selector(kind: str) -> dict[str, object]:
        choices = block_choices(kind) if kind in kind_options() else []
        value = choices[0] if choices else None
        return gr.update(choices=choices, value=value)

    block_kind_selector.change(
        update_block_selector,
        inputs=[block_kind_selector],
        outputs=[block_selector],
    )

    add_block_btn.click(
        add_block,
        inputs=[block_selector, graph_json_bridge],
        outputs=[graph_json_bridge],
    ).then(None, inputs=[graph_json_bridge], js=CANVAS_LOAD_JS)

    graph_json_bridge.change(
        on_graph_bridge_change,
        inputs=[graph_json_bridge, integration_secret_state],
        outputs=[graph_state, validation_output, integration_secret_state],
    )

    selected_node_bridge.change(
        _on_node_selected_with_visibility,
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
            prop_provider_type,
            prop_model_id,
            agent_group,
            provider_group,
            integration_group,
            output_group,
            common_bottom,
        ],
    )

    delete_node_btn.click(
        delete_selected,
        inputs=[selected_node_bridge, graph_json_bridge],
        outputs=[graph_json_bridge],
    ).then(None, inputs=[graph_json_bridge], js=CANVAS_LOAD_JS)

    set_entry_btn.click(
        set_entry,
        inputs=[selected_node_bridge, graph_json_bridge],
        outputs=[graph_json_bridge],
    ).then(None, inputs=[graph_json_bridge], js=CANVAS_LOAD_JS)

    prop_api_key.change(
        update_session_secret,
        inputs=[selected_node_bridge, prop_api_key, integration_secret_state],
        outputs=[integration_secret_state],
    )

    for component in (
        prop_name,
        prop_desc,
        prop_api_key_env,
        prop_path,
        prop_config,
        prop_provider_type,
        prop_model_id,
    ):
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
                prop_provider_type,
                prop_model_id,
            ],
            outputs=[graph_json_bridge],
        ).then(None, inputs=[graph_json_bridge], js=CANVAS_LOAD_JS)

    graph_name_input.change(
        update_graph_name,
        inputs=[graph_name_input, graph_json_bridge],
        outputs=[graph_json_bridge],
    )

    # Toolbar
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
    import_btn.click(lambda: None)  # Just opens the Import accordion

    # Import
    import_apply_btn.click(
        import_json,
        inputs=[import_input],
        outputs=[graph_json_bridge, validation_output],
    ).then(None, inputs=[graph_json_bridge], js=CANVAS_LOAD_JS)

    # Runtime
    run_runtime_btn.click(
        run_runtime,
        inputs=[graph_json_bridge, integration_secret_state, runtime_input],
        outputs=[validation_output, runtime_output, runtime_status_state],
        js=CANVAS_RUNTIME_START_JS,
    ).then(None, inputs=[runtime_status_state], js=CANVAS_RUNTIME_DONE_JS)

    # Presets
    load_workspace_btn.click(
        load_from_workspace,
        inputs=[workspace_load_selector],
        outputs=[graph_json_bridge, workspace_save_status],
    ).then(None, inputs=[graph_json_bridge], js=CANVAS_LOAD_JS)

    if active_workspace_state is not None:
        load_workspace_btn.click(
            _get_workspace_slug,
            inputs=[workspace_load_selector],
            outputs=[active_workspace_state],
        )

    delete_workspace_btn.click(
        delete_workspace_action,
        inputs=[workspace_load_selector],
        outputs=[workspace_load_selector, workspace_save_status],
    )

    save_workspace_btn.click(
        save_as_workspace,
        inputs=[graph_json_bridge, integration_secret_state, workspace_name_input],
        outputs=[workspace_save_status],
        js=CANVAS_GET_GRAPH_JS,
    ).then(
        lambda: gr.Dropdown(choices=_get_workspace_names()),
        outputs=[workspace_load_selector],
    )


_sanitize_graph_and_secrets = sanitize_graph_and_secrets

__all__ = ["SecretState", "_sanitize_graph_and_secrets", "build_agent_builder_tab"]
