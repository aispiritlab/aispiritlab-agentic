"""Gradio tab construction for the Agent Builder."""

from __future__ import annotations

import gradio as gr

from agentic_graph.canvas import CANVAS_GET_GRAPH_JS, CANVAS_LOAD_JS, build_canvas_html
from agentic_graph.serialization import graph_to_json
from agentic_graph.tab_actions import (
    add_block,
    delete_selected,
    export_json,
    generate_code,
    generate_summary,
    import_json,
    on_graph_bridge_change,
    on_node_selected,
    run_runtime,
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


def build_agent_builder_tab() -> None:
    """Build the Agent Builder UI inside the current gr.Blocks context."""
    initial = initial_graph()
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
                choices=list(kind_options()),
                value="agent",
                label="Block kind",
            )
            block_selector = gr.Dropdown(
                choices=block_choices("agent"),
                value=block_choices("agent")[0] if block_choices("agent") else None,
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
                placeholder="dispatch_mode=broadcast\nprovider_type=openai\nmodel_id=qwen3.5-4b",
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

    validation_output = gr.Markdown(value=format_validation_report(initial))
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


_sanitize_graph_and_secrets = sanitize_graph_and_secrets

__all__ = ["SecretState", "build_agent_builder_tab", "_sanitize_graph_and_secrets"]
