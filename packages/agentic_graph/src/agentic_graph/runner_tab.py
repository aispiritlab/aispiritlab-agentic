"""Runner tab — execute workspace presets and track agent activity."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import asdict

import gradio as gr

from agentic_graph.canvas import CANVAS_LOAD_JS, build_canvas_html


def _get_workspace_names() -> list[str]:
    try:
        from agentic_runtime.workspaces import list_workspaces

        return [w.name for w in list_workspaces()]
    except Exception:
        return []


def _load_preset_graph(workspace_name: str) -> tuple[str, str, str, list[dict[str, object]]]:
    """Load workspace graph JSON for the canvas + status info."""
    from agentic_runtime.workspaces import list_workspaces, load_workspace_graph

    if not workspace_name:
        return "", "Select a preset.", "", []

    slug_map = {w.name: w.slug for w in list_workspaces()}
    slug = slug_map.get(workspace_name, workspace_name)
    graph_json = load_workspace_graph(slug)
    if not graph_json or graph_json == "{}":
        return "", f"**{workspace_name}** has no saved graph.", "", []

    try:
        from agentic_graph.serialization import graph_from_json

        graph = graph_from_json(graph_json)
        agent_count = sum(1 for n in graph.nodes if n.node_type == "agent")
        status = f"Loaded **{workspace_name}** — {agent_count} agents"
    except Exception as error:
        status = f"Error: {error}"

    return graph_json, status, graph_json, []


def _run_preset(
    graph_json: str,
    message: str,
) -> tuple[str, str, list[dict[str, object]], str]:
    """Run a message through the preset graph and capture events."""
    if not graph_json or graph_json == "{}":
        return "No preset loaded.", "", [], "error"
    if not message.strip():
        return "Enter a message.", "", [], "error"

    try:
        from agentic_graph.runtime import run_graph_runtime
        from agentic_graph.serialization import graph_from_json

        graph = graph_from_json(graph_json)
        result = run_graph_runtime(graph, message.strip())
        steps_text = "\n".join(f"- {step}" for step in result.steps) if result.steps else ""
        response = result.response or "(empty response)"
        event_records = [asdict(record) for record in result.events]
        return response, steps_text, event_records, result.status
    except Exception as error:
        return f"Runtime error: {error}", "", [], "error"


def build_runner_tab(preview_events_state: gr.State | None = None) -> None:
    """Build the Runner tab UI."""
    preview_events_state = preview_events_state or gr.State(value=[])

    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("### Preset")
            runner_preset = gr.Dropdown(
                label="Active preset",
                choices=_get_workspace_names(),
                interactive=True,
            )
            with gr.Row():
                load_preset_btn = gr.Button("Load", variant="primary", size="sm")
                refresh_presets_btn = gr.Button("Refresh", size="sm")
            runner_status = gr.Markdown()

            gr.HTML('<hr style="border-color:#334155;margin:12px 0;">')
            gr.Markdown("### Controls")
            runner_message = gr.Textbox(
                label="Message",
                placeholder="Ask the agents something...",
                lines=3,
            )
            run_btn = gr.Button("Send", variant="primary")

        with gr.Column(scale=3, min_width=420):
            gr.HTML(value=build_canvas_html())
            runner_graph_bridge = gr.Textbox(
                value="",
                elem_id="runner-graph-bridge",
                elem_classes=["agent-graph-bridge"],
                visible=False,
            )

        with gr.Column(scale=1, min_width=240):
            gr.Markdown("### Output")
            runner_output = gr.Textbox(
                label="Response",
                lines=12,
                interactive=False,
            )

            gr.Markdown("### Steps")
            runner_steps = gr.Markdown()

            gr.Markdown("### History")
            runner_history = gr.Dataframe(
                headers=["#", "Input", "Output"],
                column_count=(3, "fixed"),
                interactive=False,
                wrap=True,
            )

    runner_graph_state = gr.State(value="")
    runner_turn_count = gr.State(value=0)
    runner_history_data = gr.State(value=[])

    refresh_presets_btn.click(
        lambda: gr.Dropdown(choices=_get_workspace_names()),
        outputs=[runner_preset],
    )

    load_preset_btn.click(
        _load_preset_graph,
        inputs=[runner_preset],
        outputs=[runner_graph_bridge, runner_status, runner_graph_state, preview_events_state],
    ).then(
        None,
        inputs=[runner_graph_bridge],
        js=CANVAS_LOAD_JS,
    )

    def _run_and_record(
        graph_json: str,
        message: str,
        turn_count: int,
        history: list[list[str]],
    ) -> Generator[tuple[str, str, int, list[list[str]], object, list[dict[str, object]], str]]:
        final_output, steps, event_records, status = _run_preset(graph_json, message)
        new_turn = turn_count + 1

        streamed_output = ""
        for char in final_output:
            streamed_output += char
            yield (
                streamed_output,
                steps,
                new_turn,
                history,
                gr.skip(),
                gr.skip(),
                gr.skip(),
            )

        updated_history = list(history) + [[str(new_turn), message.strip(), final_output]]
        status_message = (
            f"Run finished with status: **{status}**"
            if status == "ok"
            else f"Run failed with status: **{status}**"
        )
        yield (
            final_output,
            steps,
            new_turn,
            updated_history,
            gr.Dataframe(value=updated_history),
            event_records,
            status_message,
        )

    run_btn.click(
        _run_and_record,
        inputs=[runner_graph_state, runner_message, runner_turn_count, runner_history_data],
        outputs=[
            runner_output,
            runner_steps,
            runner_turn_count,
            runner_history_data,
            runner_history,
            preview_events_state,
            runner_status,
        ],
    )
