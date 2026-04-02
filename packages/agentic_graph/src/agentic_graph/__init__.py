"""Agentics Manager — visual agent composition, workspace runtime, and event tracking."""

from __future__ import annotations

from agentic_graph.builder import (
    AgenticGraphBuilder,
    ValidationIssue,
    generate_graph_summary,
    generate_python_code,
    render_validation_report,
    validate_graph,
)
from agentic_graph.runtime import (
    GraphRuntime,
    RuntimeEventRecord,
    RuntimeExecutionResult,
    RuntimeOutput,
    run_graph_runtime,
)
from agentic_graph.tab import build_agent_builder_tab


def main() -> None:
    """Launch the Agentics Manager Gradio app."""
    from agentic_runtime.users import create_user, list_users
    from agentic_runtime.workspaces import (
        create_workspace,
        list_workspaces,
        set_active_workspace,
    )
    from chat.styles import GLOBAL_CSS
    from chat.theme import SPIRIT_THEME
    import gradio as gr

    def _user_choices() -> list[str]:
        return [u.name for u in list_users()]

    def _workspace_choices() -> list[str]:
        return [w.name for w in list_workspaces()]

    def _ws_slug_map() -> dict[str, str]:
        return {w.name: w.slug for w in list_workspaces()}

    def _on_create_user(name: str) -> tuple[object, str]:
        name = name.strip()
        if not name:
            return gr.skip(), ""
        profile = create_user(name)
        return gr.Dropdown(choices=_user_choices(), value=profile.name), ""

    def _on_ws_switch(ws_name: str) -> str:
        slug_map = _ws_slug_map()
        slug = slug_map.get(ws_name, "default")
        set_active_workspace(slug)
        return slug

    def _on_new_workspace(name: str) -> tuple[object, str]:
        name = name.strip()
        if not name:
            return gr.skip(), ""
        preset = create_workspace(name=name, graph_json="{}")
        set_active_workspace(preset.slug)
        return gr.Dropdown(choices=_workspace_choices(), value=preset.name), preset.slug

    with gr.Blocks(fill_height=True, title="Agentics Manager") as app:
        gr.Navbar(main_page_name="Agentics Manager")

        default_workspace_name = _workspace_choices()[0] if _workspace_choices() else "Default"
        active_ws_state = gr.State(value=_ws_slug_map().get(default_workspace_name, "default"))
        preview_events_state = gr.State(value=[])

        # ── Context bar: User + Workspace ──
        with gr.Row():
            user_selector = gr.Dropdown(
                label="User",
                choices=_user_choices(),
                value=list_users()[0].name if list_users() else "Default",
                scale=2,
                min_width=120,
            )
            new_user_input = gr.Textbox(label="New user", placeholder="Name", scale=2, min_width=100)
            create_user_btn = gr.Button("Create", size="sm", scale=1)
            workspace_selector = gr.Dropdown(
                label="Workspace",
                choices=_workspace_choices(),
                value=default_workspace_name,
                scale=3,
                min_width=160,
            )
            new_ws_input = gr.Textbox(label="New workspace", placeholder="research-team", scale=2, min_width=120)
            new_ws_btn = gr.Button("+ New", size="sm", variant="primary", scale=1)

        # ── View tabs ──
        with gr.Tabs():
            with gr.Tab("Editor", id="tab-editor"):
                build_agent_builder_tab(active_workspace_state=active_ws_state)

            with gr.Tab("Runner", id="tab-runner"):
                from agentic_graph.runner_tab import build_runner_tab
                build_runner_tab(preview_events_state=preview_events_state)

            with gr.Tab("Events", id="tab-events"):
                from agentic_graph.events_tab import build_events_tab
                build_events_tab(preview_events_state=preview_events_state)

        # ── Event wiring ──
        create_user_btn.click(
            _on_create_user,
            inputs=[new_user_input],
            outputs=[user_selector, new_user_input],
        )
        workspace_selector.change(
            _on_ws_switch,
            inputs=[workspace_selector],
            outputs=[active_ws_state],
        )
        new_ws_btn.click(
            _on_new_workspace,
            inputs=[new_ws_input],
            outputs=[workspace_selector, active_ws_state],
        )

    app.queue()
    app.launch(server_name="0.0.0.0", server_port=7861, theme=SPIRIT_THEME, css=GLOBAL_CSS)


__all__ = [
    "AgenticGraphBuilder",
    "ValidationIssue",
    "build_agent_builder_tab",
    "generate_graph_summary",
    "generate_python_code",
    "GraphRuntime",
    "main",
    "render_validation_report",
    "RuntimeEventRecord",
    "RuntimeExecutionResult",
    "RuntimeOutput",
    "run_graph_runtime",
    "validate_graph",
]
