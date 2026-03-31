"""Agentic Graph builder UI, validation, and code generation helpers."""

from __future__ import annotations

from agentic_graph.builder import (
    AgenticGraphBuilder,
    ValidationIssue,
    generate_graph_summary,
    generate_python_code,
    render_validation_report,
    validate_graph,
)
from agentic_graph.runtime import GraphRuntime, RuntimeExecutionResult, RuntimeOutput, run_graph_runtime
from agentic_graph.tab import build_agent_builder_tab


def main() -> None:
    """Launch the standalone Agentic Graph Gradio app."""
    import gradio as gr

    with gr.Blocks(fill_height=True, title="Agentic Graph") as app:
        gr.Markdown("# Agentic Graph Builder")
        build_agent_builder_tab()

    app.queue()
    app.launch(server_name="0.0.0.0", server_port=7861)


__all__ = [
    "AgenticGraphBuilder",
    "ValidationIssue",
    "build_agent_builder_tab",
    "generate_graph_summary",
    "generate_python_code",
    "GraphRuntime",
    "main",
    "render_validation_report",
    "RuntimeExecutionResult",
    "RuntimeOutput",
    "run_graph_runtime",
    "validate_graph",
]
