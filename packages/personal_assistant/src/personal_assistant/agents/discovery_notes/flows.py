from __future__ import annotations

from evaluation import Flow, Flows


DEFAULT_DISCOVERY_NOTE_FLOWS = Flows(
    Flow(
        name="flow::discovery_rag_embeddings_obsidian",
        steps=(
            "search_rag_foundations",
            "search_embeddings",
            "search_obsidian_linking",
        ),
    ),
    Flow(
        name="flow::discovery_agentic_then_python_rag",
        steps=(
            "search_agentic_tools",
            "search_python_rag",
        ),
    ),
)


__all__ = ["DEFAULT_DISCOVERY_NOTE_FLOWS", "Flow", "Flows"]
