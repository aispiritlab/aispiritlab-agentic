from __future__ import annotations

from agentic.tools import Toolset


def search(query: str) -> str:
    """Search for relevant context in the RAG knowledge base.

    Args:
        query: The search query.
    """
    from personal_assistant.rag import search as rag_search

    if not query.strip():
        return "Brak treści zapytania."
    return rag_search(query)


toolset = Toolset([search])
