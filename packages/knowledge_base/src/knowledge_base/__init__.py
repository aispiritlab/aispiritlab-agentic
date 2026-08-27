"""Knowledge base — vault loading and local vector search."""

from __future__ import annotations

from pathlib import Path

from .documents import Document
from .loader import (
    load_vault_markdown_dataset,
    load_vault_markdown_dataset_after_modified,
    load_vault_markdown_dataset_by_path,
)
from .main import main
from .paths import RAG_PATH
from .store import (
    QdrantKnowledgeBase,
    close_knowledge_base,
    create_knowledge_base,
    open_knowledge_base,
    rebuild_knowledge_base,
    resync_knowledge_base,
    update_knowledge_base,
)

__all__ = [
    "RAG_PATH",
    "Document",
    "QdrantKnowledgeBase",
    "close_knowledge_base",
    "create_knowledge_base",
    "delete_note_from_kb",
    "get_knowledge_base",
    "load_vault_markdown_dataset",
    "load_vault_markdown_dataset_after_modified",
    "load_vault_markdown_dataset_by_path",
    "main",
    "open_knowledge_base",
    "rebuild_knowledge_base",
    "resync_knowledge_base",
    "resync_notes",
    "update_knowledge_base",
    "update_note_in_kb",
    "update_notes",
]


def get_knowledge_base() -> QdrantKnowledgeBase:
    """Get the RAG knowledge base for querying."""
    return open_knowledge_base(path=RAG_PATH)


def update_notes(path: Path) -> QdrantKnowledgeBase:
    return update_knowledge_base(path=path, documents=load_vault_markdown_dataset_by_path(path))


def resync_notes() -> QdrantKnowledgeBase:
    return resync_knowledge_base(path=RAG_PATH)


def update_note_in_kb(note_path: str) -> None:
    path = Path(note_path)
    if not path.exists():
        return
    documents = load_vault_markdown_dataset_by_path(path)
    kb = get_knowledge_base()
    kb.update_by_source(source=note_path, documents=documents)


def delete_note_from_kb(note_path: str) -> None:
    kb = get_knowledge_base()
    kb.delete_by_source(source=note_path)
