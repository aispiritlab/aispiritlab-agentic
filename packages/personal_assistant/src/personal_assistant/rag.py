from pathlib import Path
import re
from typing import Any
from knowledge_base import get_knowledge_base
from knowledge_base.documents import Document


_CHUNK_SUFFIX_PATTERN = re.compile(r"\s+chunk=\d+\s*$", re.IGNORECASE)


def _extract_note_name(metadata: dict[str, Any], index: int) -> str:
    source = metadata.get("source")
    if isinstance(source, str) and source.strip():
        return Path(source).stem

    header = metadata.get("header")
    if isinstance(header, str) and header.strip():
        cleaned_header = _CHUNK_SUFFIX_PATTERN.sub("", header.strip())
        stem = Path(cleaned_header).stem
        if stem:
            return stem
    return f"Document page={index}"


def build_context(documents: list[Document]) -> str:
    prepared_context = []
    for i, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        note_name = _extract_note_name(metadata, i)
        prepared_context.append(f"**{note_name}.** {doc.page_content}")
    return "\n".join(prepared_context)


def find_similar_documents(message: str) -> str:
    docs = get_knowledge_base().similarity_search(message, k=5)
    return build_context(docs)


def search(query: str) -> str:
    """Search the RAG knowledge base for context relevant to a query."""
    return find_similar_documents(query)
