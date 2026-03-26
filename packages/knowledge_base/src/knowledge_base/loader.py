from pathlib import Path

from .documents import Document
from knowledge_base.splitters.splitter import split_markdown


def _resolve_vault_path() -> Path:
    from knowledge_base.main import _get_vault_path as resolve_vault_path

    return resolve_vault_path()


def load_vault_markdown_dataset() -> list[Document]:
    vault_path = _resolve_vault_path()
    documents: list[Document] = []

    for path in vault_path.rglob("*.md"):
        if not path.is_file():
            continue
        last_modified = path.stat().st_mtime
        content = path.read_text(encoding="utf-8", errors="ignore")
        for index, chunk in enumerate(split_markdown(content.encode("utf-8")), start=1):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": str(path),
                        "header": f"{path.relative_to(vault_path)} chunk={index}",
                        "last_modified": last_modified,
                    },
                )
            )

    return documents

def load_vault_markdown_dataset_after_modified(after_modified: float | None = None) -> list[Document]:
    vault_path = _resolve_vault_path()
    documents: list[Document] = []

    for path in vault_path.rglob("*.md"):
        if not path.is_file():
            continue
        last_modified = path.stat().st_mtime
        if after_modified is not None and last_modified <= after_modified:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        for index, chunk in enumerate(split_markdown(content.encode("utf-8")), start=1):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": str(path),
                        "header": f"{path.relative_to(vault_path)} chunk={index}",
                        "last_modified": last_modified,
                    },
                )
            )

    return documents

def load_vault_markdown_dataset_by_path(path: Path) -> list[Document]:
    last_modified = path.stat().st_mtime
    content = path.read_text(encoding="utf-8", errors="ignore")
    documents = []
    for index, chunk in enumerate(split_markdown(content.encode("utf-8")), start=1):
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "source": str(path),
                    "header": f"{path.name} chunk={index}",
                    "last_modified": last_modified,
                },
            )
        )
    return documents
