import os
from pathlib import Path

import orjson

from knowledge_base.loader import load_vault_markdown_dataset
from .documents import Document
from .store import rebuild_knowledge_base

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RAG_PATH = PROJECT_ROOT / "data" / "knowledge_base"
HOME = Path.home()
PERSONALIZATION_FILES = (
    HOME / ".aispiritagent" / "personalization.json",
)
DEFAULT_OBSIDIAN_VAULT_DIRS = (
    HOME / "Obsidian",
    HOME / "Documents" / "Obsidian",
    HOME / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents",
)


def _resolve_vault_path_from_config(
    personalization_file: Path,
    vault_path: str,
) -> Path:
    resolved_vault_path = Path(vault_path).expanduser()
    if not resolved_vault_path.is_absolute():
        resolved_vault_path = (personalization_file.parent / resolved_vault_path).resolve()
    return resolved_vault_path


def _iter_obsidian_vault_roots() -> tuple[Path, ...]:
    custom_roots_raw = os.getenv("OBSIDIAN_VAULT_DIRS", "")
    custom_roots = tuple(
        Path(entry).expanduser()
        for entry in custom_roots_raw.split(os.pathsep)
        if entry.strip()
    )
    if custom_roots:
        return custom_roots
    return DEFAULT_OBSIDIAN_VAULT_DIRS


def _resolve_vault_path_from_name(vault_name: str) -> Path | None:
    for root in _iter_obsidian_vault_roots():
        candidate = root / vault_name
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None



def _get_vault_path() -> Path:
    for personalization_file in PERSONALIZATION_FILES:
        if not personalization_file.exists():
            continue

        data = orjson.loads(personalization_file.read_bytes())
        vault_path = data.get("vault_path")
        if isinstance(vault_path, str) and vault_path.strip():
            resolved_vault_path = _resolve_vault_path_from_config(
                personalization_file,
                vault_path.strip(),
            )
            if resolved_vault_path.exists() and resolved_vault_path.is_dir():
                return resolved_vault_path
            raise FileNotFoundError(f"Vault path {resolved_vault_path} does not exist.")

        vault_name = data.get("vault_name")
        if isinstance(vault_name, str) and vault_name.strip():
            resolved_vault = _resolve_vault_path_from_name(vault_name.strip())
            if resolved_vault is not None:
                return resolved_vault
            searched_roots = ", ".join(str(path) for path in _iter_obsidian_vault_roots())
            raise FileNotFoundError(
                f"Vault named {vault_name.strip()} was not found in: {searched_roots}"
            )

    locations = ", ".join(str(path) for path in PERSONALIZATION_FILES)
    raise FileNotFoundError(f"Could not find a valid personalization file. Checked: {locations}")


def initial_rag(documents: list[Document]):
    return rebuild_knowledge_base(
        path=RAG_PATH,
        documents=documents,
    )


def main():
    documents = load_vault_markdown_dataset()
    initial_rag([
        doc for doc in documents if doc.page_content.strip() != ""
    ])


if __name__ == "__main__":
    main()
