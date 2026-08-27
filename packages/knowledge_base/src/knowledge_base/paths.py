"""Filesystem locations for the knowledge base.

The data directory is resolved from ``KNOWLEDGE_BASE_PATH`` when set, so an
installed (non-editable) package — the Docker case — does not have to guess the
repository layout from ``__file__``.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["RAG_PATH", "resolve_rag_path"]

_ENV_VAR = "KNOWLEDGE_BASE_PATH"
_DEFAULT_RELATIVE = Path("data") / "knowledge_base"


def _repository_root() -> Path | None:
    """Walk up from this file looking for the workspace root, if we are in one."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "packages").is_dir():
            return parent
    return None


def resolve_rag_path() -> Path:
    configured = os.environ.get(_ENV_VAR)
    if configured:
        return Path(configured).expanduser()

    root = _repository_root()
    if root is not None:
        return root / _DEFAULT_RELATIVE

    return Path.home() / ".aispiritagent" / "knowledge_base"


RAG_PATH = resolve_rag_path()
