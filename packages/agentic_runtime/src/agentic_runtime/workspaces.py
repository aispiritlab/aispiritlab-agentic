"""Workspace management — agent graph presets with data tagging.

Each workspace is a directory under ``~/.aispiritagent/workspaces/`` containing
a ``graph.json`` (agent graph preset) and ``config.json`` (metadata).

All messages and traces carry the active workspace slug so fine-tuning
datasets and MLflow experiments can be filtered per workspace.
"""

from __future__ import annotations

import contextvars
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path.home() / ".aispiritagent"
_WORKSPACES_DIR = _ROOT / "workspaces"

# ---------------------------------------------------------------------------
# Per-request workspace context (thread-safe via contextvars)
# ---------------------------------------------------------------------------

_active_workspace: contextvars.ContextVar[str] = contextvars.ContextVar(
    "active_workspace", default="default"
)


def get_active_workspace() -> str:
    return _active_workspace.get()


def set_active_workspace(slug: str) -> contextvars.Token[str]:
    return _active_workspace.set(slug)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WorkspacePreset:
    slug: str
    name: str
    description: str
    created_at: str  # ISO 8601


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "workspace"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Workspace CRUD
# ---------------------------------------------------------------------------


def workspace_dir(slug: str) -> Path:
    return _WORKSPACES_DIR / slug


def graph_path(slug: str) -> Path:
    return workspace_dir(slug) / "graph.json"


def config_path(slug: str) -> Path:
    return workspace_dir(slug) / "config.json"


def _read_config(slug: str) -> WorkspacePreset | None:
    path = config_path(slug)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    return WorkspacePreset(
        slug=slug,
        name=data.get("name", slug),
        description=data.get("description", ""),
        created_at=data.get("created_at", ""),
    )


def _write_config(preset: WorkspacePreset) -> None:
    path = config_path(preset.slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(preset), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def list_workspaces() -> list[WorkspacePreset]:
    """Return all workspaces. Creates 'default' if none exist."""
    if not _WORKSPACES_DIR.exists():
        _ensure_default_workspace()

    presets: list[WorkspacePreset] = []
    for entry in sorted(_WORKSPACES_DIR.iterdir()):
        if not entry.is_dir():
            continue
        preset = _read_config(entry.name)
        if preset is not None:
            presets.append(preset)

    if not presets:
        _ensure_default_workspace()
        default = _read_config("default")
        if default is not None:
            presets.append(default)

    return presets


def create_workspace(
    name: str,
    graph_json: str,
    *,
    description: str = "",
) -> WorkspacePreset:
    """Create a new workspace with a graph preset."""
    slug = _slugify(name)
    existing_slugs = {p.slug for p in list_workspaces()}

    if slug in existing_slugs:
        base = slug
        counter = 2
        while slug in existing_slugs:
            slug = f"{base}-{counter}"
            counter += 1

    preset = WorkspacePreset(
        slug=slug,
        name=name.strip(),
        description=description.strip(),
        created_at=_now_iso(),
    )

    directory = workspace_dir(slug)
    directory.mkdir(parents=True, exist_ok=True)

    graph_path(slug).write_text(graph_json, encoding="utf-8")
    _write_config(preset)
    return preset


def load_workspace_graph(slug: str) -> str:
    """Load the graph JSON for a workspace. Returns empty string if not found."""
    path = graph_path(slug)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def update_workspace_graph(slug: str, graph_json: str) -> None:
    """Update the graph JSON for an existing workspace."""
    path = graph_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(graph_json, encoding="utf-8")


def delete_workspace(slug: str) -> None:
    """Delete a workspace. Cannot delete 'default'."""
    if slug == "default":
        raise ValueError("Cannot delete the default workspace.")
    import shutil

    directory = workspace_dir(slug)
    if directory.exists():
        shutil.rmtree(directory)


def get_workspace(slug: str) -> WorkspacePreset | None:
    return _read_config(slug)


# ---------------------------------------------------------------------------
# Session ID helpers
# ---------------------------------------------------------------------------


def build_session_id(user_slug: str, workspace_slug: str | None = None) -> str:
    """Build a session_id that encodes both user and workspace.

    Format: ``user:<user_slug>:ws:<workspace_slug>``
    This allows filtering messages and traces by user, workspace, or both.
    """
    ws = workspace_slug or _active_workspace.get()
    return f"user:{user_slug}:ws:{ws}"


def parse_session_id(session_id: str) -> tuple[str, str]:
    """Extract (user_slug, workspace_slug) from a session_id."""
    user = ""
    workspace = ""
    if ":ws:" in session_id:
        before_ws, workspace = session_id.rsplit(":ws:", 1)
    else:
        before_ws = session_id
    if before_ws.startswith("user:"):
        user = before_ws[5:]
    return user, workspace


# ---------------------------------------------------------------------------
# Default workspace
# ---------------------------------------------------------------------------


def _ensure_default_workspace() -> None:
    """Create the default workspace if it doesn't exist."""
    directory = workspace_dir("default")
    if directory.exists() and config_path("default").exists():
        return

    directory.mkdir(parents=True, exist_ok=True)

    preset = WorkspacePreset(
        slug="default",
        name="Default",
        description="Default workspace with Personal Assistant agents",
        created_at=_now_iso(),
    )
    _write_config(preset)

    # Create a minimal graph.json placeholder
    if not graph_path("default").exists():
        graph_path("default").write_text("{}", encoding="utf-8")
