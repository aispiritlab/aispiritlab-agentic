"""User management for multi-user sessions.

Users are stored under ``~/.aispiritagent/users/`` with one directory per user.
A ``users.json`` index keeps the user list. Legacy ``personalization.json``
at the root level is migrated to ``users/default/`` on first access.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil

from agentic_runtime.slugs import resolve_child, slugify, validate_slug

_ROOT = Path.home() / ".aispiritagent"
_USERS_INDEX = _ROOT / "users.json"
_USERS_DIR = _ROOT / "users"
_LEGACY_PERSONALIZATION = _ROOT / "personalization.json"


@dataclass(frozen=True, slots=True)
class UserProfile:
    name: str
    slug: str
    created_at: str  # ISO 8601


def _slugify(name: str) -> str:
    return slugify(name, fallback="user")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _read_index() -> list[UserProfile]:
    if not _USERS_INDEX.exists():
        return []
    try:
        raw = json.loads(_USERS_INDEX.read_text(encoding="utf-8"))
    except json.JSONDecodeError, OSError:
        return []
    if not isinstance(raw, list):
        return []
    return [
        UserProfile(name=entry["name"], slug=entry["slug"], created_at=entry.get("created_at", ""))
        for entry in raw
        if isinstance(entry, dict) and "name" in entry and "slug" in entry
    ]


def _write_index(users: list[UserProfile]) -> None:
    _USERS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    _USERS_INDEX.write_text(
        json.dumps([asdict(u) for u in users], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def user_dir(slug: str) -> Path:
    """Return the directory for a given user slug.

    Raises ``InvalidSlugError`` if the slug is not a safe path segment.
    """
    return resolve_child(_USERS_DIR, slug, kind="user slug")


def personalization_path(slug: str) -> Path:
    """Return the personalization.json path for a user."""
    return user_dir(slug) / "personalization.json"


def list_users() -> list[UserProfile]:
    """Return all registered users. Runs migration if needed."""
    _ensure_migrated()
    users = _read_index()
    if not users:
        default = _create_user_internal("Default", "default")
        return [default]
    return users


def create_user(name: str) -> UserProfile:
    """Create a new user. Returns the profile."""
    _ensure_migrated()
    slug = _slugify(name)
    users = _read_index()
    existing_slugs = {u.slug for u in users}

    if slug in existing_slugs:
        base = slug
        counter = 2
        while slug in existing_slugs:
            slug = f"{base}-{counter}"
            counter += 1

    return _create_user_internal(name, slug)


def _create_user_internal(name: str, slug: str) -> UserProfile:
    users = _read_index()
    if any(u.slug == slug for u in users):
        return next(u for u in users if u.slug == slug)

    profile = UserProfile(name=name, slug=slug, created_at=_now_iso())
    directory = user_dir(slug)
    directory.mkdir(parents=True, exist_ok=True)

    users.append(profile)
    _write_index(users)
    return profile


def delete_user(slug: str) -> None:
    """Delete a registered user and their directory.

    Refuses unknown slugs and the last remaining user, so the recursive delete
    below can only ever target a directory this module created.
    """
    validate_slug(slug, kind="user slug")
    users = _read_index()
    if not any(u.slug == slug for u in users):
        raise ValueError(f"Unknown user {slug!r}.")

    remaining = [u for u in users if u.slug != slug]
    if not remaining:
        raise ValueError("Cannot delete the last user.")

    directory = user_dir(slug)
    if directory.is_dir():
        shutil.rmtree(directory)

    _write_index(remaining)


def get_user(slug: str) -> UserProfile | None:
    """Return a user by slug, or None."""
    for u in _read_index():
        if u.slug == slug:
            return u
    return None


def default_user_slug() -> str:
    """Return the slug of the first user (after migration)."""
    users = list_users()
    return users[0].slug


# ---------------------------------------------------------------------------
# Legacy migration
# ---------------------------------------------------------------------------

_migrated = False


def _ensure_migrated() -> None:
    global _migrated
    if _migrated:
        return
    _migrated = True

    if not _LEGACY_PERSONALIZATION.exists():
        return

    # Already has users — skip migration
    if _USERS_INDEX.exists() and _read_index():
        return

    # Migrate legacy personalization to users/default/
    default_dir = user_dir("default")
    default_dir.mkdir(parents=True, exist_ok=True)

    target = default_dir / "personalization.json"
    if not target.exists():
        shutil.copy2(_LEGACY_PERSONALIZATION, target)

    # Read legacy name for the user profile
    try:
        data = json.loads(_LEGACY_PERSONALIZATION.read_text(encoding="utf-8"))
        name = data.get("name", "Default") if isinstance(data, dict) else "Default"
    except json.JSONDecodeError, OSError:
        name = "Default"

    profile = UserProfile(name=str(name) or "Default", slug="default", created_at=_now_iso())
    _write_index([profile])
