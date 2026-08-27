"""Slug generation and validation for filesystem-backed identities.

User and workspace slugs become directory names under ``~/.aispiritagent``.
They are therefore untrusted path segments: every helper that turns a slug into
a ``Path`` must go through :func:`resolve_child` so a crafted value such as
``../../..`` cannot escape its base directory.
"""

from __future__ import annotations

from pathlib import Path
import re

__all__ = [
    "InvalidSlugError",
    "is_valid_slug",
    "resolve_child",
    "slugify",
    "validate_slug",
]

SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
MAX_SLUG_LENGTH = 64


class InvalidSlugError(ValueError):
    """Raised when a slug is not a safe single path segment."""


def slugify(name: str, *, fallback: str) -> str:
    """Normalize a display name into a safe slug.

    The result always satisfies :func:`is_valid_slug`.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    slug = slug[:MAX_SLUG_LENGTH].strip("-")
    return slug or fallback


def is_valid_slug(slug: str) -> bool:
    return bool(slug) and len(slug) <= MAX_SLUG_LENGTH and SLUG_PATTERN.fullmatch(slug) is not None


def validate_slug(slug: str, *, kind: str = "slug") -> str:
    """Return ``slug`` unchanged, or raise if it is not a safe path segment."""
    if not is_valid_slug(slug):
        raise InvalidSlugError(
            f"Invalid {kind} {slug!r}: expected lowercase letters, digits and hyphens "
            f"(max {MAX_SLUG_LENGTH} characters)."
        )
    return slug


def resolve_child(base: Path, slug: str, *, kind: str = "slug") -> Path:
    """Return ``base / slug`` after proving the result stays inside ``base``.

    The pattern check alone already rejects ``..`` and separators; the
    containment check is a second, independent guard that also covers symlinked
    bases and case-insensitive filesystems.
    """
    validate_slug(slug, kind=kind)
    candidate = (base / slug).resolve()
    root = base.resolve()
    if candidate != root and not candidate.is_relative_to(root):
        raise InvalidSlugError(f"Invalid {kind} {slug!r}: resolves outside {root}.")
    return base / slug
