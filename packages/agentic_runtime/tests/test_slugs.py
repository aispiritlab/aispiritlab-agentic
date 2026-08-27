"""Guards on slugs, which become directory names under ``~/.aispiritagent``.

Slugs arrive from the chat client, so every one of these cases is a path the
UI could hand us. The traversal cases below must never produce a ``Path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_runtime.slugs import (
    MAX_SLUG_LENGTH,
    InvalidSlugError,
    is_valid_slug,
    resolve_child,
    slugify,
    validate_slug,
)

# Values a client could send that must never become a path segment.
UNSAFE_SLUGS = [
    "..",
    "../..",
    "../etc",
    "..%2f..",
    "/etc/passwd",
    "/",
    "foo/bar",
    "foo\\bar",
    ".",
    "~",
    "~/secrets",
    "",
    "   ",
    ".hidden",
    "-leading-hyphen",
    "UPPERCASE",
    "MiXeD",
    "with space",
    "with_underscore",
    "kawałek",  # non-ascii
    "null\x00byte",
    "a" * (MAX_SLUG_LENGTH + 1),
]


@pytest.mark.parametrize("slug", UNSAFE_SLUGS)
def test_unsafe_slugs_are_rejected(slug: str) -> None:
    assert is_valid_slug(slug) is False

    with pytest.raises(InvalidSlugError):
        validate_slug(slug)


@pytest.mark.parametrize(
    "slug", ["default", "jan", "jan-kowalski", "a", "u2", "a" * MAX_SLUG_LENGTH]
)
def test_safe_slugs_are_accepted(slug: str) -> None:
    assert is_valid_slug(slug) is True
    assert validate_slug(slug) == slug


def test_validate_slug_names_the_kind_in_the_error() -> None:
    with pytest.raises(InvalidSlugError, match="Invalid workspace slug"):
        validate_slug("../escape", kind="workspace slug")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Default", "default"),
        ("Jan Kowalski", "jan-kowalski"),
        ("  spaced   out  ", "spaced-out"),
        ("Already-A-Slug", "already-a-slug"),
        ("!!!leading and trailing!!!", "leading-and-trailing"),
        ("mixed123ALPHA", "mixed123alpha"),
        ("under_score", "under-score"),
    ],
)
def test_slugify_normalises_display_names(name: str, expected: str) -> None:
    assert slugify(name, fallback="user") == expected


@pytest.mark.parametrize("name", ["", "   ", "!!!", "---", "żółć", "\x00"])
def test_slugify_falls_back_when_nothing_survives(name: str) -> None:
    assert slugify(name, fallback="user") == "user"


def test_slugify_truncates_to_the_maximum_length() -> None:
    slug = slugify("x" * 200, fallback="user")

    assert len(slug) == MAX_SLUG_LENGTH
    assert is_valid_slug(slug)


def test_slugify_never_leaves_a_trailing_hyphen_after_truncation() -> None:
    # Truncating mid-separator would otherwise yield "aaa...-", which the
    # pattern still accepts but which reads as a broken name.
    slug = slugify("a" * (MAX_SLUG_LENGTH - 1) + " tail", fallback="user")

    assert not slug.endswith("-")
    assert is_valid_slug(slug)


@pytest.mark.parametrize("name", ["../../etc/passwd", "..", "/root", "a/b/c", "żażółć gęślą"])
def test_slugify_output_is_always_a_safe_slug(name: str) -> None:
    assert is_valid_slug(slugify(name, fallback="user"))


def test_resolve_child_returns_a_direct_child(tmp_path: Path) -> None:
    assert resolve_child(tmp_path, "default") == tmp_path / "default"


@pytest.mark.parametrize("slug", UNSAFE_SLUGS)
def test_resolve_child_refuses_to_leave_its_base(tmp_path: Path, slug: str) -> None:
    with pytest.raises(InvalidSlugError):
        resolve_child(tmp_path, slug, kind="user slug")


def test_resolve_child_does_not_create_anything(tmp_path: Path) -> None:
    resolve_child(tmp_path, "fresh")

    assert list(tmp_path.iterdir()) == []


def test_resolve_child_accepts_a_symlinked_base(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    # The base itself being a symlink is legitimate (e.g. a relocated home).
    assert resolve_child(link, "default") == link / "default"


def test_resolve_child_refuses_a_child_symlinked_out_of_the_base(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (base / "escape").symlink_to(outside, target_is_directory=True)

    # The name passes the pattern check; only the containment check catches it.
    assert is_valid_slug("escape")
    with pytest.raises(InvalidSlugError, match="resolves outside"):
        resolve_child(base, "escape")


def test_invalid_slug_error_is_a_value_error() -> None:
    # Callers upstream catch ValueError; keep that contract.
    assert issubclass(InvalidSlugError, ValueError)
