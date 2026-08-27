"""User and workspace storage: path containment and destructive-op guards.

Every test redirects the module-level storage constants at ``tmp_path`` first,
so nothing here can touch the real ``~/.aispiritagent``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextvars
import json
from pathlib import Path

import pytest

from agentic_runtime import users, workspaces
from agentic_runtime.slugs import InvalidSlugError

TRAVERSAL_SLUGS = ["..", "../..", "../../etc", "/etc/passwd", "foo/bar", "", "UPPER", "~"]


@pytest.fixture
def users_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "aispiritagent"
    monkeypatch.setattr(users, "_ROOT", root)
    monkeypatch.setattr(users, "_USERS_DIR", root / "users")
    monkeypatch.setattr(users, "_USERS_INDEX", root / "users.json")
    monkeypatch.setattr(users, "_LEGACY_PERSONALIZATION", root / "personalization.json")
    monkeypatch.setattr(users, "_migrated", False)
    return root


@pytest.fixture
def workspaces_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "aispiritagent"
    monkeypatch.setattr(workspaces, "_ROOT", root)
    monkeypatch.setattr(workspaces, "_WORKSPACES_DIR", root / "workspaces")
    return root


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slug", TRAVERSAL_SLUGS)
def test_user_dir_refuses_traversal(users_root: Path, slug: str) -> None:
    with pytest.raises(InvalidSlugError):
        users.user_dir(slug)


@pytest.mark.parametrize("slug", TRAVERSAL_SLUGS)
def test_personalization_path_refuses_traversal(users_root: Path, slug: str) -> None:
    with pytest.raises(InvalidSlugError):
        users.personalization_path(slug)


def test_user_dir_stays_under_the_users_directory(users_root: Path) -> None:
    path = users.user_dir("jan")

    assert path == users_root / "users" / "jan"
    assert path.parent == users_root / "users"


def test_list_users_creates_a_default_user_on_first_call(users_root: Path) -> None:
    result = users.list_users()

    assert [u.slug for u in result] == ["default"]
    assert (users_root / "users" / "default").is_dir()


def test_create_user_slugifies_and_deduplicates(users_root: Path) -> None:
    first = users.create_user("Jan Kowalski")
    second = users.create_user("Jan  kowalski")

    assert first.slug == "jan-kowalski"
    assert second.slug == "jan-kowalski-2"
    assert second.name == "Jan  kowalski"


def test_create_user_never_writes_outside_the_users_directory(users_root: Path) -> None:
    profile = users.create_user("../../../etc/passwd")

    assert profile.slug == "etc-passwd"
    assert users.user_dir(profile.slug).is_dir()
    assert not (users_root.parent / "passwd").exists()


def test_delete_user_removes_the_directory_and_the_index_entry(users_root: Path) -> None:
    users.list_users()
    victim = users.create_user("Temp")
    directory = users.user_dir(victim.slug)
    (directory / "personalization.json").write_text("{}", encoding="utf-8")

    users.delete_user(victim.slug)

    assert not directory.exists()
    assert victim.slug not in {u.slug for u in users.list_users()}


def test_delete_user_refuses_an_unknown_slug(users_root: Path) -> None:
    users.list_users()

    with pytest.raises(ValueError, match="Unknown user"):
        users.delete_user("never-registered")


def test_delete_user_refuses_the_last_remaining_user(users_root: Path) -> None:
    only = users.list_users()[0]

    with pytest.raises(ValueError, match="Cannot delete the last user"):
        users.delete_user(only.slug)

    assert users.user_dir(only.slug).is_dir()


@pytest.mark.parametrize("slug", TRAVERSAL_SLUGS)
def test_delete_user_refuses_traversal_before_touching_the_index(
    users_root: Path, slug: str
) -> None:
    users.list_users()

    with pytest.raises(InvalidSlugError):
        users.delete_user(slug)


def test_read_index_survives_a_corrupt_file(users_root: Path) -> None:
    (users_root).mkdir(parents=True, exist_ok=True)
    (users_root / "users.json").write_text("{ not json", encoding="utf-8")

    # A broken index must not crash the app; it re-seeds the default user.
    assert [u.slug for u in users.list_users()] == ["default"]


def test_read_index_skips_malformed_entries(users_root: Path) -> None:
    users_root.mkdir(parents=True, exist_ok=True)
    (users_root / "users.json").write_text(
        json.dumps([{"name": "Ok", "slug": "ok"}, {"name": "no slug"}, "junk"]),
        encoding="utf-8",
    )

    assert [u.slug for u in users.list_users()] == ["ok"]


def test_get_user_returns_none_for_unknown_slug(users_root: Path) -> None:
    users.list_users()

    assert users.get_user("default") is not None
    assert users.get_user("nope") is None


def test_legacy_personalization_is_migrated_into_the_default_user(users_root: Path) -> None:
    users_root.mkdir(parents=True, exist_ok=True)
    (users_root / "personalization.json").write_text(
        json.dumps({"name": "Stary Profil"}), encoding="utf-8"
    )

    profiles = users.list_users()

    assert [p.slug for p in profiles] == ["default"]
    assert profiles[0].name == "Stary Profil"
    migrated = users.personalization_path("default")
    assert json.loads(migrated.read_text(encoding="utf-8")) == {"name": "Stary Profil"}


# ---------------------------------------------------------------------------
# Workspaces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slug", TRAVERSAL_SLUGS)
def test_workspace_paths_refuse_traversal(workspaces_root: Path, slug: str) -> None:
    for accessor in (workspaces.workspace_dir, workspaces.graph_path, workspaces.config_path):
        with pytest.raises(InvalidSlugError):
            accessor(slug)


def test_list_workspaces_creates_a_default_workspace(workspaces_root: Path) -> None:
    result = workspaces.list_workspaces()

    assert [w.slug for w in result] == ["default"]
    assert workspaces.graph_path("default").read_text(encoding="utf-8") == "{}"


def test_create_workspace_deduplicates_slugs(workspaces_root: Path) -> None:
    first = workspaces.create_workspace("Badania", "{}", description="  opis  ")
    second = workspaces.create_workspace("badania", "{}")

    assert first.slug == "badania"
    assert first.description == "opis"
    assert second.slug == "badania-2"


def test_workspace_graph_round_trips(workspaces_root: Path) -> None:
    preset = workspaces.create_workspace("Grafy", '{"nodes": []}')

    assert workspaces.load_workspace_graph(preset.slug) == '{"nodes": []}'

    workspaces.update_workspace_graph(preset.slug, '{"nodes": [1]}')
    assert workspaces.load_workspace_graph(preset.slug) == '{"nodes": [1]}'


def test_load_workspace_graph_returns_empty_for_a_missing_workspace(
    workspaces_root: Path,
) -> None:
    assert workspaces.load_workspace_graph("nieistniejacy") == ""


def test_delete_workspace_refuses_the_default(workspaces_root: Path) -> None:
    workspaces.list_workspaces()

    with pytest.raises(ValueError, match="Cannot delete the default workspace"):
        workspaces.delete_workspace("default")

    assert workspaces.workspace_dir("default").is_dir()


def test_delete_workspace_refuses_an_unknown_slug(workspaces_root: Path) -> None:
    with pytest.raises(ValueError, match="Unknown workspace"):
        workspaces.delete_workspace("never-created")


@pytest.mark.parametrize("slug", TRAVERSAL_SLUGS)
def test_delete_workspace_refuses_traversal(workspaces_root: Path, slug: str) -> None:
    with pytest.raises(InvalidSlugError):
        workspaces.delete_workspace(slug)


def test_delete_workspace_removes_the_directory(workspaces_root: Path) -> None:
    preset = workspaces.create_workspace("Tymczasowy", "{}")
    directory = workspaces.workspace_dir(preset.slug)

    workspaces.delete_workspace(preset.slug)

    assert not directory.exists()


def test_read_config_survives_a_corrupt_config(workspaces_root: Path) -> None:
    workspaces.create_workspace("Zepsuty", "{}")
    workspaces.config_path("zepsuty").write_text("{ not json", encoding="utf-8")

    assert workspaces.get_workspace("zepsuty") is None


# ---------------------------------------------------------------------------
# Session identifiers and the active-workspace context
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("user_slug", "workspace_slug"),
    [("default", "default"), ("jan-kowalski", "badania"), ("u2", "ws-2")],
)
def test_session_id_round_trips(user_slug: str, workspace_slug: str) -> None:
    session_id = workspaces.build_session_id(user_slug, workspace_slug)

    assert session_id == f"user:{user_slug}:ws:{workspace_slug}"
    assert workspaces.parse_session_id(session_id) == (user_slug, workspace_slug)


def test_build_session_id_defaults_to_the_active_workspace() -> None:
    token = workspaces.set_active_workspace("badania")
    try:
        assert workspaces.build_session_id("jan") == "user:jan:ws:badania"
    finally:
        workspaces._active_workspace.reset(token)


def test_parse_session_id_tolerates_a_legacy_id_without_a_workspace() -> None:
    assert workspaces.parse_session_id("user:jan") == ("jan", "")
    assert workspaces.parse_session_id("freeform") == ("", "")


def test_active_workspace_is_isolated_per_context() -> None:
    # Gradio serves requests concurrently; a global would leak one user's
    # workspace into another's session.
    def read_in_own_context(slug: str) -> str:
        context = contextvars.copy_context()

        def _set_and_read() -> str:
            workspaces.set_active_workspace(slug)
            return workspaces.get_active_workspace()

        return context.run(_set_and_read)

    with ThreadPoolExecutor(max_workers=4) as pool:
        observed = list(pool.map(read_in_own_context, ["a", "b", "c", "d"]))

    assert observed == ["a", "b", "c", "d"]
    assert workspaces.get_active_workspace() == "default"
