"""Where the knowledge base looks for its data directory.

Resolution order matters: an installed (non-editable) package inside Docker has
no repository layout to walk, so ``KNOWLEDGE_BASE_PATH`` has to win, and the
``~/.aispiritagent`` fallback has to exist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from knowledge_base import paths


def test_the_environment_variable_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(paths._ENV_VAR, str(tmp_path / "kb"))

    assert paths.resolve_rag_path() == tmp_path / "kb"


def test_the_environment_variable_is_tilde_expanded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(paths._ENV_VAR, "~/custom-kb")

    assert paths.resolve_rag_path() == Path.home() / "custom-kb"


def test_an_empty_environment_variable_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    # An unset variable and one set to "" must behave the same; compose files
    # routinely produce the latter.
    monkeypatch.setenv(paths._ENV_VAR, "")

    assert paths.resolve_rag_path() != Path("")


def test_the_repository_layout_is_used_when_running_from_a_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(paths._ENV_VAR, raising=False)

    resolved = paths.resolve_rag_path()
    root = paths._repository_root()

    assert root is not None, "the test suite runs from a checkout"
    assert resolved == root / "data" / "knowledge_base"


def test_the_repository_root_holds_both_markers() -> None:
    root = paths._repository_root()

    assert root is not None
    assert (root / "pyproject.toml").exists()
    assert (root / "packages").is_dir()


def test_an_installed_package_falls_back_to_the_home_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The Docker case: no pyproject.toml anywhere above site-packages.
    monkeypatch.delenv(paths._ENV_VAR, raising=False)
    monkeypatch.setattr(paths, "_repository_root", lambda: None)

    assert paths.resolve_rag_path() == Path.home() / ".aispiritagent" / "knowledge_base"


def test_the_environment_variable_still_wins_outside_a_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(paths, "_repository_root", lambda: None)
    monkeypatch.setenv(paths._ENV_VAR, str(tmp_path))

    assert paths.resolve_rag_path() == tmp_path


def test_the_module_constant_matches_the_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(paths._ENV_VAR, raising=False)

    assert paths.RAG_PATH == paths.resolve_rag_path()


def test_resolution_does_not_create_the_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "not-yet"
    monkeypatch.setenv(paths._ENV_VAR, str(target))

    paths.resolve_rag_path()

    assert not target.exists()
