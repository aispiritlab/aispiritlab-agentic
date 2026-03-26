from agentic import git_tracer


def _configure_commit_identity(repo) -> None:
    with repo.config_writer() as config:
        config.set_value("user", "name", "Test User")
        config.set_value("user", "email", "test@example.com")


def _configure_commit_identity_env(monkeypatch) -> None:
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Test User")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.com")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Test User")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "test@example.com")


def test_initial_tracking_project_commits_existing_files(tmp_path, monkeypatch) -> None:
    _configure_commit_identity_env(monkeypatch)
    (tmp_path / "personalization.json").write_text('{"name":"Ala"}', encoding="utf-8")

    repo = git_tracer.initial_tracking_project(tmp_path)

    assert repo.head.commit.message.strip() == "Initial commit"
    assert repo.git.ls_files("--", "personalization.json").strip() == "personalization.json"


def test_commit_decorator_stages_untracked_files(tmp_path, monkeypatch) -> None:
    _configure_commit_identity_env(monkeypatch)
    repo = git_tracer.initial_tracking_project(tmp_path)
    _configure_commit_identity(repo)

    @git_tracer.commit(lambda: tmp_path)
    def add_note() -> str:
        (tmp_path / "Ania ma kota.md").write_text("Mirek", encoding="utf-8")
        return "ok"

    assert add_note() == "ok"
    assert repo.head.commit.message.strip() == "Commit from add_note"
    assert repo.git.ls_files("--", "Ania ma kota.md").strip() == "Ania ma kota.md"


def test_commit_decorator_skips_empty_commit(tmp_path, monkeypatch) -> None:
    _configure_commit_identity_env(monkeypatch)
    repo = git_tracer.initial_tracking_project(tmp_path)
    _configure_commit_identity(repo)

    @git_tracer.commit(lambda: tmp_path)
    def write_same_content() -> str:
        (tmp_path / "note.md").write_text("same content", encoding="utf-8")
        return "ok"

    assert write_same_content() == "ok"
    first_commit = repo.head.commit.hexsha

    assert write_same_content() == "ok"
    assert repo.head.commit.hexsha == first_commit
