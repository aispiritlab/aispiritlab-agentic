from pathlib import Path
import subprocess

import orjson

from personal_assistant.agents.discovery_notes import tools as discovery_tools
from personal_assistant.agents.manage_notes import tools as note_tools


def _set_personalization_file(
    tmp_path: Path,
    monkeypatch,
    *,
    vault_path: Path | None = None,
    vault_name: str | None = None,
) -> None:
    personalization_path = tmp_path / "personalization.json"
    payload: dict[str, str] = {}
    if vault_path is not None:
        payload["vault_path"] = str(vault_path)
    if vault_name is not None:
        payload["vault_name"] = vault_name
    personalization_path.write_bytes(orjson.dumps(payload))
    monkeypatch.setattr(note_tools, "PERSONALIZATION_FILE", personalization_path)


def test_list_notes_lists_markdown_notes_sorted(monkeypatch, tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "b.md").write_text("B", encoding="utf-8")
    (vault / "a.md").write_text("A", encoding="utf-8")
    (vault / "ignore.txt").write_text("X", encoding="utf-8")
    _set_personalization_file(tmp_path, monkeypatch, vault_path=vault)

    result = note_tools.list_notes()

    assert result == "Notatki:\n- a\n- b"


def test_add_note_creates_new_note(monkeypatch, tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    _set_personalization_file(tmp_path, monkeypatch, vault_path=vault)

    result = note_tools.add_note("Zakupy", "Mleko")

    assert result == "Notatka Zakupy dodana."
    assert (vault / "Zakupy.md").read_text(encoding="utf-8") == "Mleko"


def test_add_note_appends_to_existing_note_with_newline_separator(monkeypatch, tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Zakupy.md").write_text("Mleko", encoding="utf-8")
    _set_personalization_file(tmp_path, monkeypatch, vault_path=vault)

    result = note_tools.add_note("Zakupy", "Jajka")

    assert result == "Notatka Zakupy zaktualizowana."
    assert (vault / "Zakupy.md").read_text(encoding="utf-8") == "Mleko\nJajka"


def test_edit_note_appends_to_existing_note(monkeypatch, tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Projekt.md").write_text("Plan", encoding="utf-8")
    _set_personalization_file(tmp_path, monkeypatch, vault_path=vault)

    result = note_tools.edit_note("Projekt", "Termin jutro")

    assert result == "Notatka Projekt zaktualizowana."
    assert (vault / "Projekt.md").read_text(encoding="utf-8") == "Plan\nTermin jutro"


def test_edit_note_creates_note_when_missing(monkeypatch, tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    _set_personalization_file(tmp_path, monkeypatch, vault_path=vault)

    result = note_tools.edit_note("Nowa", "Start")

    assert result == "Notatka Nowa dodana."
    assert (vault / "Nowa.md").read_text(encoding="utf-8") == "Start"


def test_list_notes_returns_empty_message_when_no_notes(monkeypatch, tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    _set_personalization_file(tmp_path, monkeypatch, vault_path=vault)

    result = note_tools.list_notes()

    assert result == "Brak notatek."


def test_get_note_accepts_rag_chunk_identifier(monkeypatch, tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "superasna.md").write_text("ala ma kota", encoding="utf-8")
    _set_personalization_file(tmp_path, monkeypatch, vault_path=vault)

    result = note_tools.get_note("superasna.md chunk=1")

    assert result == "Notatka superasna:\nala ma kota"


def test_manage_note_toolset_includes_crud_tools() -> None:
    assert note_tools.toolset.has_tool("list_notes")
    assert note_tools.toolset.has_tool("edit_note")
    assert note_tools.toolset.has_tool("add_note")
    assert note_tools.toolset.has_tool("get_note")


def test_discovery_toolset_includes_search() -> None:
    assert discovery_tools.toolset.has_tool("search")


def test_search_uses_rag_search(monkeypatch) -> None:
    import agentic_runtime.rag as rag

    monkeypatch.setattr(rag, "search", lambda query: f"wynik:{query}")

    result = discovery_tools.search("python")

    assert result == "wynik:python"


def test_search_rejects_empty_query() -> None:
    assert discovery_tools.search("   ") == "Brak treści zapytania."


def test_add_note_uses_obsidian_cli_with_vault_name(monkeypatch, tmp_path: Path) -> None:
    _set_personalization_file(tmp_path, monkeypatch, vault_name="MyVault")
    calls: list[list[str]] = []

    def fake_run(args, check, capture_output, text):  # noqa: ANN001
        assert check is False
        assert capture_output is True
        assert text is True
        calls.append(list(args))
        command = args[2]
        if command == "read":
            return subprocess.CompletedProcess(args=args, returncode=1, stdout="", stderr="not found")
        if command == "create":
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(note_tools.subprocess, "run", fake_run)

    result = note_tools.add_note("Zakupy", "Mleko")

    assert result == "Notatka Zakupy dodana."
    assert calls == [
        ["obsidian", "vault=MyVault", "read", "file=Zakupy"],
        ["obsidian", "vault=MyVault", "create", "name=Zakupy", "content=Mleko"],
    ]


def test_list_notes_uses_obsidian_cli_with_vault_name(monkeypatch, tmp_path: Path) -> None:
    _set_personalization_file(tmp_path, monkeypatch, vault_name="MyVault")
    calls: list[list[str]] = []

    def fake_run(args, check, capture_output, text):  # noqa: ANN001
        assert check is False
        assert capture_output is True
        assert text is True
        calls.append(list(args))
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="b.md\na.md\nfolder/c.md\n",
            stderr="",
        )

    monkeypatch.setattr(note_tools.subprocess, "run", fake_run)

    result = note_tools.list_notes()

    assert result == "Notatki:\n- a\n- b\n- c"
    assert calls == [
        ["obsidian", "vault=MyVault", "files", "ext=md"],
    ]
