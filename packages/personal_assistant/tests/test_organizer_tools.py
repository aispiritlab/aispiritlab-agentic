from pathlib import Path
import subprocess

import orjson

from personal_assistant.agents.manage_notes import tools as note_tools
from personal_assistant.messaging.events import CreatedNote
from personal_assistant.agents.organizer import tools as organizer_tools


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


def test_tag_note_adds_frontmatter_tag_to_filesystem_note(monkeypatch, tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "Projekt.md"
    note.write_text("Plan sprintu", encoding="utf-8")
    _set_personalization_file(tmp_path, monkeypatch, vault_path=vault)

    result = organizer_tools.tag_note("Projekt", "para/project")

    assert result == "Notatka Projekt otagowana jako para/project."
    assert note.read_text(encoding="utf-8") == (
        "---\n"
        "tags:\n"
        "- para/project\n"
        "---\n"
        "Plan sprintu"
    )


def test_tag_note_merges_existing_tags_without_duplicates(monkeypatch, tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "Zdrowie.md"
    note.write_text(
        "---\n"
        "status: active\n"
        "tags:\n"
        "- para/area\n"
        "---\n"
        "Badania kontrolne",
        encoding="utf-8",
    )
    _set_personalization_file(tmp_path, monkeypatch, vault_path=vault)

    result = organizer_tools.tag_note("Zdrowie", "para/area")

    assert result == "Notatka Zdrowie ma już tag para/area."
    assert note.read_text(encoding="utf-8").startswith("---\nstatus: active\n")


def test_tag_note_uses_obsidian_cli_tags_and_overwrite(monkeypatch, tmp_path: Path) -> None:
    _set_personalization_file(tmp_path, monkeypatch, vault_name="MyVault")
    calls: list[list[str]] = []

    def fake_run(args, check, capture_output, text):  # noqa: ANN001
        assert check is False
        assert capture_output is True
        assert text is True
        calls.append(list(args))
        command = args[2]
        if command == "read":
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout="---\ntags:\n- para/resource\n---\nTreść",
                stderr="",
            )
        if command == "tags":
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout='["para/resource"]',
                stderr="",
            )
        if command == "create":
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout="",
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(note_tools.subprocess, "run", fake_run)

    result = organizer_tools.tag_note("Research", "para/archive")

    assert result == "Notatka Research otagowana jako para/archive."
    assert calls[0] == ["obsidian", "vault=MyVault", "read", "file=Research"]
    assert calls[1] == ["obsidian", "vault=MyVault", "tags", "file=Research", "format=json"]
    assert calls[2][0:4] == ["obsidian", "vault=MyVault", "create", "name=Research"]
    assert calls[2][-1] == "overwrite"


def test_organizer_decider_formats_created_note_as_user_message() -> None:
    from personal_assistant.deciders import make_organizer_decider
    from agentic_runtime.messaging.messages import UserMessage

    decider = make_organizer_decider()
    result = decider(
        CreatedNote(
            note_name="Projekt",
            note_content="Plan sprintu",
        )
    )

    assert len(result) == 1
    assert isinstance(result[0], UserMessage)
    assert "Projekt" in result[0].text
    assert "Plan sprintu" in result[0].text
