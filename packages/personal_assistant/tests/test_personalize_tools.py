from pathlib import Path

import orjson

from personal_assistant.agents.personalize import tools as personalize_tools


def test_update_personalization_rejects_unknown_vault(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(personalize_tools, "HOME", tmp_path)
    monkeypatch.setattr(
        personalize_tools,
        "_verify_vault_name",
        lambda vault_name: (False, f"Nie udało się zweryfikować vaulta '{vault_name}'."),
    )

    result = personalize_tools.update_personalization(name="Ala", vault_name="Arcans")

    assert result == "Nie udało się zweryfikować vaulta 'Arcans'."
    assert not (tmp_path / ".aispiritagent" / "personalization.json").exists()


def test_update_personalization_saves_vault_name_when_verified(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(personalize_tools, "HOME", tmp_path)
    monkeypatch.setattr(personalize_tools, "_verify_vault_name", lambda _: (True, ""))
    monkeypatch.setattr(personalize_tools.git_tracer, "initial_tracking_project", lambda _: None)
    monkeypatch.setattr(personalize_tools, "load_vault_markdown_dataset", lambda: [])
    monkeypatch.setattr(personalize_tools, "initial_rag", lambda _: None)

    result = personalize_tools.update_personalization(name="Ala", vault_name="Arcans")

    assert result == "Personalizacja zapisana."
    personalization_file = tmp_path / ".aispiritagent" / "personalization.json"
    payload = orjson.loads(personalization_file.read_bytes())
    assert payload == {"name": "Ala", "vault_name": "Arcans"}
