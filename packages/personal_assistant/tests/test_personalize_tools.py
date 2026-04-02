from pathlib import Path

import orjson
from personal_assistant.agents.personalize import tools as personalize_tools


def test_update_personalization_rejects_unknown_vault(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "personalization.json"
    monkeypatch.setattr(personalize_tools, "_user_personalization_path", lambda _user=None: target)
    monkeypatch.setattr(
        personalize_tools,
        "_verify_vault_name",
        lambda vault_name: (False, f"Nie udało się zweryfikować vaulta '{vault_name}'."),
    )

    result = personalize_tools.update_personalization(name="Ala", vault_name="Arcans")

    assert result == "Nie udało się zweryfikować vaulta 'Arcans'."
    assert not target.exists()


def test_update_personalization_saves_vault_name_when_verified(
    monkeypatch, tmp_path: Path
) -> None:
    target = tmp_path / "user" / "personalization.json"
    monkeypatch.setattr(personalize_tools, "_user_personalization_path", lambda _user=None: target)
    monkeypatch.setattr(personalize_tools, "_verify_vault_name", lambda _: (True, ""))
    monkeypatch.setattr(personalize_tools.git_tracer, "initial_tracking_project", lambda _: None)

    result = personalize_tools.update_personalization(name="Ala", vault_name="Arcans")

    assert result == "Personalizacja zapisana."
    payload = orjson.loads(target.read_bytes())
    assert payload == {"name": "Ala", "vault_name": "Arcans"}
