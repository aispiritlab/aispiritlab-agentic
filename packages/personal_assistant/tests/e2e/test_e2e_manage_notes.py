from __future__ import annotations

import json

import pytest

from .conftest import assert_step_efficiency

pytestmark = pytest.mark.workflow_smoke


@pytest.fixture()
def notes_callback(tmp_path, monkeypatch):
    from personal_assistant.agents.manage_notes import tools as note_tools

    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    personalization_file = tmp_path / "personalization.json"
    personalization_file.write_text(
        json.dumps({"vault_path": str(vault_path)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(note_tools, "PERSONALIZATION_FILE", personalization_file)

    from personal_assistant.agents.manage_notes.evaluation import NotesAgentEvalCallback

    callback = NotesAgentEvalCallback()
    callback.reset()
    callback._agent.start()
    yield callback
    callback.close()


_SCENARIO_NAMES = ("add_note", "read_note", "list_notes")


def _get_scenarios():
    from personal_assistant.agents.manage_notes.evaluation import NOTES_TOOL_SCENARIOS

    return [s for s in NOTES_TOOL_SCENARIOS if s.name in _SCENARIO_NAMES]


@pytest.mark.parametrize(
    "scenario_name", _SCENARIO_NAMES, ids=_SCENARIO_NAMES
)
def test_manage_notes_e2e(notes_callback, scenario_name):
    from personal_assistant.agents.manage_notes.evaluation import NOTES_TOOL_SCENARIOS

    scenario = next(s for s in NOTES_TOOL_SCENARIOS if s.name == scenario_name)

    if scenario.prefill_messages:
        notes_callback.prime(scenario.prefill_messages)

    response = notes_callback.run(scenario.user_message)

    assert isinstance(response, str)
    assert len(response) > 0

    assert_step_efficiency(
        user_input=scenario.user_message,
        output=response,
        agent_name="manage_notes_agent",
        tool_name=scenario.tool_name,
        parameters=scenario.parameters,
    )
