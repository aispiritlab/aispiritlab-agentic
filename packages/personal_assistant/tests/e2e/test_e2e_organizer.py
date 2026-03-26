from __future__ import annotations

import pytest

from .conftest import assert_step_efficiency

pytestmark = pytest.mark.workflow_smoke


@pytest.fixture()
def organizer_callback(monkeypatch):
    from personal_assistant.agents.manage_notes import tools as note_tools

    monkeypatch.setattr(
        note_tools,
        "_run_obsidian_command",
        lambda *_args, **_kwargs: (True, "ok"),
    )

    from personal_assistant.agents.organizer.evaluation import OrganizerEvalCallback

    callback = OrganizerEvalCallback()
    callback.reset()
    yield callback
    callback.close()


_SCENARIO_NAMES = ("classify_project_note", "classify_resource_note")


@pytest.mark.parametrize(
    "scenario_name", _SCENARIO_NAMES, ids=_SCENARIO_NAMES
)
def test_organizer_e2e(organizer_callback, scenario_name):
    from personal_assistant.agents.organizer.evaluation import ORGANIZER_TOOL_SCENARIOS

    scenario = next(s for s in ORGANIZER_TOOL_SCENARIOS if s.name == scenario_name)

    response = organizer_callback.run(scenario.user_message)

    assert isinstance(response, str)
    assert len(response) > 0

    assert_step_efficiency(
        user_input=scenario.user_message,
        output=response,
        agent_name="organizer_agent",
        tool_name=scenario.tool_name,
        parameters=scenario.parameters,
    )
