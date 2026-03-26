from __future__ import annotations

import pytest

from .conftest import assert_step_efficiency

pytestmark = pytest.mark.workflow_smoke


@pytest.fixture()
def discovery_callback(monkeypatch):
    from personal_assistant.agents.discovery_notes import tools as discovery_tools

    monkeypatch.setattr(
        discovery_tools,
        "search",
        lambda query: f"Wyniki: {query}",
    )

    from personal_assistant.agents.discovery_notes.evaluation import DiscoveryNotesEvalCallback

    callback = DiscoveryNotesEvalCallback()
    callback.reset()
    yield callback
    callback.close()


_SCENARIO_NAMES = ("search_rag_foundations", "search_embeddings")


@pytest.mark.parametrize(
    "scenario_name", _SCENARIO_NAMES, ids=_SCENARIO_NAMES
)
def test_discovery_notes_e2e(discovery_callback, scenario_name):
    from personal_assistant.agents.discovery_notes.evaluation import DISCOVERY_NOTES_TOOL_SCENARIOS

    scenario = next(
        s for s in DISCOVERY_NOTES_TOOL_SCENARIOS if s.name == scenario_name
    )

    response = discovery_callback.run(scenario.user_message)

    assert isinstance(response, str)
    assert len(response) > 0

    assert_step_efficiency(
        user_input=scenario.user_message,
        output=response,
        agent_name="discovery_notes_agent",
        tool_name=scenario.tool_name,
        parameters=scenario.parameters,
    )
