from __future__ import annotations

import pytest

pytestmark = pytest.mark.workflow_smoke

_SCENARIO_NAMES = (
    "route_personalize_name",
    "route_manage_notes_add",
    "route_discovery_search",
    "route_sage_decision",
)


@pytest.fixture()
def router_callback():
    from personal_assistant.agents.router.evaluation import RouterEvalCallback

    callback = RouterEvalCallback()
    yield callback


@pytest.mark.parametrize(
    "scenario_name", _SCENARIO_NAMES, ids=_SCENARIO_NAMES
)
def test_router_e2e(router_callback, scenario_name):
    from personal_assistant.agents.router.evaluation import ROUTER_TOOL_SCENARIOS

    scenario = next(s for s in ROUTER_TOOL_SCENARIOS if s.name == scenario_name)

    response = router_callback.run(scenario.user_message)

    assert isinstance(response, str)
    assert len(response) > 0

    route = response.strip().lower()
    assert route == scenario.expected_output
