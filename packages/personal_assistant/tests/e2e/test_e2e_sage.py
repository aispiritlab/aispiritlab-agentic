from __future__ import annotations

import pytest

from .conftest import assert_step_efficiency

pytestmark = pytest.mark.workflow_smoke


def test_sage_e2e():
    from agentic_runtime.messaging.messages import UserMessage
    from personal_assistant.agents.sage.sage_workflow import SageWorkflow

    workflow = SageWorkflow(tracer=None, context=None)
    user_text = "Pomoz mi zdecydowac czy kupic dom czy mieszkanie"
    response = workflow.handle(UserMessage(text=user_text))

    assert isinstance(response, str)
    assert len(response) > 0

    assert_step_efficiency(
        user_input=user_text,
        output=response,
        agent_name="sage_agent",
    )
