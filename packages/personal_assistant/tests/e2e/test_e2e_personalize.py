from __future__ import annotations

import pytest

pytestmark = pytest.mark.workflow_smoke


def test_personalize_e2e(monkeypatch):
    from personal_assistant.agents.personalize import tools as personalize_tools

    monkeypatch.setattr(
        personalize_tools,
        "update_personalization",
        lambda name, vault_name, vault_path=None: "Personalizacja zapisana.",
    )
    monkeypatch.setattr(
        personalize_tools,
        "_verify_vault_name",
        lambda vault_name: (True, ""),
    )

    from agentic_runtime.messaging.messages import UserMessage
    from personal_assistant.agents.personalize.personlize_workflow import PersonalizeWorkflow

    workflow = PersonalizeWorkflow(tracer=None, context=None)
    user_text = "Mam na imie Mateusz"
    response = workflow.handle(UserMessage(text=user_text))

    assert isinstance(response, str)
    assert len(response) > 0
