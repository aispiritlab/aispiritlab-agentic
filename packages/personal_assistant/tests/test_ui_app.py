from __future__ import annotations

import personal_assistant.ui.app as app_module


def test_generate_response_passes_workspace_to_agent_call(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def _fake_ai_spirit_agent(
        message: str, user: str | None = None, workspace: str | None = None
    ) -> str:
        captured["message"] = message
        captured["user"] = user or ""
        captured["workspace"] = workspace or ""
        return "done"

    monkeypatch.setattr(app_module, "ai_spirit_agent", _fake_ai_spirit_agent)

    history = [{"role": "user", "content": "Hello world"}]

    list(app_module.generate_response(history, "Agenci", "alice", "research"))

    assert captured == {
        "message": "Hello world",
        "user": "alice",
        "workspace": "research",
    }
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] == "done"
