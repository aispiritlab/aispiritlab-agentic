from __future__ import annotations

import personal_assistant as personal_assistant_module
import personal_assistant.runtime as runtime_module
from personal_assistant.settings import settings


class _StubRuntime:
    def __init__(self, user_slug: str = "default", workspace_slug: str = "default") -> None:
        self.user_slug = user_slug
        self.workspace_slug = workspace_slug
        self.session_id = f"user:{user_slug}:ws:{workspace_slug}"
        self.stopped = False

    def start(self) -> str:
        return "hello"

    def stop(self) -> None:
        self.stopped = True

    def run(self, text: str) -> str:
        return text

    def run_chat(self, text: str) -> str:
        return text

    def run_generate_image(self, text: str, images=None):
        del images
        return text

    def reset_chat(self) -> None:
        return None

    def clear_personalization_history(self) -> None:
        return None


def test_get_runtime_caches_per_user_and_workspace(monkeypatch) -> None:
    personal_assistant_module.shutdown_application()
    monkeypatch.setattr(settings, "agentic_transport", "in_memory", raising=False)
    monkeypatch.setattr(runtime_module, "PARuntime", _StubRuntime)

    runtime_a1 = personal_assistant_module.get_runtime(user="alice", workspace="research")
    runtime_a2 = personal_assistant_module.get_runtime(user="alice", workspace="research")
    runtime_b = personal_assistant_module.get_runtime(user="alice", workspace="ops")

    assert runtime_a1 is runtime_a2
    assert runtime_a1 is not runtime_b
    assert runtime_a1.session_id == "user:alice:ws:research"
    assert runtime_b.session_id == "user:alice:ws:ops"

    personal_assistant_module.shutdown_application()


def test_drop_runtime_sessions_filters_by_user_and_workspace(monkeypatch) -> None:
    personal_assistant_module.shutdown_application()
    monkeypatch.setattr(settings, "agentic_transport", "in_memory", raising=False)
    monkeypatch.setattr(runtime_module, "PARuntime", _StubRuntime)

    runtime_a1 = personal_assistant_module.get_runtime(user="alice", workspace="research")
    runtime_a2 = personal_assistant_module.get_runtime(user="alice", workspace="ops")
    runtime_b = personal_assistant_module.get_runtime(user="bob", workspace="research")

    personal_assistant_module.drop_runtime_sessions(user="alice")

    assert runtime_a1.stopped is True
    assert runtime_a2.stopped is True
    assert runtime_b.stopped is False

    personal_assistant_module.drop_runtime_sessions(workspace="research")

    assert runtime_b.stopped is True

    personal_assistant_module.shutdown_application()
