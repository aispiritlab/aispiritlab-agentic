import threading
import time

import pytest

from agentic.agent import Agent
from agentic.models import ModelConfig, ModelProvider
from agentic.models.response import ModelResponse
from agentic.prompts import GemmaPromptBuilder


def test_model_provider_returns_loaded_model_instance() -> None:
    loaded_instance = object()

    class StubProvider:
        @classmethod
        def load(cls, model_name: str, config: object) -> object:
            assert model_name == "working-model"
            return loaded_instance

    provider = ModelProvider(
        "working-model",
        supported_providers={"mlx": StubProvider},
    )

    assert provider.model is loaded_instance


def test_agent_surfaces_model_provider_load_error(monkeypatch) -> None:
    provider = ModelProvider("broken-model")

    def _raise_load_error() -> tuple[object, object]:
        raise FileNotFoundError("weights missing")

    monkeypatch.setattr(provider, "_build_model_instance", _raise_load_error)
    agent = Agent(
        model_provider=provider,
        prompt_builder=GemmaPromptBuilder(system_prompt="SYSTEM"),
    )

    with pytest.raises(RuntimeError, match="weights missing"):
        agent.run("hej")

    assert provider.get_load_error("model") == "FileNotFoundError: weights missing"


def test_model_provider_shares_backend_and_keeps_per_instance_config() -> None:
    load_backend_calls: list[str] = []
    close_backend_calls: list[object] = []
    shared_backend = object()

    class FakeModel:
        def __init__(self, backend: object, config: ModelConfig) -> None:
            self.backend = backend
            self.config = config
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class SharedProvider:
        @classmethod
        def load_backend(cls, model_name: str) -> object:
            load_backend_calls.append(model_name)
            return shared_backend

        @classmethod
        def build_model(
            cls,
            backend: object,
            model_name: str,
            config: ModelConfig,
        ) -> FakeModel:
            assert model_name == "shared-model"
            return FakeModel(backend, config)

        @classmethod
        def close_backend(cls, backend: object) -> None:
            close_backend_calls.append(backend)

    first_provider = ModelProvider(
        "shared-model",
        supported_providers={"mlx": SharedProvider},
        config=ModelConfig(max_tokens=32),
    )
    second_provider = ModelProvider(
        "shared-model",
        supported_providers={"mlx": SharedProvider},
        config=ModelConfig(max_tokens=64),
    )

    first_model = first_provider.model
    second_model = second_provider.model

    assert load_backend_calls == ["shared-model"]
    assert first_model is not second_model
    assert first_model.backend is shared_backend
    assert second_model.backend is shared_backend
    assert first_model.config.max_tokens == 32
    assert second_model.config.max_tokens == 64

    first_provider.close()
    assert first_model.closed is True
    assert close_backend_calls == []

    second_provider.close()
    assert second_model.closed is True
    assert close_backend_calls == [shared_backend]


def test_model_provider_shutdown_all_releases_shared_backends() -> None:
    closed_backends: list[object] = []
    shared_backend = object()

    class FakeModel:
        def __init__(self, backend: object) -> None:
            self.backend = backend

    class SharedProvider:
        @classmethod
        def load_backend(cls, model_name: str) -> object:
            assert model_name == "shutdown-model"
            return shared_backend

        @classmethod
        def build_model(
            cls,
            backend: object,
            model_name: str,
            config: ModelConfig,
        ) -> FakeModel:
            del model_name, config
            return FakeModel(backend)

        @classmethod
        def close_backend(cls, backend: object) -> None:
            closed_backends.append(backend)

    provider = ModelProvider(
        "shutdown-model",
        supported_providers={"mlx": SharedProvider},
    )

    _ = provider.model
    provider.shutdown_all()

    assert closed_backends == [shared_backend]


def test_model_provider_builds_instance_once_per_provider_under_concurrency() -> None:
    load_backend_calls: list[str] = []
    build_calls: list[str] = []
    close_backend_calls: list[object] = []
    shared_backend = object()
    build_started = threading.Event()
    release_build = threading.Event()
    results: list[object] = []

    class FakeModel:
        def __init__(self, backend: object) -> None:
            self.backend = backend

        def close(self) -> None:
            return None

    class SharedProvider:
        @classmethod
        def load_backend(cls, model_name: str) -> object:
            load_backend_calls.append(model_name)
            return shared_backend

        @classmethod
        def build_model(
            cls,
            backend: object,
            model_name: str,
            config: ModelConfig,
        ) -> FakeModel:
            del model_name, config
            build_calls.append("build")
            build_started.set()
            assert release_build.wait(timeout=1), "Timed out waiting to release build"
            return FakeModel(backend)

        @classmethod
        def close_backend(cls, backend: object) -> None:
            close_backend_calls.append(backend)

    provider = ModelProvider(
        "shared-model",
        supported_providers={"mlx": SharedProvider},
    )

    def _load_model() -> None:
        results.append(provider.model)

    first = threading.Thread(target=_load_model)
    second = threading.Thread(target=_load_model)
    first.start()
    assert build_started.wait(timeout=1), "Timed out waiting for the first build to start"
    second.start()
    time.sleep(0.05)
    release_build.set()
    first.join(timeout=1)
    second.join(timeout=1)

    assert load_backend_calls == ["shared-model"]
    assert build_calls == ["build"]
    assert len(results) == 2
    assert results[0] is results[1]

    provider.close()
    assert close_backend_calls == [shared_backend]


def test_model_provider_serializes_inference_for_shared_backend() -> None:
    close_backend_calls: list[object] = []

    class SharedBackend:
        def __init__(self) -> None:
            self.counter_lock = threading.Lock()
            self.active = 0
            self.max_active = 0

    shared_backend = SharedBackend()

    class FakeModel:
        def __init__(
            self,
            backend: SharedBackend,
            config: ModelConfig,
            *,
            inference_lock: threading.Lock | None = None,
        ) -> None:
            del config
            self.backend = backend
            self._inference_lock = inference_lock or threading.Lock()

        def response(self, prompt: str) -> ModelResponse:
            del prompt
            with self._inference_lock:
                with self.backend.counter_lock:
                    self.backend.active += 1
                    self.backend.max_active = max(
                        self.backend.max_active,
                        self.backend.active,
                    )
                time.sleep(0.05)
                with self.backend.counter_lock:
                    self.backend.active -= 1
            return ModelResponse(text="ok")

        def close(self) -> None:
            return None

    class SharedProvider:
        @classmethod
        def load_backend(cls, model_name: str) -> SharedBackend:
            assert model_name == "shared-model"
            return shared_backend

        @classmethod
        def build_model(
            cls,
            backend: SharedBackend,
            model_name: str,
            config: ModelConfig,
            *,
            inference_lock: threading.Lock | None = None,
        ) -> FakeModel:
            assert model_name == "shared-model"
            return FakeModel(backend, config, inference_lock=inference_lock)

        @classmethod
        def close_backend(cls, backend: SharedBackend) -> None:
            close_backend_calls.append(backend)

    first_provider = ModelProvider(
        "shared-model",
        supported_providers={"mlx": SharedProvider},
    )
    second_provider = ModelProvider(
        "shared-model",
        supported_providers={"mlx": SharedProvider},
    )

    first_model = first_provider.model
    second_model = second_provider.model
    barrier = threading.Barrier(2)
    threads = [
        threading.Thread(target=lambda: (barrier.wait(), first_model.response("first"))),
        threading.Thread(target=lambda: (barrier.wait(), second_model.response("second"))),
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)

    assert shared_backend.max_active == 1

    first_provider.close()
    second_provider.close()
    assert close_backend_calls == [shared_backend]
