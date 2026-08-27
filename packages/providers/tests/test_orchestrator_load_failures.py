"""Load-failure caching, invalidation and shared-backend refcounting.

A model server that is down must not wedge the app: ``get()`` degrades to
``None``, remembers the failure briefly so every request does not pay the
timeout, and retries on its own once ``load_retry_seconds`` elapses.
``invalidate()`` is the manual short-circuit.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Any, ClassVar, cast

import pytest

from providers.models.config import ModelConfig
from providers.orchestrator import ModelProvider


@pytest.fixture(autouse=True)
def isolate_class_state() -> Iterator[None]:
    """Keep the process-wide backend caches from leaking between tests."""
    saved_backends = dict(ModelProvider._shared_backends)
    saved_locks = dict(ModelProvider._load_locks)
    ModelProvider._shared_backends.clear()
    ModelProvider._load_locks.clear()
    yield
    ModelProvider._shared_backends.clear()
    ModelProvider._shared_backends.update(saved_backends)
    ModelProvider._load_locks.clear()
    ModelProvider._load_locks.update(saved_locks)


class FakeModel:
    def __init__(self, name: str | None) -> None:
        self.name = name
        self.closed = 0
        self.inference_lock: threading.Lock | None = None

    def close(self) -> None:
        self.closed += 1


class ImmediateRetryProvider(ModelProvider):
    """A provider that never caches a failure, so a retry is observable."""

    load_retry_seconds: ClassVar[float] = 0.0


class SimpleProvider:
    """A provider with no shared backend: one instance per load."""

    loads = 0

    @classmethod
    def load(cls, name: str | None, config: ModelConfig) -> FakeModel:
        del config
        cls.loads += 1
        return FakeModel(name)


class BrokenProvider:
    loads = 0

    @classmethod
    def load(cls, name: str | None, config: ModelConfig) -> FakeModel:
        del name, config
        cls.loads += 1
        raise ConnectionRefusedError("model server is down")


def _provider(provider_cls: type, name: str | None = "test-model") -> ModelProvider:
    return ModelProvider(
        name,
        model_provider_type="mlx",
        supported_providers={"mlx": provider_cls},
    )


# ---------------------------------------------------------------------------
# Failure caching
# ---------------------------------------------------------------------------


def test_a_failed_load_degrades_to_none_instead_of_raising() -> None:
    BrokenProvider.loads = 0

    assert _provider(BrokenProvider).get() is None


def test_a_failed_load_is_remembered_so_every_request_does_not_retry() -> None:
    BrokenProvider.loads = 0
    provider = _provider(BrokenProvider)

    for _ in range(5):
        assert provider.get() is None

    assert BrokenProvider.loads == 1


def test_the_load_error_is_reported_with_its_type() -> None:
    BrokenProvider.loads = 0
    provider = _provider(BrokenProvider)
    provider.get()

    assert provider.get_load_error() == "ConnectionRefusedError: model server is down"


def test_no_load_error_is_reported_before_the_first_attempt() -> None:
    assert _provider(BrokenProvider).get_load_error() is None


def test_invalidate_makes_the_next_get_retry_immediately() -> None:
    BrokenProvider.loads = 0
    provider = _provider(BrokenProvider)
    provider.get()
    assert BrokenProvider.loads == 1

    provider.invalidate()

    assert provider.get_load_error() is None
    assert provider.get() is None
    assert BrokenProvider.loads == 2


def test_invalidate_is_harmless_when_nothing_failed() -> None:
    provider = _provider(SimpleProvider)

    provider.invalidate()

    assert provider.get_load_error() is None


def test_the_cached_failure_expires_and_the_model_is_picked_up() -> None:
    # The real scenario: the app starts before the LLM server does.
    class FlakyProvider:
        loads = 0
        healthy = False

        @classmethod
        def load(cls, name: str | None, config: ModelConfig) -> FakeModel:
            del config
            cls.loads += 1
            if not cls.healthy:
                raise ConnectionRefusedError("not up yet")
            return FakeModel(name)

    provider = ImmediateRetryProvider(
        "test-model",
        model_provider_type="mlx",
        supported_providers={"mlx": FlakyProvider},
    )

    assert provider.get() is None

    FlakyProvider.healthy = True
    model = provider.get()

    assert model is not None
    assert FlakyProvider.loads == 2


def test_a_successful_load_clears_a_previous_failure() -> None:
    class FlakyProvider:
        healthy = False

        @classmethod
        def load(cls, name: str | None, config: ModelConfig) -> FakeModel:
            del config
            if not cls.healthy:
                raise ConnectionRefusedError("not up yet")
            return FakeModel(name)

    provider = ImmediateRetryProvider(
        "test-model",
        model_provider_type="mlx",
        supported_providers={"mlx": FlakyProvider},
    )
    provider.get()
    assert provider.get_load_error() is not None

    FlakyProvider.healthy = True
    provider.get()

    assert provider.get_load_error() is None


def test_a_missing_model_name_fails_softly() -> None:
    class SharedProvider:
        @staticmethod
        def load_backend(name: str) -> object:
            return object()

        @staticmethod
        def build_model(backend: object, name: str | None, config: ModelConfig) -> FakeModel:
            del backend, config
            return FakeModel(name)

    provider = _provider(SharedProvider, name=None)

    assert provider.get() is None
    assert "not configured" in (provider.get_load_error() or "")


def test_an_unsupported_provider_type_fails_softly() -> None:
    provider = ModelProvider(
        "m",
        model_provider_type=cast("Any", "nope"),
        supported_providers={},
    )

    assert provider.get() is None
    assert "Unsupported model provider type" in (provider.get_load_error() or "")


# ---------------------------------------------------------------------------
# Allowed attribute names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["get", "invalidate", "get_load_error"])
def test_only_known_model_slots_are_addressable(method: str) -> None:
    provider = _provider(SimpleProvider)

    with pytest.raises(AttributeError):
        getattr(provider, method)("__dict__")


def test_attribute_access_proxies_to_get() -> None:
    SimpleProvider.loads = 0
    provider = _provider(SimpleProvider)

    assert provider.model is not None
    assert provider.voice_model is not None

    with pytest.raises(AttributeError):
        provider.something_else  # noqa: B018


def test_a_failed_attribute_access_returns_none_rather_than_raising() -> None:
    assert _provider(BrokenProvider).model is None


# ---------------------------------------------------------------------------
# Caching and lifecycle
# ---------------------------------------------------------------------------


def test_a_loaded_model_is_cached() -> None:
    SimpleProvider.loads = 0
    provider = _provider(SimpleProvider)

    first = provider.get()
    second = provider.get()

    assert first is second
    assert SimpleProvider.loads == 1


def test_model_and_voice_model_are_cached_separately() -> None:
    SimpleProvider.loads = 0
    provider = _provider(SimpleProvider)

    assert provider.get("model") is not provider.get("voice_model")
    assert SimpleProvider.loads == 2


def test_session_yields_the_model_and_keeps_it_afterwards() -> None:
    provider = _provider(SimpleProvider)

    with provider.session() as model:
        assert model is not None

    assert provider.get() is model


def test_session_yields_none_when_the_model_is_unavailable() -> None:
    with _provider(BrokenProvider).session() as model:
        assert model is None


def test_close_releases_the_cached_instances() -> None:
    SimpleProvider.loads = 0
    provider = _provider(SimpleProvider)
    model = cast("FakeModel | None", provider.get())
    assert model is not None

    provider.close()

    assert model.closed == 1
    assert provider.get() is not model
    assert SimpleProvider.loads == 2


def test_close_clears_a_pending_load_failure() -> None:
    provider = _provider(BrokenProvider)
    provider.get()

    provider.close()

    assert provider.get_load_error() is None


# ---------------------------------------------------------------------------
# Shared backends
# ---------------------------------------------------------------------------


class SharedBackendProvider:
    backends_opened = 0
    backends_closed = 0

    @classmethod
    def load_backend(cls, name: str) -> object:
        cls.backends_opened += 1
        return {"name": name}

    @staticmethod
    def build_model(
        backend: object,
        name: str | None,
        config: ModelConfig,
        *,
        inference_lock: threading.Lock | None = None,
    ) -> FakeModel:
        del backend, config
        model = FakeModel(name)
        model.inference_lock = inference_lock
        return model

    @classmethod
    def close_backend(cls, backend: object) -> None:
        del backend
        cls.backends_closed += 1


def _shared(name: str = "shared-model") -> ModelProvider:
    return ModelProvider(
        name,
        model_provider_type="mlx",
        supported_providers={"mlx": SharedBackendProvider},
    )


def test_two_providers_for_one_model_share_a_single_backend() -> None:
    SharedBackendProvider.backends_opened = 0
    first, second = _shared(), _shared()

    first.get()
    second.get()

    assert SharedBackendProvider.backends_opened == 1


def test_the_backend_stays_open_while_another_provider_still_holds_it() -> None:
    SharedBackendProvider.backends_opened = 0
    SharedBackendProvider.backends_closed = 0
    first, second = _shared(), _shared()
    first.get()
    second.get()

    first.close()
    assert SharedBackendProvider.backends_closed == 0

    second.close()
    assert SharedBackendProvider.backends_closed == 1


def test_different_models_get_different_backends() -> None:
    SharedBackendProvider.backends_opened = 0

    _shared("model-a").get()
    _shared("model-b").get()

    assert SharedBackendProvider.backends_opened == 2


def test_the_shared_inference_lock_is_handed_to_every_instance() -> None:
    first, second = _shared(), _shared()

    model_a = cast("FakeModel | None", first.get())
    model_b = cast("FakeModel | None", second.get())

    assert model_a is not None and model_b is not None
    assert model_a.inference_lock is model_b.inference_lock


def test_a_backend_is_loaded_once_under_concurrent_first_use() -> None:
    SharedBackendProvider.backends_opened = 0
    providers = [_shared() for _ in range(8)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        models = list(pool.map(lambda p: p.get(), providers))

    assert all(model is not None for model in models)
    assert SharedBackendProvider.backends_opened == 1


def test_loading_one_model_does_not_block_loading_another() -> None:
    # Per-cache-key load locks: a 40 s load of model A must not stall model B.
    barrier = threading.Barrier(2, timeout=5)

    class RendezvousProvider(SharedBackendProvider):
        @classmethod
        def load_backend(cls, name: str) -> object:
            barrier.wait()
            return {"name": name}

    def make(name: str) -> ModelProvider:
        return ModelProvider(
            name,
            model_provider_type="mlx",
            supported_providers={"mlx": RendezvousProvider},
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        models = list(pool.map(lambda n: make(n).get(), ["model-a", "model-b"]))

    assert all(model is not None for model in models)
    assert not barrier.broken


def test_a_failing_build_releases_the_shared_backend() -> None:
    class FailingBuildProvider(SharedBackendProvider):
        @staticmethod
        def build_model(backend, name, config, *, inference_lock=None):  # type: ignore[no-untyped-def]
            del backend, name, config, inference_lock
            raise RuntimeError("build failed")

    FailingBuildProvider.backends_closed = 0
    provider = ModelProvider(
        "leaky",
        model_provider_type="mlx",
        supported_providers={"mlx": FailingBuildProvider},
    )

    assert provider.get() is None
    # Refcount must have gone back to zero, closing the backend.
    assert FailingBuildProvider.backends_closed == 1
    assert ModelProvider._shared_backends == {}


def test_shutdown_all_closes_every_shared_backend() -> None:
    SharedBackendProvider.backends_closed = 0
    _shared("model-a").get()
    _shared("model-b").get()

    ModelProvider.shutdown_all()

    assert SharedBackendProvider.backends_closed == 2
    assert ModelProvider._shared_backends == {}
