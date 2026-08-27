from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
import inspect
from threading import Lock
from time import monotonic
from typing import ClassVar, Literal

from structlog import get_logger

from providers._proto import TextModel
from providers.api import OpenAIProvider
from providers.basic import BasicProvider
from providers.mlx import MlxAudioProvider, MlxProvider
from providers.mlx.mlx_vlm import MlxVlmProvider
from providers.models.config import DEFAULT_MODEL_CONFIG, ModelConfig
from providers.onnx import OnnxProvider
from providers.sglang import SGLangProvider
from providers.transformers import TransformersProvider
from providers.vllm import VLLMProvider

logger = get_logger(__name__)

ModelProviderType = Literal[
    "onnx", "mlx", "mlx-audio", "mlx-vlm", "openai", "basic", "vllm", "transformers", "sglang"
]

SUPPORTED_PROVIDERS = {
    "basic": BasicProvider,
    "mlx-audio": MlxAudioProvider,
    "mlx-vlm": MlxVlmProvider,
    "mlx": MlxProvider,
    "onnx": OnnxProvider,
    "openai": OpenAIProvider,
    "vllm": VLLMProvider,
    "transformers": TransformersProvider,
    "sglang": SGLangProvider,
}


@dataclass(slots=True)
class _SharedBackendEntry:
    backend: object
    provider_cls: object
    inference_lock: Lock = field(default_factory=Lock)
    ref_count: int = 0


@dataclass(slots=True)
class _LoadFailure:
    """A cached load error, kept only until ``expires_at``."""

    message: str
    expires_at: float


class ModelProvider:
    _allowed_model_attrs = frozenset({"model", "voice_model"})
    _shared_backends: ClassVar[
        dict[tuple[str | None, ModelProviderType], _SharedBackendEntry]
    ] = {}
    _shared_lock: ClassVar[Lock] = Lock()
    # Per-cache-key locks, so loading model A does not block loading model B.
    _load_locks: ClassVar[dict[tuple[str | None, ModelProviderType], Lock]] = {}
    _load_locks_guard: ClassVar[Lock] = Lock()

    #: How long a failed load is remembered before the next attempt retries.
    load_retry_seconds: ClassVar[float] = 30.0

    def __init__(
        self,
        name: str | None = None,
        *,
        model_provider_type: ModelProviderType = "mlx",
        supported_providers=None,
        config: ModelConfig = DEFAULT_MODEL_CONFIG,
    ):
        if supported_providers is None:
            supported_providers = SUPPORTED_PROVIDERS
        self._model_name = name
        self._model_provider_type = model_provider_type
        self._instance_lock = Lock()
        self._instance_cache: dict[str, TextModel] = {}
        self._shared_backend_keys: dict[str, tuple[str | None, ModelProviderType]] = {}
        self._load_errors: dict[str, _LoadFailure] = {}
        self._config = config
        self._supported_providers = supported_providers

    def _cache_key(self) -> tuple[str | None, ModelProviderType]:
        return self._model_name, self._model_provider_type

    def _provider_cls(self):
        provider_cls = self._supported_providers.get(self._model_provider_type)
        if provider_cls is None:
            raise RuntimeError(f"Unsupported model provider type: {self._model_provider_type}")
        return provider_cls

    def _load_backend(self) -> object:
        if self._model_name is None:
            raise AttributeError(
                "Text model name is not configured for this ModelProvider instance."
            )
        provider_cls = self._provider_cls()
        load_backend = getattr(provider_cls, "load_backend", None)
        if callable(load_backend):
            return load_backend(self._model_name)
        return provider_cls.load(self._model_name, self._config)

    @staticmethod
    def _build_model_accepts_inference_lock(build_model: Callable[..., object]) -> bool:
        try:
            params = inspect.signature(build_model).parameters.values()
        except TypeError, ValueError:
            return False

        return any(
            parameter.name == "inference_lock" or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in params
        )

    @classmethod
    def _load_lock_for(cls, cache_key: tuple[str | None, ModelProviderType]) -> Lock:
        with cls._load_locks_guard:
            lock = cls._load_locks.get(cache_key)
            if lock is None:
                lock = Lock()
                cls._load_locks[cache_key] = lock
            return lock

    def _acquire_shared_backend(
        self,
    ) -> tuple[tuple[str | None, ModelProviderType], _SharedBackendEntry]:
        cache_key = self._cache_key()
        provider_cls = self._provider_cls()

        with self._shared_lock:
            entry = self._shared_backends.get(cache_key)
            if entry is not None:
                entry.ref_count += 1
                return cache_key, entry

        # Load outside the class-wide lock: model loading takes tens of seconds
        # and must not block every other provider in the process.
        with self._load_lock_for(cache_key):
            with self._shared_lock:
                entry = self._shared_backends.get(cache_key)
                if entry is not None:
                    entry.ref_count += 1
                    return cache_key, entry

            backend = self._load_backend()

            with self._shared_lock:
                entry = self._shared_backends.get(cache_key)
                if entry is None:
                    entry = _SharedBackendEntry(backend=backend, provider_cls=provider_cls)
                    self._shared_backends[cache_key] = entry
                entry.ref_count += 1
                return cache_key, entry

    def _release_shared_backend(self, cache_key: tuple[str | None, ModelProviderType]) -> None:
        provider_cls: object | None = None
        backend_to_close: object | None = None

        with self._shared_lock:
            entry = self._shared_backends.get(cache_key)
            if entry is None:
                return
            entry.ref_count -= 1
            if entry.ref_count > 0:
                return
            backend_to_close = entry.backend
            provider_cls = entry.provider_cls
            del self._shared_backends[cache_key]

        close_backend = getattr(provider_cls, "close_backend", None)
        if callable(close_backend) and backend_to_close is not None:
            close_backend(backend_to_close)

    def _build_model_instance(
        self,
    ) -> tuple[TextModel, tuple[str | None, ModelProviderType] | None]:
        provider_cls = self._provider_cls()
        build_model = getattr(provider_cls, "build_model", None)
        if not callable(build_model):
            return provider_cls.load(self._model_name, self._config), None

        cache_key, entry = self._acquire_shared_backend()
        build_kwargs = {}
        if self._build_model_accepts_inference_lock(build_model):
            build_kwargs["inference_lock"] = entry.inference_lock
        try:
            instance = build_model(
                entry.backend,
                self._model_name,
                self._config,
                **build_kwargs,
            )
        except Exception:
            self._release_shared_backend(cache_key)
            raise
        return instance, cache_key

    def get(self, name: str = "model") -> TextModel | None:
        """Return the loaded model, or ``None`` when it is unavailable.

        A failed load is remembered for ``load_retry_seconds`` only, so a model
        server that comes up after the app started is picked up automatically.
        """
        if name not in self._allowed_model_attrs:
            raise AttributeError(name)

        with self._instance_lock:
            cached = self._instance_cache.get(name)
            if cached is not None:
                return cached

            failure = self._load_errors.get(name)
            if failure is not None and monotonic() < failure.expires_at:
                return None
            self._load_errors.pop(name, None)

            try:
                instance, cache_key = self._build_model_instance()
            except Exception as error:
                logger.warning(
                    "model_load_failed",
                    model_name=self._model_name,
                    model_provider_type=self._model_provider_type,
                    error_type=type(error).__name__,
                    error_message=str(error),
                    retry_in_seconds=self.load_retry_seconds,
                )
                self._load_errors[name] = _LoadFailure(
                    message=f"{type(error).__name__}: {error}",
                    expires_at=monotonic() + self.load_retry_seconds,
                )
                self.__dict__.pop(name, None)
                return None

            self._instance_cache[name] = instance
            if cache_key is not None:
                self._shared_backend_keys[name] = cache_key
            self._load_errors.pop(name, None)
            self.__setattr__(name, instance)
            return instance

    def invalidate(self, name: str = "model") -> None:
        """Forget a cached load failure so the next ``get()`` retries at once."""
        if name not in self._allowed_model_attrs:
            raise AttributeError(name)
        with self._instance_lock:
            self._load_errors.pop(name, None)

    def get_load_error(self, name: str = "model") -> str | None:
        if name not in self._allowed_model_attrs:
            raise AttributeError(name)

        with self._instance_lock:
            failure = self._load_errors.get(name)
            return failure.message if failure is not None else None

    @contextmanager
    def session(self, name: str = "model") -> Iterator[TextModel | None]:
        """Scope a model for one unit of work.

        The instance is cached and shared, so nothing is torn down on exit; the
        context manager exists so callers have one obvious place to acquire a
        model and so a future pooled implementation can release it here.
        """
        yield self.get(name)

    def close(self) -> None:
        with self._instance_lock:
            cached_instances = list(self._instance_cache.items())
            shared_backend_keys = list(self._shared_backend_keys.values())
            self._instance_cache.clear()
            self._shared_backend_keys.clear()
            self._load_errors.clear()
            for name, _instance in cached_instances:
                self.__dict__.pop(name, None)

        for _name, instance in cached_instances:
            close = getattr(instance, "close", None)
            if callable(close):
                close()

        for cache_key in shared_backend_keys:
            self._release_shared_backend(cache_key)

    @classmethod
    def shutdown_all(cls) -> None:
        with cls._shared_lock:
            shared_entries = list(cls._shared_backends.items())
            cls._shared_backends.clear()

        for (model_name, provider_type), entry in shared_entries:
            provider_cls = entry.provider_cls
            close_backend = getattr(provider_cls, "close_backend", None)
            if callable(close_backend):
                close_backend(entry.backend)
            logger.debug(
                "model_backend_shutdown",
                model_name=model_name,
                model_provider_type=provider_type,
            )

    def __getattr__(self, name: str) -> TextModel | None:
        if name in self._allowed_model_attrs:
            return self.get(name)
        raise AttributeError(name)
