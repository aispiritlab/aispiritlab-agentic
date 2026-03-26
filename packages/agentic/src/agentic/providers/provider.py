from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import inspect
from threading import Lock
from typing import Iterator, Literal

from structlog import get_logger

from agentic.models.config import ModelConfig
from agentic.providers.api import OpenAIProvider
from agentic.providers.mlx import MlxAudioProvider, MlxProvider
from agentic.providers.mlx.mlx_vlm import MlxVlmProvider
from agentic.providers.onnx import OnnxAsrProvider

logger = get_logger(__name__)

ModelProviderType = Literal["onnx", "mlx", "mlx-audio", "mlx-vlm", "openai"]

SUPPORTED_PROVIDERS = {
    "mlx-audio": MlxAudioProvider,
    "mlx-vlm": MlxVlmProvider,
    "mlx": MlxProvider,
    "onnx": OnnxAsrProvider,
    "openai": OpenAIProvider,
}


@dataclass(slots=True)
class _SharedBackendEntry:
    backend: object
    provider_cls: object
    inference_lock: Lock = field(default_factory=Lock)
    ref_count: int = 0


class ModelProvider:
    _allowed_model_attrs = frozenset({"model", "voice_model"})
    _shared_backends: dict[tuple[str | None, ModelProviderType], _SharedBackendEntry] = {}
    _shared_lock = Lock()

    def __init__(
        self,
        name: str | None = None,
        *,
        model_provider_type: ModelProviderType = "mlx",
        supported_providers=None,
        config: ModelConfig = ModelConfig(),
    ):
        if supported_providers is None:
            supported_providers = SUPPORTED_PROVIDERS
        self._model_name = name
        self._model_provider_type = model_provider_type
        self._instance_lock = Lock()
        self._instance_cache: dict[str, object | None] = {}
        self._shared_backend_keys: dict[str, tuple[str | None, ModelProviderType]] = {}
        self._load_errors: dict[str, str] = {}
        self._config = config
        self._supported_providers = supported_providers

    def _cache_key(self) -> tuple[str | None, ModelProviderType]:
        return self._model_name, self._model_provider_type

    def _provider_cls(self):
        provider_cls = self._supported_providers.get(self._model_provider_type)
        if provider_cls is None:
            raise RuntimeError(
                f"Unsupported model provider type: {self._model_provider_type}"
            )
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
    def _build_model_accepts_inference_lock(build_model: object) -> bool:
        try:
            params = inspect.signature(build_model).parameters.values()
        except (TypeError, ValueError):
            return False

        return any(
            parameter.name == "inference_lock"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in params
        )

    def _acquire_shared_backend(
        self,
    ) -> tuple[tuple[str | None, ModelProviderType], _SharedBackendEntry]:
        cache_key = self._cache_key()
        provider_cls = self._provider_cls()
        with self._shared_lock:
            entry = self._shared_backends.get(cache_key)
            if entry is None:
                entry = _SharedBackendEntry(
                    backend=self._load_backend(),
                    provider_cls=provider_cls,
                )
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

    def _build_model_instance(self) -> tuple[object, tuple[str | None, ModelProviderType] | None]:
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

    def get(self, name: str = "model") -> object | None:
        if name not in self._allowed_model_attrs:
            raise AttributeError(name)

        with self._instance_lock:
            if name in self._instance_cache:
                return self._instance_cache[name]

            try:
                instance, cache_key = self._build_model_instance()
            except Exception as error:
                error_text = f"{type(error).__name__}: {error}"
                logger.warning(
                    "model_load_failed",
                    model_name=self._model_name,
                    model_provider_type=self._model_provider_type,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
                self._instance_cache[name] = None
                self._load_errors[name] = error_text
                self.__setattr__(name, None)
                return None

            self._instance_cache[name] = instance
            if cache_key is not None:
                self._shared_backend_keys[name] = cache_key
            self._load_errors.pop(name, None)
            self.__setattr__(name, instance)
            return instance

    def get_load_error(self, name: str = "model") -> str | None:
        if name not in self._allowed_model_attrs:
            raise AttributeError(name)

        with self._instance_lock:
            return self._load_errors.get(name)

    @contextmanager
    def session(self, name: str = "model") -> Iterator[object | None]:
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

    def __getattr__(self, name: str) -> object:
        if name in self._allowed_model_attrs:
            return self.get(name)
        raise AttributeError(name)
