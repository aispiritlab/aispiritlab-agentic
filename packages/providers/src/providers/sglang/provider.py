"""SGLang model provider with configurable inference strategies."""

from __future__ import annotations

from threading import Lock
from typing import Any, ClassVar

from providers._proto import ProviderProto
from providers.models.config import ModelConfig
from providers.sglang.config import SglangProviderConfig
from providers.sglang.sglang_model import SglangNativeModel, SglangOpenAIModel


class SGLangProvider(ProviderProto):
    model_provider_type: ClassVar[str] = "sglang"

    _config: ClassVar[SglangProviderConfig] = SglangProviderConfig()

    @classmethod
    def configure(
        cls,
        config: SglangProviderConfig | None = None,
        /,
        **kwargs: Any,
    ) -> None:
        cls._config = config if config is not None else SglangProviderConfig(**kwargs)

    @classmethod
    def load_backend(cls, model_name: str) -> object:
        match cls._config.inference_strategy:
            case "native":
                return cls._load_native_backend(model_name)
            case "openai_compatible":
                return cls._load_openai_backend()

    @classmethod
    def _load_native_backend(cls, model_name: str) -> object:
        from sglang import Engine

        return Engine(
            model_path=model_name,
            tp_size=cls._config.tp_size,
            mem_fraction_static=cls._config.mem_fraction_static,
        )

    @classmethod
    def _load_openai_backend(cls) -> object:
        from openai import OpenAI

        return OpenAI(
            base_url=cls._config.base_url,
            api_key=cls._config.api_key,
        )

    @classmethod
    def build_model(
        cls,
        backend: object,
        model_name: str,
        config: ModelConfig,
        *,
        inference_lock: Lock | None = None,
    ) -> SglangNativeModel | SglangOpenAIModel:
        match cls._config.inference_strategy:
            case "native":
                return SglangNativeModel(
                    backend, model_name, config=config, inference_lock=inference_lock
                )
            case "openai_compatible":
                return SglangOpenAIModel(
                    model_name, backend, config=config, inference_lock=inference_lock
                )

    @classmethod
    def close_backend(cls, backend: object) -> None:
        match cls._config.inference_strategy:
            case "native":
                shutdown = getattr(backend, "shutdown", None)
                if callable(shutdown):
                    shutdown()
            case "openai_compatible":
                close = getattr(backend, "close", None)
                if callable(close):
                    close()

    @classmethod
    def load(cls, model_name: str, config: ModelConfig) -> SglangNativeModel | SglangOpenAIModel:
        return cls.build_model(cls.load_backend(model_name), model_name, config)
