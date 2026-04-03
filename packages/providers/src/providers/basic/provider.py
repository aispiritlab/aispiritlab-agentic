"""Basic OpenAI-compatible API provider.

Drop-in provider for any OpenAI-compatible endpoint (Groq, Together,
Fireworks, local servers, etc.).  Subclass and override
``model_provider_type`` / ``_config`` to create specialised variants
(see ``OpenAIProvider``).
"""

from __future__ import annotations

from threading import Lock
from typing import Any, ClassVar

from providers._proto import ProviderProto
from providers.api.api_model import ApiModel
from providers.api.http_client import HttpClient
from providers.basic.config import HttpProviderConfig
from providers.models.config import ModelConfig


class BasicProvider(ProviderProto):
    model_provider_type: ClassVar[str] = "basic"

    _config: ClassVar[HttpProviderConfig] = HttpProviderConfig()

    @classmethod
    def configure(
        cls,
        config: HttpProviderConfig | None = None,
        /,
        **kwargs: Any,
    ) -> None:
        cls._config = config if config is not None else HttpProviderConfig(**kwargs)

    @classmethod
    def load_backend(cls, model_name: str) -> HttpClient:
        del model_name
        return HttpClient(
            cls._config.base_url,
            api_key=cls._config.api_key,
            timeout=cls._config.timeout,
        )

    @classmethod
    def build_model(
        cls,
        backend: HttpClient,
        model_name: str,
        config: ModelConfig,
        *,
        inference_lock: Lock | None = None,
    ) -> ApiModel:
        return ApiModel(
            model_name,
            backend,
            config=config,
            inference_lock=inference_lock,
        )

    @classmethod
    def close_backend(cls, backend: HttpClient) -> None:
        backend.close()

    @classmethod
    def load(cls, model_name: str, config: ModelConfig) -> ApiModel:
        return cls.build_model(cls.load_backend(model_name), model_name, config)
