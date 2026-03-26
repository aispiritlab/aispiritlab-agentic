"""OpenAI-compatible API provider."""

from __future__ import annotations

from threading import Lock

from agentic.models.config import ModelConfig
from agentic.providers.api.api_model import ApiModel
from agentic.providers.api.http_client import HttpClient


class OpenAIProvider:
    model_provider_type = "openai"

    _base_url: str = "http://localhost:1234"
    _api_key: str | None = None
    _timeout: float = 120.0

    @classmethod
    def configure(
        cls,
        base_url: str = "http://localhost:1234",
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        cls._base_url = base_url
        cls._api_key = api_key
        cls._timeout = timeout

    @classmethod
    def load_backend(cls, model_name: str) -> HttpClient:
        del model_name
        return HttpClient(
            cls._base_url,
            api_key=cls._api_key,
            timeout=cls._timeout,
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
