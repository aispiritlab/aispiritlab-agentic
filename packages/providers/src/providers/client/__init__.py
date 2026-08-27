"""Client layer for communicating with inference servers."""

from __future__ import annotations

from typing import Any, Literal

from providers.client._http_inference_client import HttpInferenceClient
from providers.client._openai_client import OpenAIInferenceClient
from providers.schema.config import InferenceConfig
from providers.schema.response import InferenceResponse

type ClientBackend = Literal["openai", "httpx"]

__all__ = [
    "ClientBackend",
    "HttpInferenceClient",
    "OpenAIInferenceClient",
    "ProviderClient",
]


class ProviderClient:
    """Factory that creates the appropriate client based on backend choice."""

    def __init__(
        self,
        base_url: str,
        *,
        backend: ClientBackend = "openai",
        api_key: str = "no-key",
        model: str = "default",
        config: InferenceConfig | None = None,
    ) -> None:
        if backend not in ("openai", "httpx"):
            raise ValueError(f"Unsupported backend: {backend}")
        config = config or InferenceConfig()
        self._backend_type = backend
        if backend == "openai":
            self._openai: OpenAIInferenceClient | None = OpenAIInferenceClient(
                base_url,
                api_key=api_key,
                model=model,
                config=config,
            )
            self._http: HttpInferenceClient | None = None
        else:
            self._openai = None
            self._http = HttpInferenceClient(
                base_url,
                api_key=api_key,
                model=model,
                config=config,
            )

    def chat(self, prompt: str | list[dict[str, str]], **kwargs: Any) -> InferenceResponse:
        if self._openai is not None:
            return self._openai.chat(prompt, **kwargs)
        if self._http is None:
            raise RuntimeError("ProviderClient has no configured backend.")
        return self._http.chat(prompt, **kwargs)

    async def achat(self, prompt: str | list[dict[str, str]], **kwargs: Any) -> InferenceResponse:
        if self._openai is not None:
            return await self._openai.achat(prompt, **kwargs)
        raise NotImplementedError("Async chat is only supported with the 'openai' backend")

    def close(self) -> None:
        if self._openai is not None:
            self._openai.close()
        if self._http is not None:
            self._http.close()

    async def aclose(self) -> None:
        if self._openai is not None:
            await self._openai.aclose()
        if self._http is not None:
            self._http.close()
