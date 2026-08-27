"""OpenAI library client for local inference servers."""

from __future__ import annotations

import time
from typing import Any

from openai import AsyncOpenAI, OpenAI
from structlog import get_logger

from providers.schema.config import InferenceConfig
from providers.schema.response import InferenceResponse

logger = get_logger(__name__)


class OpenAIInferenceClient:
    """Uses openai.OpenAI / openai.AsyncOpenAI with base_url pointing to local server."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "no-key",
        model: str = "default",
        config: InferenceConfig | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._config = config or InferenceConfig()
        self._client = OpenAI(base_url=f"{self._base_url}/v1", api_key=api_key)
        self._async_client: AsyncOpenAI | None = None

    def _get_async_client(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                base_url=f"{self._base_url}/v1", api_key=self._api_key
            )
        return self._async_client

    def _build_messages(self, prompt: str | list[dict[str, str]]) -> list[dict[str, str]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return prompt

    def chat(self, prompt: str | list[dict[str, str]], **kwargs: Any) -> InferenceResponse:
        messages = self._build_messages(prompt)
        logger.debug("openai_chat_request", model=self._model, messages_count=len(messages))

        started = time.monotonic()
        response = self._client.chat.completions.create(
            model=kwargs.pop("model", self._model),
            messages=messages,  # type: ignore[arg-type]
            temperature=kwargs.pop("temperature", self._config.temperature),
            top_p=kwargs.pop("top_p", self._config.top_p),
            max_tokens=kwargs.pop("max_tokens", self._config.max_tokens),
            presence_penalty=kwargs.pop("presence_penalty", self._config.presence_penalty),
            **kwargs,
        )
        latency_ms = round((time.monotonic() - started) * 1000, 2)

        text = ""
        finish_reason = ""
        if response.choices:
            text = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or ""

        usage = response.usage
        return InferenceResponse(
            text=text.strip(),
            model=response.model,
            request_id=response.id,
            finish_reason=finish_reason,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            latency_ms=latency_ms,
        )

    async def achat(self, prompt: str | list[dict[str, str]], **kwargs: Any) -> InferenceResponse:
        messages = self._build_messages(prompt)
        logger.debug("openai_async_chat_request", model=self._model, messages_count=len(messages))

        started = time.monotonic()
        response = await self._get_async_client().chat.completions.create(
            model=kwargs.pop("model", self._model),
            messages=messages,  # type: ignore[arg-type]
            temperature=kwargs.pop("temperature", self._config.temperature),
            top_p=kwargs.pop("top_p", self._config.top_p),
            max_tokens=kwargs.pop("max_tokens", self._config.max_tokens),
            presence_penalty=kwargs.pop("presence_penalty", self._config.presence_penalty),
            **kwargs,
        )
        latency_ms = round((time.monotonic() - started) * 1000, 2)

        text = ""
        finish_reason = ""
        if response.choices:
            text = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or ""

        usage = response.usage
        return InferenceResponse(
            text=text.strip(),
            model=response.model,
            request_id=response.id,
            finish_reason=finish_reason,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            latency_ms=latency_ms,
        )

    def close(self) -> None:
        self._client.close()

    async def aclose(self) -> None:
        self._client.close()
        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None
