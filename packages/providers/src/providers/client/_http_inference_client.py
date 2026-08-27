"""Lightweight httpx-based inference client (no openai dependency)."""

from __future__ import annotations

import time
from typing import Any

from structlog import get_logger

from providers.api.http_client import HttpClient
from providers.api.openai_schema import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)
from providers.schema.config import InferenceConfig
from providers.schema.response import InferenceResponse

logger = get_logger(__name__)


class HttpInferenceClient:
    """Uses HttpClient + openai_schema dataclasses. No openai dependency."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        model: str = "default",
        config: InferenceConfig | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._config = config or InferenceConfig()
        self._client = HttpClient(base_url, api_key=api_key, timeout=timeout)

    def _build_messages(self, prompt: str | list[dict[str, str]]) -> list[ChatMessage]:
        if isinstance(prompt, str):
            return [ChatMessage(role="user", content=prompt)]
        return [ChatMessage(role=m["role"], content=m["content"]) for m in prompt]

    def chat(self, prompt: str | list[dict[str, str]], **kwargs: Any) -> InferenceResponse:
        messages = self._build_messages(prompt)
        logger.debug("http_chat_request", model=self._model, messages_count=len(messages))

        request = ChatCompletionRequest(
            model=kwargs.pop("model", self._model),
            messages=messages,
            temperature=kwargs.pop("temperature", self._config.temperature),
            top_p=kwargs.pop("top_p", self._config.top_p),
            max_tokens=kwargs.pop("max_tokens", self._config.max_tokens),
            presence_penalty=kwargs.pop("presence_penalty", self._config.presence_penalty),
        )

        started = time.monotonic()
        data = self._client.post("/v1/chat/completions", data=request.to_dict())
        latency_ms = round((time.monotonic() - started) * 1000, 2)

        resp = ChatCompletionResponse.from_dict(data)

        text = ""
        finish_reason = ""
        if resp.choices:
            text = resp.choices[0].message.content.strip()
            finish_reason = resp.choices[0].finish_reason

        return InferenceResponse(
            text=text,
            model=self._model,
            request_id=resp.id,
            finish_reason=finish_reason,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            total_tokens=resp.usage.total_tokens,
            latency_ms=latency_ms,
        )

    def close(self) -> None:
        self._client.close()
