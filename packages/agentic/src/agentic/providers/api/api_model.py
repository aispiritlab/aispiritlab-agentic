"""API-based model using OpenAI-compatible endpoints."""

from __future__ import annotations

import time
from threading import Lock
from typing import Any

from structlog import get_logger

from agentic.models.config import ModelConfig
from agentic.models.response import ModelResponse
from agentic.providers.api.http_client import HttpClient
from agentic.providers.api.openai_schema import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)

logger = get_logger(__name__)


class ApiModel:
    def __init__(
        self,
        model_name: str,
        client: HttpClient,
        config: ModelConfig = ModelConfig(),
        *,
        inference_lock: Lock | None = None,
    ) -> None:
        self._model_name = model_name
        self._client = client
        self._config = config
        self._inference_lock = inference_lock or Lock()

    def response(self, prompt: str | list[dict[str, str]], **kwargs: Any) -> ModelResponse:
        if self._client is None:
            raise RuntimeError("Model client has been closed.")

        if isinstance(prompt, str):
            messages = [ChatMessage(role="user", content=prompt)]
        else:
            messages = [ChatMessage(role=m["role"], content=m["content"]) for m in prompt]

        sampling = self._config.sampling_profile
        request = ChatCompletionRequest(
            model=self._model_name,
            messages=messages,
            temperature=sampling.temperature if sampling else 0.7,
            top_p=sampling.top_p if sampling else 0.8,
            max_tokens=self._config.max_tokens,
            presence_penalty=sampling.presence_penalty if sampling else 0.0,
        )

        with self._inference_lock:
            logger.debug("api_request", model=self._model_name, messages_count=len(messages))
            started = time.monotonic()
            data = self._client.post("/v1/chat/completions", data=request.to_dict())
            latency_ms = round((time.monotonic() - started) * 1000, 2)
            resp = ChatCompletionResponse.from_dict(data)

        text = ""
        finish_reason = ""
        if resp.choices:
            text = resp.choices[0].message.content.strip()
            finish_reason = resp.choices[0].finish_reason

        return ModelResponse(
            text=text,
            model=self._model_name,
            request_id=resp.id,
            finish_reason=finish_reason,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            total_tokens=resp.usage.total_tokens,
            latency_ms=latency_ms,
        )

    def close(self) -> None:
        self._client = None
