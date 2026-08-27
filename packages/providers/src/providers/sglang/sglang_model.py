"""Model wrappers for SGLang inference strategies."""

from __future__ import annotations

from threading import Lock
import time
from typing import Any

from structlog import get_logger

from providers.models.config import DEFAULT_MODEL_CONFIG, ModelConfig
from providers.models.response import ModelResponse

logger = get_logger(__name__)


class SglangNativeModel:
    def __init__(
        self,
        engine: object,
        model_name: str,
        config: ModelConfig = DEFAULT_MODEL_CONFIG,
        *,
        inference_lock: Lock | None = None,
    ) -> None:
        self._engine = engine
        self._model_name = model_name
        self._config = config
        self._inference_lock = inference_lock or Lock()

    def _build_sampling_params(self, **kwargs: Any) -> dict[str, Any]:
        sampling = self._config.sampling_profile
        params: dict[str, Any] = {
            "max_new_tokens": self._config.max_tokens,
            "temperature": sampling.temperature if sampling else 0.7,
            "top_p": sampling.top_p if sampling else 0.8,
            "top_k": int(sampling.top_k) if sampling and sampling.top_k > 0 else 0,
            "min_p": sampling.min_p if sampling else 0.0,
            "repetition_penalty": sampling.repetition_penalty if sampling else 1.0,
            "presence_penalty": sampling.presence_penalty if sampling else 0.0,
        }
        for key in (
            "frequency_penalty",
            "json_schema",
            "regex",
            "ebnf",
            "stop",
            "stop_token_ids",
            "n",
        ):
            if key in kwargs:
                params[key] = kwargs.pop(key)
        return params

    def response(self, prompt: str | list[dict[str, str]], **kwargs: Any) -> ModelResponse:
        if self._engine is None:
            raise RuntimeError("Engine has been closed.")

        sampling_params = self._build_sampling_params(**kwargs)

        with self._inference_lock:
            logger.debug("sglang_native_request", model=self._model_name)
            started = time.monotonic()

            if isinstance(prompt, list):
                output = self._engine.generate(
                    prompt=[prompt],
                    sampling_params=sampling_params,
                )
            else:
                output = self._engine.generate(
                    prompt=prompt,
                    sampling_params=sampling_params,
                )

            latency_ms = round((time.monotonic() - started) * 1000, 2)

        text = output.get("text", "").strip()
        meta = output.get("meta_info", {})
        prompt_tokens = meta.get("prompt_tokens", 0)
        completion_tokens = meta.get("completion_tokens", 0)
        finish_reason = meta.get("finish_reason", {})
        if isinstance(finish_reason, dict):
            finish_reason = finish_reason.get("type", "stop")

        return ModelResponse(
            text=text,
            model=self._model_name,
            request_id=meta.get("id", ""),
            finish_reason=str(finish_reason),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
        )

    def close(self) -> None:
        self._engine = None


class SglangOpenAIModel:
    def __init__(
        self,
        model_name: str,
        client: object,
        config: ModelConfig = DEFAULT_MODEL_CONFIG,
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
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": m["role"], "content": m["content"]} for m in prompt]

        sampling = self._config.sampling_profile

        create_kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "temperature": sampling.temperature if sampling else 0.7,
            "top_p": sampling.top_p if sampling else 0.8,
            "max_tokens": self._config.max_tokens,
            "presence_penalty": sampling.presence_penalty if sampling else 0.0,
        }

        extra_body: dict[str, Any] = {}
        if sampling and sampling.repetition_penalty != 1.0:
            extra_body["repetition_penalty"] = sampling.repetition_penalty
        if sampling and sampling.min_p > 0.0:
            extra_body["min_p"] = sampling.min_p
        if sampling and sampling.top_k > 0:
            extra_body["top_k"] = sampling.top_k
        for key in (
            "frequency_penalty",
            "json_schema",
            "regex",
            "ebnf",
            "stop",
            "n",
            "min_p",
            "top_k",
            "repetition_penalty",
        ):
            if key in kwargs:
                extra_body[key] = kwargs.pop(key)
        if extra_body:
            create_kwargs["extra_body"] = extra_body

        with self._inference_lock:
            logger.debug("sglang_openai_request", model=self._model_name)
            started = time.monotonic()

            completion = self._client.chat.completions.create(**create_kwargs)

            latency_ms = round((time.monotonic() - started) * 1000, 2)

        text = ""
        finish_reason = ""
        if completion.choices:
            text = (completion.choices[0].message.content or "").strip()
            finish_reason = completion.choices[0].finish_reason or ""

        usage = completion.usage
        return ModelResponse(
            text=text,
            model=self._model_name,
            request_id=completion.id,
            finish_reason=finish_reason,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            latency_ms=latency_ms,
        )

    def close(self) -> None:
        self._client = None
