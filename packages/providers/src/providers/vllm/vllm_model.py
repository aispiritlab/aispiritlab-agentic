"""Model wrappers for vLLM inference strategies."""

from __future__ import annotations

from threading import Lock
import time
from typing import Any

from structlog import get_logger

from providers.models.config import DEFAULT_MODEL_CONFIG, ModelConfig
from providers.models.response import ModelResponse

logger = get_logger(__name__)


class VllmNativeModel:
    def __init__(
        self,
        backend: object,
        model_name: str,
        config: ModelConfig = DEFAULT_MODEL_CONFIG,
        *,
        inference_lock: Lock | None = None,
    ) -> None:
        self._llm = backend
        self._model_name = model_name
        self._config = config
        self._inference_lock = inference_lock or Lock()

    def _build_sampling_params(self) -> object:
        from vllm import SamplingParams

        sampling = self._config.sampling_profile
        return SamplingParams(
            temperature=sampling.temperature if sampling else 0.7,
            top_p=sampling.top_p if sampling else 0.8,
            top_k=int(sampling.top_k) if sampling and sampling.top_k > 0 else -1,
            min_p=sampling.min_p if sampling else 0.0,
            max_tokens=self._config.max_tokens,
            repetition_penalty=sampling.repetition_penalty if sampling else 1.0,
            presence_penalty=sampling.presence_penalty if sampling else 0.0,
        )

    def response(self, prompt: str | list[dict[str, str]], **kwargs: Any) -> ModelResponse:
        if self._llm is None:
            raise RuntimeError("Model has been closed.")

        sampling_params = kwargs.pop("sampling_params", None) or self._build_sampling_params()

        with self._inference_lock:
            logger.debug("vllm_native_request", model=self._model_name)
            started = time.monotonic()

            if isinstance(prompt, list):
                messages = [{"role": m["role"], "content": m["content"]} for m in prompt]
                outputs = self._llm.chat(messages=messages, sampling_params=sampling_params)
            else:
                outputs = self._llm.generate([prompt], sampling_params=sampling_params)

            latency_ms = round((time.monotonic() - started) * 1000, 2)

        text = outputs[0].outputs[0].text.strip()
        completion_tokens = len(outputs[0].outputs[0].token_ids)
        prompt_tokens = len(outputs[0].prompt_token_ids)

        return ModelResponse(
            text=text,
            model=self._model_name,
            request_id=outputs[0].request_id,
            finish_reason=str(outputs[0].outputs[0].finish_reason),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
        )

    def close(self) -> None:
        self._llm = None


class VllmOpenAIModel:
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
        for key in ("frequency_penalty", "stop", "n", "min_p", "top_k", "repetition_penalty"):
            if key in kwargs:
                extra_body[key] = kwargs.pop(key)
        if extra_body:
            create_kwargs["extra_body"] = extra_body

        with self._inference_lock:
            logger.debug("vllm_openai_request", model=self._model_name)
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


class VllmRayModel:
    def __init__(
        self,
        processor: object,
        model_name: str,
        config: ModelConfig = DEFAULT_MODEL_CONFIG,
        *,
        inference_lock: Lock | None = None,
        dataset_factory: object | None = None,
    ) -> None:
        self._processor = processor
        self._model_name = model_name
        self._config = config
        self._inference_lock = inference_lock or Lock()
        self._dataset_factory = dataset_factory

    def _create_dataset(self, items: list[dict[str, str]]) -> object:
        if self._dataset_factory is not None:
            return self._dataset_factory(items)
        import ray

        return ray.data.from_items(items)

    def response(self, prompt: str | list[dict[str, str]], **kwargs: Any) -> ModelResponse:
        if self._processor is None:
            raise RuntimeError("Ray processor has been closed.")

        if isinstance(prompt, list):
            text_prompt = prompt[-1]["content"]
        else:
            text_prompt = prompt

        with self._inference_lock:
            logger.debug("vllm_ray_request", model=self._model_name)
            started = time.monotonic()

            ds = self._create_dataset([{"prompt": text_prompt}])
            result_ds = self._processor(ds)
            results = result_ds.take_all()

            latency_ms = round((time.monotonic() - started) * 1000, 2)

        text = str(results[0].get("generated_text", "")).strip()

        return ModelResponse(
            text=text,
            model=self._model_name,
            latency_ms=latency_ms,
        )

    def close(self) -> None:
        self._processor = None
