"""Model wrapper for HuggingFace Transformers inference."""

from __future__ import annotations

from threading import Lock
import time
from typing import Any

from structlog import get_logger

from providers.models.config import DEFAULT_MODEL_CONFIG, ModelConfig
from providers.models.response import ModelResponse

logger = get_logger(__name__)


class TransformersModel:
    def __init__(
        self,
        backend: tuple[object, object],
        model_name: str,
        config: ModelConfig = DEFAULT_MODEL_CONFIG,
        *,
        inference_lock: Lock | None = None,
    ) -> None:
        self._model, self._tokenizer = backend
        self._model_name = model_name
        self._config = config
        self._inference_lock = inference_lock or Lock()

    def response(self, prompt: str | list[dict[str, str]], **kwargs: Any) -> ModelResponse:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model has been closed.")

        if isinstance(prompt, list):
            text_input = self._tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            text_input = prompt

        with self._inference_lock:
            logger.debug("transformers_request", model=self._model_name)
            started = time.monotonic()

            inputs = self._tokenizer(
                [text_input],
                return_tensors="pt",
                padding=True,
            ).to(self._model.device)

            input_length = inputs.input_ids.shape[1]
            sampling = self._config.sampling_profile

            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._config.max_tokens,
                do_sample=True,
                temperature=sampling.temperature if sampling else 0.7,
                top_p=sampling.top_p if sampling else 0.8,
                top_k=int(sampling.top_k) if sampling and sampling.top_k > 0 else 50,
                repetition_penalty=sampling.repetition_penalty if sampling else 1.0,
            )

            new_tokens = generated_ids[:, input_length:]
            text = self._tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

            latency_ms = round((time.monotonic() - started) * 1000, 2)

        completion_tokens = new_tokens.shape[1]
        return ModelResponse(
            text=text,
            model=self._model_name,
            prompt_tokens=input_length,
            completion_tokens=completion_tokens,
            total_tokens=input_length + completion_tokens,
            latency_ms=latency_ms,
        )

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
