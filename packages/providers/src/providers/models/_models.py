from __future__ import annotations

from pathlib import Path
from threading import Lock
import time
from typing import Any

from structlog import get_logger

from .config import ModelConfig
from .response import ModelResponse

logger = get_logger(__name__)


class Model:
    def __init__(
        self,
        provider: tuple[object, object],
        config: ModelConfig = ModelConfig(),
        *,
        inference_lock: Lock | None = None,
    ):
        self._model, self._tokenizer = provider
        self._inference_lock = inference_lock or Lock()
        self._config = config

    def response(self, prompt: str | list[dict[str, str]], **kwargs: Any) -> ModelResponse:
        from mlx_lm import generate

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model has been closed.")

        if isinstance(prompt, str):
            template_message = prompt
        else:
            template_message = self._tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                tokenize=False,
            )

        with self._inference_lock:
            logger.debug("prompt", prompt=template_message)
            started = time.monotonic()
            text = generate(
                self._model,
                self._tokenizer,
                prompt=template_message,
                verbose=False,
                max_tokens=self._config.max_tokens,
            ).strip()
            latency_ms = round((time.monotonic() - started) * 1000, 2)

        return ModelResponse(text=text, latency_ms=latency_ms)

    def close(self) -> None:
        self._model = None
        self._tokenizer = None


class VLModel:
    def __init__(
        self,
        provider: tuple[object, object],
        config: ModelConfig = ModelConfig(),
        *,
        inference_lock: Lock | None = None,
    ):
        self._model, self._tokenizer = provider
        self._inference_lock = inference_lock or Lock()
        self._config = config

    def response(self, prompt: str | list[dict[str, str]], **kwargs: Any) -> ModelResponse:
        from mlx_vlm import generate

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model has been closed.")

        if isinstance(prompt, str):
            template_message = prompt
        else:
            template_message = self._tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                tokenize=False,
            )

        with self._inference_lock:
            logger.debug("prompt", prompt=template_message)
            started = time.monotonic()
            result = generate(
                self._model,
                self._tokenizer,
                prompt=template_message,
                image=kwargs.get("image"),
                audio=kwargs.get("audio"),
                verbose=False,
                max_tokens=self._config.max_tokens,
            )
            text = (result.text if hasattr(result, "text") else str(result)).strip()
            latency_ms = round((time.monotonic() - started) * 1000, 2)

        return ModelResponse(text=text, latency_ms=latency_ms)

    def close(self) -> None:
        self._model = None
        self._tokenizer = None


class VoiceModel:
    def __init__(
        self,
        provider: object,
        config: ModelConfig = ModelConfig(),
        *,
        inference_lock: Lock | None = None,
    ):
        self._model = provider
        self._inference_lock = inference_lock or Lock()
        self._config = config

    def response(self, audio: str | Path | object, **kwargs: Any) -> str:
        if self._model is None:
            raise RuntimeError("Voice model has been closed.")

        with self._inference_lock:
            return self._model.generate(audio, language="pl", **kwargs)

    def close(self) -> None:
        self._model = None
