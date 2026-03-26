"""Lightweight direct vision-language call using the MLX VLM provider."""

from __future__ import annotations

from typing import cast

from agentic.models import ModelConfig, ModelProvider
from agentic.models.response import ModelResponse
from agentic.observability import LLMTracer, NoopLLMTracer

_DEFAULT_SYSTEM_PROMPT = "Jesteś pomocnym asystentem AI. Odpowiadaj po polsku."


class VLMCall:
    """Direct VLM chat for uploaded image analysis."""

    def __init__(
        self,
        model_name: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        tracer: LLMTracer | None = None,
    ) -> None:
        self._model_provider = ModelProvider(
            model_name,
            model_provider_type="mlx-vlm",
            config=ModelConfig(max_tokens=max_tokens),
        )
        self._tracer = tracer or NoopLLMTracer()
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._history: list[dict[str, str]] = []

    def _build_prompt(self, message: str) -> list[dict[str, str]]:
        prompt: list[dict[str, str]] = []
        if self._system_prompt:
            prompt.append({"role": "system", "content": self._system_prompt})
        prompt.extend(self._history)
        prompt.append({"role": "user", "content": message})
        return prompt

    def respond(self, message: str, *, images: str | list[str] | None = None) -> ModelResponse:
        prompt = self._build_prompt(message)

        with self._model_provider.session("model") as model:
            if model is None:
                load_error = self._model_provider.get_load_error("model")
                if load_error:
                    raise RuntimeError(f"Model is not available for inference: {load_error}")
                raise RuntimeError("Model is not available for inference.")

            response = self._tracer.llm(
                name="vlm-call",
                model=getattr(self._model_provider, "_model_name", None) or "",
                messages=cast(list[dict[str, object]], prompt),
                invoke=lambda: model.response(prompt, image=images),
            )

        self._history.append({"role": "user", "content": message})
        self._history.append({"role": "assistant", "content": response.text})
        return response

    def call(self, message: str, *, images: str | list[str] | None = None) -> str:
        return self.respond(message, images=images).text

    def reset(self) -> None:
        self._history.clear()

    def close(self) -> None:
        self.reset()
        self._model_provider.close()
