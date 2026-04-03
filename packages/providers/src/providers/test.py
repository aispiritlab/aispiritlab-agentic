"""Test model provider for deterministic, LLM-free unit tests.

Example usage::

    from providers.test import TestModelProvider

    provider = TestModelProvider(responses=["Hello!", '{"name": "create_note", "parameters": {"title": "Test"}}'])
    with provider.session("model") as model:
        result = model.response("prompt")
        assert result.text == "Hello!"

Or with ``Agent``::

    agent = Agent(
        model_provider=TestModelProvider(responses=["I'll create that note for you."]),
        prompt_builder=my_builder,
    )
    result = agent.run("Create a note")
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from providers.models.response import ModelResponse


@dataclass(slots=True)
class TestModelConfig:
    """Configuration for the test model."""

    responses: list[str] = field(default_factory=lambda: ["Test response"])
    model_name: str = "test-model"
    prompt_tokens: int = 10
    completion_tokens: int = 20
    latency_ms: float = 1.0
    finish_reason: str = "stop"
    response_factory: Callable[[str | list[dict[str, str]]], str] | None = None


class _TestModel:
    """Model instance returned by TestModelProvider.session()."""

    def __init__(self, config: TestModelConfig) -> None:
        self._config = config
        self._call_count = 0
        self._call_history: list[str | list[dict[str, str]]] = []

    def response(self, prompt: str | list[dict[str, str]], **kwargs: Any) -> ModelResponse:
        self._call_history.append(prompt)

        if self._config.response_factory is not None:
            text = self._config.response_factory(prompt)
        else:
            idx = self._call_count % len(self._config.responses)
            text = self._config.responses[idx]

        self._call_count += 1
        total = self._config.prompt_tokens + self._config.completion_tokens
        return ModelResponse(
            text=text,
            model=self._config.model_name,
            prompt_tokens=self._config.prompt_tokens,
            completion_tokens=self._config.completion_tokens,
            total_tokens=total,
            latency_ms=self._config.latency_ms,
            finish_reason=self._config.finish_reason,
        )

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def call_history(self) -> list[str | list[dict[str, str]]]:
        return list(self._call_history)

    def close(self) -> None:
        pass


class TestModelProvider:
    """Drop-in replacement for ``ModelProvider`` in tests.

    Supports the same ``session()`` / ``get()`` / ``close()`` interface
    as the real provider, but returns deterministic responses without
    loading any model.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        config: TestModelConfig | None = None,
        response_factory: Callable[[str | list[dict[str, str]]], str] | None = None,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            cfg_kwargs: dict[str, Any] = {}
            if responses is not None:
                cfg_kwargs["responses"] = responses
            if response_factory is not None:
                cfg_kwargs["response_factory"] = response_factory
            self._config = TestModelConfig(**cfg_kwargs)

        self._model = _TestModel(self._config)
        self._model_name = self._config.model_name

    @contextmanager
    def session(self, name: str = "model") -> Iterator[_TestModel]:
        yield self._model

    def get(self, name: str = "model") -> _TestModel:
        return self._model

    def get_load_error(self, name: str = "model") -> str | None:
        return None

    @property
    def model(self) -> _TestModel:
        return self._model

    @property
    def call_count(self) -> int:
        return self._model.call_count

    @property
    def call_history(self) -> list[str | list[dict[str, str]]]:
        return self._model.call_history

    def close(self) -> None:
        self._model.close()
