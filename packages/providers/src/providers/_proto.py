from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Protocol, runtime_checkable

from providers.models.config import ModelConfig
from providers.models.response import ModelResponse


@runtime_checkable
class TextModel(Protocol):
    """A loaded model that answers a prompt."""

    def response(self, prompt: str | list[dict[str, str]], **kwargs: Any) -> ModelResponse: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Base provider configuration — marker type for all configs."""


@runtime_checkable
class ProviderProto(Protocol):
    @classmethod
    def load_backend(cls, model_name: str) -> object: ...

    @classmethod
    def build_model(
        cls,
        backend: object,
        model_name: str,
        config: ModelConfig,
        *,
        inference_lock: Lock | None = None,
    ) -> object: ...

    @classmethod
    def close_backend(cls, backend: object) -> None: ...

    @classmethod
    def load(cls, model_name: str, config: ModelConfig) -> object: ...
