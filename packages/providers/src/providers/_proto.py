from __future__ import annotations

from threading import Lock
from typing import ClassVar, Protocol, runtime_checkable

from providers.models.config import ModelConfig


@runtime_checkable
class ProviderProto(Protocol):
    model_provider_type: ClassVar[str]

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
