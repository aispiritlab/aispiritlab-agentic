"""ONNX Runtime ASR provider."""

from __future__ import annotations

from threading import Lock
from typing import ClassVar

from providers._proto import ProviderProto
from providers.models.config import ModelConfig


class OnnxProvider(ProviderProto):
    model_provider_type: ClassVar[str] = "onnx"

    @classmethod
    def load_backend(cls, model_name: str) -> object:
        import onnx_asr

        return onnx_asr.load_model(model_name, quantization="int8")

    @classmethod
    def build_model(
        cls,
        backend: object,
        model_name: str,
        config: ModelConfig,
        *,
        inference_lock: Lock | None = None,
    ) -> object:
        del model_name, config, inference_lock
        return backend

    @classmethod
    def close_backend(cls, backend: object) -> None:
        close = getattr(backend, "close", None)
        if callable(close):
            close()

    @classmethod
    def load(cls, model_name: str, config: ModelConfig) -> object:
        return cls.build_model(cls.load_backend(model_name), model_name, config)
