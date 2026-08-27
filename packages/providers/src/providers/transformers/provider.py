"""HuggingFace Transformers model provider."""

from __future__ import annotations

from threading import Lock
from typing import Any, ClassVar

from providers._proto import ProviderProto
from providers.models.config import ModelConfig
from providers.transformers.config import TransformersProviderConfig
from providers.transformers.transformers_model import TransformersModel


class TransformersProvider(ProviderProto):
    model_provider_type: ClassVar[str] = "transformers"

    _config: ClassVar[TransformersProviderConfig] = TransformersProviderConfig()

    @classmethod
    def configure(
        cls,
        config: TransformersProviderConfig | None = None,
        /,
        **kwargs: Any,
    ) -> None:
        cls._config = config if config is not None else TransformersProviderConfig(**kwargs)

    @classmethod
    def _resolve_torch_dtype(cls) -> object | None:
        if cls._config.torch_dtype is None:
            return None
        import torch

        dtype_map: dict[str, object] = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(cls._config.torch_dtype)

    @classmethod
    def _build_quantization_config(cls) -> object | None:
        if cls._config.quantization is None:
            return None
        from transformers import BitsAndBytesConfig

        match cls._config.quantization:
            case "4bit":
                return BitsAndBytesConfig(load_in_4bit=True)
            case "8bit":
                return BitsAndBytesConfig(load_in_8bit=True)
            case _:
                return None

    @classmethod
    def load_backend(cls, model_name: str) -> tuple[object, object]:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, object] = {"device_map": cls._config.device_map}

        torch_dtype = cls._resolve_torch_dtype()
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        quantization_config = cls._build_quantization_config()
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        return model, tokenizer

    @classmethod
    def build_model(
        cls,
        backend: tuple[object, object],
        model_name: str,
        config: ModelConfig,
        *,
        inference_lock: Lock | None = None,
    ) -> TransformersModel:
        return TransformersModel(backend, model_name, config=config, inference_lock=inference_lock)

    @classmethod
    def close_backend(cls, backend: tuple[object, object]) -> None:
        del backend
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @classmethod
    def load(cls, model_name: str, config: ModelConfig) -> TransformersModel:
        return cls.build_model(cls.load_backend(model_name), model_name, config)
