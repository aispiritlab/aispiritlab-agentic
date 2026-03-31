"""HuggingFace Transformers model provider."""

from __future__ import annotations

from threading import Lock

from agentic.models.config import ModelConfig
from agentic.providers.transformers.transformers_model import TransformersModel


class TransformersProvider:
    model_provider_type = "transformers"

    _device_map: str = "auto"
    _torch_dtype: str | None = None
    _quantization: str | None = None

    @classmethod
    def configure(
        cls,
        device_map: str = "auto",
        torch_dtype: str | None = None,
        quantization: str | None = None,
    ) -> None:
        cls._device_map = device_map
        cls._torch_dtype = torch_dtype
        cls._quantization = quantization

    @classmethod
    def _resolve_torch_dtype(cls) -> object | None:
        if cls._torch_dtype is None:
            return None
        import torch

        dtype_map: dict[str, object] = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(cls._torch_dtype)

    @classmethod
    def _build_quantization_config(cls) -> object | None:
        if cls._quantization is None:
            return None
        from transformers import BitsAndBytesConfig

        match cls._quantization:
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

        model_kwargs: dict[str, object] = {"device_map": cls._device_map}

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
        return TransformersModel(
            backend, model_name, config=config, inference_lock=inference_lock
        )

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
