"""vLLM model provider with configurable inference strategies."""

from __future__ import annotations

from threading import Lock
from typing import Any, ClassVar

from providers._proto import ProviderProto
from providers.models.config import ModelConfig
from providers.vllm.config import VllmProviderConfig
from providers.vllm.vllm_model import VllmNativeModel, VllmOpenAIModel, VllmRayModel


class VLLMProvider(ProviderProto):
    model_provider_type: ClassVar[str] = "vllm"

    _config: ClassVar[VllmProviderConfig] = VllmProviderConfig()

    @classmethod
    def configure(
        cls,
        config: VllmProviderConfig | None = None,
        /,
        **kwargs: Any,
    ) -> None:
        cls._config = config if config is not None else VllmProviderConfig(**kwargs)

    @classmethod
    def load_backend(cls, model_name: str) -> object:
        match cls._config.inference_strategy:
            case "native":
                return cls._load_native_backend(model_name)
            case "openai_compatible":
                return cls._load_openai_backend()
            case "ray":
                return cls._load_ray_backend(model_name)
            case "cli":
                raise NotImplementedError("CLI inference is in progress.")

    @classmethod
    def _load_native_backend(cls, model_name: str) -> object:
        from vllm import LLM

        return LLM(
            model=model_name,
            tensor_parallel_size=cls._config.tensor_parallel_size,
            gpu_memory_utilization=cls._config.gpu_memory_utilization,
        )

    @classmethod
    def _load_openai_backend(cls) -> object:
        from openai import OpenAI

        return OpenAI(
            base_url=cls._config.base_url,
            api_key=cls._config.api_key,
        )

    @classmethod
    def _load_ray_backend(cls, model_name: str) -> object:
        from ray.data.llm import vLLMEngineProcessorConfig, build_processor

        config = vLLMEngineProcessorConfig(
            model_source=model_name,
            engine_kwargs={"tensor_parallel_size": cls._config.tensor_parallel_size},
        )
        return build_processor(
            config,
            preprocess=lambda row: dict(
                messages=[{"role": "user", "content": row["prompt"]}],
                sampling_params={"max_tokens": 512},
            ),
            postprocess=lambda row: dict(generated_text=row["generated_text"]),
        )

    @classmethod
    def build_model(
        cls,
        backend: object,
        model_name: str,
        config: ModelConfig,
        *,
        inference_lock: Lock | None = None,
    ) -> VllmNativeModel | VllmOpenAIModel | VllmRayModel:
        match cls._config.inference_strategy:
            case "native":
                return VllmNativeModel(
                    backend, model_name, config=config, inference_lock=inference_lock
                )
            case "openai_compatible":
                return VllmOpenAIModel(
                    model_name, backend, config=config, inference_lock=inference_lock
                )
            case "ray":
                return VllmRayModel(
                    backend, model_name, config=config, inference_lock=inference_lock
                )
            case "cli":
                raise NotImplementedError("CLI inference is in progress.")

    @classmethod
    def close_backend(cls, backend: object) -> None:
        match cls._config.inference_strategy:
            case "native":
                del backend
            case "openai_compatible":
                close = getattr(backend, "close", None)
                if callable(close):
                    close()
            case "ray":
                del backend
            case "cli":
                pass

    @classmethod
    def load(
        cls, model_name: str, config: ModelConfig
    ) -> VllmNativeModel | VllmOpenAIModel | VllmRayModel:
        return cls.build_model(cls.load_backend(model_name), model_name, config)
