"""vLLM model provider with configurable inference strategies."""

from __future__ import annotations

from threading import Lock
from typing import Literal

from providers.models.config import ModelConfig
from providers.vllm.vllm_model import VllmNativeModel, VllmOpenAIModel, VllmRayModel

VllmInferenceStrategy = Literal["native", "openai_compatible", "ray", "cli"]


class VLLMProvider:
    model_provider_type = "vllm"

    _inference_strategy: VllmInferenceStrategy = "native"
    _base_url: str = "http://localhost:8000/v1"
    _api_key: str = "token-abc123"
    _tensor_parallel_size: int = 1
    _gpu_memory_utilization: float = 0.9

    @classmethod
    def configure(
        cls,
        inference_strategy: VllmInferenceStrategy = "native",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "token-abc123",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        cls._inference_strategy = inference_strategy
        cls._base_url = base_url
        cls._api_key = api_key
        cls._tensor_parallel_size = tensor_parallel_size
        cls._gpu_memory_utilization = gpu_memory_utilization

    @classmethod
    def load_backend(cls, model_name: str) -> object:
        match cls._inference_strategy:
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
            tensor_parallel_size=cls._tensor_parallel_size,
            gpu_memory_utilization=cls._gpu_memory_utilization,
        )

    @classmethod
    def _load_openai_backend(cls) -> object:
        from openai import OpenAI

        return OpenAI(
            base_url=cls._base_url,
            api_key=cls._api_key,
        )

    @classmethod
    def _load_ray_backend(cls, model_name: str) -> object:
        from ray.data.llm import vLLMEngineProcessorConfig, build_processor

        config = vLLMEngineProcessorConfig(
            model_source=model_name,
            engine_kwargs={"tensor_parallel_size": cls._tensor_parallel_size},
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
        match cls._inference_strategy:
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
        match cls._inference_strategy:
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
