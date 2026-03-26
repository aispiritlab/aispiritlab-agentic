from threading import Lock

from agentic.models._models import VLModel
from agentic.models.config import ModelConfig
from agentic.providers.mlx.memory import clear_mlx_cache


class MlxVlmProvider:
    model_provider_type = "mlx-vlm"

    @classmethod
    def load_backend(cls, model_name: str) -> tuple[object, object]:
        from mlx_vlm import load

        return load(model_name)

    @classmethod
    def build_model(
        cls,
        backend: tuple[object, object],
        model_name: str,
        config: ModelConfig,
        *,
        inference_lock: Lock | None = None,
    ) -> VLModel:
        del model_name
        return VLModel(backend, config=config, inference_lock=inference_lock)

    @classmethod
    def close_backend(cls, backend: tuple[object, object]) -> None:
        del backend
        clear_mlx_cache()

    @classmethod
    def load(cls, model_name: str, config: ModelConfig) -> VLModel:
        return cls.build_model(cls.load_backend(model_name), model_name, config)
