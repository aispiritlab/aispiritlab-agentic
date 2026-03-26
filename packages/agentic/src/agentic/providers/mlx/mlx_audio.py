from threading import Lock

from agentic.models._models import VoiceModel
from agentic.models.config import ModelConfig
from agentic.providers.mlx.memory import clear_mlx_cache


class MlxAudioProvider:
    model_provider_type = "mlx-audio"

    @classmethod
    def load_backend(cls, model_name: str) -> object:
        from mlx_audio.stt.utils import load_model

        return load_model(model_name)

    @classmethod
    def build_model(
        cls,
        backend: object,
        model_name: str,
        config: ModelConfig,
        *,
        inference_lock: Lock | None = None,
    ) -> VoiceModel:
        del model_name
        return VoiceModel(backend, config=config, inference_lock=inference_lock)

    @classmethod
    def close_backend(cls, backend: object) -> None:
        del backend
        clear_mlx_cache()

    @classmethod
    def load(cls, model_name: str, config: ModelConfig) -> VoiceModel:
        return cls.build_model(cls.load_backend(model_name), model_name, config)
