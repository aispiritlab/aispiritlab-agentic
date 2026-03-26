from agentic.models.config import ModelConfig


class OnnxAsrProvider:
    model_provider_type = "onnx"

    @classmethod
    def load_backend(cls, model_name: str):
        import onnx_asr

        return onnx_asr.load_model(model_name, quantization="int8")

    @classmethod
    def build_model(cls, backend: object, model_name: str, config: ModelConfig):
        del model_name, config
        return backend

    @classmethod
    def close_backend(cls, backend: object) -> None:
        close = getattr(backend, "close", None)
        if callable(close):
            close()

    @classmethod
    def load(cls, model_name: str, config: ModelConfig):
        return cls.build_model(cls.load_backend(model_name), model_name, config)


__all__ = ["OnnxAsrProvider"]
