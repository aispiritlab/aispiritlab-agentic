"""ONNX Runtime ASR provider."""

from providers.onnx.provider import OnnxProvider

OnnxAsrProvider = OnnxProvider  # backward-compat alias

__all__ = ["OnnxAsrProvider", "OnnxProvider"]
