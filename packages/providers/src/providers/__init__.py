"""Inference provider management — model providers, orchestrator, and subprocess clients."""

from providers._platform import PlatformInfo, detect_platform
from providers._proto import ProviderProto
from providers._settings import ProviderSettings
from providers.models.config import ModelConfig
from providers.models.response import ModelResponse
from providers.orchestrator import ModelProvider, ModelProviderType
from providers.schema import InferenceConfig, InferenceResponse

__all__ = [
    "InferenceConfig",
    "InferenceResponse",
    "ModelConfig",
    "ModelProvider",
    "ModelProviderType",
    "ModelResponse",
    "PlatformInfo",
    "ProviderProto",
    "ProviderSettings",
    "detect_platform",
    "main",
]


def main() -> None:
    from providers._cli import run

    run()
