"""Inference provider management — model providers, orchestrator, and subprocess clients."""

from providers._platform import PlatformInfo, detect_platform
from providers._proto import ProviderConfig, ProviderProto
from providers._settings import ProviderSettings
from providers.basic import BasicProvider
from providers.basic.config import HttpProviderConfig
from providers.models.config import ModelConfig
from providers.models.response import ModelResponse
from providers.orchestrator import ModelProvider, ModelProviderType
from providers.schema import InferenceConfig, InferenceResponse
from providers.sglang import SGLangProvider

__all__ = [
    "BasicProvider",
    "HttpProviderConfig",
    "InferenceConfig",
    "InferenceResponse",
    "ModelConfig",
    "ModelProvider",
    "ModelProviderType",
    "ModelResponse",
    "PlatformInfo",
    "ProviderConfig",
    "ProviderProto",
    "ProviderSettings",
    "SGLangProvider",
    "detect_platform",
    "main",
]


def main() -> None:
    from providers._cli import run

    run()
