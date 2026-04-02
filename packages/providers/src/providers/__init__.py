"""Subprocess-based inference provider management."""

from providers._platform import PlatformInfo, detect_platform
from providers._settings import ProviderSettings
from providers.schema import InferenceConfig, InferenceResponse

__all__ = [
    "InferenceConfig",
    "InferenceResponse",
    "PlatformInfo",
    "ProviderSettings",
    "detect_platform",
    "main",
]


def main() -> None:
    from providers._cli import run

    run()
