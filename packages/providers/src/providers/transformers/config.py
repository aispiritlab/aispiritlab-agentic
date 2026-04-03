"""Transformers provider configuration."""

from __future__ import annotations

from dataclasses import dataclass

from providers._proto import ProviderConfig


@dataclass(frozen=True, slots=True)
class TransformersProviderConfig(ProviderConfig):
    """HuggingFace Transformers provider configuration."""

    device_map: str = "auto"
    torch_dtype: str | None = None
    quantization: str | None = None
