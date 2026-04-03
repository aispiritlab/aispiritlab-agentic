"""SGLang provider configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from providers.basic.config import HttpProviderConfig


@dataclass(frozen=True, slots=True)
class SglangProviderConfig(HttpProviderConfig):
    """SGLang provider configuration with per-strategy defaults."""

    base_url: str = "http://localhost:30000/v1"
    api_key: str | None = "EMPTY"
    inference_strategy: Literal["native", "openai_compatible"] = "openai_compatible"
    tp_size: int = 1
    mem_fraction_static: float = 0.88
