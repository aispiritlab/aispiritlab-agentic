"""vLLM provider configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from providers.basic.config import HttpProviderConfig


@dataclass(frozen=True, slots=True)
class VllmProviderConfig(HttpProviderConfig):
    """vLLM provider configuration with per-strategy defaults."""

    base_url: str = "http://localhost:8000/v1"
    api_key: str | None = "token-abc123"
    inference_strategy: Literal["native", "openai_compatible", "ray", "cli"] = "native"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
