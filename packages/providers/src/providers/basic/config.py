"""HTTP provider configuration shared by all OpenAI-compatible providers."""

from __future__ import annotations

from dataclasses import dataclass

from providers._proto import ProviderConfig


@dataclass(frozen=True, slots=True)
class HttpProviderConfig(ProviderConfig):
    """Configuration for HTTP-based providers."""

    base_url: str = "http://localhost:8080"
    api_key: str | None = None
    timeout: float = 120.0
