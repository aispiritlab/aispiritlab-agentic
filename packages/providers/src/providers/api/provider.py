"""OpenAI-compatible API provider (LM Studio default)."""

from __future__ import annotations

from typing import ClassVar

from providers.basic.config import HttpProviderConfig
from providers.basic.provider import BasicProvider


class OpenAIProvider(BasicProvider):
    model_provider_type: ClassVar[str] = "openai"

    _config: ClassVar[HttpProviderConfig] = HttpProviderConfig(base_url="http://localhost:1234")
