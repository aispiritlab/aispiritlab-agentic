from __future__ import annotations

from agentic.prompts import ChatPromptBuilder, QwenPromptBuilder
from agentic.providers.provider import ModelProviderType


def build_specialized_prompt_builder(
    *,
    system_prompt: str,
    model_provider_type: ModelProviderType,
) -> ChatPromptBuilder | QwenPromptBuilder:
    if model_provider_type == "openai":
        return ChatPromptBuilder(system_prompt=system_prompt)
    return QwenPromptBuilder(system_prompt=system_prompt)
