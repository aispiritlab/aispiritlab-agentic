from __future__ import annotations

from typing import Optional

from deepeval.models import LocalModel
from pydantic import BaseModel


class FixedLlamaCppModel(LocalModel):
    """Wrapper for llama.cpp (via Deepeval LocalModel) returning content only."""

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[BaseModel] = None,
    ) -> str | BaseModel:
        result = await super().a_generate(prompt, schema)
        if isinstance(result, tuple):
            return result[0]
        return result

    def generate(
        self,
        prompt: str,
        schema: Optional[BaseModel] = None,
    ) -> str | BaseModel:
        result = super().generate(prompt, schema)
        if isinstance(result, tuple):
            return result[0]
        return result
