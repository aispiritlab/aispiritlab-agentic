"""Lightweight direct-chat agent using the OpenAI-compatible API provider."""

from __future__ import annotations

from agentic.core_agent import CoreAgentResponse, CoreAgentic
from agentic.models.config import ModelConfig
from agentic.observability import LLMTracer
from agentic.prompts import ChatPromptBuilder

_DEFAULT_SYSTEM_PROMPT = "Jesteś pomocnym asystentem AI. Odpowiadaj po polsku."


class LLMCall:
    """Direct LLM chat via OpenAI-compatible API.

    Used as:
    - Fallback when the router can't match a workflow
    - Standalone chat mode bypassing the agent/workflow system
    """

    def __init__(
        self,
        model_name: str,
        *,
        system_prompt: str | None = None,
        external_prompt_name: str | None = None,
        max_tokens: int = 1024,
        tracer: LLMTracer | None = None,
    ) -> None:
        prompt_builder = ChatPromptBuilder(
            system_prompt=system_prompt or (None if external_prompt_name else _DEFAULT_SYSTEM_PROMPT),
            external_prompt_name=external_prompt_name,
        )
        self._agent = CoreAgentic(
            model_id=model_name,
            prompt_builder=prompt_builder,
            model_provider_type="openai",
            config=ModelConfig(max_tokens=max_tokens),
            tracer=tracer,
        )

    def respond(self, message: str) -> CoreAgentResponse:
        return self._agent.respond(message)

    def call(self, message: str) -> str:
        return self.respond(message).output

    def reset(self) -> None:
        self._agent.reset()

    def close(self) -> None:
        self.reset()
        self._agent.close()
