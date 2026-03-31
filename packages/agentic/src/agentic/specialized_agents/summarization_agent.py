from __future__ import annotations

from agentic.core_agent import CoreAgentic
from agentic.metadata import Description
from agentic.models import ModelConfig
from agentic.prompts import QwenPromptBuilder
from agentic.providers.provider import ModelProviderType

_SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a summarization specialist. "
    "Given a piece of text, produce a concise summary that preserves all key facts, "
    "entities, and actionable details. "
    "Keep the summary clear and well-structured. Do not add information that is not in the source text."
)


class SummarizationAgent(CoreAgentic):
    """Produces concise summaries of input text."""

    description = Description(
        agent_name="summarizer",
        description="Summarizes text into concise outputs.",
        capabilities=("summarization", "text-processing"),
    )

    def __init__(
        self,
        model_id: str,
        *,
        model_provider_type: ModelProviderType = "mlx",
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            prompt_builder=QwenPromptBuilder(system_prompt=_SUMMARIZATION_SYSTEM_PROMPT),
            config=config or ModelConfig(max_tokens=512, generation_mode="nothinking"),
            model_provider_type=model_provider_type,
        )

    def summarize(self, text: str) -> str:
        """Summarize the given text.

        Args:
            text: The text to summarize.

        Returns:
            A concise summary.
        """
        self._agent.clear_history()
        return self.call(f"Summarize the following text:\n\n{text}")
