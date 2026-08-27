from __future__ import annotations

from agentic.core_agent import CoreAgentic
from agentic.metadata import Description
from agentic.prompts import PromptTemplate
from agentic.specialized_agents._prompt_builders import build_specialized_prompt_builder
from providers.models import ModelConfig
from providers.orchestrator import ModelProviderType

_ROUTER_SYSTEM_PROMPT = (
    "You are a router. Given the list of available agents and the user message, "
    "output ONLY the agent name that should handle this request. "
    "Output nothing else — just the agent name."
)


class RouterAgent(CoreAgentic):
    """Routes messages to the best matching agent based on descriptions."""

    description = Description(
        agent_name="router",
        description="Routes messages to the appropriate agent.",
        capabilities=("routing",),
    )

    def __init__(
        self,
        model_id: str,
        *,
        model_provider_type: ModelProviderType = "mlx",
    ) -> None:
        super().__init__(
            model_id=model_id,
            prompt_builder=build_specialized_prompt_builder(
                system_prompt=_ROUTER_SYSTEM_PROMPT,
                model_provider_type=model_provider_type,
            ),
            config=ModelConfig(max_tokens=20, generation_mode="orchestration"),
            model_provider_type=model_provider_type,
        )

    def route(self, message: str, available_agents: str) -> str:
        """Route a message to the best matching agent.

        Args:
            message: The user message to route.
            available_agents: Formatted string listing available agents and their descriptions.

        Returns:
            The name of the agent that should handle this message.
        """
        self._agent.clear_history()
        prompt = PromptTemplate(
            template=("Available agents:\n{available_agents}\n\nUser message: {message}\n"),
            context_variables=["available_agents", "message"],
        )
        formatted = prompt.format(
            available_agents=available_agents,
            message=message,
        )
        response = self.respond(formatted)
        return response.output.replace("<think>", "").replace("</think>", "").strip()
