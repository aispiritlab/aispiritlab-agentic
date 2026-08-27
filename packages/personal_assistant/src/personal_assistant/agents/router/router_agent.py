from __future__ import annotations

from agentic.core_agent import CoreAgentic, CoreAgentResponse
from agentic.prompts import PromptTemplate, QwenPromptBuilder
from agentic_runtime.trace import create_tracer  # framework utility
from personal_assistant.settings import settings
from providers.models import ModelConfig
from registry import Prompts


class RouterAgent(CoreAgentic):
    _WELCOME_MESSAGE = (
        "Cześć! Mogę dodać, edytować, odczytać i listować notatki, wyszukiwać "
        "informacje semantycznie albo pomóc w podejmowaniu decyzji. Co chcesz zrobić?"
    )

    def __init__(self) -> None:
        super().__init__(
            model_id=settings.orchestration_model_name,
            prompt_builder=QwenPromptBuilder(external_prompt_name=Prompts.DECISION),
            tracer=create_tracer(enabled=True),
            config=ModelConfig(max_tokens=20, generation_mode="orchestration"),
            welcome_message=self._WELCOME_MESSAGE,
        )

    def call(self, message: str) -> str:
        response = self.respond(message)
        return response.output.replace("<think>", "").replace("</think>", "").strip()

    def route_response(
        self,
        message: str,
        available_workflows_summary: str,
    ) -> CoreAgentResponse:
        # Held across clear + respond: otherwise a concurrent turn can wipe the
        # history this call just prepared, or read the other turn's messages.
        with self.turn_lock:
            return self._route_response_locked(message, available_workflows_summary)

    def _route_response_locked(
        self,
        message: str,
        available_workflows_summary: str,
    ) -> CoreAgentResponse:
        self._agent.clear_history()
        prompt = PromptTemplate(
            template=(
                "Dostępni agenci:\n"
                "{available_workflows_summary}\n\n"
                "Wiadomość użytkownika: {message}\n"
            ),
            context_variables=["available_workflows_summary", "message"],
        )
        formatted = prompt.format(
            available_workflows_summary=available_workflows_summary,
            message=message,
        )
        response = self.respond(formatted)
        return CoreAgentResponse(
            result=response.result,
            output=response.output.replace("<think>", "").replace("</think>", "").strip(),
            tool_results=response.tool_results,
        )

    def route(self, message: str, available_workflows_summary: str) -> str:
        return self.route_response(message, available_workflows_summary).output.strip()
