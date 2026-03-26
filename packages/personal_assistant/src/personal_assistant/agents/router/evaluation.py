from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agentic.models import ModelConfig, ModelProvider
from agentic.prompts import GemmaPromptBuilder, PromptTemplate, QwenPromptBuilder
from evaluation import EvaluationDefinition, ToolScenario, serialize_scenarios_to_json

from personal_assistant.settings import settings

from ..discovery_notes.detective_agent import DiscoveryNotesAgent
from ..manage_notes.manage_notes_agent import ManageNotesAgent
from ..personalize.personalize_agent import PersonalizeAgent
from ..sage.sage_agent import SageAgent
from .flows import DEFAULT_ROUTER_FLOWS


ROUTER_TOOL_SCENARIOS: tuple[ToolScenario, ...] = (
    ToolScenario(
        name="route_personalize_name",
        user_message="Mam na imię Mateusz.",
        expected_output="personalize",
    ),
    ToolScenario(
        name="route_personalize_vault",
        user_message="Mój vault w Obsidianie nazywa się SecondBrain.",
        expected_output="personalize",
    ),
    ToolScenario(
        name="route_manage_notes_add",
        user_message="Dodaj notatkę o nazwie Zakupy z treścią Mleko i chleb.",
        expected_output="manage_notes",
    ),
    ToolScenario(
        name="route_manage_notes_read",
        user_message="Odczytaj notatkę Projekt Atlas.",
        expected_output="manage_notes",
    ),
    ToolScenario(
        name="route_manage_notes_list",
        user_message="Pokaż wszystkie notatki.",
        expected_output="manage_notes",
    ),
    ToolScenario(
        name="route_discovery_search",
        user_message="Znajdź w moich notatkach informacje o RAG i embeddings.",
        expected_output="discovery_notes",
    ),
    ToolScenario(
        name="route_sage_decision",
        user_message=(
            "Pomóż mi zdecydować, czy zatrudnić freelancera, czy "
            "budować zespół wewnętrzny."
        ),
        expected_output="sage",
    ),
    ToolScenario(
        name="route_sage_house_purchase",
        user_message="Chcialbym zakupic dom i potrzebuje pomocy",
        expected_output="sage",
    ),
    ToolScenario(
        name="route_sage_house_vs_flat",
        user_message="Nie wiem czy kupić dom czy mieszkanie, pomóż podjąć decyzję.",
        expected_output="sage",
    ),
    ToolScenario(
        name="route_sage_compare_options",
        user_message="Porównaj opcje leasingu i zakupu auta.",
        expected_output="sage",
    ),
)


def _default_available_workflows_summary() -> str:
    specs = (
        ("personalize", "PersonalizeWorkflow", PersonalizeAgent.description),
        ("manage_notes", "ManageNotesWorkflow", ManageNotesAgent.description),
        ("discovery_notes", "DiscoveryNotesWorkflow", DiscoveryNotesAgent.description),
        ("sage", "SageWorkflow", SageAgent.description),
    )
    lines = []
    for agent_name, workflow_name, description in specs:
        capabilities = ", ".join(description.capabilities)
        lines.append(
            f"- {agent_name} (workflow: {workflow_name}): "
            f"{description.description} (capabilities: {capabilities})"
        )
    return "\n".join(lines)


class RouterEvalCallback:
    _ROUTE_TEMPLATE = PromptTemplate(
        template=(
            "Dostępni agenci:\n"
            "{available_workflows_summary}\n\n"
            "Wiadomość użytkownika: {message}\n"
        ),
        context_variables=["available_workflows_summary", "message"],
    )

    def __init__(self, runtime_options: Mapping[str, Any] | None = None) -> None:
        self._runtime_options = dict(runtime_options or {})
        self._prompt_builder = QwenPromptBuilder(system_prompt=_load_router_prompt())
        self._model_provider = ModelProvider(
            settings.orchestration_model_name,
            config=ModelConfig(max_tokens=20),
        )
        self._available_workflows_summary = str(
            self._runtime_options.get(
                "available_workflows_summary",
                _default_available_workflows_summary(),
            )
        )

    def reset(self) -> None:
        return None

    def prime(self, messages: Sequence[str]) -> None:
        return None

    def run(self, user_input: str, prompt_text: str | None = None) -> str:
        prompt_builder = (
            GemmaPromptBuilder(system_prompt=prompt_text)
            if prompt_text is not None
            else self._prompt_builder
        )
        route_message = self._ROUTE_TEMPLATE.format(
            available_workflows_summary=self._available_workflows_summary,
            message=user_input,
        )
        prompt = prompt_builder.build_prompt(route_message, toolsets=None)

        with self._model_provider.session("model") as model:
            if model is None:
                raise RuntimeError("Model is not available for inference.")
            response = model.response(prompt).text
        return response.replace("<think>", "").replace("</think>", "").strip()


def _load_router_prompt() -> str:
    from registry.prompts import DECISION_PROMPT

    return DECISION_PROMPT


def _create_router_callback(
    runtime_options: Mapping[str, Any] | None = None,
) -> RouterEvalCallback:
    return RouterEvalCallback(runtime_options)


ROUTER_EVALUATION = EvaluationDefinition(
    name="router",
    scenarios=ROUTER_TOOL_SCENARIOS,
    flows=DEFAULT_ROUTER_FLOWS,
    prompt_loader=_load_router_prompt,
    agent_callback_factory=_create_router_callback,
    assistant_greeting=(
        "Cześć! Mogę dodać, edytować, odczytać i listować notatki, wyszukiwać "
        "informacje semantycznie albo pomóc w podejmowaniu decyzji. Co chcesz zrobić?"
    ),
    scenarios_example=serialize_scenarios_to_json(ROUTER_TOOL_SCENARIOS),
)


__all__ = [
    "ROUTER_EVALUATION",
    "ROUTER_TOOL_SCENARIOS",
    "RouterEvalCallback",
]
