from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agentic.message import UserMessage
from agentic.prompts import GemmaPromptBuilder, QwenPromptBuilder
from agentic.tools import Toolsets
from evaluation import EvaluationDefinition, ToolScenario, serialize_scenarios_to_json

from personal_assistant.settings import settings

from .flows import DEFAULT_ORGANIZER_FLOWS
from .organizer_agent import OrganizerAgent
from .tools import toolset as organizer_toolset


ORGANIZER_TOOL_SCENARIOS: tuple[ToolScenario, ...] = (
    ToolScenario(
        name="classify_project_note",
        user_message=(
            "Nazwa notatki: Projekt Atlas\n"
            "Treść notatki:\n"
            "Plan sprintu, milestone i lista zadań do wdrożenia MVP w tym miesiącu.\n"
        ),
        tool_name="tag_note",
        parameters={"note_name": "Projekt Atlas", "tag": "para/project"},
    ),
    ToolScenario(
        name="classify_area_note",
        user_message=(
            "Nazwa notatki: Zdrowie i nawyki\n"
            "Treść notatki:\n"
            "Cotygodniowy plan treningowy i rutyna monitorowania zdrowia.\n"
        ),
        tool_name="tag_note",
        parameters={"note_name": "Zdrowie i nawyki", "tag": "para/area"},
    ),
    ToolScenario(
        name="classify_resource_note",
        user_message=(
            "Nazwa notatki: Notatki o RAG\n"
            "Treść notatki:\n"
            "Zbiór linków, definicji i porównań strategii retrieval oraz chunkingu.\n"
        ),
        tool_name="tag_note",
        parameters={"note_name": "Notatki o RAG", "tag": "para/resource"},
    ),
    ToolScenario(
        name="classify_archive_note",
        user_message=(
            "Nazwa notatki: Projekt CRM 2023\n"
            "Treść notatki:\n"
            "Projekt zamknięty. Podsumowanie lessons learned i decyzja o archiwizacji.\n"
        ),
        tool_name="tag_note",
        parameters={"note_name": "Projekt CRM 2023", "tag": "para/archive"},
    ),
    ToolScenario(
        name="classify_ambiguous_note",
        user_message=(
            "Nazwa notatki: Pomysły na automatyzacje\n"
            "Treść notatki:\n"
            "Luźne inspiracje i checklisty do wykorzystania w przyszłości.\n"
        ),
        tool_name="tag_note",
        parameters={"note_name": "Pomysły na automatyzacje", "tag": "para/resource"},
    ),
)


class OrganizerToolResultSimulator:
    def __init__(self) -> None:
        self._tags_by_note: dict[str, set[str]] = {}

    def apply(self, tool_name: str, parameters: Mapping[str, Any]) -> str:
        if tool_name != "tag_note":
            return f"Error: tool '{tool_name}' does not exist."

        note_name = str(parameters.get("note_name", "")).strip()
        tag = str(parameters.get("tag", "")).strip().lstrip("#")
        if not note_name:
            return "Brak nazwy notatki."
        if not tag:
            return "Brak tagu do dodania."

        existing = self._tags_by_note.setdefault(note_name, set())
        if tag in existing:
            return f"Notatka {note_name} ma już tag {tag}."

        existing.add(tag)
        return f"Notatka {note_name} otagowana jako {tag}."


class OrganizerEvalCallback:
    def __init__(self, runtime_options: Mapping[str, Any] | None = None) -> None:
        self._runtime_options = dict(runtime_options or {})
        self._agent = OrganizerAgent(
            model_id=settings.model_name,
            prompt_builder=QwenPromptBuilder(system_prompt=_load_organizer_prompt()),
            toolsets=Toolsets([organizer_toolset]),
        )
        self._original_prompt_builder = self._agent._agent.system_prompt

    @property
    def _inner_agent(self):
        return self._agent._agent

    def reset(self) -> None:
        self._agent.reset()

    def prime(self, messages: Sequence[str]) -> None:
        for message in messages:
            if isinstance(message, str) and message.strip():
                self.run(message)

    def run(self, user_input: str, prompt_text: str | None = None) -> str:
        agent = self._inner_agent
        if prompt_text is not None:
            agent.system_prompt = GemmaPromptBuilder(system_prompt=prompt_text)
        else:
            agent.system_prompt = self._original_prompt_builder

        history_text = (
            agent._history.conversation_text(agent._context.max_history_messages)
            if agent._context.add_history_to_context
            else ""
        )
        message_with_history = "\n".join(
            turn
            for turn in [history_text, UserMessage(user_input).as_turn()]
            if turn
        )
        prompt = agent.system_prompt.build_prompt(
            message_with_history,
            toolsets=agent._toolsets,
        )
        with agent._model_provider.session("model") as model:
            if model is None:
                raise RuntimeError("Model is not available for inference.")
            response = model.response(prompt).text
        agent._history.store(user_input, response)
        return response

    def close(self) -> None:
        self._inner_agent.system_prompt = self._original_prompt_builder
        self._agent.reset()


def _load_organizer_prompt() -> str:
    from registry.prompts import ORGANIZER_PROMPT

    return ORGANIZER_PROMPT


def _create_organizer_callback(
    runtime_options: Mapping[str, Any] | None = None,
) -> OrganizerEvalCallback:
    return OrganizerEvalCallback(runtime_options)


ORGANIZER_EVALUATION = EvaluationDefinition(
    name="organizer",
    scenarios=ORGANIZER_TOOL_SCENARIOS,
    flows=DEFAULT_ORGANIZER_FLOWS,
    prompt_loader=_load_organizer_prompt,
    agent_callback_factory=_create_organizer_callback,
    conversation_simulator_factory=OrganizerToolResultSimulator,
    assistant_greeting=(
        "Cześć! Sklasyfikuję notatkę metodą PARA i dodam odpowiedni tag."
    ),
    scenarios_example=serialize_scenarios_to_json(ORGANIZER_TOOL_SCENARIOS),
)


__all__ = [
    "ORGANIZER_EVALUATION",
    "ORGANIZER_TOOL_SCENARIOS",
    "OrganizerEvalCallback",
    "OrganizerToolResultSimulator",
]
