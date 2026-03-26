from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agentic.message import UserMessage
from agentic.prompts import GemmaPromptBuilder
from agentic.prompts import QwenPromptBuilder
from agentic.tools import Toolsets
from evaluation import EvaluationDefinition, ToolScenario, serialize_scenarios_to_json

from personal_assistant.settings import settings

from .flows import DEFAULT_NOTE_FLOWS
from .manage_notes_agent import ManageNotesAgent
from .tools import toolset as manage_notes_toolset


NOTES_TOOL_SCENARIOS: tuple[ToolScenario, ...] = (
    ToolScenario(
        name="add_note",
        user_message="Dodaj notatkę o nazwie Zakupy z treścią Mleko i chleb.",
        tool_name="add_note",
        parameters={
            "note_name": "Zakupy",
            "note": "Mleko i chleb",
        },
    ),
    ToolScenario(
        name="read_note",
        user_message="Odczytaj notatkę o nazwie Zakupy.",
        tool_name="get_note",
        parameters={"note_name": "Zakupy"},
    ),
    ToolScenario(
        name="list_notes",
        user_message="Wyświetl listę moich notatek.",
        tool_name="list_notes",
        parameters={},
    ),
    ToolScenario(
        name="list_notes_no_diacritics",
        user_message="Wyswietl notatki",
        tool_name="list_notes",
        parameters={},
    ),
    ToolScenario(
        name="list_notes_typo",
        user_message="poakz notatki",
        tool_name="list_notes",
        parameters={},
    ),
    ToolScenario(
        name="read_note_no_diacritics",
        user_message="Pokaz notatke bron z tekstem",
        tool_name="get_note",
        parameters={"note_name": "bron z tekstem"},
    ),
    ToolScenario(
        name="read_note_simple",
        user_message="odczytaj notatke Mirek",
        tool_name="get_note",
        parameters={"note_name": "Mirek"},
    ),
    ToolScenario(
        name="add_note_o_tresci_simple_name",
        user_message="Utworz notatke jestem o tresci co tam sie dzieje w domu.",
        tool_name="add_note",
        parameters={
            "note_name": "jestem",
            "note": "co tam sie dzieje w domu.",
        },
    ),
    ToolScenario(
        name="add_note_o_tresci_multiword_name",
        user_message="Utworz notatke dom rodzinny o tresci co slychac u rodzicow",
        tool_name="add_note",
        parameters={
            "note_name": "dom rodzinny",
            "note": "co slychac u rodzicow",
        },
    ),
    ToolScenario(
        name="add_note_dodaj_o_tresci_variant",
        user_message="Dodaj notatke Plan weekendu o tresci kino i spacer",
        tool_name="add_note",
        parameters={
            "note_name": "Plan weekendu",
            "note": "kino i spacer",
        },
    ),
    ToolScenario(
        name="add_note_z_nazwa_o_tresci_embedded_quotes",
        user_message='Dodaj notatkę z nazwą Masterox o treści "Tsrtrst"Trstrst',
        tool_name="add_note",
        parameters={
            "note_name": "Masterox",
            "note": '"Tsrtrst"Trstrst',
        },
    ),
    ToolScenario(
        name="edit_note_explicit",
        user_message="Edytuj notatkę o nazwie Zakupy, dopisz Jajka.",
        tool_name="edit_note",
        parameters={
            "note_name": "Zakupy",
            "note": "Jajka",
        },
    ),
    ToolScenario(
        name="edit_note_no_diacritics",
        user_message="edytuj notatke zakupy dopisz maslo",
        tool_name="edit_note",
        parameters={
            "note_name": "zakupy",
            "note": "maslo",
        },
    ),
    ToolScenario(
        name="add_note_short_form",
        user_message="Utworz notatke super z tsrtrststdbvstb",
        tool_name="add_note",
        parameters={
            "note_name": "super",
            "note": "tsrtrststdbvstb",
        },
    ),
    ToolScenario(
        name="read_note_after_add",
        user_message="odczytaj notatke super",
        tool_name="get_note",
        parameters={"note_name": "super"},
    ),
    ToolScenario(
        name="read_note_display_variant_masterox",
        user_message="Wyświetl notatkę Masterox",
        tool_name="get_note",
        parameters={"note_name": "Masterox"},
    ),
    ToolScenario(
        name="read_this_note_from_context",
        user_message="odczytaj te notatke",
        tool_name="get_note",
        parameters={"note_name": "super"},
        prefill_messages=(
            "Utworz notatke super z tsrtrststdbvstb",
        ),
    ),
    ToolScenario(
        name="edit_this_note_from_context",
        user_message="edytuj te notatke dopisz abc",
        tool_name="edit_note",
        parameters={"note_name": "super", "note": "abc"},
        prefill_messages=(
            "Utworz notatke super z tsrtrststdbvstb",
        ),
    ),
    ToolScenario(
        name="read_last_note_from_context",
        user_message="pokaz ostatnia notatke",
        tool_name="get_note",
        parameters={"note_name": "super"},
        prefill_messages=(
            "Utworz notatke super z tsrtrststdbvstb",
        ),
    ),
    ToolScenario(
        name="read_last_note_from_context_diacritics_masterox",
        user_message="Wyświetl ostatnią notatkę.",
        tool_name="get_note",
        parameters={"note_name": "Masterox"},
        prefill_messages=(
            'Dodaj notatkę z nazwą Masterox o treści "Tsrtrst"Trstrst',
        ),
    ),
    ToolScenario(
        name="edit_last_note_from_context",
        user_message="dopisz do ostatniej notatki jeszcze maslo",
        tool_name="edit_note",
        parameters={"note_name": "super", "note": "jeszcze maslo"},
        prefill_messages=(
            "Utworz notatke super z tsrtrststdbvstb",
            "odczytaj notatke super",
        ),
    ),
)


class NotesToolResultSimulator:
    def __init__(self) -> None:
        self._notes: dict[str, str] = {}

    def apply(self, tool_name: str, parameters: Mapping[str, Any]) -> str:
        if tool_name in {"add_note", "edit_note"}:
            return self._apply_add_or_edit(parameters)
        if tool_name == "get_note":
            return self._apply_get(parameters)
        if tool_name == "list_notes":
            return self._apply_list()
        return f"Error: tool '{tool_name}' does not exist."

    def _apply_add_or_edit(self, parameters: Mapping[str, Any]) -> str:
        note_name = str(parameters.get("note_name", ""))
        note_fragment = str(parameters.get("note", ""))
        existing = self._notes.get(note_name)
        if existing is None:
            self._notes[note_name] = note_fragment
            return f"Notatka {note_name} dodana."

        prefix = "" if not existing or note_fragment.startswith("\n") else "\n"
        self._notes[note_name] = f"{existing}{prefix}{note_fragment}"
        return f"Notatka {note_name} zaktualizowana."

    def _apply_get(self, parameters: Mapping[str, Any]) -> str:
        note_name = str(parameters.get("note_name", ""))
        if note_name not in self._notes:
            return f"Notatka {note_name} nie istnieje."
        return f"Notatka {note_name}:\n{self._notes[note_name]}"

    def _apply_list(self) -> str:
        if not self._notes:
            return "Brak notatek."
        notes = sorted(self._notes.keys())
        return "Notatki:\n" + "\n".join(f"- {name}" for name in notes)


class NotesAgentEvalCallback:
    def __init__(self, runtime_options: Mapping[str, Any] | None = None) -> None:
        self._runtime_options = dict(runtime_options or {})
        self._agent = ManageNotesAgent(
            model_id=settings.model_name,
            prompt_builder=QwenPromptBuilder(system_prompt=_load_note_prompt()),
            toolsets=Toolsets([manage_notes_toolset]),
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


def _load_note_prompt() -> str:
    from registry.prompts import MANAGE_NOTES_PROMPT

    return MANAGE_NOTES_PROMPT


def _create_notes_callback(
    runtime_options: Mapping[str, Any] | None = None,
) -> NotesAgentEvalCallback:
    return NotesAgentEvalCallback(runtime_options)


NOTES_EVALUATION = EvaluationDefinition(
    name="notes",
    scenarios=NOTES_TOOL_SCENARIOS,
    flows=DEFAULT_NOTE_FLOWS,
    prompt_loader=_load_note_prompt,
    agent_callback_factory=_create_notes_callback,
    conversation_simulator_factory=NotesToolResultSimulator,
    assistant_greeting=(
        "Cześć! Mogę dodać notatkę, edytować notatkę, odczytać notatkę albo "
        "wyświetlić listę notatek. Co chcesz zrobić?"
    ),
    scenarios_example=serialize_scenarios_to_json(NOTES_TOOL_SCENARIOS),
)


__all__ = [
    "NOTES_EVALUATION",
    "NOTES_TOOL_SCENARIOS",
    "NotesAgentEvalCallback",
    "NotesToolResultSimulator",
]
