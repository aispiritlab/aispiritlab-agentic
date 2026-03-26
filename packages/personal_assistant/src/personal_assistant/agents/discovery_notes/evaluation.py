from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agentic.message import UserMessage
from agentic.prompts import GemmaPromptBuilder, QwenPromptBuilder
from agentic.tools import Toolsets
from evaluation import EvaluationDefinition, ToolScenario, serialize_scenarios_to_json

from personal_assistant.settings import settings

from .detective_agent import DiscoveryNotesAgent
from .flows import DEFAULT_DISCOVERY_NOTE_FLOWS
from .tools import toolset as discovery_notes_toolset


DISCOVERY_NOTES_TOOL_SCENARIOS: tuple[ToolScenario, ...] = (
    ToolScenario(
        name="search_rag_foundations",
        user_message="Znajdź informacje o podstawach RAG.",
        tool_name="search",
        parameters={"query": "podstawach RAG"},
    ),
    ToolScenario(
        name="search_embeddings",
        user_message="Wyszukaj notatki o embeddings i chunkingu.",
        tool_name="search",
        parameters={"query": "embeddings i chunkingu"},
    ),
    ToolScenario(
        name="search_obsidian_linking",
        user_message="Szukaj wiedzy o linkowaniu notatek w Obsidianie.",
        tool_name="search",
        parameters={"query": "linkowaniu notatek w Obsidianie"},
    ),
    ToolScenario(
        name="search_agentic_tools",
        user_message="Znajdz notatki o narzedziach agentowych.",
        tool_name="search",
        parameters={"query": "narzedziach agentowych"},
    ),
    ToolScenario(
        name="search_python_rag",
        user_message="Wyszukaj materiały o Pythonie i RAG.",
        tool_name="search",
        parameters={"query": "Pythonie i RAG"},
    ),
)


class DiscoveryNotesToolResultSimulator:
    def apply(self, tool_name: str, parameters: Mapping[str, Any]) -> str:
        if tool_name != "search":
            return f"Error: tool '{tool_name}' does not exist."
        query = str(parameters.get("query", "")).strip()
        if not query:
            return "Brak treści zapytania."
        return f"Wyniki dla zapytania: {query}"


class DiscoveryNotesEvalCallback:
    def __init__(self, runtime_options: Mapping[str, Any] | None = None) -> None:
        self._runtime_options = dict(runtime_options or {})
        self._agent = DiscoveryNotesAgent(
            model_id=settings.model_name,
            prompt_builder=QwenPromptBuilder(system_prompt=_load_discovery_notes_prompt()),
            toolsets=Toolsets([discovery_notes_toolset]),
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


def _load_discovery_notes_prompt() -> str:
    from registry.prompts import DISCOVERY_NOTES_PROMPT

    return DISCOVERY_NOTES_PROMPT


def _create_discovery_notes_callback(
    runtime_options: Mapping[str, Any] | None = None,
) -> DiscoveryNotesEvalCallback:
    return DiscoveryNotesEvalCallback(runtime_options)


DISCOVERY_NOTES_EVALUATION = EvaluationDefinition(
    name="discovery_notes",
    scenarios=DISCOVERY_NOTES_TOOL_SCENARIOS,
    flows=DEFAULT_DISCOVERY_NOTE_FLOWS,
    prompt_loader=_load_discovery_notes_prompt,
    agent_callback_factory=_create_discovery_notes_callback,
    conversation_simulator_factory=DiscoveryNotesToolResultSimulator,
    assistant_greeting=(
        "Cześć! Pomogę Ci znaleźć notatki semantycznie. Jakie hasło mam wyszukać?"
    ),
    scenarios_example=serialize_scenarios_to_json(DISCOVERY_NOTES_TOOL_SCENARIOS),
)


__all__ = [
    "DISCOVERY_NOTES_EVALUATION",
    "DISCOVERY_NOTES_TOOL_SCENARIOS",
    "DiscoveryNotesEvalCallback",
    "DiscoveryNotesToolResultSimulator",
]
