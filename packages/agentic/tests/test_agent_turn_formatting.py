from contextlib import contextmanager

from agentic.agent import Agent
from agentic.message import SystemMessage, ToolMessage
from agentic.models.response import ModelResponse
from agentic.prompts import GemmaPromptBuilder
from agentic.tools import Toolset, Toolsets


class FakeModel:
    def __init__(self, responses: list[str]):
        self._responses = iter(responses)
        self.prompts: list[str] = []

    def response(self, prompt: str) -> ModelResponse:
        self.prompts.append(prompt)
        return ModelResponse(text=next(self._responses))


class FakeProvider:
    def __init__(self, model: FakeModel):
        self._model = model

    @contextmanager
    def session(self, name: str = "model"):
        yield self._model


class FakeMemory:
    def __init__(self, summary_text: str):
        self.summary_text = summary_text
        self.summary_calls = 0

    def retrieve(self, key: str, length: int) -> str:
        del key, length
        return ""

    def store(self, key: str, value: str) -> None:
        del key, value

    def summary(self) -> str:
        self.summary_calls += 1
        return self.summary_text


def add_note(note_name: str, note: str) -> str:
    return "Notatka dodana."


def get_note(note_name: str) -> str:
    return f"Notatka {note_name}:\n..."


def test_tool_message_uses_valid_closing_tag() -> None:
    assert ToolMessage("ok").as_turn() == "<tool_call_response>ok\n</tool_call_response>"


def test_agent_keeps_user_turn_separate_after_tool_call() -> None:
    first_response = '{"name":"add_note","parameters":{"note_name":"marara","note":"tresci marara"}}'
    second_response = '{"name":"get_note","parameters":{"note_name":"marara"}}'
    model = FakeModel([first_response, second_response])
    agent = Agent(
        model_provider=FakeProvider(model),
        prompt_builder=GemmaPromptBuilder(system_prompt="SYSTEM"),
        toolsets=Toolsets([Toolset([add_note, get_note])]),
    )
    agent.history.add(SystemMessage("Cześć!"))

    first = agent.run("Dodaj notatke marara o tresci marara")
    assert len(first.tool_calls) == 1
    assert first.content == first_response

    first_exec = agent.toolsets.run_tool(first.tool_calls[0])
    assert first_exec is not None
    assert first_exec.output == "Notatka dodana."

    second = agent.run("Wyswietl ta notatke")
    assert len(second.tool_calls) == 1

    prompt = model.prompts[1]
    assert "<end_of_turn>Wyswietl ta notatke" not in prompt
    assert "<start_of_turn>user\nWyswietl ta notatke\n<end_of_turn>" in prompt
    assert "<tool_call_response>" not in prompt


def test_agent_keeps_staged_memory_context_outside_rendered_prompt() -> None:
    memory = FakeMemory("future memory summary")
    agent = Agent(
        model_provider=FakeProvider(FakeModel(["unused"])),
        prompt_builder=GemmaPromptBuilder(system_prompt="SYSTEM"),
        memory=memory,
    )
    agent.history.store("Wczesniejsze pytanie", "Wczesniejsza odpowiedz")

    prompt_context = agent._gather_context("Aktualne pytanie", agent.context)
    prompt = agent._build_prompt(prompt_context)

    assert memory.summary_calls == 1
    assert prompt_context.memory_context == "future memory summary"
    assert "future memory summary" not in prompt
    assert "<start_of_turn>user\nWczesniejsze pytanie\n<end_of_turn>" in prompt
    assert "<start_of_turn>model\nWczesniejsza odpowiedz\n<end_of_turn>" in prompt
    assert "<start_of_turn>user\nAktualne pytanie\n<end_of_turn>" in prompt


def test_agent_renders_tool_message_as_tool_turn_and_stores_it_in_history() -> None:
    model = FakeModel(["Koncowa odpowiedz."])
    agent = Agent(
        model_provider=FakeProvider(model),
        prompt_builder=GemmaPromptBuilder(system_prompt="SYSTEM"),
    )
    agent.history.store(
        "Dodaj notatke marara o tresci marara",
        '{"name":"add_note","parameters":{"note_name":"marara","note":"tresci marara"}}',
    )

    result = agent.run(ToolMessage("Notatka dodana."))

    assert result.content == "Koncowa odpowiedz."
    prompt = model.prompts[0]
    assert "<tool_call_response>Notatka dodana.\n</tool_call_response>" in prompt

    next_prompt_context = agent._gather_context("Wyswietl ta notatke", agent.context)
    next_prompt = agent._build_prompt(next_prompt_context)
    assert "<tool_call_response>Notatka dodana.\n</tool_call_response>" in next_prompt
