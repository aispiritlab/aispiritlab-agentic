from contextlib import contextmanager

from agentic.agent import Agent
from agentic.models.response import ModelResponse
from agentic.prompts import GemmaPromptBuilder
from agentic.tools import ToolCallCommand, Toolset, Toolsets


def add_note(note_name: str, note: str) -> str:
    return f"saved:{note_name}:{note}"


class FakeModel:
    def __init__(self, response_text: str):
        self._response_text = response_text

    def response(self, prompt: str) -> ModelResponse:
        return ModelResponse(text=self._response_text)


class FakeProvider:
    def __init__(self, model: FakeModel):
        self._model = model

    @contextmanager
    def session(self, name: str = "model"):
        yield self._model


def test_agent_returns_tool_call_without_executing() -> None:
    raw_response = '{"name":"add_note","parameters":{"note_name":"Pizza","note":"Pepperoni"}}'
    model = FakeModel(raw_response)
    agent = Agent(
        model_provider=FakeProvider(model),
        prompt_builder=GemmaPromptBuilder(system_prompt="test"),
        toolsets=Toolsets([Toolset([add_note])]),
    )

    result = agent.run("zapisz")

    assert result.tool_call == ("add_note", {"note_name": "Pizza", "note": "Pepperoni"})
    assert result.tool_calls == [("add_note", {"note_name": "Pizza", "note": "Pepperoni"})]
    assert result.content == raw_response


def test_toolsets_parse_and_execute_command_from_tool_call() -> None:
    raw_response = '{"name":"add_note","parameters":{"note_name":"Pizza","note":"Pepperoni"}}'
    model = FakeModel(raw_response)
    agent = Agent(
        model_provider=FakeProvider(model),
        prompt_builder=GemmaPromptBuilder(system_prompt="test"),
        toolsets=Toolsets([Toolset([add_note])]),
    )

    result = agent.run("zapisz")
    tool_call = result.tool_calls[0]

    command = agent.toolsets.parse_tool(tool_call)
    assert command is not None
    assert isinstance(command, ToolCallCommand)
    assert command.function_name == "add_note"
    assert command.parameters == {"note_name": "Pizza", "note": "Pepperoni"}

    tool_result = agent.toolsets.execute(command)
    assert tool_result.output == "saved:Pizza:Pepperoni"


def test_toolsets_execute_returns_validation_error() -> None:
    model = FakeModel('{"name":"add_note","parameters":{"note_name":"Pizza"}}')
    agent = Agent(
        model_provider=FakeProvider(model),
        prompt_builder=GemmaPromptBuilder(system_prompt="test"),
        toolsets=Toolsets([Toolset([add_note])]),
    )

    result = agent.run("zapisz")
    assert result.tool_call == ("add_note", {"note_name": "Pizza"})

    command = agent.toolsets.parse_tool(result.tool_calls[0])
    assert command is not None
    tool_result = agent.toolsets.execute(command)
    assert "brak wymaganych parametrów" in tool_result.output
