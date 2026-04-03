from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

from agentic.agent import Agent, AgentResult
from agentic.capabilities import AbstractCapability, HookContext
from agentic.core_agent import CoreAgentResponse, CoreAgentic
from agentic.exceptions import DEFAULT_RETRY, RetryPolicy
from agentic.message import ToolMessage
from agentic.prompts import GemmaPromptBuilder
from agentic.structured_output import StructuredOutput
from agentic.usage import RequestUsage, UsageLimits
from agentic.workflow.messages import ConversationData, RecordedMessageMetadata, UserMessage
from agentic.workflow.reactor import MultiTurnLLMReactor
from providers.models.response import ModelResponse


class FakeModel:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.prompts: list[str] = []

    def response(self, prompt: str) -> ModelResponse:
        self.prompts.append(prompt)
        return ModelResponse(
            text=next(self._responses),
            model="test-model",
            prompt_tokens=5,
            completion_tokens=7,
            total_tokens=12,
            latency_ms=1.0,
        )


class FakeProvider:
    def __init__(self, model: FakeModel) -> None:
        self._model = model

    @contextmanager
    def session(self, name: str = "model"):
        del name
        yield self._model


@dataclass(frozen=True)
class StructuredAnswer:
    value: int


class TrackingCapability(AbstractCapability):
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.hook_runs: list[tuple[str, int]] = []

    def get_instructions(self, context: HookContext) -> str | None:
        self.hook_runs.append(("instructions", context.turn))
        return "ALWAYS APPLY CAPABILITY RULES"

    def before_tool_execute(
        self,
        tool_name: str,
        parameters: dict[str, object],
        context: HookContext,
    ) -> dict[str, object]:
        del tool_name, context
        updated = dict(parameters)
        updated["value"] = str(updated["value"]).upper()
        return updated

    def after_tool_execute(
        self,
        tool_name: str,
        result: str,
        context: HookContext,
    ) -> str:
        del tool_name, context
        return result + "!"

    def on_error(self, error: Exception, context: HookContext) -> None:
        self.errors.append(f"{context.run_id}:{error}")


def _make_core_agent(agent: Agent) -> CoreAgentic:
    core = object.__new__(CoreAgentic)
    core._agent = agent
    core._tracer = None
    core._retry_policy = DEFAULT_RETRY
    core._welcome_message = ""
    return core


def test_agent_accepts_typed_structured_output_and_preserves_response_text() -> None:
    model = FakeModel(['{"value": 3}'])
    agent = Agent(
        model_provider=FakeProvider(model),
        prompt_builder=GemmaPromptBuilder(system_prompt="SYSTEM"),
        structured_output=StructuredOutput(StructuredAnswer),
    )

    result = agent.run("podaj wynik")

    assert result.content == StructuredAnswer(value=3)
    assert result.content_text == '{"value": 3}'

    next_prompt_context = agent._gather_context("kolejne pytanie", agent.context)
    next_prompt = agent._build_prompt(next_prompt_context)
    assert '{"value": 3}' in next_prompt


def test_agent_retries_invalid_structured_output_and_counts_usage_per_attempt() -> None:
    model = FakeModel(["not json", '{"value": 5}'])
    agent = Agent(
        model_provider=FakeProvider(model),
        prompt_builder=GemmaPromptBuilder(system_prompt="SYSTEM"),
        structured_output=StructuredOutput(StructuredAnswer),
        retry_policy=RetryPolicy(max_retries=1),
    )

    result = agent.run("napraw odpowiedz")

    assert result.content == StructuredAnswer(value=5)
    assert len(model.prompts) == 2
    assert "Validation error:" in model.prompts[1]
    assert "Original request:" in model.prompts[1]
    assert "napraw odpowiedz" in model.prompts[1]
    assert agent.run_usage.requests == 2


def test_agent_usage_limits_reset_for_each_run() -> None:
    model = FakeModel(["ok", "ok"])
    agent = Agent(
        model_provider=FakeProvider(model),
        prompt_builder=GemmaPromptBuilder(system_prompt="SYSTEM"),
        usage_limits=UsageLimits(request_limit=1),
    )

    first = agent.run("pierwsze")
    second = agent.run("drugie")

    assert first.content == "ok"
    assert second.content == "ok"
    assert agent.run_usage.requests == 1


def test_core_agent_uses_capability_instructions_and_tool_hooks() -> None:
    def echo(value: str) -> str:
        return value

    capability = TrackingCapability()
    model = FakeModel(['{"name":"echo","parameters":{"value":"hello"}}'])
    agent = Agent(
        model_provider=FakeProvider(model),
        prompt_builder=GemmaPromptBuilder(system_prompt="SYSTEM"),
        tools=[echo],
        capabilities=[capability],
    )
    core = _make_core_agent(agent)

    response = core.respond("uruchom narzedzie")

    assert response.output == "HELLO!"
    assert "ALWAYS APPLY CAPABILITY RULES" in model.prompts[0]
    assert capability.errors == []


def test_multi_turn_reactor_resets_usage_for_each_invoke() -> None:
    class FakeAgent:
        def __init__(self) -> None:
            self.calls = 0

        def respond(self, message: str | ToolMessage) -> CoreAgentResponse:
            self.calls += 1
            del message
            return CoreAgentResponse(
                result=AgentResult(
                    content="ok",
                    run_id=f"run-{self.calls}",
                    request_usage=RequestUsage(total_tokens=1),
                ),
                output="ok",
            )

    reactor = MultiTurnLLMReactor(
        agent=FakeAgent(),  # type: ignore[arg-type]
        usage_limits=UsageLimits(request_limit=1),
    )
    command = UserMessage(
        data=ConversationData(role="user", text="hej"),
        metadata=RecordedMessageMetadata(runtime_id="rt-1"),
    )

    first = reactor.invoke(command)
    second = reactor.invoke(command)

    assert first.data.text == "ok"
    assert second.data.text == "ok"
    assert reactor.run_usage.requests == 1
