from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


def render_tool_call(tool_name: str, parameters: Mapping[str, Any]) -> str:
    payload = {
        "name": tool_name,
        "parameters": dict(parameters),
    }
    content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"<tool_call>\n{content}\n</tool_call>"


@dataclass(frozen=True)
class Flow:
    name: str
    steps: tuple[str, ...]


class Flows:
    def __init__(self, *flows: Flow) -> None:
        self._flows = tuple(flows)

    @classmethod
    def from_iterable(cls, flows: Iterable[Flow]) -> "Flows":
        return cls(*tuple(flows))

    @property
    def values(self) -> tuple[Flow, ...]:
        return self._flows

    def __iter__(self):
        return iter(self._flows)

    def __len__(self) -> int:
        return len(self._flows)

    def __bool__(self) -> bool:
        return bool(self._flows)


@dataclass(frozen=True)
class ToolScenario:
    name: str
    user_message: str
    tool_name: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    expected_output: str | None = None
    prefill_messages: tuple[str, ...] = ()

    @property
    def user_message_pl(self) -> str:
        return self.user_message

    @property
    def expected_tool_call(self) -> str:
        return self.resolved_expected_output()

    def resolved_expected_output(self) -> str:
        if self.expected_output is not None:
            return self.expected_output
        if self.tool_name is None:
            raise ValueError(
                f"Scenario '{self.name}' is missing `tool_name` or `expected_output`."
            )
        return render_tool_call(self.tool_name, self.parameters)

    def require_tool_name(self) -> str:
        if self.tool_name is None:
            raise ValueError(
                f"Scenario '{self.name}' does not define `tool_name`; "
                "conversation rendering requires it."
            )
        return self.tool_name

    def as_prompt_optimization_item(self) -> dict[str, Any]:
        item: dict[str, Any] = {
            "name": self.name,
            "user_message": self.user_message,
        }
        if self.prefill_messages:
            item["prefill_messages"] = list(self.prefill_messages)
        if self.expected_output is not None:
            item["expected_output"] = self.expected_output
        elif self.tool_name is not None:
            item["tool_name"] = self.tool_name
            item["parameters"] = dict(self.parameters)
        return item


class AgentEvalCallback(Protocol):
    def reset(self) -> None: ...

    def prime(self, messages: Sequence[str]) -> None: ...

    def run(self, user_input: str, prompt_text: str | None = None) -> str: ...


class ToolResultSimulator(Protocol):
    def apply(self, tool_name: str, parameters: Mapping[str, Any]) -> str: ...


AgentCallbackFactory = Callable[[Mapping[str, Any] | None], AgentEvalCallback]
ToolResultSimulatorFactory = Callable[[], ToolResultSimulator]
PromptLoader = Callable[[], str]


@dataclass(frozen=True)
class EvaluationDefinition:
    name: str
    scenarios: tuple[ToolScenario, ...]
    flows: Flows | None = None
    prompt_text: str | None = None
    prompt_loader: PromptLoader | None = None
    agent_callback_factory: AgentCallbackFactory | None = None
    conversation_simulator_factory: ToolResultSimulatorFactory | None = None
    assistant_greeting: str = (
        "Cześć! Mogę pomóc wywołać odpowiednie narzędzie. Co chcesz zrobić?"
    )
    scenarios_example: str | None = None

    def resolve_prompt_text(self) -> str:
        if self.prompt_text is not None:
            return self.prompt_text
        if self.prompt_loader is None:
            raise ValueError(
                f"Evaluation definition '{self.name}' does not provide prompt text."
            )
        prompt_text = self.prompt_loader()
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ValueError(
                f"Evaluation definition '{self.name}' returned an empty prompt."
            )
        return prompt_text

    def create_agent_callback(
        self,
        runtime_options: Mapping[str, Any] | None = None,
    ) -> AgentEvalCallback:
        if self.agent_callback_factory is None:
            raise ValueError(
                f"Evaluation definition '{self.name}' does not provide an agent callback."
            )
        return self.agent_callback_factory(runtime_options)
