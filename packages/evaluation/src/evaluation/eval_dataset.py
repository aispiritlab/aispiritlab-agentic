from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from deepeval.dataset import Golden

from evaluation.contracts import EvaluationDefinition, Flows, ToolResultSimulator, ToolScenario


@dataclass(frozen=True)
class ConversationStep:
    source_scenario_name: str
    user_message: str
    expected_output: str
    tool_name: str | None
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ConversationScenario:
    name: str
    steps: tuple[ConversationStep, ...]
    assistant_greeting: str

    def render_transcript(
        self,
        simulator: ToolResultSimulator | None = None,
    ) -> str:
        lines: list[str] = [
            f"assistant: {self.assistant_greeting}",
        ]
        for step in self.steps:
            lines.extend(
                [
                    "",
                    f"user: {step.user_message}",
                    "",
                    f"assistant: {step.expected_output}",
                ]
            )
            tool_result = _simulate_tool_result(simulator, step)
            if tool_result is not None:
                lines.extend(
                    [
                        "",
                        f"assistant: {tool_result}",
                    ]
                )
        return "\n".join(lines)

    def as_examples(
        self,
        simulator: ToolResultSimulator | None = None,
    ) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = [
            {
                "role": "assistant",
                "content": self.assistant_greeting,
            }
        ]
        for step in self.steps:
            examples.extend(
                [
                    {
                        "role": "user",
                        "content": step.user_message,
                        "source_scenario_name": step.source_scenario_name,
                    },
                    {
                        "role": "assistant",
                        "content": step.expected_output,
                        "kind": "model_tool_call",
                    },
                ]
            )
            tool_result = _simulate_tool_result(simulator, step)
            if tool_result is not None:
                examples.append(
                    {
                        "role": "assistant",
                        "content": tool_result,
                        "kind": "tool_result",
                    }
                )
        return examples


def build_goldens_from_scenarios(
    scenarios: Iterable[ToolScenario],
) -> list[Golden]:
    resolved = tuple(scenarios)
    _validate_scenarios(resolved)
    return [
        Golden(
            input=scenario.user_message,
            expected_output=scenario.resolved_expected_output(),
            additional_metadata=_build_metadata(scenario),
        )
        for scenario in resolved
    ]


def build_goldens_from_flows(
    scenarios: Iterable[ToolScenario],
    flows: Flows,
) -> list[Golden]:
    resolved = tuple(scenarios)
    _validate_scenarios(resolved)
    scenario_by_name = _build_scenarios_by_name(resolved)

    goldens: list[Golden] = []
    for flow in flows:
        history: list[str] = []
        for step_name in flow.steps:
            scenario = scenario_by_name.get(step_name)
            if scenario is None:
                raise ValueError(
                    f"Flow '{flow.name}' contains unknown scenario: {step_name!r}."
                )
            prefill_messages = _normalize_messages([*history, *scenario.prefill_messages])
            goldens.append(
                Golden(
                    input=scenario.user_message,
                    expected_output=scenario.resolved_expected_output(),
                    additional_metadata={
                        "prefill_messages": prefill_messages
                    }
                    if prefill_messages
                    else None,
                )
            )
            history.append(scenario.user_message)
    if not goldens:
        raise ValueError(
            "No flow goldens were built. Provide at least one flow with known scenario names."
        )
    return goldens


def build_prompt_optimization_goldens(
    definition: EvaluationDefinition,
) -> list[Golden]:
    if definition.flows is None:
        return build_goldens_from_scenarios(definition.scenarios)
    return build_goldens_from_flows(definition.scenarios, definition.flows)


def build_conversation_scenarios(
    definition: EvaluationDefinition,
) -> tuple[ConversationScenario, ...]:
    scenarios = definition.scenarios
    _validate_scenarios(scenarios)
    scenario_by_name = _build_scenarios_by_name(scenarios)
    scenario_by_user_message = _build_scenarios_by_user_message(scenarios)
    conversations: list[ConversationScenario] = []
    seen_names: set[str] = set()

    def add_conversation(name: str, scenario_names: list[str]) -> None:
        if name in seen_names:
            return
        steps = tuple(
            _build_step(scenario_by_name[scenario_name])
            for scenario_name in scenario_names
        )
        conversations.append(
            ConversationScenario(
                name=name,
                steps=steps,
                assistant_greeting=definition.assistant_greeting,
            )
        )
        seen_names.add(name)

    for scenario in scenarios:
        expanded_steps: list[str] = []
        for prefill_message in scenario.prefill_messages:
            prefill = scenario_by_user_message.get(prefill_message)
            if prefill is None:
                raise ValueError(
                    "Unknown prefill message in scenarios: "
                    f"{prefill_message!r} (scenario={scenario.name!r})"
                )
            expanded_steps.append(prefill.name)
        expanded_steps.append(scenario.name)
        add_conversation(f"single::{scenario.name}", expanded_steps)

    if definition.flows is not None:
        for flow in definition.flows:
            for step_name in flow.steps:
                if step_name not in scenario_by_name:
                    raise ValueError(
                        f"Flow '{flow.name}' contains unknown scenario: {step_name!r}."
                    )
            add_conversation(flow.name, list(flow.steps))

    return tuple(conversations)


def build_conversation_examples(
    definition: EvaluationDefinition,
) -> list[dict[str, Any]]:
    return [
        {
            "name": conversation.name,
            "steps": [step.source_scenario_name for step in conversation.steps],
            "messages": conversation.as_examples(
                simulator=(
                    definition.conversation_simulator_factory()
                    if definition.conversation_simulator_factory is not None
                    else None
                )
            ),
        }
        for conversation in build_conversation_scenarios(definition)
    ]


def _build_step(scenario: ToolScenario) -> ConversationStep:
    return ConversationStep(
        source_scenario_name=scenario.name,
        user_message=scenario.user_message,
        expected_output=scenario.resolved_expected_output(),
        tool_name=scenario.tool_name,
        parameters=dict(scenario.parameters),
    )


def _build_metadata(scenario: ToolScenario) -> dict[str, Any] | None:
    messages = _normalize_messages(scenario.prefill_messages)
    if not messages:
        return None
    return {"prefill_messages": messages}


def _normalize_messages(messages: Iterable[str]) -> list[str]:
    return [
        message.strip()
        for message in messages
        if isinstance(message, str) and message.strip()
    ]


def _build_scenarios_by_name(
    scenarios: tuple[ToolScenario, ...],
) -> dict[str, ToolScenario]:
    mapping: dict[str, ToolScenario] = {}
    duplicates: list[str] = []
    for scenario in scenarios:
        if scenario.name in mapping:
            duplicates.append(scenario.name)
            continue
        mapping[scenario.name] = scenario
    if duplicates:
        dupes = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"Duplicate ToolScenario.name entries: {dupes}")
    return mapping


def _build_scenarios_by_user_message(
    scenarios: tuple[ToolScenario, ...],
) -> dict[str, ToolScenario]:
    mapping: dict[str, ToolScenario] = {}
    duplicates: list[str] = []
    for scenario in scenarios:
        if scenario.user_message in mapping:
            duplicates.append(scenario.user_message)
            continue
        mapping[scenario.user_message] = scenario
    if duplicates:
        dupes = "; ".join(sorted(set(duplicates)))
        raise ValueError(
            "Duplicate ToolScenario.user_message entries (prefill expansion would "
            f"be ambiguous): {dupes}"
        )
    return mapping


def _validate_scenarios(scenarios: tuple[ToolScenario, ...]) -> None:
    if not scenarios:
        raise ValueError("At least one scenario is required.")
    _build_scenarios_by_name(scenarios)
    _build_scenarios_by_user_message(scenarios)


def _simulate_tool_result(
    simulator: ToolResultSimulator | None,
    step: ConversationStep,
) -> str | None:
    if simulator is None or step.tool_name is None:
        return None
    return simulator.apply(step.tool_name, step.parameters)
