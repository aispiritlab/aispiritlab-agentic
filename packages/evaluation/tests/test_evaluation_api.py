from __future__ import annotations

import pytest
from deepeval.dataset import Golden
from deepeval.prompt import Prompt

import evaluation
from evaluation import (
    EvaluationDefinition,
    Flow,
    Flows,
    ToolScenario,
    build_conversation_scenarios,
    build_goldens_from_flows,
    load_evaluation_definition,
    optimize_prompt_text,
)
from evaluation import notes_prompt_optimization_miprov2 as optimization_module


def test_public_api_exports_generic_contracts() -> None:
    exported = set(evaluation.__all__)

    assert "Flow" in exported
    assert "Flows" in exported
    assert "ToolScenario" in exported
    assert "EvaluationDefinition" in exported
    assert "load_evaluation_definition" in exported
    assert "optimize_prompt_text" in exported


def test_build_goldens_from_flows_merges_history_and_prefill() -> None:
    scenarios = (
        ToolScenario(
            name="first",
            user_message="dodaj notatke",
            tool_name="add_note",
            parameters={"note_name": "A"},
        ),
        ToolScenario(
            name="second",
            user_message="pokaz ja",
            tool_name="get_note",
            parameters={"note_name": "A"},
            prefill_messages=("prefill-a", "prefill-b"),
        ),
    )

    goldens = build_goldens_from_flows(
        scenarios,
        Flows(Flow(name="flow::main", steps=("first", "second"))),
    )

    assert len(goldens) == 2
    assert goldens[1].additional_metadata == {
        "prefill_messages": ["dodaj notatke", "prefill-a", "prefill-b"]
    }


def test_build_conversation_scenarios_rejects_unknown_prefill_message() -> None:
    definition = EvaluationDefinition(
        name="broken",
        scenarios=(
            ToolScenario(
                name="broken",
                user_message="pokaz ostatnia notatke",
                tool_name="get_note",
                parameters={"note_name": "X"},
                prefill_messages=("nieznana wiadomosc",),
            ),
        ),
    )

    with pytest.raises(ValueError, match="Unknown prefill message"):
        build_conversation_scenarios(definition)


def test_load_evaluation_definition_resolves_notes_definition() -> None:
    definition = load_evaluation_definition(
        "agentic_runtime.manage_notes.evaluation:NOTES_EVALUATION"
    )

    assert definition.name == "notes"
    assert definition.flows is not None
    assert definition.scenarios


def test_optimize_prompt_text_uses_callback_contract(monkeypatch) -> None:
    events: list[object] = []

    class DummyCallback:
        def reset(self) -> None:
            events.append("reset")

        def prime(self, messages) -> None:
            events.append(("prime", tuple(messages)))

        def run(self, user_input: str, prompt_text: str | None = None) -> str:
            events.append(("run", user_input, prompt_text))
            return "model-output"

        def close(self) -> None:
            events.append("close")

    class FakePromptOptimizer:
        def __init__(self, *, model_callback, **kwargs) -> None:
            self._model_callback = model_callback

        def optimize(self, prompt: Prompt, goldens: list[Golden]) -> Prompt:
            self._model_callback(prompt, goldens[0])
            return Prompt(text_template="optimized prompt")

    monkeypatch.setattr(optimization_module, "PromptOptimizer", FakePromptOptimizer)
    monkeypatch.setattr(
        optimization_module,
        "FixedOpenRouterModel",
        lambda **kwargs: object(),
    )

    definition = EvaluationDefinition(
        name="dummy",
        scenarios=(
            ToolScenario(
                name="scenario",
                user_message="uzyj narzedzia",
                tool_name="tool",
                parameters={"x": "1"},
                prefill_messages=("prefill",),
            ),
        ),
        prompt_text="prompt bazowy",
        agent_callback_factory=lambda runtime_options=None: DummyCallback(),
    )

    optimized_prompt = optimize_prompt_text(
        definition=definition,
        prompt_text="prompt bazowy",
        openrouter_model="openai/gpt-4o-mini",
        goldens=[
            Golden(
                input="uzyj narzedzia",
                expected_output="wynik",
                additional_metadata={"prefill_messages": ["prefill"]},
            )
        ],
    )

    assert optimized_prompt == "optimized prompt"
    assert events == [
        "reset",
        ("prime", ("prefill",)),
        ("run", "uzyj narzedzia", "prompt bazowy"),
        "close",
    ]
