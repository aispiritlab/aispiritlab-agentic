from __future__ import annotations

import importlib
import json
import os
import re
from contextlib import contextmanager
from typing import Any

import pytest
from deepeval.metrics import StepEfficiencyMetric, ToolCorrectnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams

import agentic.prompts as agentic_prompts
from personal_assistant.agents.manage_notes.evaluation import NOTES_TOOL_SCENARIOS
from registry.prompts import GREETING_PROMPT, MANAGE_NOTES_PROMPT, Prompts

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DEEPEVAL_METRICS", "").lower() not in {"1", "true", "yes"},
    reason="Set RUN_DEEPEVAL_METRICS=1 to run deepeval metric integration tests.",
)


AVAILABLE_NOTE_TOOLS = ("add_note", "edit_note", "get_note", "list_notes")


def _extract_last_user_turn(prompt: str) -> str:
    turns = re.findall(
        r"<start_of_turn>user\n(.*?)\n<end_of_turn>",
        prompt,
        flags=re.DOTALL,
    )
    if not turns:
        raise AssertionError("Brak wiadomości użytkownika w promptcie Gemma.")
    return turns[-1].strip()


def _extract_trace_from_efficiency_prompt(prompt: str) -> dict[str, Any]:
    marker = "TRACE:"
    start = prompt.rfind(marker)
    if start < 0:
        return {}

    trace_text = prompt[start + len(marker):]
    if "JSON:" in trace_text:
        trace_text = trace_text.split("JSON:", maxsplit=1)[0]
    trace_text = trace_text.strip()

    try:
        data = json.loads(trace_text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


class _DeterministicNotesModel:
    _routes = {
        scenario.user_message_pl: scenario.expected_tool_call
        for scenario in NOTES_TOOL_SCENARIOS
    }

    def response(self, prompt: str) -> str:
        user_message = _extract_last_user_turn(prompt)
        try:
            return self._routes[user_message]
        except KeyError as error:
            raise AssertionError(
                f"Brak przygotowanego przykładu dla wiadomości: {user_message!r}"
            ) from error


class _DeterministicEvalJudge(DeepEvalBaseLLM):
    def load_model(self) -> "_DeterministicEvalJudge":
        return self

    def generate(self, prompt: str, schema=None) -> Any:
        if schema is None:
            return "{}"

        schema_name = getattr(schema, "__name__", "")
        if schema_name == "Task":
            return schema(task="Obsłuż żądanie użytkownika dotyczące notatek.")

        if schema_name == "EfficiencyVerdict":
            trace = _extract_trace_from_efficiency_prompt(prompt)
            child_spans = trace.get("children", []) if isinstance(trace, dict) else []
            score = 1.0 if len(child_spans) == 1 else 0.5
            reason = (
                "Agent wykonał jeden konieczny krok narzędziowy."
                if score == 1.0
                else "Wykryto nadmiarowe kroki."
            )
            return schema(score=score, reason=reason)

        if schema_name == "ToolSelectionScore":
            return schema(
                score=1.0,
                reason="Wybrane narzędzie jest zgodne z intencją użytkownika.",
            )

        raise AssertionError(f"Nieobsługiwany schemat ewaluacji: {schema_name}")

    async def a_generate(self, prompt: str, schema=None) -> Any:
        return self.generate(prompt, schema=schema)

    def get_model_name(self) -> str:
        return "deterministic-eval-judge"


def _registry_prompt_loader(name: str) -> str:
    if name == Prompts.MANAGE_NOTES:
        return MANAGE_NOTES_PROMPT
    if name == Prompts.GREETING:
        return GREETING_PROMPT
    return MANAGE_NOTES_PROMPT


@pytest.fixture()
def notes_manager(monkeypatch, tmp_path):
    monkeypatch.setattr(agentic_prompts, "get_prompt", _registry_prompt_loader)

    import agentic_runtime.trace as trace_module

    monkeypatch.setattr(trace_module, "init_tracing", lambda: "test-run")

    from personal_assistant.agents.manage_notes import tools as note_tools

    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    personalization_file = tmp_path / "personalization.json"
    personalization_file.write_text(
        json.dumps({"vault_path": str(vault_path)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(note_tools, "PERSONALIZATION_FILE", personalization_file)

    import agentic_runtime as runtime_module

    runtime_module = importlib.reload(runtime_module)
    runtime_module._reset_runtime()
    runtime = runtime_module.get_runtime()

    fake_model = _DeterministicNotesModel()

    @contextmanager
    def fake_session(name: str = "model"):  # noqa: ARG001
        yield fake_model

    monkeypatch.setattr(
        runtime.note_workflow._agent._model_provider,
        "session",
        fake_session,
    )

    runtime.note_workflow._agent.reset()
    runtime.note_workflow._agent.start()
    return runtime


def _run_and_extract_tool_call(notes_manager, user_input: str) -> tuple[str, str, dict[str, Any]]:
    model_reply = notes_manager.note_workflow._agent._agent.run(user_input)
    if model_reply.tool_call is None:
        raise AssertionError(f"Brak tool_call dla wejścia: {user_input!r}")
    tool_name, parameters = model_reply.tool_call
    return model_reply.content or "", tool_name, parameters


def _build_tool_test_case(
    user_input: str,
    output: str,
    called_tool_name: str,
    called_parameters: dict[str, Any],
    expected_tool_name: str,
    expected_parameters: dict[str, Any],
) -> LLMTestCase:
    return LLMTestCase(
        input=user_input,
        actual_output=output,
        tools_called=[
            ToolCall(name=called_tool_name, input_parameters=called_parameters),
        ],
        expected_tools=[
            ToolCall(name=expected_tool_name, input_parameters=expected_parameters),
        ],
    )


def _build_efficiency_trace(
    user_input: str,
    output: str,
    tool_name: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": "notes_agent",
        "type": "agent",
        "input": {"input": user_input},
        "output": output,
        "children": [
            {
                "name": tool_name,
                "type": "tool",
                "input": {"inputParameters": parameters},
                "output": "ok",
                "children": [],
            }
        ],
    }


def test_notes_agent_uses_registry_note_prompt(notes_manager) -> None:
    system_prompt = notes_manager.note_workflow._agent._agent.system_prompt.system_prompt

    assert "Jesteś asystentem do zarządzania notatkami." in system_prompt
    assert "Odpowiadasz wyłącznie po polsku." in system_prompt
    assert "Przykłady:" in system_prompt
    assert "edit_note" in system_prompt


def test_tool_correctness_for_add_read_list_and_edit_paths(notes_manager) -> None:
    metric = ToolCorrectnessMetric(
        available_tools=[ToolCall(name=name) for name in AVAILABLE_NOTE_TOOLS],
        threshold=1.0,
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        should_exact_match=True,
        should_consider_ordering=True,
        async_mode=False,
        model=_DeterministicEvalJudge(),
    )

    for scenario in NOTES_TOOL_SCENARIOS:
        output, called_tool_name, called_parameters = _run_and_extract_tool_call(
            notes_manager,
            scenario.user_message_pl,
        )

        test_case = _build_tool_test_case(
            user_input=scenario.user_message_pl,
            output=output,
            called_tool_name=called_tool_name,
            called_parameters=called_parameters,
            expected_tool_name=scenario.tool_name,
            expected_parameters=scenario.parameters,
        )

        score = metric.measure(test_case)

        assert called_tool_name == scenario.tool_name
        assert score == pytest.approx(1.0)
        assert metric.success


def test_step_efficiency_for_add_read_list_and_edit_paths(notes_manager) -> None:
    metric = StepEfficiencyMetric(
        threshold=0.9,
        async_mode=False,
        model=_DeterministicEvalJudge(),
    )

    for scenario in NOTES_TOOL_SCENARIOS:
        output, called_tool_name, called_parameters = _run_and_extract_tool_call(
            notes_manager,
            scenario.user_message_pl,
        )

        test_case = LLMTestCase(
            input=scenario.user_message_pl,
            actual_output=output,
        )
        test_case._trace_dict = _build_efficiency_trace(
            user_input=scenario.user_message_pl,
            output=output,
            tool_name=called_tool_name,
            parameters=called_parameters,
        )

        score = metric.measure(test_case)

        assert score == pytest.approx(1.0)
        assert metric.success
