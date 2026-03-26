from __future__ import annotations

import json
import os
from typing import Any

import pytest
from agentic.models import ModelProvider
from deepeval.metrics import StepEfficiencyMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from registry.prompts import (
    DECISION_PROMPT,
    DISCOVERY_NOTES_PROMPT,
    GREETING_PROMPT,
    MANAGE_NOTES_PROMPT,
    ORGANIZER_PROMPT,
    SAGE_PROMPT,
    Prompts,
)

# ---------------------------------------------------------------------------
# Prompt constants map (avoids MLflow dependency)
# ---------------------------------------------------------------------------

_PROMPT_CONSTANTS: dict[str, str] = {
    Prompts.GREETING: GREETING_PROMPT,
    Prompts.MANAGE_NOTES: MANAGE_NOTES_PROMPT,
    Prompts.DISCOVERY_NOTES: DISCOVERY_NOTES_PROMPT,
    Prompts.SAGE: SAGE_PROMPT,
    Prompts.ORGANIZER: ORGANIZER_PROMPT,
    Prompts.DECISION: DECISION_PROMPT,
}


def _registry_prompt_loader(name: str) -> str:
    return _PROMPT_CONSTANTS.get(name, MANAGE_NOTES_PROMPT)


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes"}


def _ensure_model_available(model_name: str) -> None:
    provider = ModelProvider(model_name)
    with provider.session("model") as model:
        if model is not None:
            return
    error = provider.get_load_error("model") or "unknown load error"
    pytest.skip(f"Live smoke model '{model_name}' is not available: {error}")


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="session")
def workflow_smoke_enabled():
    if _truthy_env("RUN_WORKFLOW_SMOKE_TESTS") or _truthy_env("RUN_E2E_TESTS"):
        return
    pytest.skip("Set RUN_WORKFLOW_SMOKE_TESTS=1 to run workflow smoke tests.")


@pytest.fixture(autouse=True, scope="session")
def e2e_model_override():
    """Override all model settings to use Qwen/Qwen3.5-2B."""
    from agentic_runtime.settings import settings

    original_model = settings.model_name
    original_orchestration = settings.orchestration_model_name
    original_thinking = settings.thinkink_model

    settings.model_name = "Qwen/Qwen3.5-2B"
    settings.orchestration_model_name = "Qwen/Qwen3.5-2B"
    settings.thinkink_model = "Qwen/Qwen3.5-2B"

    yield

    settings.model_name = original_model
    settings.orchestration_model_name = original_orchestration
    settings.thinkink_model = original_thinking


@pytest.fixture(autouse=True, scope="session")
def workflow_smoke_model_ready(workflow_smoke_enabled, e2e_model_override):
    from agentic_runtime.settings import settings

    for model_name in {
        settings.model_name,
        settings.orchestration_model_name,
        settings.thinkink_model,
    }:
        _ensure_model_available(model_name)


@pytest.fixture(autouse=True, scope="session")
def noop_tracing():
    """Disable MLflow tracing for e2e tests."""
    import agentic_runtime.trace as trace_module

    original = trace_module.init_tracing
    trace_module.init_tracing = lambda: "e2e-test-run"
    yield
    trace_module.init_tracing = original


@pytest.fixture(autouse=True, scope="session")
def prompt_loader():
    """Load prompts from constants instead of MLflow registry."""
    import registry.prompts as registry_prompts_module

    original = registry_prompts_module.get_prompt

    registry_prompts_module.get_prompt = _registry_prompt_loader

    yield

    registry_prompts_module.get_prompt = original


# ---------------------------------------------------------------------------
# Deterministic eval judge (from evaluation test suite)
# ---------------------------------------------------------------------------


def _extract_trace_from_efficiency_prompt(prompt: str) -> dict[str, Any]:
    marker = "TRACE:"
    start = prompt.rfind(marker)
    if start < 0:
        return {}
    trace_text = prompt[start + len(marker) :]
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


class _DeterministicEvalJudge(DeepEvalBaseLLM):
    def load_model(self) -> "_DeterministicEvalJudge":
        return self

    def generate(self, prompt: str, schema=None) -> Any:
        if schema is None:
            return "{}"

        schema_name = getattr(schema, "__name__", "")
        if schema_name == "Task":
            return schema(task="Obsłuż żądanie użytkownika.")

        if schema_name == "EfficiencyVerdict":
            trace = _extract_trace_from_efficiency_prompt(prompt)
            child_spans = trace.get("children", []) if isinstance(trace, dict) else []
            score = 1.0 if len(child_spans) <= 1 else 0.5
            reason = (
                "Agent wykonał konieczne kroki."
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


# ---------------------------------------------------------------------------
# Step efficiency helpers
# ---------------------------------------------------------------------------


def build_efficiency_trace(
    *,
    agent_name: str,
    user_input: str,
    output: str,
    tool_name: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a trace dict for StepEfficiencyMetric."""
    children: list[dict[str, Any]] = []
    if tool_name is not None:
        children.append(
            {
                "name": tool_name,
                "type": "tool",
                "input": {"inputParameters": parameters or {}},
                "output": "ok",
                "children": [],
            }
        )
    return {
        "name": agent_name,
        "type": "agent",
        "input": {"input": user_input},
        "output": output,
        "children": children,
    }


def assert_step_efficiency(
    *,
    user_input: str,
    output: str,
    agent_name: str,
    tool_name: str | None = None,
    parameters: dict[str, Any] | None = None,
    threshold: float = 0.5,
) -> None:
    """Assert that a single agent interaction passes StepEfficiencyMetric."""
    metric = StepEfficiencyMetric(
        threshold=threshold,
        async_mode=False,
        model=_DeterministicEvalJudge(),
    )
    test_case = LLMTestCase(
        input=user_input,
        actual_output=output,
    )
    test_case._trace_dict = build_efficiency_trace(
        agent_name=agent_name,
        user_input=user_input,
        output=output,
        tool_name=tool_name,
        parameters=parameters,
    )
    score = metric.measure(test_case)
    assert score >= threshold, (
        f"StepEfficiencyMetric score {score} < threshold {threshold}"
    )
