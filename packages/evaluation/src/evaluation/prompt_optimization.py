from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from evaluation.contracts import EvaluationDefinition, ToolScenario, render_tool_call
from evaluation.eval_dataset import build_goldens_from_scenarios
from evaluation.notes_prompt_optimization_miprov2 import optimize_prompt_text


def serialize_scenarios_to_json(
    scenarios: Sequence[ToolScenario],
    *,
    indent: int = 2,
) -> str:
    return json.dumps(
        [scenario.as_prompt_optimization_item() for scenario in scenarios],
        ensure_ascii=False,
        indent=indent,
    )


def parse_scenarios_json(scenarios_json: str) -> tuple[ToolScenario, ...]:
    raw = scenarios_json.strip()
    if not raw:
        raise ValueError("Scenario JSON cannot be empty.")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid scenario JSON: {error}") from error

    if not isinstance(data, list):
        raise ValueError("Scenarios must be encoded as a JSON array.")
    if not data:
        raise ValueError("Provide at least one test scenario.")

    scenarios: list[ToolScenario] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Scenario #{index}: every item must be a JSON object.")

        user_message = _extract_user_message(item, index=index)
        prefill_messages = _extract_prefill_messages(item, index=index)
        expected_output = _build_expected_output(item, index=index)
        tool_name = item.get("tool_name")
        parameters = item.get("parameters", {})
        if tool_name is not None and not isinstance(tool_name, str):
            raise ValueError(f"Scenario #{index}: `tool_name` must be a string.")
        if not isinstance(parameters, dict):
            raise ValueError(f"Scenario #{index}: `parameters` must be a JSON object.")

        scenarios.append(
            ToolScenario(
                name=_extract_name(item, index=index),
                user_message=user_message,
                tool_name=tool_name.strip() if isinstance(tool_name, str) else None,
                parameters=parameters,
                expected_output=expected_output,
                prefill_messages=tuple(prefill_messages),
            )
        )
    return tuple(scenarios)


def run_prompt_optimization(
    *,
    definition: EvaluationDefinition,
    prompt_to_optimize: str,
    scenarios_json: str,
    openrouter_model: str,
    openrouter_api_key: str,
    num_candidates: float | int,
    num_trials: float | int,
    runtime_options: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    resolved_prompt = prompt_to_optimize.strip()
    if not resolved_prompt:
        return "Wklej prompt do optymalizacji.", ""

    resolved_model = openrouter_model.strip() or "openai/gpt-4o-mini"
    resolved_api_key = openrouter_api_key.strip()

    try:
        resolved_num_candidates = _to_positive_int(num_candidates, "num_candidates")
        resolved_num_trials = _to_positive_int(num_trials, "num_trials")
        scenarios = parse_scenarios_json(scenarios_json)
        optimized_prompt = optimize_prompt_text(
            definition=definition,
            prompt_text=resolved_prompt,
            openrouter_model=resolved_model,
            openrouter_api_key=resolved_api_key or None,
            num_candidates=resolved_num_candidates,
            num_trials=resolved_num_trials,
            goldens=build_goldens_from_scenarios(scenarios),
            runtime_options=runtime_options,
        )
    except Exception as error:
        return f"Błąd optymalizacji: {error}", ""

    return (
        "Optymalizacja zakończona. "
        f"Scenariusze: {len(scenarios)}, model: {resolved_model}.",
        optimized_prompt,
    )


def _extract_name(item: dict[str, object], *, index: int) -> str:
    name = item.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Scenario #{index}: `name` must be a non-empty string.")
    return name.strip()


def _extract_user_message(item: dict[str, object], *, index: int) -> str:
    for field_name in ("input", "user_message", "message", "user_message_pl"):
        value = item.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(
        f"Scenario #{index}: provide the user message in `input`, `user_message`, "
        "`message`, or `user_message_pl`."
    )


def _extract_prefill_messages(item: dict[str, object], *, index: int) -> list[str]:
    raw_prefill = item.get("prefill_messages", [])
    if raw_prefill is None:
        return []
    if not isinstance(raw_prefill, list):
        raise ValueError(f"Scenario #{index}: `prefill_messages` must be a list.")
    messages: list[str] = []
    for message in raw_prefill:
        if not isinstance(message, str):
            raise ValueError(
                f"Scenario #{index}: `prefill_messages` can contain only strings."
            )
        if message.strip():
            messages.append(message.strip())
    return messages


def _build_expected_output(item: dict[str, object], *, index: int) -> str | None:
    expected_output = item.get("expected_output")
    if expected_output is not None:
        if isinstance(expected_output, dict):
            tool_name = expected_output.get("name") or expected_output.get("tool_name")
            parameters = expected_output.get("parameters", {})
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError(
                    f"Scenario #{index}: `expected_output.name` must be a string."
                )
            if not isinstance(parameters, dict):
                raise ValueError(
                    f"Scenario #{index}: `expected_output.parameters` must be an object."
                )
            return render_tool_call(tool_name.strip(), parameters)
        if not isinstance(expected_output, str) or not expected_output.strip():
            raise ValueError(
                f"Scenario #{index}: `expected_output` must be a non-empty string."
            )
        return expected_output.strip()

    tool_name = item.get("tool_name")
    if tool_name is None:
        return None
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError(f"Scenario #{index}: `tool_name` must be a non-empty string.")
    parameters = item.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError(f"Scenario #{index}: `parameters` must be an object.")
    return render_tool_call(tool_name.strip(), parameters)


def _to_positive_int(value: float | int, field_name: str) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Pole `{field_name}` musi być liczbą całkowitą.") from error
    if resolved <= 0:
        raise ValueError(f"Pole `{field_name}` musi być większe od zera.")
    return resolved
