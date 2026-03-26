from __future__ import annotations

import importlib

from evaluation.contracts import EvaluationDefinition


def load_evaluation_definition(spec: str) -> EvaluationDefinition:
    module_name, separator, attribute_name = spec.partition(":")
    if not separator or not module_name.strip() or not attribute_name.strip():
        raise ValueError(
            "Definition spec must use the `module:attribute` format, "
            "for example `agentic_runtime.manage_notes.evaluation:NOTES_EVALUATION`."
        )

    module = importlib.import_module(module_name.strip())
    value = getattr(module, attribute_name.strip(), None)
    if value is None:
        raise ValueError(
            f"Definition attribute '{attribute_name.strip()}' was not found in "
            f"module '{module_name.strip()}'."
        )

    if callable(value) and not isinstance(value, EvaluationDefinition):
        value = value()

    if not isinstance(value, EvaluationDefinition):
        raise TypeError(
            f"Definition '{spec}' must resolve to EvaluationDefinition, "
            f"got {type(value).__name__}."
        )
    return value
