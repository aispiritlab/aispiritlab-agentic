"""Structured output: parse and validate LLM responses into typed dataclasses."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import json
from typing import Any, get_type_hints

import structlog

from agentic.exceptions import ModelRetry

logger = structlog.get_logger(__name__)


type OutputValidator = Callable[[Any], Any]


def _extract_json(text: str) -> str:
    """Extract JSON from model response, handling markdown code blocks."""
    stripped = text.strip()
    if stripped.startswith("```json"):
        stripped = stripped.removeprefix("```json").strip()
    elif stripped.startswith("```"):
        stripped = stripped.removeprefix("```").strip()
    if stripped.endswith("```"):
        stripped = stripped.removesuffix("```").strip()
    return stripped


def _coerce_value(value: Any, expected_type: Any) -> Any:
    """Best-effort coercion of a parsed JSON value to the expected type."""
    if value is None:
        return value
    if isinstance(expected_type, type):
        if expected_type is str and not isinstance(value, str):
            return str(value)
        if expected_type is int and isinstance(value, (str, float)):
            return int(value)
        if expected_type is float and isinstance(value, (str, int)):
            return float(value)
        if expected_type is bool and isinstance(value, (str, int)):
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes")
            return bool(value)
    return value


class StructuredOutput:
    """Parse LLM text responses into typed dataclass instances.

    Supports:
      - Frozen dataclasses as output types
      - JSON extraction from markdown-wrapped responses
      - Custom output validators
      - Retry signal on validation failure (raises ``ModelRetry``)
    """

    def __init__(
        self,
        output_type: type | None = None,
        *,
        validators: list[OutputValidator] | None = None,
    ) -> None:
        self._output_type = output_type
        self._validators = validators or []
        self._is_dataclass = output_type is not None and dataclasses.is_dataclass(output_type)

    @property
    def output_type(self) -> type | None:
        return self._output_type

    def is_structured(self, response: str) -> bool:
        """Return True if we should attempt structured parsing."""
        if self._output_type is None:
            return False
        if self._is_dataclass:
            return True
        # Check if response looks like JSON
        stripped = _extract_json(response)
        return stripped.startswith(("{", "["))

    def parse(self, response: str) -> Any:
        """Parse and validate the response.

        Raises ``ModelRetry`` if parsing or validation fails, so the
        framework can feed the error back to the model.
        """
        if self._output_type is None:
            return response

        json_text = _extract_json(response)

        try:
            raw = json.loads(json_text)
        except json.JSONDecodeError as e:
            raise ModelRetry(
                f"Invalid JSON in response: {e}. "
                f"Please respond with valid JSON matching the schema."
            ) from e

        if self._is_dataclass:
            result = self._build_dataclass(raw)
        else:
            result = raw

        for validator in self._validators:
            result = validator(result)

        return result

    def _build_dataclass(self, raw: Any) -> Any:
        """Build a dataclass instance from parsed JSON dict."""
        if not isinstance(raw, dict):
            raise ModelRetry(
                f"Expected a JSON object for {self._output_type.__name__}, "
                f"got {type(raw).__name__}. Please respond with a JSON object."
            )

        hints = get_type_hints(self._output_type)
        fields = {f.name for f in dataclasses.fields(self._output_type)}

        kwargs: dict[str, Any] = {}
        errors: list[str] = []

        for field_name in fields:
            if field_name in raw:
                expected = hints.get(field_name)
                kwargs[field_name] = _coerce_value(raw[field_name], expected)
            elif field_name not in {
                f.name
                for f in dataclasses.fields(self._output_type)
                if f.default is not dataclasses.MISSING
                or f.default_factory is not dataclasses.MISSING
            }:
                errors.append(f"missing required field '{field_name}'")

        extra = set(raw.keys()) - fields
        if extra:
            errors.append(f"unexpected fields: {', '.join(sorted(extra))}")

        if errors:
            detail = "; ".join(errors)
            field_list = ", ".join(sorted(fields))
            raise ModelRetry(
                f"Output validation failed for {self._output_type.__name__}: {detail}. "
                f"Expected fields: {field_list}"
            )

        try:
            return self._output_type(**kwargs)
        except (TypeError, ValueError) as e:
            raise ModelRetry(f"Failed to construct {self._output_type.__name__}: {e}") from e

    def json_schema(self) -> dict[str, Any] | None:
        """Generate a JSON schema for the output type (for prompt injection)."""
        if not self._is_dataclass or self._output_type is None:
            return None

        hints = get_type_hints(self._output_type)
        properties: dict[str, Any] = {}
        required: list[str] = []

        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
        }

        for f in dataclasses.fields(self._output_type):
            hint = hints.get(f.name, str)
            json_type = type_map.get(hint, "string") if isinstance(hint, type) else "string"
            prop: dict[str, Any] = {"type": json_type}
            if f.metadata and "description" in f.metadata:
                prop["description"] = f.metadata["description"]
            properties[f.name] = prop
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                required.append(f.name)

        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required
        return schema
