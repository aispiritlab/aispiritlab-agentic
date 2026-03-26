from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

import structlog

from agentic.observability import LLMTracer, NoopLLMTracer

from ._tools import Command, JsonRepairer, Tool, ToolCall, ToolCallCommand, ToolContext


logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ToolRunResult:
    tool_call: ToolCall
    output: str


class Toolset:
    def __init__(self, tools: Sequence[Callable | Tool]):
        self._tools: list[Tool] = []
        for tool in tools:
            if isinstance(tool, Tool):
                self._tools.append(tool)
            elif callable(tool):
                self._tools.append(Tool(tool))
            else:
                raise ValueError(f"Expected Tool or callable, got {type(tool)}")

    def execute(
        self,
        function_name: str,
        parameters: dict[str, Any],
        *,
        tool_context: ToolContext | None = None,
    ) -> Any:
        for tool in self._tools:
            if tool.name == function_name:
                self._validate_parameters(tool, function_name, parameters)
                return tool.call(parameters, tool_context=tool_context)
        raise ValueError(f"Tool '{function_name}' not found in toolset")

    @staticmethod
    def _validate_parameters(tool: Tool, function_name: str, parameters: dict[str, Any]) -> None:
        provided = set(parameters.keys())
        missing = sorted(tool.required_parameters - provided)
        unexpected = sorted(provided - tool.all_parameters)

        if missing:
            missing_params = ", ".join(missing)
            raise ValueError(
                f"Błąd: brak wymaganych parametrów dla narzędzia '{function_name}': {missing_params}."
            )

        if unexpected:
            unexpected_params = ", ".join(unexpected)
            raise ValueError(
                f"Błąd: nieznane parametry dla narzędzia '{function_name}': {unexpected_params}."
            )

    def has_tool(self, function_name: str) -> bool:
        return any(tool.name == function_name for tool in self._tools)

    @property
    def tools(self) -> tuple[Tool, ...]:
        return tuple(self._tools)


class Toolsets(Sequence[Toolset]):
    def __init__(
        self,
        toolsets: Sequence[Toolset] | None = None,
        *,
        json_repairer: JsonRepairer | None = None,
    ):
        self._toolsets = list(toolsets or [])
        self._json_repairer = json_repairer
        self._validate_unique_tool_names()
        self._tool_names = tuple(
            tool.name
            for toolset in self._toolsets
            for tool in toolset.tools
        )
        self._command_to_tool: dict[type[Command], str] = {}
        for toolset in self._toolsets:
            for tool_def in toolset.tools:
                if tool_def.command_class is not None:
                    self._command_to_tool[tool_def.command_class] = tool_def.name

    @classmethod
    def from_sources(
        cls,
        *,
        tools: Sequence[Callable | Tool] | None = None,
        toolsets: Sequence[Toolset] | Toolsets | None = None,
        json_repairer: JsonRepairer | None = None,
    ) -> Toolsets:
        merged_toolsets: list[Toolset] = []
        if tools:
            merged_toolsets.append(Toolset(tools))

        if isinstance(toolsets, Toolsets):
            merged_toolsets.extend(toolsets._toolsets)
            if json_repairer is None:
                json_repairer = toolsets._json_repairer
        elif toolsets:
            merged_toolsets.extend(toolsets)

        return cls(merged_toolsets, json_repairer=json_repairer)

    def __len__(self) -> int:
        return len(self._toolsets)

    def __getitem__(self, index: int) -> Toolset:
        return self._toolsets[index]

    def _validate_unique_tool_names(self) -> None:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for toolset in self._toolsets:
            for tool in toolset.tools:
                if tool.name in seen:
                    duplicates.add(tool.name)
                seen.add(tool.name)

        if duplicates:
            duplicate_names = ", ".join(sorted(duplicates))
            raise ValueError(f"Duplicate tool names found: {duplicate_names}")

    def tool_exists(self, function_name: str) -> bool:
        return any(toolset.has_tool(function_name) for toolset in self._toolsets)

    def detect_tool(self, payload: Any) -> bool:
        if not self._tool_names:
            return False

        if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[0], str):
            return payload[0] in self._tool_names

        if isinstance(payload, dict):
            for key in ("name", "tool", "function"):
                value = payload.get(key)
                if isinstance(value, str) and value in self._tool_names:
                    return True
            return False

        if not isinstance(payload, str):
            return False

        return any(tool_name in payload for tool_name in self._tool_names)

    @staticmethod
    def is_tool_error(result: str) -> bool:
        return result.startswith("Błąd:") or result.startswith("Error:")

    @staticmethod
    def _coerce_tool_call(payload: Any, repairer: JsonRepairer | None = None) -> ToolCall | None:
        return Tool.parse_tool_definition(payload, repairer=repairer)

    def parse_tool(self, payload: Any) -> Command | None:
        """Parse model response into a typed Command without executing."""
        tool_call = self._coerce_tool_call(payload, repairer=self._json_repairer)
        if tool_call is None:
            return None
        function_name, parameters = tool_call
        for toolset in self._toolsets:
            for tool_def in toolset.tools:
                if tool_def.name == function_name:
                    return tool_def.create_command(parameters)
        return None

    def execute(
        self,
        command: Command,
        *,
        tool_context: ToolContext | None = None,
        tracer: LLMTracer | None = None,
    ) -> ToolRunResult:
        """Execute an already-parsed Command."""
        resolved_tracer = tracer or NoopLLMTracer()
        if isinstance(command, ToolCallCommand):
            function_name = command.function_name
            params = dict(command.parameters)
        else:
            cmd_type = type(command)
            if cmd_type not in self._command_to_tool:
                raise ValueError(f"No tool registered for command type {cmd_type.__name__}")
            function_name = self._command_to_tool[cmd_type]
            params = asdict(command)

        tool_call_tuple: ToolCall = (function_name, params)
        with resolved_tracer.step(
            name=f"tool.{function_name}",
            span_type="TOOL",
            input=params,
            attributes={"tool_name": function_name},
        ) as span:
            for toolset in self._toolsets:
                if toolset.has_tool(function_name):
                    try:
                        result = toolset.execute(function_name, params, tool_context=tool_context)
                    except Exception as error:
                        error_text = str(error)
                        span.update(level="ERROR", output={"error": error_text})
                        if self.is_tool_error(error_text):
                            return ToolRunResult(tool_call=tool_call_tuple, output=error_text)
                        return ToolRunResult(tool_call=tool_call_tuple, output=f"Error: {error_text}")
                    output = "" if result is None else str(result)
                    span.update(output={"output": output[:500]})
                    return ToolRunResult(tool_call=tool_call_tuple, output=output)

        return ToolRunResult(
            tool_call=tool_call_tuple,
            output=f"Error: tool '{function_name}' does not exist.",
        )

    def run_tool(
        self,
        payload: Any,
        *,
        tool_context: ToolContext | None = None,
        tracer: LLMTracer | None = None,
    ) -> ToolRunResult | None:
        """Backward compat: parse + execute in one step."""
        command = self.parse_tool(payload)
        logger.info("run_tool", command=command)
        if command is None:
            tool_call = self._coerce_tool_call(payload, repairer=self._json_repairer)
            if tool_call is not None:
                function_name = tool_call[0]
                return ToolRunResult(
                    tool_call=tool_call,
                    output=f"Error: tool '{function_name}' does not exist.",
                )
            return None
        return self.execute(command, tool_context=tool_context, tracer=tracer)
