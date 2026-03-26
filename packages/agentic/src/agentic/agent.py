from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
from typing import Any, AsyncGenerator, Callable, Sequence
from uuid import uuid4

import structlog

from agentic.history import History
from agentic.memory import InMemory, Memory
from agentic.message import Message, SystemMessage, UserMessage
from agentic.models import ModelProvider
from agentic.models.response import ModelResponse
from agentic.observability import LLMTracer, NoopLLMTracer
from agentic.prompts import PromptBuilder as PromptBuilder
from agentic.response_parser import ResponseParser
from agentic.structured_output import StructuredOutput
from agentic.tools import JsonRepairer, Tool, ToolCall, Toolset, Toolsets, build_chat_tools

logger = structlog.get_logger(__name__)

ToolOutput = str


@dataclass(frozen=True)
class PromptSnapshot:
    text: str
    prompt_name: str | None = None
    prompt_hash: str = ""
    tool_schema: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class Context:
    add_history_to_context: bool = True
    max_history_messages: int = 20
    observability_tags: dict[str, Any] | None = None


@dataclass(frozen=True)
class PromptContext:
    message: str
    history_text: str = ""
    memory_context: str = ""


@dataclass(frozen=True)
class PromptArtifacts:
    prompt: str | list[dict[str, str]]
    system_prompt_text: str
    prompt_name: str | None = None
    prompt_hash: str = ""
    tool_schema: list[dict[str, Any]] = field(default_factory=list)

    def __contains__(self, item: object) -> bool:
        return isinstance(self.prompt, str) and isinstance(item, str) and item in self.prompt

    def __str__(self) -> str:
        if isinstance(self.prompt, str):
            return self.prompt
        return str(self.prompt)


class AgentResult:
    def __init__(
        self,
        content: str,
        *,
        reasoning: str = "",
        tool_calls: list[ToolCall] | None = None,
        usage: dict[str, Any] | None = None,
        trace_id: str | None = None,
        run_id: str | None = None,
        prompt_snapshot: PromptSnapshot | None = None,
    ) -> None:
        self.content = content
        self.reasoning = reasoning
        self.tool_calls: list[ToolCall] = tool_calls or []
        self.usage = usage or {}
        self.trace_id = trace_id
        self.run_id = run_id
        self.prompt_snapshot = prompt_snapshot

    @property
    def tool_call(self) -> ToolCall | None:
        if not self.tool_calls:
            return None
        return self.tool_calls[0]


def _message_preview(message: str, max_length: int = 128) -> str:
    if len(message) <= max_length:
        return message
    return f"{message[:max_length]}..."


def _serialize_tool_schema(toolsets: Toolsets) -> list[dict[str, Any]]:
    schema: list[dict[str, Any]] = []
    for toolset in toolsets:
        for tool in toolset.tools:
            schema.append(
                {
                    "name": tool.name,
                    "description": tool.doc,
                    "arguments": [dict(argument) for argument in tool.args],
                }
            )
    return schema


def _build_trace_messages(
    message_text: str,
    prompt_artifacts: PromptArtifacts,
) -> list[dict[str, Any]]:
    if isinstance(prompt_artifacts.prompt, list):
        return [dict(message) for message in prompt_artifacts.prompt]
    messages: list[dict[str, Any]] = []
    if prompt_artifacts.system_prompt_text:
        messages.append({"role": "system", "content": prompt_artifacts.system_prompt_text})
    messages.append({"role": "user", "content": message_text})
    return messages


class Agent:
    def __init__(
        self,
        model_provider: ModelProvider,
        *,
        prompt_builder: PromptBuilder,
        context: Context | None = None,
        memory: Memory | None = None,
        history: History | None = None,
        tools: Sequence[Callable | Tool] | None = None,
        toolsets: Toolsets | Sequence[Toolset] | None = None,
        json_repairer: JsonRepairer | None = None,
        structured_output: StructuredOutput | None = None,
        tracer: LLMTracer | None = None,
    ) -> None:
        self._model_provider = model_provider
        self._toolsets = Toolsets.from_sources(
            tools=tools,
            toolsets=toolsets,
            json_repairer=json_repairer,
        )
        self._prompt_builder = prompt_builder
        self._context = context or Context(add_history_to_context=True)
        self._memory = memory or InMemory()
        self._history = history or History()
        self._agent_id = str(uuid4())
        self._tracer = tracer or NoopLLMTracer()
        self._response_parser = ResponseParser(self._toolsets, structured_output)

    @property
    def history(self) -> History:
        return self._history

    @history.setter
    def history(self, value: History) -> None:
        self._history = value

    @property
    def toolsets(self) -> Toolsets:
        return self._toolsets

    @property
    def context(self) -> Context:
        return self._context

    @property
    def system_prompt(self) -> PromptBuilder:
        return self._prompt_builder

    @system_prompt.setter
    def system_prompt(self, value: PromptBuilder) -> None:
        self._prompt_builder = value

    def _render_system_prompt(self) -> tuple[str, list[dict[str, Any]]]:
        tool_prompt = (
            "\n".join(
                self._prompt_builder.tools_instruction(toolset)
                for toolset in self._toolsets
                if toolset.tools
            )
            if self._toolsets
            else ""
        )
        system_template = self._prompt_builder.system_prompt or ""
        rendered = system_template.replace("{tools}", tool_prompt).strip()
        return rendered, _serialize_tool_schema(self._toolsets)

    def _gather_context(self, message_text: str, current_context: Context) -> PromptContext:
        history_text = (
            self._history.conversation_text(current_context.max_history_messages)
            if current_context.add_history_to_context
            else ""
        )
        # Keep the staged memory API active so custom Memory implementations can
        # adopt the future prompt-context contract before prompt injection lands.
        memory_context = self._memory.summary()
        return PromptContext(
            message=message_text,
            history_text=history_text,
            memory_context=memory_context,
        )

    @staticmethod
    def _render_input_turn(message: str | Message) -> str:
        if isinstance(message, Message):
            as_turn = getattr(message, "as_turn", None)
            if callable(as_turn):
                return as_turn()
            return str(message)
        return UserMessage(str(message)).as_turn()

    def _build_prompt(
        self,
        prompt_context: PromptContext,
        current_message: str | Message | None = None,
    ) -> PromptArtifacts:
        # TODO: Inject prompt_context.memory_context once memory summaries are part
        # of the end-to-end prompt contract. For now prompts intentionally include
        # only conversation history and the active user turn.
        rendered_input_turn = self._render_input_turn(
            prompt_context.message if current_message is None else current_message
        )
        message_with_history = "\n".join(
            turn
            for turn in [
                prompt_context.history_text,
                rendered_input_turn,
            ]
            if turn
        )
        system_prompt_text, tool_schema = self._render_system_prompt()
        prompt = self._prompt_builder.build_prompt(message_with_history, toolsets=self._toolsets)
        prompt_name = getattr(self._prompt_builder, "external_prompt_name", None)
        return PromptArtifacts(
            prompt=prompt,
            system_prompt_text=system_prompt_text,
            prompt_name=prompt_name or None,
            prompt_hash=hashlib.sha256(system_prompt_text.encode("utf-8")).hexdigest(),
            tool_schema=tool_schema,
        )

    def _call_model(self, prompt: str | list[dict[str, str]], **kwargs: Any) -> ModelResponse:
        with self._model_provider.session("model") as model:
            if model is None:
                load_error: str | None = None
                get_load_error = getattr(self._model_provider, "get_load_error", None)
                if callable(get_load_error):
                    load_error = get_load_error("model")
                if load_error:
                    raise RuntimeError(
                        f"Model is not available for inference: {load_error}"
                    )
                raise RuntimeError("Model is not available for inference.")
            return model.response(prompt, **kwargs)

    def _to_result(
        self,
        model_response: ModelResponse,
        *,
        run_id: str,
        prompt_artifacts: PromptArtifacts,
        trace_id: str | None,
    ) -> AgentResult:
        parsed = self._response_parser.parse(model_response.text)
        return AgentResult(
            content=parsed.content,
            reasoning=parsed.reasoning,
            tool_calls=list(parsed.tool_calls),
            usage={
                "prompt_tokens": model_response.prompt_tokens,
                "completion_tokens": model_response.completion_tokens,
                "total_tokens": model_response.total_tokens,
                "latency_ms": model_response.latency_ms,
                "model": model_response.model,
                "finish_reason": model_response.finish_reason,
            },
            trace_id=trace_id,
            run_id=run_id,
            prompt_snapshot=PromptSnapshot(
                text=prompt_artifacts.system_prompt_text,
                prompt_name=prompt_artifacts.prompt_name,
                prompt_hash=prompt_artifacts.prompt_hash,
                tool_schema=list(prompt_artifacts.tool_schema),
            ),
        )

    def _store_history(self, message: str | Message, response_content: str) -> None:
        if isinstance(message, Message):
            self._history.add(message)
        else:
            self._history.add(UserMessage(message))
        self._history.add(SystemMessage(response_content))

    def run(
        self,
        message: str | Message,
        ctx: Context | None = None,
        *,
        images: str | list[str] | None = None,
    ) -> AgentResult:
        current_context = ctx or self._context
        message_text = str(message)
        run_id = str(uuid4())

        model_kwargs: dict[str, Any] = {}
        if images is not None:
            model_kwargs["image"] = images

        with self._tracer.agent(
            name="agent.run",
            agent_id=self._agent_id,
            input=_message_preview(message_text),
            attributes={
                "run_id": run_id,
                "history_enabled": current_context.add_history_to_context,
            },
        ) as span:
            prompt_context = self._gather_context(message_text, current_context)
            prompt_artifacts = self._build_prompt(prompt_context, message)
            trace_messages = _build_trace_messages(message_text, prompt_artifacts)
            chat_tools = build_chat_tools(prompt_artifacts.tool_schema)

            model_response = self._tracer.llm(
                name="llm-call",
                model=getattr(self._model_provider, "_model_name", None) or "",
                messages=trace_messages,
                tools=chat_tools,
                extra_attributes={
                    "agentic.prompt_hash": prompt_artifacts.prompt_hash,
                    "agentic.tool_count": len(prompt_artifacts.tool_schema),
                    **(
                        {"agentic.prompt_name": prompt_artifacts.prompt_name}
                        if prompt_artifacts.prompt_name
                        else {}
                    ),
                },
                invoke=lambda: self._call_model(prompt_artifacts.prompt, **model_kwargs),
            )

            result = self._to_result(
                model_response,
                run_id=run_id,
                prompt_artifacts=prompt_artifacts,
                trace_id=self._tracer.current_trace_id,
            )
            span.update(output={"content": result.content[:200]})
            self._store_history(message, result.content)
            return result

    async def arun(self, message: str | Message, ctx: Context | None = None) -> AgentResult:
        return await asyncio.to_thread(self.run, message, ctx)

    async def astream(
        self,
        message: str | Message,
        ctx: Context | None = None,
    ) -> AsyncGenerator[dict[str, str | AgentResult], None]:
        result = await self.arun(message, ctx)
        yield {"type": "result", "result": result}

    def clear_history(self) -> None:
        self._history.clear()
