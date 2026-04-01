from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Literal, Mapping, Protocol

logger = logging.getLogger(__name__)

from agentic.models.response import ModelResponse


def _sanitize_tags(tags: Mapping[str, Any] | None) -> dict[str, str]:
    if not tags:
        return {}

    sanitized: dict[str, str] = {}
    for key, value in tags.items():
        if value is None:
            continue
        sanitized[str(key)] = str(value)
    return sanitized


def _truncate_text(value: str, *, max_length: int = 500) -> str:
    if len(value) <= max_length:
        return value
    return f"{value[:max_length]}..."


def _preview_value(value: Any) -> Any:
    if isinstance(value, str):
        return _truncate_text(value)
    if isinstance(value, Mapping):
        return {str(key): _preview_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_preview_value(item) for item in value[:20]]
    if isinstance(value, tuple):
        return tuple(_preview_value(item) for item in value[:20])
    return value


def _safe_int(value: Any) -> int:
    if not isinstance(value, int):
        return 0
    return max(value, 0)


ContentMode = Literal["none", "preview", "full"]


@dataclass(slots=True)
class _SummaryFrame:
    prefix: str
    llm_call_count: int = 0
    tool_call_count: int = 0
    tool_names: set[str] = field(default_factory=set)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    models: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# SpanHandle: yielded by context managers so callers can annotate spans
# ---------------------------------------------------------------------------


class SpanHandle(Protocol):
    def update(
        self,
        *,
        output: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        level: str | None = None,
    ) -> None: ...


class NoopSpanHandle:
    def update(
        self,
        *,
        output: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        level: str | None = None,
    ) -> None:
        pass


_NOOP_HANDLE = NoopSpanHandle()


class MlflowSpanHandle:
    def __init__(self, span: Any) -> None:
        self._span = span

    def update(
        self,
        *,
        output: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        level: str | None = None,
    ) -> None:
        if self._span is None:
            return
        if output is not None:
            self._span.set_outputs(output)
        if metadata is not None:
            for key, value in metadata.items():
                self._span.set_attribute(str(key), value)
        if level is not None:
            self._span.set_attribute("level", level)


# ---------------------------------------------------------------------------
# TracingContext: structured multi-tenant tags/metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TracingContext:
    session_id: str = ""
    user_id: str = ""
    runtime_id: str = ""
    turn_id: str = ""
    workflow: str = ""
    domain: str = ""
    tags: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TraceSnapshot: active trace/span identity propagated to messages/storage
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TraceSnapshot:
    session_id: str = ""
    trace_id: str = ""
    span_id: str = ""
    parent_span_id: str = ""
    span_name: str = ""
    span_type: str = ""


def build_trace_snapshot(
    trace: TraceSnapshot | None = None,
    *,
    session_id: str = "",
    trace_id: str | None = None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    span_name: str | None = None,
    span_type: str | None = None,
) -> TraceSnapshot | None:
    resolved_trace_id = trace.trace_id if trace is not None and trace.trace_id else (trace_id or "")
    resolved_span_id = trace.span_id if trace is not None and trace.span_id else (span_id or "")
    resolved_parent_span_id = (
        trace.parent_span_id if trace is not None and trace.parent_span_id else (parent_span_id or "")
    )
    resolved_span_name = trace.span_name if trace is not None and trace.span_name else (span_name or "")
    resolved_span_type = trace.span_type if trace is not None and trace.span_type else (span_type or "")
    resolved_session_id = trace.session_id if trace is not None and trace.session_id else session_id

    if not any(
        (
            resolved_session_id,
            resolved_trace_id,
            resolved_span_id,
            resolved_parent_span_id,
            resolved_span_name,
            resolved_span_type,
        )
    ):
        return None

    return TraceSnapshot(
        session_id=resolved_session_id,
        trace_id=resolved_trace_id,
        span_id=resolved_span_id,
        parent_span_id=resolved_parent_span_id,
        span_name=resolved_span_name,
        span_type=resolved_span_type,
    )


# ---------------------------------------------------------------------------
# LLMTracer Protocol
# ---------------------------------------------------------------------------


class LLMTracer(Protocol):
    @contextmanager
    def workflow(
        self,
        *,
        name: str,
        session_id: str,
        user_id: str = "",
        metadata: Mapping[str, Any] | None = None,
        tags: Mapping[str, str] | None = None,
        input: Any | None = None,
        tracing_context: TracingContext | None = None,
    ) -> Iterator[SpanHandle]: ...

    @contextmanager
    def agent(
        self,
        *,
        name: str,
        agent_id: str = "",
        input: Any | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[SpanHandle]: ...

    @contextmanager
    def step(
        self,
        *,
        name: str,
        input: Any | None = None,
        attributes: Mapping[str, Any] | None = None,
        span_type: str = "CHAIN",
    ) -> Iterator[SpanHandle]: ...

    def llm(
        self,
        *,
        name: str,
        model: str,
        messages: list[dict[str, Any]],
        invoke: Callable[..., ModelResponse],
        **kwargs: Any,
    ) -> ModelResponse: ...

    @property
    def current_trace(self) -> TraceSnapshot | None: ...

    @property
    def current_trace_id(self) -> str | None: ...

    def get_trace_url(self) -> str | None: ...

    def flush(self) -> None: ...

    def shutdown(self, timeout_seconds: float = 5.0) -> None: ...


# ---------------------------------------------------------------------------
# NoopLLMTracer
# ---------------------------------------------------------------------------


class NoopLLMTracer:
    @contextmanager
    def workflow(
        self,
        *,
        name: str,
        session_id: str,
        user_id: str = "",
        metadata: Mapping[str, Any] | None = None,
        tags: Mapping[str, str] | None = None,
        input: Any | None = None,
        tracing_context: TracingContext | None = None,
    ) -> Iterator[SpanHandle]:
        yield _NOOP_HANDLE

    @contextmanager
    def agent(
        self,
        *,
        name: str,
        agent_id: str = "",
        input: Any | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[SpanHandle]:
        yield _NOOP_HANDLE

    @contextmanager
    def step(
        self,
        *,
        name: str,
        input: Any | None = None,
        attributes: Mapping[str, Any] | None = None,
        span_type: str = "CHAIN",
    ) -> Iterator[SpanHandle]:
        yield _NOOP_HANDLE

    def llm(
        self,
        *,
        name: str,
        model: str,
        messages: list[dict[str, Any]],
        invoke: Callable[..., ModelResponse],
        **kwargs: Any,
    ) -> ModelResponse:
        return invoke()

    @property
    def current_trace(self) -> TraceSnapshot | None:
        return None

    @property
    def current_trace_id(self) -> str | None:
        return None

    def get_trace_url(self) -> str | None:
        return None

    def flush(self) -> None:
        pass

    def shutdown(self, timeout_seconds: float = 5.0) -> None:
        pass


# ---------------------------------------------------------------------------
# MlflowLLMTracer
# ---------------------------------------------------------------------------


class MlflowLLMTracer:
    def __init__(
        self,
        *,
        auto_flush: bool = True,
        terminate_on_flush: bool = False,
        tracking_uri: str | None = None,
        content_mode: ContentMode = "preview",
    ) -> None:
        self.auto_flush = auto_flush
        self.terminate_on_flush = terminate_on_flush
        self._tracking_uri = tracking_uri
        self._content_mode: ContentMode = content_mode
        self._summary_state = threading.local()
        self._trace_state = threading.local()
        self._mlflow: Any | None = None
        self._span_type_enum: Any | None = None
        try:
            import mlflow
            from mlflow.entities import SpanType

            self._mlflow = mlflow
            self._span_type_enum = SpanType
        except Exception:
            self._mlflow = None
            self._span_type_enum = None

    def _resolve_span_type(self, span_type: str) -> Any | None:
        if self._span_type_enum is None:
            return None
        mapping = {
            "CHAIN": self._span_type_enum.CHAIN,
            "TOOL": self._span_type_enum.TOOL,
            "AGENT": self._span_type_enum.AGENT,
            "CHAT_MODEL": self._span_type_enum.CHAT_MODEL,
            "RETRIEVER": self._span_type_enum.RETRIEVER,
            "EMBEDDING": self._span_type_enum.EMBEDDING,
            "PARSER": self._span_type_enum.PARSER,
            "RERANKER": self._span_type_enum.RERANKER,
        }
        return mapping.get(span_type.upper(), self._span_type_enum.CHAIN)

    def _resolve_content_mode(self, override: Any | None = None) -> ContentMode:
        candidate = self._content_mode if override is None else str(override).lower()
        if candidate in {"none", "preview", "full"}:
            return candidate  # type: ignore[return-value]
        return "preview"

    def _summary_stack(self) -> list[_SummaryFrame]:
        stack = getattr(self._summary_state, "stack", None)
        if stack is None:
            stack = []
            self._summary_state.stack = stack
        return stack

    def _session_stack(self) -> list[str]:
        stack = getattr(self._trace_state, "session_stack", None)
        if stack is None:
            stack = []
            self._trace_state.session_stack = stack
        return stack

    def _push_session_id(self, session_id: str) -> None:
        self._session_stack().append(session_id)

    def _pop_session_id(self, session_id: str) -> None:
        stack = self._session_stack()
        if stack and stack[-1] == session_id:
            stack.pop()
            return
        try:
            stack.remove(session_id)
        except ValueError:
            logger.warning("Attempted to pop non-existent session_id=%s", session_id)

    def _current_session_id(self) -> str:
        stack = self._session_stack()
        return stack[-1] if stack else ""

    def _push_summary(self, prefix: str) -> _SummaryFrame:
        frame = _SummaryFrame(prefix=prefix)
        self._summary_stack().append(frame)
        return frame

    def _pop_summary(self, frame: _SummaryFrame) -> None:
        stack = self._summary_stack()
        if stack and stack[-1] is frame:
            stack.pop()
            return
        try:
            stack.remove(frame)
        except ValueError:
            pass

    def _record_llm_call(self, model: str, result: ModelResponse) -> None:
        input_tokens = _safe_int(result.prompt_tokens)
        output_tokens = _safe_int(result.completion_tokens)
        resolved_model = result.model or model
        for frame in self._summary_stack():
            frame.llm_call_count += 1
            frame.total_input_tokens += input_tokens
            frame.total_output_tokens += output_tokens
            if resolved_model:
                frame.models.add(str(resolved_model))

    def _record_tool_call(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        tool_name = ""
        if attributes is not None:
            raw_tool_name = attributes.get("tool_name")
            if raw_tool_name:
                tool_name = str(raw_tool_name)
        if not tool_name and name.startswith("tool."):
            tool_name = name.split(".", maxsplit=1)[1]
        if not tool_name:
            tool_name = name
        for frame in self._summary_stack():
            frame.tool_call_count += 1
            frame.tool_names.add(tool_name)

    @staticmethod
    def _apply_summary_attributes(span: Any, frame: _SummaryFrame) -> None:
        tool_names = sorted(frame.tool_names)[:20]
        total_tokens = frame.total_input_tokens + frame.total_output_tokens
        span.set_attribute(f"{frame.prefix}.llm_call_count", frame.llm_call_count)
        span.set_attribute(f"{frame.prefix}.iteration_count", frame.llm_call_count)
        span.set_attribute(f"{frame.prefix}.tool_call_count", frame.tool_call_count)
        span.set_attribute(f"{frame.prefix}.unique_tool_count", len(frame.tool_names))
        if tool_names:
            span.set_attribute(f"{frame.prefix}.tool_names", tool_names)
        span.set_attribute("agentic.usage.total_input_tokens", frame.total_input_tokens)
        span.set_attribute("agentic.usage.total_output_tokens", frame.total_output_tokens)
        span.set_attribute("agentic.usage.total_tokens", total_tokens)
        models = sorted(frame.models)
        if models:
            span.set_attribute("agentic.model.primary", models[0])
            span.set_attribute("agentic.model.all", models)

    def _set_chat_tools(self, span: Any, tools: list[dict[str, Any]]) -> None:
        if not tools or self._mlflow is None:
            return
        tracing_module = getattr(self._mlflow, "tracing", None)
        if tracing_module is None:
            return
        helper = getattr(tracing_module, "set_span_chat_tools", None)
        if not callable(helper):
            return
        try:
            helper(span, tools)
        except Exception:
            pass

    @contextmanager
    def _open_span(
        self,
        name: str,
        *,
        span_type: str = "CHAIN",
        input: Any | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[Any]:
        if self._mlflow is None or not hasattr(self._mlflow, "start_span"):
            yield None
            return

        kwargs: dict[str, Any] = {"name": name}
        resolved = self._resolve_span_type(span_type)
        if resolved is not None:
            kwargs["span_type"] = resolved

        span_manager: Any | None = None
        try:
            span_manager = self._mlflow.start_span(**kwargs)
            span = span_manager.__enter__()
        except Exception:
            yield None
            return

        try:
            if input is not None:
                span.set_inputs(input if isinstance(input, dict) else {"input": input})
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(str(key), value)
            yield span
        except BaseException as error:
            try:
                span_manager.__exit__(type(error), error, error.__traceback__)
            except Exception:
                pass
            raise
        else:
            try:
                span_manager.__exit__(None, None, None)
            except Exception:
                pass

    def _wrap_handle(self, span: Any) -> SpanHandle:
        if span is not None:
            return MlflowSpanHandle(span)
        return _NOOP_HANDLE

    def _snapshot_from_span(self, span: Any | None) -> TraceSnapshot | None:
        if span is None:
            return None
        trace_id = getattr(span, "trace_id", None)
        span_id = getattr(span, "span_id", None)
        parent_span_id = getattr(span, "parent_id", None)
        span_name = getattr(span, "name", None)
        span_type = getattr(span, "span_type", None)
        return TraceSnapshot(
            session_id=self._current_session_id(),
            trace_id="" if trace_id is None else str(trace_id),
            span_id="" if span_id is None else str(span_id),
            parent_span_id="" if parent_span_id is None else str(parent_span_id),
            span_name="" if span_name is None else str(span_name),
            span_type="" if span_type is None else str(span_type),
        )

    @contextmanager
    def workflow(
        self,
        *,
        name: str,
        session_id: str,
        user_id: str = "",
        metadata: Mapping[str, Any] | None = None,
        tags: Mapping[str, str] | None = None,
        input: Any | None = None,
        tracing_context: TracingContext | None = None,
    ) -> Iterator[SpanHandle]:
        frame = self._push_summary("agentic.run")
        effective_session = (
            tracing_context.session_id
            if tracing_context and tracing_context.session_id
            else session_id
        )
        try:
            self._push_session_id(effective_session)
            with self._open_span(name, span_type="CHAIN", input=input) as span:
                if self._mlflow is not None and span is not None:
                    effective_user = (
                        tracing_context.user_id
                        if tracing_context and tracing_context.user_id
                        else user_id
                    )
                    trace_metadata: dict[str, str] = {
                        "mlflow.trace.session": effective_session,
                    }
                    if effective_user:
                        trace_metadata["mlflow.trace.user"] = effective_user
                    if tracing_context and tracing_context.metadata:
                        trace_metadata.update(_sanitize_tags(tracing_context.metadata))
                    if metadata:
                        trace_metadata.update(_sanitize_tags(metadata))
                    if tracing_context and tracing_context.runtime_id:
                        trace_metadata["agentic.runtime_id"] = tracing_context.runtime_id
                    if tracing_context and tracing_context.turn_id:
                        trace_metadata["agentic.turn_id"] = tracing_context.turn_id
                    if tracing_context and tracing_context.workflow:
                        trace_metadata["agentic.workflow"] = tracing_context.workflow
                    if tracing_context and tracing_context.domain:
                        trace_metadata["agentic.domain"] = tracing_context.domain

                    effective_tags: dict[str, str] = {}
                    if tracing_context and tracing_context.tags:
                        effective_tags.update(tracing_context.tags)
                    if tags:
                        effective_tags.update(tags)

                    try:
                        self._mlflow.update_current_trace(
                            metadata=trace_metadata,
                            tags=_sanitize_tags(effective_tags),
                        )
                    except Exception:
                        pass
                try:
                    yield self._wrap_handle(span)
                finally:
                    if span is not None:
                        self._apply_summary_attributes(span, frame)
        finally:
            self._pop_session_id(effective_session)
            self._pop_summary(frame)
            if self.auto_flush:
                self.flush()

    @contextmanager
    def agent(
        self,
        *,
        name: str,
        agent_id: str = "",
        input: Any | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[SpanHandle]:
        agent_attrs = dict(attributes or {})
        if agent_id:
            agent_attrs["agent_id"] = agent_id
        frame = self._push_summary("agentic.agent")
        try:
            with self._open_span(name, span_type="AGENT", input=input, attributes=agent_attrs) as span:
                try:
                    yield self._wrap_handle(span)
                finally:
                    if span is not None:
                        self._apply_summary_attributes(span, frame)
        finally:
            self._pop_summary(frame)

    @contextmanager
    def step(
        self,
        *,
        name: str,
        input: Any | None = None,
        attributes: Mapping[str, Any] | None = None,
        span_type: str = "CHAIN",
    ) -> Iterator[SpanHandle]:
        if span_type.upper() == "TOOL":
            self._record_tool_call(name, attributes)
        with self._open_span(name, span_type=span_type, input=input, attributes=attributes) as span:
            yield self._wrap_handle(span)

    def llm(
        self,
        *,
        name: str,
        model: str,
        messages: list[dict[str, Any]],
        invoke: Callable[..., ModelResponse],
        **kwargs: Any,
    ) -> ModelResponse:
        content_mode = self._resolve_content_mode(kwargs.pop("content_mode", None))
        tools = kwargs.pop("tools", None)
        extra_attributes = dict(kwargs.pop("extra_attributes", {}) or {})
        with self._open_span(name, span_type="CHAT_MODEL") as span:
            if span is not None:
                if content_mode == "full":
                    span.set_inputs({"messages": messages, "model": model, "params": kwargs})
                elif content_mode == "preview":
                    span.set_inputs(
                        {
                            "messages": _preview_value(messages),
                            "model": model,
                            "params": _preview_value(kwargs),
                        }
                    )
                else:
                    span.set_inputs({"message_count": len(messages), "model": model})
                if content_mode != "none" and isinstance(tools, list):
                    self._set_chat_tools(span, tools)
                for key, value in extra_attributes.items():
                    span.set_attribute(str(key), value)

            started = time.perf_counter()
            result = invoke()
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            self._record_llm_call(model, result)
            resolved_model = result.model or model

            if span is not None:
                if content_mode == "full":
                    span.set_outputs({"text": result.text})
                elif content_mode == "preview":
                    span.set_outputs({"text": _truncate_text(result.text)})
                span.set_attribute("latency_ms", latency_ms)
                span.set_attribute("gen_ai.request.model", model)
                if resolved_model:
                    span.set_attribute("gen_ai.response.model", resolved_model)
                if result.finish_reason:
                    span.set_attribute("finish_reason", result.finish_reason)
                if result.request_id:
                    span.set_attribute("request_id", result.request_id)
                if result.prompt_tokens:
                    span.set_attribute("gen_ai.usage.input_tokens", result.prompt_tokens)
                    span.set_attribute("prompt_tokens", result.prompt_tokens)
                if result.completion_tokens:
                    span.set_attribute("gen_ai.usage.output_tokens", result.completion_tokens)
                    span.set_attribute("completion_tokens", result.completion_tokens)
                if result.total_tokens:
                    span.set_attribute("total_tokens", result.total_tokens)
                if resolved_model:
                    span.set_attribute("model", resolved_model)

            return result

    @property
    def current_trace(self) -> TraceSnapshot | None:
        if self._mlflow is None:
            return None
        try:
            span = self._mlflow.get_current_active_span()
        except (AttributeError, RuntimeError, OSError):
            return None
        return self._snapshot_from_span(span)

    @property
    def current_trace_id(self) -> str | None:
        snapshot = self.current_trace
        if snapshot is not None and snapshot.trace_id:
            return snapshot.trace_id
        if self._mlflow is None:
            return None
        try:
            span = self._mlflow.get_current_active_span()
            if span is None:
                return None
            request_id = getattr(span, "request_id", None)
            if request_id is not None:
                return str(request_id)
        except Exception:
            return None
        return None

    def get_trace_url(self) -> str | None:
        snapshot = self.current_trace
        trace_id = snapshot.trace_id if snapshot is not None else self.current_trace_id
        if trace_id is None or self._tracking_uri is None:
            return None
        base = self._tracking_uri.rstrip("/")
        return f"{base}/#/traces/{trace_id}"

    def flush(self) -> None:
        if self._mlflow is None:
            return

        def _do_flush() -> None:
            try:
                from mlflow.tracking.fluent import _get_trace_exporter

                exporter = _get_trace_exporter()
                if exporter is not None and hasattr(exporter, "force_flush"):
                    exporter.force_flush(timeout_millis=5000)
                elif exporter is not None and hasattr(exporter, "_async_queue"):
                    exporter._async_queue.flush(terminate=self.terminate_on_flush)
            except Exception:
                pass

        thread = threading.Thread(target=_do_flush, daemon=True)
        thread.start()

    def shutdown(self, timeout_seconds: float = 5.0) -> None:
        if self._mlflow is None:
            return
        try:
            from mlflow.tracking.fluent import _get_trace_exporter

            exporter = _get_trace_exporter()
            if exporter is not None and hasattr(exporter, "force_flush"):
                exporter.force_flush(timeout_millis=int(timeout_seconds * 1000))
            elif exporter is not None and hasattr(exporter, "_async_queue"):
                exporter._async_queue.flush(terminate=True)
        except Exception:
            pass


def build_tracer(
    *,
    enabled: bool = True,
    backend: str = "mlflow",
    tracking_uri: str | None = None,
    content_mode: ContentMode = "preview",
) -> LLMTracer:
    if not enabled:
        return NoopLLMTracer()
    if backend == "mlflow":
        return MlflowLLMTracer(tracking_uri=tracking_uri, content_mode=content_mode)
    return NoopLLMTracer()
