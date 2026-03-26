import pytest

from agentic.models.response import ModelResponse
from agentic.observability import (
    MlflowLLMTracer,
    MlflowSpanHandle,
    NoopLLMTracer,
    NoopSpanHandle,
    TracingContext,
)


class _FakeSpan:
    def __init__(self) -> None:
        self.inputs: dict | None = None
        self.outputs: dict | None = None
        self.attributes: dict = {}
        self.trace_id: str = "fake-trace-id"

    def set_inputs(self, inputs: dict) -> None:
        self.inputs = inputs

    def set_outputs(self, outputs: dict) -> None:
        self.outputs = outputs

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def __enter__(self) -> "_FakeSpan":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeTracing:
    @staticmethod
    def set_span_chat_tools(span: _FakeSpan, tools: list[dict]) -> None:
        span.set_attribute("mlflow.chat.tools", tools)


class _FakeMlflow:
    def __init__(self) -> None:
        self.spans: list[_FakeSpan] = []
        self._active_span: _FakeSpan | None = None
        self.flushed = False
        self.trace_metadata: dict = {}
        self.trace_tags: dict = {}
        self.tracing = _FakeTracing()

    def start_span(self, **kwargs) -> _FakeSpan:
        span = _FakeSpan()
        self.spans.append(span)
        self._active_span = span
        return span

    def get_current_active_span(self) -> _FakeSpan | None:
        return self._active_span

    def update_current_trace(self, metadata=None, tags=None) -> None:
        if metadata:
            self.trace_metadata.update(metadata)
        if tags:
            self.trace_tags.update(tags)

    def flush_trace_async_logging(self, terminate: bool = False) -> None:
        self.flushed = True


def _make_tracer(
    tracking_uri: str | None = None,
) -> tuple[MlflowLLMTracer, _FakeMlflow]:
    tracer = MlflowLLMTracer(tracking_uri=tracking_uri)
    fake = _FakeMlflow()
    tracer._mlflow = fake
    tracer._span_type_enum = None
    return tracer, fake


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------


def test_mlflow_tracer_preserves_body_exception() -> None:
    tracer, _ = _make_tracer()

    with pytest.raises(ValueError, match="boom"):
        with tracer.agent(name="test", agent_id="a1"):
            raise ValueError("boom")


def test_mlflow_tracer_workflow_sets_session_metadata() -> None:
    tracer, fake = _make_tracer()

    with tracer.workflow(name="sage", session_id="sess-1", user_id="user-1"):
        pass

    assert fake.trace_metadata["mlflow.trace.session"] == "sess-1"
    assert fake.trace_metadata["mlflow.trace.user"] == "user-1"


def test_mlflow_tracer_llm_captures_response() -> None:
    tracer, fake = _make_tracer()

    response = ModelResponse(
        text="hello",
        model="test-model",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        finish_reason="stop",
        request_id="req-1",
    )

    result = tracer.llm(
        name="llm-call",
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        invoke=lambda: response,
    )

    assert result.text == "hello"
    assert len(fake.spans) == 1
    span = fake.spans[0]
    assert span.inputs == {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "test-model",
        "params": {},
    }
    assert span.outputs == {"text": "hello"}
    assert span.attributes["model"] == "test-model"
    assert span.attributes["prompt_tokens"] == 10
    assert span.attributes["completion_tokens"] == 5
    assert span.attributes["total_tokens"] == 15
    assert span.attributes["finish_reason"] == "stop"
    assert span.attributes["request_id"] == "req-1"
    assert span.attributes["gen_ai.request.model"] == "test-model"
    assert span.attributes["gen_ai.response.model"] == "test-model"
    assert span.attributes["gen_ai.usage.input_tokens"] == 10
    assert span.attributes["gen_ai.usage.output_tokens"] == 5


def test_mlflow_tracer_content_mode_none_skips_prompt_and_response_bodies() -> None:
    tracer = MlflowLLMTracer(content_mode="none")
    fake = _FakeMlflow()
    tracer._mlflow = fake
    tracer._span_type_enum = None

    response = ModelResponse(text="hello", model="test-model")

    tracer.llm(
        name="llm-call",
        model="test-model",
        messages=[{"role": "user", "content": "sensitive prompt"}],
        invoke=lambda: response,
    )

    span = fake.spans[0]
    assert span.inputs == {"message_count": 1, "model": "test-model"}
    assert span.outputs is None


def test_mlflow_tracer_records_chat_tools_when_available() -> None:
    tracer, fake = _make_tracer()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Lookup a note",
                "parameters": {"type": "object", "properties": {"name": {"type": "string"}}},
            },
        }
    ]

    tracer.llm(
        name="llm-call",
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        invoke=lambda: ModelResponse(text="hello", model="test-model"),
    )

    assert fake.spans[0].attributes["mlflow.chat.tools"] == tools


def test_mlflow_tracer_rolls_up_summary_attributes() -> None:
    tracer, fake = _make_tracer()

    with tracer.workflow(name="workflow", session_id="sess-1"):
        with tracer.agent(name="agent.run", agent_id="agent-1"):
            tracer.llm(
                name="llm-call",
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
                invoke=lambda: ModelResponse(
                    text="hello",
                    model="test-model",
                    prompt_tokens=10,
                    completion_tokens=5,
                ),
            )
            with tracer.step(name="tool.lookup", span_type="TOOL", attributes={"tool_name": "lookup"}):
                pass

    workflow_span = fake.spans[0]
    agent_span = fake.spans[1]
    assert workflow_span.attributes["agentic.run.llm_call_count"] == 1
    assert workflow_span.attributes["agentic.run.iteration_count"] == 1
    assert workflow_span.attributes["agentic.run.tool_call_count"] == 1
    assert workflow_span.attributes["agentic.run.unique_tool_count"] == 1
    assert workflow_span.attributes["agentic.usage.total_input_tokens"] == 10
    assert workflow_span.attributes["agentic.usage.total_output_tokens"] == 5
    assert workflow_span.attributes["agentic.usage.total_tokens"] == 15
    assert workflow_span.attributes["agentic.model.primary"] == "test-model"
    assert workflow_span.attributes["agentic.model.all"] == ["test-model"]
    assert agent_span.attributes["agentic.agent.llm_call_count"] == 1
    assert agent_span.attributes["agentic.agent.tool_call_count"] == 1
    assert agent_span.attributes["agentic.agent.tool_names"] == ["lookup"]


def test_noop_tracer_llm_calls_invoke() -> None:
    tracer = NoopLLMTracer()
    response = ModelResponse(text="ok")
    result = tracer.llm(
        name="llm",
        model="m",
        messages=[],
        invoke=lambda: response,
    )
    assert result.text == "ok"


def test_noop_tracer_context_managers() -> None:
    tracer = NoopLLMTracer()
    with tracer.workflow(name="w", session_id="s"):
        with tracer.agent(name="a"):
            with tracer.step(name="s"):
                pass
    assert tracer.current_trace_id is None


def test_noop_tracer_flush_is_noop() -> None:
    tracer = NoopLLMTracer()
    tracer.flush()


# ---------------------------------------------------------------------------
# SpanHandle tests
# ---------------------------------------------------------------------------


def test_noop_span_handle_update_is_noop() -> None:
    handle = NoopSpanHandle()
    handle.update(output={"x": 1}, metadata={"k": "v"}, level="ERROR")


def test_mlflow_span_handle_sets_outputs() -> None:
    span = _FakeSpan()
    handle = MlflowSpanHandle(span)
    handle.update(output={"result": "ok"})
    assert span.outputs == {"result": "ok"}


def test_mlflow_span_handle_sets_metadata() -> None:
    span = _FakeSpan()
    handle = MlflowSpanHandle(span)
    handle.update(metadata={"key": "val"})
    assert span.attributes["key"] == "val"


def test_mlflow_span_handle_sets_level() -> None:
    span = _FakeSpan()
    handle = MlflowSpanHandle(span)
    handle.update(level="ERROR")
    assert span.attributes["level"] == "ERROR"


def test_mlflow_span_handle_none_span_is_safe() -> None:
    handle = MlflowSpanHandle(None)
    handle.update(output={"x": 1}, metadata={"k": "v"}, level="ERROR")


# ---------------------------------------------------------------------------
# SpanHandle yielded from context managers
# ---------------------------------------------------------------------------


def test_workflow_yields_span_handle() -> None:
    tracer, fake = _make_tracer()
    with tracer.workflow(name="w", session_id="s") as handle:
        assert isinstance(handle, MlflowSpanHandle)
        handle.update(output={"status": "done"})
    assert fake.spans[0].outputs == {"status": "done"}


def test_agent_yields_span_handle() -> None:
    tracer, fake = _make_tracer()
    with tracer.agent(name="a") as handle:
        assert isinstance(handle, MlflowSpanHandle)
        handle.update(output={"content": "hi"})
    assert fake.spans[0].outputs == {"content": "hi"}


def test_step_yields_span_handle() -> None:
    tracer, fake = _make_tracer()
    with tracer.step(name="s") as handle:
        assert isinstance(handle, MlflowSpanHandle)
        handle.update(output={"result": "ok"})
    assert fake.spans[0].outputs == {"result": "ok"}


def test_noop_tracer_yields_noop_handle() -> None:
    tracer = NoopLLMTracer()
    with tracer.workflow(name="w", session_id="s") as handle:
        assert isinstance(handle, NoopSpanHandle)
    with tracer.agent(name="a") as handle:
        assert isinstance(handle, NoopSpanHandle)
    with tracer.step(name="s") as handle:
        assert isinstance(handle, NoopSpanHandle)


# ---------------------------------------------------------------------------
# Trace URL generation
# ---------------------------------------------------------------------------


def test_get_trace_url_returns_none_without_active_span() -> None:
    tracer, _ = _make_tracer(tracking_uri="http://localhost:5001")
    # No active span — _active_span is None after _make_tracer
    tracer._mlflow._active_span = None  # type: ignore[union-attr]
    assert tracer.get_trace_url() is None


def test_get_trace_url_builds_mlflow_url() -> None:
    tracer, fake = _make_tracer(tracking_uri="http://localhost:5001")
    fake._active_span = _FakeSpan()
    assert tracer.get_trace_url() == "http://localhost:5001/#/traces/fake-trace-id"


def test_get_trace_url_strips_trailing_slash() -> None:
    tracer, fake = _make_tracer(tracking_uri="http://localhost:5001/")
    fake._active_span = _FakeSpan()
    assert tracer.get_trace_url() == "http://localhost:5001/#/traces/fake-trace-id"


def test_get_trace_url_returns_none_without_tracking_uri() -> None:
    tracer, fake = _make_tracer(tracking_uri=None)
    fake._active_span = _FakeSpan()
    assert tracer.get_trace_url() is None


def test_noop_get_trace_url_returns_none() -> None:
    tracer = NoopLLMTracer()
    assert tracer.get_trace_url() is None


# ---------------------------------------------------------------------------
# TracingContext
# ---------------------------------------------------------------------------


def test_tracing_context_merges_with_workflow_params() -> None:
    tracer, fake = _make_tracer()
    ctx = TracingContext(
        session_id="ctx-sess",
        user_id="ctx-user",
        tags={"env": "staging"},
        metadata={"tenant": "acme"},
    )

    with tracer.workflow(
        name="w",
        session_id="explicit-sess",
        user_id="explicit-user",
        tags={"version": "1"},
        tracing_context=ctx,
    ):
        pass

    # TracingContext session/user win over explicit params
    assert fake.trace_metadata["mlflow.trace.session"] == "ctx-sess"
    assert fake.trace_metadata["mlflow.trace.user"] == "ctx-user"
    # TracingContext metadata is included
    assert fake.trace_metadata["tenant"] == "acme"
    # Both tag sources are merged
    assert fake.trace_tags["env"] == "staging"
    assert fake.trace_tags["version"] == "1"


def test_tracing_context_falls_back_to_explicit_params() -> None:
    tracer, fake = _make_tracer()
    ctx = TracingContext()  # all defaults (empty)

    with tracer.workflow(
        name="w",
        session_id="explicit-sess",
        user_id="explicit-user",
        tracing_context=ctx,
    ):
        pass

    # Empty TracingContext → explicit params used
    assert fake.trace_metadata["mlflow.trace.session"] == "explicit-sess"
    assert fake.trace_metadata["mlflow.trace.user"] == "explicit-user"
