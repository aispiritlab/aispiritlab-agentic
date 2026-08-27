from dataclasses import dataclass
import json
from typing import Any, Protocol

from agentic.agent import Agent, AgentResult
from agentic.exceptions import DEFAULT_RETRY, ModelRetry, RetryPolicy
from agentic.message import Message, ToolMessage
from agentic.observability import LLMTracer
from agentic.prompts import PromptBuilder
from agentic.tools import ToolRunResult, Toolsets
from providers.models import ModelConfig
from providers.orchestrator import ModelProvider, ModelProviderType


class Agentic(Protocol):
    def call(self, message: str) -> str: ...

    def start(self) -> str: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class CoreAgentResponse:
    result: AgentResult
    output: str
    tool_results: tuple[ToolRunResult, ...] = ()


def _retry_as_exception(result: ToolRunResult) -> ModelRetry:
    """Wrap a retryable ToolRunResult into a ModelRetry for policy checking."""
    return ModelRetry(result.output)


class CoreAgentic(Agentic):
    def __init__(
        self,
        model_id: str,
        prompt_builder: PromptBuilder,
        toolsets: Toolsets | None = None,
        welcome_message: str | None = "",
        tracer: LLMTracer | None = None,
        *,
        model_provider_type: ModelProviderType = "mlx",
        config: ModelConfig | None = None,
        retry_policy: RetryPolicy = DEFAULT_RETRY,
        **kwargs: Any,
    ) -> None:
        self._model_provider = ModelProvider(
            model_id,
            model_provider_type=model_provider_type,
            config=config or ModelConfig(),
        )
        self._tracer = tracer
        self._retry_policy = retry_policy

        self._agent = Agent(
            model_provider=self._model_provider,
            prompt_builder=prompt_builder,
            toolsets=toolsets,
            tracer=tracer,
            retry_policy=retry_policy,
            **kwargs,
        )
        self._welcome_message = welcome_message

    def respond(
        self,
        message: str | Message,
        *,
        images: str | list[str] | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> CoreAgentResponse:
        with self._agent.turn_lock:
            return self._respond_locked(message, images=images, retry_policy=retry_policy)

    def _respond_locked(
        self,
        message: str | Message,
        *,
        images: str | list[str] | None,
        retry_policy: RetryPolicy | None,
    ) -> CoreAgentResponse:
        policy = retry_policy or self._retry_policy
        model_reply = self._agent.run(message, images=images)
        if not model_reply.tool_calls:
            return CoreAgentResponse(result=model_reply, output=model_reply.content_text)

        outputs: list[str] = []
        tool_results: list[ToolRunResult] = []
        attempt = 0

        for tool_call in model_reply.tool_calls:
            run_result = self._agent.run_tool(
                tool_call,
                tracer=self._tracer,
                run_id=model_reply.run_id,
            )
            if run_result is None:
                continue

            while run_result.retry and policy.should_retry(
                attempt, _retry_as_exception(run_result)
            ):
                attempt += 1
                tool_name, tool_args = run_result.tool_call
                retry_msg = ToolMessage(
                    f"Tool '{tool_name}' failed: {run_result.output}\n"
                    f"Arguments: {json.dumps(tool_args, ensure_ascii=False)}\n"
                    f"Please fix and try again."
                )
                model_reply = self._agent.run(retry_msg)
                if not model_reply.tool_calls:
                    return CoreAgentResponse(
                        result=model_reply,
                        output=model_reply.content_text,
                        tool_results=tuple(tool_results),
                    )
                run_result = self._agent.run_tool(
                    model_reply.tool_calls[0],
                    tracer=self._tracer,
                    run_id=model_reply.run_id,
                )
                if run_result is None:
                    break

            if run_result is not None:
                tool_results.append(run_result)
                outputs.append(run_result.output)

        return CoreAgentResponse(
            result=model_reply,
            output="\n".join(outputs),
            tool_results=tuple(tool_results),
        )

    def call(self, message: str) -> str:
        return self.respond(message).output

    def preload_model(self) -> None:
        model = self._model_provider.model
        if model is not None:
            return

        load_error = self._model_provider.get_load_error("model")
        if load_error:
            raise RuntimeError(f"Model is not available for inference: {load_error}")
        raise RuntimeError("Model is not available for inference.")

    def start(self) -> str:
        if self._welcome_message is None:
            raise NotImplementedError("Workflow does not define a welcome message.")
        self._agent.clear_history()
        return self._welcome_message

    @property
    def turn_lock(self):
        """Serialises turns on the underlying agent."""
        return self._agent.turn_lock

    def reset(self) -> None:
        self._agent.clear_history()

    def close(self) -> None:
        self._agent.clear_history()
        self._model_provider.close()
