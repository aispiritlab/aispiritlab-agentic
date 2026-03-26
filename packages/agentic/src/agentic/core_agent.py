from dataclasses import dataclass
from typing import Any, Protocol

from agentic.agent import Agent, AgentResult
from agentic.message import Message
from agentic.models import ModelProvider, ModelConfig
from agentic.observability import LLMTracer
from agentic.prompts import PromptBuilder
from agentic.providers.provider import ModelProviderType
from agentic.tools import ToolRunResult, Toolsets


class Agentic(Protocol):
    def call(self, message: str) -> str:
        ...

    def start(self) -> str:
        ...

    def reset(self) -> None:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True, slots=True)
class CoreAgentResponse:
    result: AgentResult
    output: str
    tool_results: tuple[ToolRunResult, ...] = ()


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
        config: ModelConfig = ModelConfig(),
        **kwargs: Any,
    ) -> None:
        self._model_provider = ModelProvider(model_id, model_provider_type=model_provider_type, config=config)
        self._tracer = tracer

        self._agent = Agent(
            model_provider=self._model_provider,
            prompt_builder=prompt_builder,
            toolsets=toolsets,
            tracer=tracer,
        )
        self._welcome_message = welcome_message

    def respond(
        self,
        message: str | Message,
        *,
        images: str | list[str] | None = None,
    ) -> CoreAgentResponse:
        model_reply = self._agent.run(message, images=images)
        if not model_reply.tool_calls:
            return CoreAgentResponse(result=model_reply, output=model_reply.content)

        outputs: list[str] = []
        tool_results: list[ToolRunResult] = []
        for tool_call in model_reply.tool_calls:
            run_result = self._agent.toolsets.run_tool(tool_call, tracer=self._tracer)
            if run_result is None:
                continue
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

    def reset(self) -> None:
        self._agent.clear_history()

    def close(self) -> None:
        self._agent.clear_history()
        self._model_provider.close()
