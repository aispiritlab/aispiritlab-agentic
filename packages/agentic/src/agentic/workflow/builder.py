from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from agentic.metadata import Description

from agentic.workflow._workflow import AgenticWorkflow
from agentic.workflow.consumer import ConsumerConfig
from agentic.workflow.execution import WorkflowExecution
from agentic.workflow.messages import Message, UserCommand, UserMessage
from agentic.workflow.reactor import (
    Decider,
    LLMReactor,
    LLMResponse,
    MultiTurnLLMReactor,
    TechnicalRoutingFn,
)
from agentic.workflow.routing import make_llm_routing
from agentic.workflow.runner import run_workflow


type InputMapper = Callable[[Message], UserMessage | None]
type ResponseEventEmitter = Callable[[LLMResponse], Sequence[Message]]
type StartHook = Callable[[], str]
type ResetHook = Callable[[], None]
type CloseHook = Callable[[], None]


def passthrough_decider(message: Message) -> Sequence[Message]:
    if isinstance(message, UserMessage):
        return [message]
    return []


@dataclass(slots=True)
class ConfiguredWorkflow(AgenticWorkflow):
    description: Description
    inputs: tuple[str, ...]
    _routing_fn: TechnicalRoutingFn
    _decider: Decider
    _input_mapper: InputMapper
    _config: ConsumerConfig | None = None
    _on_start: StartHook | None = None
    _on_reset: ResetHook | None = None
    _on_close: CloseHook | None = None

    def handle(self, message: Message) -> WorkflowExecution | str:
        if isinstance(message, UserCommand):
            if message.name == "start" and self._on_start is not None:
                return self._on_start()
            if message.name == "reset":
                if self._on_reset is not None:
                    self._on_reset()
                return ""
            return ""

        prepared = self._input_mapper(message)
        if prepared is None:
            return ""
        return run_workflow(
            message=prepared,
            decider=self._decider,
            routing_fn=self._routing_fn,
            config=self._config,
        )

    def close(self) -> None:
        if self._on_close is not None:
            self._on_close()


class WorkflowBuilder:
    def __init__(self, name: str) -> None:
        self._name = name
        self._agent: object | None = None
        self._inputs: tuple[str, ...] = ("UserMessage", "UserCommand")
        self._reactor_mode = "llm"
        self._reactor_post_process: Callable[[str], str] | None = None
        self._reactor_max_turns = 10
        self._input_mapper: InputMapper | None = None
        self._event_emitter: ResponseEventEmitter | None = None
        self._decider: Decider | None = None
        self._config: ConsumerConfig | None = None
        self._on_start: StartHook | None = None
        self._on_reset: ResetHook | None = None
        self._on_close: CloseHook | None = None
        self._description: Description | None = None

    def agent(self, agent: object) -> WorkflowBuilder:
        self._agent = agent
        close = getattr(agent, "close", None)
        if callable(close):
            self._on_close = close
        return self

    def inputs(self, *message_types: str | type[Message]) -> WorkflowBuilder:
        self._inputs = tuple(
            message_type if isinstance(message_type, str) else message_type.__name__
            for message_type in message_types
        )
        return self

    def reactor(
        self,
        mode: str,
        *,
        post_process: Callable[[str], str] | None = None,
        max_turns: int = 10,
    ) -> WorkflowBuilder:
        self._reactor_mode = mode
        self._reactor_post_process = post_process
        self._reactor_max_turns = max_turns
        return self

    def on_start(self, hook: StartHook) -> WorkflowBuilder:
        self._on_start = hook
        return self

    def on_reset(self, hook: ResetHook) -> WorkflowBuilder:
        self._on_reset = hook
        return self

    def on_close(self, hook: CloseHook) -> WorkflowBuilder:
        self._on_close = hook
        return self

    def map_input(self, mapper: InputMapper) -> WorkflowBuilder:
        self._input_mapper = mapper
        return self

    def emit_events(self, emitter: ResponseEventEmitter) -> WorkflowBuilder:
        self._event_emitter = emitter
        return self

    def decider(self, decider: Decider) -> WorkflowBuilder:
        self._decider = decider
        return self

    def config(self, config: ConsumerConfig) -> WorkflowBuilder:
        self._config = config
        return self

    def describe(
        self,
        description: str,
        *,
        capabilities: Iterable[str] = (),
    ) -> WorkflowBuilder:
        self._description = Description(
            agent_name=self._name,
            description=description,
            capabilities=tuple(capabilities),
        )
        return self

    def build(self) -> ConfiguredWorkflow:
        if self._agent is None:
            raise ValueError("WorkflowBuilder requires .agent(...) before .build().")
        if self._decider is not None and self._event_emitter is not None:
            raise ValueError("Use either .decider(...) or .emit_events(...), not both.")

        reactor = self._build_reactor(self._agent)
        routing = make_llm_routing(reactor)
        decider = self._resolve_decider()
        input_mapper = self._input_mapper or _default_input_mapper
        on_start = self._on_start or _callable_attr(self._agent, "start")
        on_reset = self._on_reset or _callable_attr(self._agent, "reset")

        return ConfiguredWorkflow(
            description=self._resolve_description(self._agent),
            inputs=self._inputs,
            _routing_fn=routing,
            _decider=decider,
            _input_mapper=input_mapper,
            _config=self._config,
            _on_start=on_start,
            _on_reset=on_reset,
            _on_close=self._on_close,
        )

    def _resolve_description(self, agent: object) -> Description:
        if self._description is not None:
            return self._description
        description = getattr(agent, "description", None)
        if isinstance(description, Description):
            return description
        return Description(
            agent_name=self._name,
            description=f"{self._name} workflow",
            capabilities=(),
        )

    def _build_reactor(self, agent: object) -> object:
        if self._reactor_mode == "llm":
            return LLMReactor(agent=agent)  # type: ignore[arg-type]
        if self._reactor_mode == "multiturn_llm":
            return MultiTurnLLMReactor(
                agent=agent,  # type: ignore[arg-type]
                max_turns=self._reactor_max_turns,
                post_process=self._reactor_post_process,
            )
        raise ValueError(f"Unsupported reactor mode: {self._reactor_mode}")

    def _resolve_decider(self) -> Decider:
        if self._decider is not None:
            return self._decider
        if self._event_emitter is None:
            return passthrough_decider

        emitter = self._event_emitter

        def eventful_decider(message: Message) -> Sequence[Message]:
            if isinstance(message, UserMessage):
                return [message]
            if isinstance(message, LLMResponse):
                return tuple(emitter(message))
            return []

        return eventful_decider


def define_workflow(name: str) -> WorkflowBuilder:
    return WorkflowBuilder(name)


def _default_input_mapper(message: Message) -> UserMessage | None:
    if isinstance(message, UserMessage):
        return message
    return None


def _callable_attr(target: object, name: str) -> Callable | None:
    value = getattr(target, name, None)
    if callable(value):
        return value
    return None
