from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable
import uuid

from agentic.metadata import Description
from agentic.observability import LLMTracer, NoopLLMTracer

from agentic.workflow._workflow import AgenticWorkflow
from agentic.workflow.message_bus import InMemoryMessageBus
from agentic.workflow.messages import Message, UserCommand, UserMessage
from agentic.workflow.output_handler import WorkflowOutputHandler
from agentic.workflow.turn_execution import TurnExecutor, TurnPlan, coerce_reply_text


type WorkflowHandler = Callable[[Message], Any]


@dataclass(slots=True)
class FunctionWorkflow(AgenticWorkflow):
    description: Description
    _handler: WorkflowHandler
    inputs: tuple[str, ...] = ("UserMessage",)

    def handle(self, message: Message) -> Any:
        return self._handler(message)

    def close(self) -> None:
        return None


class WorkflowRuntime:
    def __init__(
        self,
        *,
        bus: InMemoryMessageBus | None = None,
        tracer: LLMTracer | None = None,
        max_inline_bytes: int = 4096,
        chunk_bytes: int = 4096,
    ) -> None:
        self.bus = bus or InMemoryMessageBus()
        self._tracer = tracer or NoopLLMTracer()
        self.turn_executor = TurnExecutor(
            bus=self.bus,
            tracer=self._tracer,
            max_inline_bytes=max_inline_bytes,
            chunk_bytes=chunk_bytes,
        )
        self.workflows: dict[str, AgenticWorkflow] = {}

    def register_workflow(
        self,
        name: str,
        workflow: AgenticWorkflow | WorkflowHandler,
    ) -> AgenticWorkflow:
        normalized = self._coerce_workflow(name, workflow)
        self.workflows[name] = normalized
        return normalized

    def register_output_handler(self, handler: WorkflowOutputHandler) -> None:
        self.bus.register_output_handler(handler)

    def publish(self, message: Message) -> list[str]:
        return self.bus.publish(message)

    def publish_many(self, messages: list[Message]) -> list[str]:
        return self.bus.publish_many(messages)

    def flush_output_handlers(self) -> list[str]:
        return self.bus.flush_output_handlers()

    @property
    def message_log(self) -> list[Message]:
        return list(self.bus.messages)

    def publish_workflow_execution(
        self,
        *,
        incoming: UserMessage,
        workflow_name: str,
        execution: Any,
    ) -> str | None:
        return self.turn_executor.publish_workflow_execution(
            incoming=incoming,
            workflow_name=workflow_name,
            execution=execution,
        )

    def execute_turn(self, plan: TurnPlan) -> str:
        return self.turn_executor.execute(plan)

    def execute_workflow(
        self,
        workflow_name: str,
        incoming: UserMessage,
        *,
        selected_workflow: str | None = None,
        pre_messages: tuple[Message, ...] = (),
        trace_name: str | None = None,
        lifecycle_domain: str | None = None,
        lifecycle_target: str | None = None,
        output_agent_name: str | None = None,
    ) -> str:
        workflow = self.workflows[workflow_name]
        message = replace(
            incoming,
            domain=lifecycle_domain or workflow_name,
            target=lifecycle_target if lifecycle_target is not None else (incoming.target or workflow_name),
            runtime_id=incoming.runtime_id or self._new_id(),
            turn_id=incoming.turn_id or self._new_id(),
        )
        return self.execute_turn(
            TurnPlan(
                incoming=message,
                handler=workflow.handle,
                trace_name=trace_name or workflow_name,
                lifecycle_domain=lifecycle_domain or workflow_name,
                lifecycle_target=lifecycle_target if lifecycle_target is not None else message.target,
                lifecycle_workflow_name=workflow_name,
                output_agent_name=output_agent_name or workflow_name,
                selected_workflow=selected_workflow,
                pre_messages=pre_messages,
            )
        )

    def handle(self, message: Message) -> str:
        if isinstance(message, UserMessage):
            workflow_name = message.target or message.domain
            if workflow_name in self.workflows:
                return self.execute_workflow(workflow_name, message)

        if isinstance(message, UserCommand):
            self.bus.publish(message)
            workflow_name = message.target or message.domain
            if workflow_name in self.workflows:
                return coerce_reply_text(self.workflows[workflow_name].handle(message))

        replies = self.bus.publish(message)
        return coerce_reply_text(replies[-1] if replies else "")

    def run_text(
        self,
        text: str,
        workflow_name: str,
        *,
        runtime_id: str | None = None,
        turn_id: str | None = None,
        source: str = "user",
    ) -> str:
        return self.execute_workflow(
            workflow_name,
            UserMessage(
                runtime_id=runtime_id or self._new_id(),
                turn_id=turn_id or self._new_id(),
                domain=workflow_name,
                source=source,
                target=workflow_name,
                text=text,
            ),
        )

    def close(self) -> None:
        self.bus.close()

    @staticmethod
    def _new_id() -> str:
        return str(uuid.uuid7())

    @staticmethod
    def _coerce_workflow(
        name: str,
        workflow: AgenticWorkflow | WorkflowHandler,
    ) -> AgenticWorkflow:
        if hasattr(workflow, "handle"):
            return workflow  # type: ignore[return-value]
        return FunctionWorkflow(
            description=Description(
                agent_name=name,
                description=f"{name} workflow",
                capabilities=(),
            ),
            _handler=workflow,
        )
