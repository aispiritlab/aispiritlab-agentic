"""Generic agent orchestration runtime.

Accepts workflows, a router, and output handlers as constructor parameters.
No application-specific agents are hardcoded here.
"""

from __future__ import annotations

from dataclasses import replace
import threading
from typing import Any, Callable, Protocol
import uuid

from agentic.providers.api import OpenAIProvider
from agentic.workflow import WorkflowRuntime
from agentic.workflow._workflow import AgenticWorkflow
from structlog import get_logger

from agentic_runtime.messaging.message_bus import InMemoryMessageBus
from agentic_runtime.messaging.messages import (
    AssistantMessage,
    Message,
    PromptSnapshot,
    UserCommand,
    UserMessage,
)
from agentic_runtime.storage.sqlite_store import SQLiteMessageStore

from .execution import WorkflowExecution
from .output_handler import WorkflowOutputHandler
from .settings import Settings, settings as default_settings
from .trace import create_tracer, init_tracing
from .turn_execution import TurnExecutor, TurnPlan, coerce_reply_text

logger = get_logger(__name__)


class RouterProtocol(Protocol):
    """Minimal protocol for a workflow router."""

    def route(self, message: str, available_workflows_summary: str) -> str: ...

    def start(self) -> str: ...

    def close(self) -> None: ...


class AgenticRuntime:
    """Generic agent orchestration runtime.

    Accepts pre-built workflows, a router, and output handlers.
    Application-specific logic belongs in a subclass or wrapper (e.g. PARuntime).
    """

    def __init__(
        self,
        *,
        workflows: list[AgenticWorkflow],
        router: RouterProtocol,
        output_handlers: list[WorkflowOutputHandler] | None = None,
        workflow_filter: Callable[[AgenticWorkflow], bool] | None = None,
        on_stop: Callable[[], None] | None = None,
        settings: Settings | None = None,
    ) -> None:
        _settings = settings or default_settings
        init_tracing()
        self._stop_lock = threading.Lock()
        self._stopped = False
        self._tracer = create_tracer(enabled=True)
        self.runtime_id = self._new_runtime_id()
        self.store = SQLiteMessageStore(
            path=_settings.message_store_path,
            batch_size=_settings.message_store_batch_size,
            flush_interval_seconds=_settings.message_store_flush_interval_seconds,
        )
        self.bus = InMemoryMessageBus(store=self.store)
        self._workflow_runtime = WorkflowRuntime(
            bus=self.bus,
            tracer=self._tracer,
            max_inline_bytes=_settings.message_stream_inline_bytes,
            chunk_bytes=_settings.message_stream_chunk_bytes,
        )
        self._turn_executor = self._workflow_runtime.turn_executor
        self.router = router
        self._workflow_filter = workflow_filter
        self._on_stop = on_stop

        self.workflows: dict[str, AgenticWorkflow] = {}
        for workflow in workflows:
            self._workflow_runtime.register_workflow(workflow.description.agent_name, workflow)
        self.workflows = self._workflow_runtime.workflows

        for handler in output_handlers or []:
            self._workflow_runtime.register_output_handler(handler)

        OpenAIProvider.configure(
            base_url=_settings.api_base_url,
            api_key=_settings.api_key,
            timeout=_settings.api_timeout,
        )

    @staticmethod
    def _new_runtime_id() -> str:
        return str(uuid.uuid7())

    @staticmethod
    def _new_turn_id() -> str:
        return str(uuid.uuid7())

    def _available_workflows_summary(self) -> str:
        lines = []
        for workflow in self._routable_workflows().values():
            description = workflow.description
            capabilities = ", ".join(description.capabilities)
            workflow_name = workflow.__class__.__name__
            lines.append(
                f"- {description.agent_name} (workflow: {workflow_name}): "
                f"{description.description} (capabilities: {capabilities})"
            )
        return "\n".join(lines)

    def _routable_workflows(self) -> dict[str, AgenticWorkflow]:
        routable: dict[str, AgenticWorkflow] = {}
        for workflow in self.workflows.values():
            if self._workflow_filter and not self._workflow_filter(workflow):
                continue
            routable[workflow.description.agent_name] = workflow
        return routable

    def _resolve_workflow(self, text: str) -> AgenticWorkflow | None:
        available_summary = self._available_workflows_summary()
        workflow_name = self.router.route(text, available_summary)
        print(f"Resolved workflow: {workflow_name}")
        return self._routable_workflows().get(workflow_name)

    @staticmethod
    def _unknown_workflow_message(available_workflows_summary: str) -> str:
        return (
            "No matching agent found for this message.\n"
            "Available agents:\n"
            f"{available_workflows_summary}"
        )

    def _get_turn_executor(self) -> TurnExecutor:
        runtime = getattr(self, "_workflow_runtime", None)
        if runtime is not None:
            return runtime.turn_executor
        executor = getattr(self, "_turn_executor", None)
        if executor is None:
            executor = TurnExecutor(bus=self.bus, tracer=self._tracer)
            self._turn_executor = executor
        return executor

    def _publish_workflow_execution(
        self,
        *,
        incoming: UserMessage,
        workflow_name: str,
        execution: WorkflowExecution,
    ) -> str | None:
        runtime = getattr(self, "_workflow_runtime", None)
        if runtime is not None:
            return runtime.publish_workflow_execution(
                incoming=incoming,
                workflow_name=workflow_name,
                execution=execution,
            )
        return self._get_turn_executor().publish_workflow_execution(
            incoming=incoming,
            workflow_name=workflow_name,
            execution=execution,
        )

    def start(self) -> str:
        return self.router.start()

    def run(self, text: str) -> str:
        return self.handle(
            UserMessage(
                runtime_id=self.runtime_id,
                domain="general",
                source="user",
                text=text,
            )
        )

    def stop(self) -> None:
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True

        errors: list[Exception] = []

        def _run_shutdown_step(name: str, action) -> None:  # noqa: ANN001
            try:
                action()
            except Exception as error:
                logger.warning("runtime_shutdown_step_failed", step=name, error=str(error))
                errors.append(error)

        if self._on_stop:
            _run_shutdown_step("on_stop", self._on_stop)

        _run_shutdown_step("router", self.router.close)

        for workflow_name, workflow in self.workflows.items():
            close = getattr(workflow, "close", None)
            if callable(close):
                _run_shutdown_step(f"workflow:{workflow_name}", close)

        _run_shutdown_step("bus", self.bus.close)
        _run_shutdown_step("tracer", self._tracer.shutdown)

        if errors:
            raise errors[0]

    def _build_router_messages(
        self,
        *,
        message: UserMessage,
        turn_id: str,
        workflow_name: str,
        router_response: Any | None,
    ) -> tuple[Message, ...]:
        if router_response is None:
            return ()

        messages: list[Message] = []
        if router_response.result.prompt_snapshot is not None:
            snapshot = router_response.result.prompt_snapshot
            messages.append(
                PromptSnapshot(
                    runtime_id=message.runtime_id,
                    turn_id=turn_id,
                    domain="routing",
                    source="router",
                    target=message.source,
                    text=snapshot.text,
                    payload={"tool_schema": list(snapshot.tool_schema)},
                    prompt_name=snapshot.prompt_name,
                    prompt_hash=snapshot.prompt_hash,
                    agent_run_id=router_response.result.run_id,
                )
            )
        messages.append(
            AssistantMessage(
                runtime_id=message.runtime_id,
                turn_id=turn_id,
                domain="routing",
                source="router",
                target=message.source,
                text=workflow_name,
                agent_run_id=router_response.result.run_id,
            )
        )
        return tuple(messages)

    def _run_general_fallback(self, message: UserMessage) -> WorkflowExecution:
        return WorkflowExecution(
            text=self._unknown_workflow_message(self._available_workflows_summary())
        )

    def _plan_general_turn(self, message: UserMessage) -> TurnPlan:
        available_summary = self._available_workflows_summary()
        route_response = getattr(self.router, "route_response", None)
        if callable(route_response):
            router_response = route_response(message.text, available_summary)
            workflow_name = router_response.output.strip()
        else:
            workflow_name = self.router.route(message.text, available_summary)
            router_response = None
        workflow = self._routable_workflows().get(workflow_name)
        turn_id = self._new_turn_id()
        pre_messages = self._build_router_messages(
            message=message,
            turn_id=turn_id,
            workflow_name=workflow_name,
            router_response=router_response,
        )

        if workflow is None:
            return TurnPlan(
                incoming=UserMessage(
                    runtime_id=message.runtime_id,
                    turn_id=turn_id,
                    domain="general",
                    source=message.source,
                    text=message.text,
                ),
                handler=self._run_general_fallback,
                trace_name="general",
                lifecycle_domain="general",
                lifecycle_target=None,
                lifecycle_workflow_name="general",
                output_agent_name="assistant",
                pre_messages=pre_messages,
            )

        return TurnPlan(
            incoming=UserMessage(
                runtime_id=message.runtime_id,
                turn_id=turn_id,
                domain=workflow_name,
                source=message.source,
                target=workflow_name,
                text=message.text,
            ),
            handler=workflow.handle,
            trace_name=workflow_name,
            lifecycle_domain=workflow_name,
            lifecycle_target=workflow_name,
            lifecycle_workflow_name=workflow_name,
            output_agent_name=workflow_name,
            selected_workflow=workflow_name,
            pre_messages=pre_messages,
        )

    def _handle_general_user_message(self, message: UserMessage) -> str:
        return self._get_turn_executor().execute(self._plan_general_turn(message))

    def _plan_targeted_turn(self, message: UserMessage) -> TurnPlan:
        target = message.target
        assert target is not None
        turn_id = message.turn_id or self._new_turn_id()
        workflow = self.workflows[target]
        return TurnPlan(
            incoming=replace(message, turn_id=turn_id, domain=target),
            handler=workflow.handle,
            trace_name=target,
            lifecycle_domain=target,
            lifecycle_target=target,
            lifecycle_workflow_name=target,
            output_agent_name=target,
        )

    def _handle_targeted_message(self, message: UserMessage) -> str:
        return self._get_turn_executor().execute(self._plan_targeted_turn(message))

    def handle(self, message: Message) -> str:
        if isinstance(message, UserMessage) and message.domain == "general" and message.source == "user":
            return self._handle_general_user_message(message)

        if isinstance(message, UserMessage):
            target = message.target
            if target and target in self.workflows:
                return self._handle_targeted_message(message)

        if isinstance(message, UserCommand):
            self.bus.publish(message)
            target = message.domain or message.target
            if target and target in self.workflows:
                reply = self.workflows[target].handle(message)
                return coerce_reply_text(reply)

        replies = self.bus.publish(message)
        return coerce_reply_text(replies[-1] if replies else "")

    def handle_message(self, text: str) -> str:
        return self.run(text)

    def close(self) -> None:
        self.stop()
