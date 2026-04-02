"""Personal Assistant runtime — concrete configuration of AgenticRuntime."""

from __future__ import annotations

import threading
from typing import Any
import uuid

from providers.image.mflux import ImageGenerationResult, MfluxImageCall
from agentic.llm_call import LLMCall
from providers.api import OpenAIProvider
from agentic.workflow import WorkflowRuntime
from agentic.workflow._workflow import AgenticWorkflow
from agentic.workflow.messages import (
    AssistantMessage,
    ConversationData,
    Message,
    PromptSnapshot,
    RecordedMessageMetadata,
    UserCommand,
    UserMessage,
)
from agentic_runtime.execution import WorkflowExecution
from agentic_runtime.messaging.message_bus import InMemoryMessageBus
from agentic_runtime.storage.sqlite_store import SQLiteMessageStore
from agentic_runtime.trace import create_tracer, init_tracing
from agentic_runtime.turn_execution import TurnExecutor, TurnPlan, coerce_reply_text
from agentic_runtime.workflow_descriptors import render_workflow_descriptors
from agentic_runtime.workspaces import build_session_id
from structlog import get_logger

from personal_assistant.agents.discovery_notes.detective_workflow import DiscoveryNotesWorkflow
from personal_assistant.agents.manage_notes.manage_notes_workflow import ManageNotesWorkflow
from personal_assistant.agents.organizer.organizer_workflow import OrganizerWorkflow
from personal_assistant.agents.personalize.personlize_workflow import PersonalizeWorkflow
from personal_assistant.agents.personalize.tools import is_personalization_finished
from personal_assistant.agents.router.router_agent import RouterAgent
from personal_assistant.agents.sage.sage_workflow import SageWorkflow
from personal_assistant.output_handlers import (
    KnowledgeBaseTaskRunner,
    build_organizer_output_handler,
    build_rag_output_handler,
)
from personal_assistant.settings import settings

logger = get_logger(__name__)


def _close_knowledge_base() -> None:
    try:
        from knowledge_base import close_knowledge_base
    except ModuleNotFoundError:
        return
    close_knowledge_base()


def _resync_notes() -> None:
    try:
        from knowledge_base import resync_notes
    except ModuleNotFoundError:
        return
    resync_notes()


close_knowledge_base = _close_knowledge_base
resync_notes = _resync_notes


class WorkflowContext:
    def __init__(self, bus: InMemoryMessageBus) -> None:
        self.bus = bus


class PARuntime:
    """Personal Assistant runtime — wraps generic workflow infrastructure with PA agents."""

    def __init__(
        self,
        user_slug: str = "default",
        workspace_slug: str = "default",
    ) -> None:
        init_tracing()
        self._stop_lock = threading.Lock()
        self._stopped = False
        self._tracer = create_tracer(enabled=True)
        self.runtime_id = self._new_runtime_id()
        self.user_slug = user_slug
        self.user_name = user_slug
        self.workspace_slug = workspace_slug
        self.session_id = build_session_id(user_slug, workspace_slug)
        self.store = SQLiteMessageStore(
            path=settings.message_store_path,
            batch_size=settings.message_store_batch_size,
            flush_interval_seconds=settings.message_store_flush_interval_seconds,
        )
        self.bus = InMemoryMessageBus(store=self.store)
        self._workflow_runtime = WorkflowRuntime(
            bus=self.bus,
            tracer=self._tracer,
            max_inline_bytes=settings.message_stream_inline_bytes,
            chunk_bytes=settings.message_stream_chunk_bytes,
        )
        self._turn_executor = self._workflow_runtime.turn_executor
        self.router = RouterAgent()
        workflows = self._build_workflows(self.bus)
        (
            self.personalize_workflow,
            self.note_workflow,
            self.discovery_notes_workflow,
            self.sage_workflow,
            self.organizer_workflow,
        ) = workflows
        self.workflows: dict[str, AgenticWorkflow] = {}
        for workflow in workflows:
            self._workflow_runtime.register_workflow(workflow.description.agent_name, workflow)
        self.workflows = self._workflow_runtime.workflows
        self._kb_task_runner = KnowledgeBaseTaskRunner(resync=resync_notes)
        self._workflow_runtime.register_output_handler(
            build_organizer_output_handler(self.organizer_workflow)
        )
        self._workflow_runtime.register_output_handler(build_rag_output_handler(self._kb_task_runner))

        OpenAIProvider.configure(
            base_url=settings.api_base_url,
            api_key=settings.api_key,
            timeout=settings.api_timeout,
        )

        self.llm_call: LLMCall | None = None
        if settings.chat_model_name:
            self.llm_call = LLMCall(model_name=settings.chat_model_name, tracer=self._tracer)
        self.image_call = MfluxImageCall(
            model_name=settings.image_model_name,
            quantize=settings.image_model_quantize,
            width=settings.image_width,
            height=settings.image_height,
            steps=settings.image_steps,
            output_dir=settings.image_output_dir,
        )
        self._kb_task_runner.submit_resync()

    def _build_workflows(
        self,
        bus: InMemoryMessageBus,
    ) -> tuple[
        PersonalizeWorkflow,
        ManageNotesWorkflow,
        DiscoveryNotesWorkflow,
        SageWorkflow,
        OrganizerWorkflow,
    ]:
        context = WorkflowContext(bus)
        return (
            PersonalizeWorkflow(
                inputs=["UserMessage", "UserCommand"],
                tracer=self._tracer,
                context=context,
            ),
            ManageNotesWorkflow(
                inputs=["UserMessage", "UserCommand"],
                tracer=self._tracer,
                context=context,
            ),
            DiscoveryNotesWorkflow(
                inputs=["UserMessage", "UserCommand"],
                tracer=self._tracer,
                context=context,
            ),
            SageWorkflow(
                inputs=["UserMessage", "UserCommand"],
                tracer=self._tracer,
                context=context,
            ),
            OrganizerWorkflow(
                inputs=["CreatedNote", "UserCommand", "UserMessage"],
                tracer=self._tracer,
                context=context,
            ),
        )

    @staticmethod
    def _new_runtime_id() -> str:
        return str(uuid.uuid7())

    @staticmethod
    def _new_turn_id() -> str:
        return str(uuid.uuid7())

    def _available_workflows_summary(self) -> str:
        return render_workflow_descriptors(list(self._routable_workflows().values()))

    def _routable_workflows(self) -> dict[str, AgenticWorkflow]:
        user_slug = getattr(self, "user_slug", getattr(self, "user_name", "default"))
        personalization_finished = is_personalization_finished(user_slug)
        routable: dict[str, AgenticWorkflow] = {}
        for workflow in self.workflows.values():
            description = workflow.description
            if description.agent_name == "organizer":
                continue
            if description.agent_name == "personalize" and personalization_finished:
                continue
            routable[description.agent_name] = workflow
        return routable

    def _resolve_workflow(self, text: str) -> AgenticWorkflow | None:
        available_summary = self._available_workflows_summary()
        workflow_name = self.router.route(text, available_summary)
        print(f"Resolved workflow: {workflow_name}")
        return self._routable_workflows().get(workflow_name)

    @staticmethod
    def _unknown_workflow_message(available_workflows_summary: str) -> str:
        return (
            "Nie wiem jeszcze, ktorego agenta wybrac dla tej wiadomosci.\n"
            "Dostepni agenci:\n"
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

    def get_initial_greeting(self) -> str:
        return self.start()

    def run(self, text: str) -> str:
        return self.handle(
            UserMessage(
                data=ConversationData(role="user", text=text),
                metadata=RecordedMessageMetadata(
                    runtime_id=self.runtime_id,
                    session_id=self.session_id,
                    domain="general",
                    source="user",
                ),
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

        _run_shutdown_step("knowledge_base_tasks", self._kb_task_runner.close)
        _run_shutdown_step("router", self.router.close)

        for workflow_name, workflow in self.workflows.items():
            close = getattr(workflow, "close", None)
            if callable(close):
                _run_shutdown_step(f"workflow:{workflow_name}", close)

        if self.llm_call is not None:
            _run_shutdown_step("llm_call", self.llm_call.close)
        _run_shutdown_step("image_call", self.image_call.close)
        _run_shutdown_step("knowledge_base", close_knowledge_base)
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
                    data=ConversationData(
                        role="system",
                        text=snapshot.text,
                        payload={"tool_schema": list(snapshot.tool_schema)},
                    ),
                    metadata=RecordedMessageMetadata(
                        runtime_id=message.metadata.runtime_id,
                        turn_id=turn_id,
                        domain="routing",
                        source="router",
                        target=message.metadata.source,
                        prompt_name=snapshot.prompt_name,
                        prompt_hash=snapshot.prompt_hash,
                        agent_run_id=router_response.result.run_id,
                    ),
                )
            )
        messages.append(
            AssistantMessage(
                data=ConversationData(role="assistant", text=workflow_name),
                metadata=RecordedMessageMetadata(
                    runtime_id=message.metadata.runtime_id,
                    turn_id=turn_id,
                    domain="routing",
                    source="router",
                    target=message.metadata.source,
                    agent_run_id=router_response.result.run_id,
                ),
            )
        )
        return tuple(messages)

    def _run_general_fallback(self, message: UserMessage) -> WorkflowExecution:
        if self.llm_call is not None:
            response = self.llm_call.respond(
                message.data.text if isinstance(message.data, ConversationData) and message.data.text else ""
            )
            return WorkflowExecution(
                text=response.output,
                agent_result=response.result,
                tool_results=response.tool_results,
            )
        return WorkflowExecution(
            text=self._unknown_workflow_message(self._available_workflows_summary())
        )

    def _plan_general_turn(self, message: UserMessage) -> TurnPlan:
        available_summary = self._available_workflows_summary()
        route_response = getattr(self.router, "route_response", None)
        message_text = message.data.text if isinstance(message.data, ConversationData) else ""
        if callable(route_response):
            router_response = route_response(message_text, available_summary)
            workflow_name = router_response.output.strip()
        else:
            workflow_name = self.router.route(message_text, available_summary)
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
                    data=ConversationData(role="user", text=message_text),
                    metadata=RecordedMessageMetadata(
                        runtime_id=message.metadata.runtime_id,
                        turn_id=turn_id,
                        domain="general",
                        source=message.metadata.source,
                    ),
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
                data=ConversationData(role="user", text=message_text),
                metadata=RecordedMessageMetadata(
                    runtime_id=message.metadata.runtime_id,
                    turn_id=turn_id,
                    domain=workflow_name,
                    source=message.metadata.source,
                    target=workflow_name,
                ),
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
        target = message.metadata.target
        assert target is not None
        turn_id = message.metadata.turn_id or self._new_turn_id()
        workflow = self.workflows[target]
        return TurnPlan(
            incoming=message.with_metadata(turn_id=turn_id, domain=target),
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
        if (
            isinstance(message, UserMessage)
            and message.metadata.domain == "general"
            and message.metadata.source == "user"
        ):
            return self._handle_general_user_message(message)

        if isinstance(message, UserMessage):
            target = message.metadata.target
            if target and target in self.workflows:
                return self._handle_targeted_message(message)

        if isinstance(message, UserCommand):
            self.bus.publish(message)
            target = message.metadata.domain or message.metadata.target
            if target and target in self.workflows:
                reply = self.workflows[target].handle(message)
                return coerce_reply_text(reply)

        replies = self.bus.publish(message)
        return coerce_reply_text(replies[-1] if replies else "")

    def handle_message(self, text: str) -> str:
        return self.run(text)

    def clear_personalization_history(self) -> None:
        self.bus.clear()
        self.runtime_id = self._new_runtime_id()
        self.personalize_workflow.handle(
            UserCommand(
                type="reset",
                metadata=RecordedMessageMetadata(
                    runtime_id=self.runtime_id,
                    domain="personalize",
                    source="runtime",
                ),
            )
        )
        self.note_workflow.handle(
            UserCommand(
                type="reset",
                metadata=RecordedMessageMetadata(
                    runtime_id=self.runtime_id,
                    domain="manage_notes",
                    source="runtime",
                ),
            )
        )
        self.discovery_notes_workflow.handle(
            UserCommand(
                type="reset",
                metadata=RecordedMessageMetadata(
                    runtime_id=self.runtime_id,
                    domain="discovery_notes",
                    source="runtime",
                ),
            )
        )
        self.sage_workflow.handle(
            UserCommand(
                type="reset",
                metadata=RecordedMessageMetadata(
                    runtime_id=self.runtime_id,
                    domain="sage",
                    source="runtime",
                ),
            )
        )
        self.organizer_workflow.handle(
            UserCommand(
                type="reset",
                metadata=RecordedMessageMetadata(
                    runtime_id=self.runtime_id,
                    domain="organizer",
                    source="runtime",
                ),
            )
        )

    def run_chat(self, text: str) -> str:
        """Direct chat mode — bypasses router, goes straight to LLMCall."""
        if self.llm_call is None:
            return "Chat model is not configured. Set CHAT_MODEL_NAME in .env."
        return self.llm_call.call(text)

    def run_generate_image(
        self,
        text: str,
        images: str | list[str] | None = None,
    ) -> ImageGenerationResult:
        """Direct image mode — bypasses router, goes straight to MFLUX text-to-image."""
        del images
        return self.image_call.generate_image(text)

    def reset_chat(self) -> None:
        """Clear direct chat conversation history."""
        if self.llm_call is not None:
            self.llm_call.reset()

    def close(self) -> None:
        self.stop()
