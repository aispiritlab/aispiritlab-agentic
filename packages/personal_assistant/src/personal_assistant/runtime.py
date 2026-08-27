"""Personal Assistant runtime — concrete configuration of AgenticRuntime."""

from __future__ import annotations

from collections.abc import Sequence

from structlog import get_logger

from agentic.llm_call import LLMCall
from agentic.workflow._workflow import AgenticWorkflow
from agentic.workflow.messages import (
    ConversationData,
    Message,
    RecordedMessageMetadata,
    UserCommand,
)
from agentic_runtime.execution import WorkflowExecution
from agentic_runtime.messaging.message_bus import InMemoryMessageBus
from agentic_runtime.output_handler import WorkflowOutputHandler
from agentic_runtime.runtime import AgenticRuntime, RuntimeServices, ShutdownStep
from agentic_runtime.workspaces import build_session_id
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
from providers.image.mflux import ImageGenerationResult, MfluxImageCall

logger = get_logger(__name__)

# Agents the router must never pick: organizer reacts to events, personalize
# disappears once the user has finished onboarding.
_EVENT_DRIVEN_AGENTS = frozenset({"organizer"})


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


class PARuntime(AgenticRuntime):
    """Personal Assistant runtime — the five PA agents on the generic runtime."""

    def __init__(
        self,
        user_slug: str = "default",
        workspace_slug: str = "default",
    ) -> None:
        self.user_slug = user_slug
        self.user_name = user_slug
        self.workspace_slug = workspace_slug
        self._kb_task_runner = KnowledgeBaseTaskRunner(resync=resync_notes)

        super().__init__(
            workflows=self._build_workflows,
            router=RouterAgent(),
            output_handlers=self._build_output_handlers,
            workflow_filter=self._is_routable,
            # Late-bound so a replaced task runner is still the one closed.
            on_stop=lambda: self._kb_task_runner.close(),
            session_id=build_session_id(user_slug, workspace_slug),
            settings=settings,
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

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def _build_workflows(self, services: RuntimeServices) -> Sequence[AgenticWorkflow]:
        context = WorkflowContext(services.bus)
        tracer = services.tracer
        self.personalize_workflow = PersonalizeWorkflow(
            inputs=["UserMessage", "UserCommand"],
            tracer=tracer,
            context=context,
        )
        self.note_workflow = ManageNotesWorkflow(
            inputs=["UserMessage", "UserCommand"],
            tracer=tracer,
            context=context,
        )
        self.discovery_notes_workflow = DiscoveryNotesWorkflow(
            inputs=["UserMessage", "UserCommand"],
            tracer=tracer,
            context=context,
        )
        self.sage_workflow = SageWorkflow(
            inputs=["UserMessage", "UserCommand"],
            tracer=tracer,
            context=context,
        )
        self.organizer_workflow = OrganizerWorkflow(
            inputs=["CreatedNote", "UserCommand", "UserMessage"],
            tracer=tracer,
            context=context,
        )
        return (
            self.personalize_workflow,
            self.note_workflow,
            self.discovery_notes_workflow,
            self.sage_workflow,
            self.organizer_workflow,
        )

    def _build_output_handlers(
        self,
        services: RuntimeServices,
    ) -> Sequence[WorkflowOutputHandler]:
        del services
        return (
            build_organizer_output_handler(self.organizer_workflow),
            build_rag_output_handler(self._kb_task_runner),
        )

    def _is_routable(self, workflow: AgenticWorkflow) -> bool:
        agent_name = workflow.description.agent_name
        if agent_name in _EVENT_DRIVEN_AGENTS:
            return False
        return not (agent_name == "personalize" and is_personalization_finished(self.user_slug))

    def _extra_shutdown_steps(self) -> Sequence[ShutdownStep]:
        steps: list[ShutdownStep] = []
        if self.llm_call is not None:
            steps.append(("llm_call", self.llm_call.close))
        steps.append(("image_call", self.image_call.close))
        steps.append(("knowledge_base", close_knowledge_base))
        return steps

    # ------------------------------------------------------------------
    # PA-specific behaviour
    # ------------------------------------------------------------------

    def _run_general_fallback(self, message: Message) -> WorkflowExecution:
        """Unrouted messages fall through to the plain chat model when configured."""
        if self.llm_call is not None:
            text = (
                message.data.text
                if isinstance(message.data, ConversationData) and message.data.text
                else ""
            )
            response = self.llm_call.respond(text)
            return WorkflowExecution(
                text=response.output,
                agent_result=response.result,
                tool_results=response.tool_results,
            )
        return super()._run_general_fallback(message)

    def clear_personalization_history(self) -> None:
        self.bus.clear()
        self.runtime_id = self._new_runtime_id()
        for domain, workflow in (
            ("personalize", self.personalize_workflow),
            ("manage_notes", self.note_workflow),
            ("discovery_notes", self.discovery_notes_workflow),
            ("sage", self.sage_workflow),
            ("organizer", self.organizer_workflow),
        ):
            workflow.handle(
                UserCommand(
                    type="reset",
                    metadata=RecordedMessageMetadata(
                        runtime_id=self.runtime_id,
                        session_id=self.session_id,
                        domain=domain,
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
