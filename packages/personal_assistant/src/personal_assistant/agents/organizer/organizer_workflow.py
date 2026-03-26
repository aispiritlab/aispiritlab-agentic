from __future__ import annotations

from structlog import get_logger
from agentic.prompts import PromptTemplate
from agentic.prompts import QwenPromptBuilder
from agentic.tools import Toolsets
from agentic.workflow import WorkflowBuilder
from agentic.workflow._workflow import AgenticWorkflow
from registry import Prompts

from agentic.workflow.messages import Message, UserCommand, UserMessage
from agentic_runtime.execution import WorkflowExecution
from agentic_runtime.messaging.consumer import MessageConsumer
from agentic_runtime.messaging.message_stream import InMemoryMessageStream
from agentic_runtime.reactor import LLMReactor
from agentic_runtime.routing import make_llm_routing
from agentic_runtime.workflow_runner import _build_execution

from personal_assistant.deciders import make_organizer_decider
from personal_assistant.messaging.events import CreatedNote
from personal_assistant.settings import settings

from .organizer_agent import OrganizerAgent
from .tools import toolset as organizer_toolset

logger = get_logger(__name__)


class OrganizerWorkflow(AgenticWorkflow):
    def __init__(self, *args, **kwargs) -> None:
        tracer = kwargs.pop("tracer", None)
        context = kwargs.pop("context", None)
        self.context = context
        self.inputs = kwargs.pop("inputs", [])
        self._agent = OrganizerAgent(
            model_id=settings.model_name,
            prompt_builder=QwenPromptBuilder(external_prompt_name=Prompts.ORGANIZER),
            toolsets=Toolsets([organizer_toolset]),
            tracer=tracer,
            context=context,
            **kwargs,
        )
        self._reactor = LLMReactor(agent=self._agent)
        self._routing = make_llm_routing(self._reactor)
        self._decider = make_organizer_decider()

        def _map_input(message: Message) -> UserMessage | None:
            if isinstance(message, CreatedNote):
                payload = PromptTemplate(
                    template=(
                        "Nazwa notatki: {note_name}\n"
                        "Treść notatki:\n"
                        "{note_content}\n"
                    ),
                    context_variables=["note_name", "note_content"],
                ).format(
                    note_name=message.note_name,
                    note_content=message.note_content,
                )
                return UserMessage(
                    text=payload,
                    domain=message.domain,
                    runtime_id=message.runtime_id,
                    turn_id=message.turn_id,
                    source=message.source,
                    target=self._agent.description.agent_name,
                )
            if isinstance(message, UserMessage):
                return message
            return None

        self._workflow = (
            WorkflowBuilder(self._agent.description.agent_name)
            .agent(self._agent)
            .inputs(*self.inputs)
            .map_input(_map_input)
            .build()
        )
        self.description = self._workflow.description

    def handle(self, message: Message) -> WorkflowExecution | str:
        workflow = getattr(self, "_workflow", None)
        if workflow is not None:
            return workflow.handle(message)
        logger.info(f"Handling message: {message}")
        if isinstance(message, UserCommand):
            if message.name == "start":
                return self._agent.start()
            if message.name == "reset":
                self._agent.reset()
            return ""
        if isinstance(message, (CreatedNote, UserMessage)):
            stream = InMemoryMessageStream()
            consumer = MessageConsumer()
            stream.append(message)
            consumer.consume(stream, self._decider, self._routing)
            return _build_execution(stream.all_messages())
        return ""

    def close(self) -> None:
        workflow = getattr(self, "_workflow", None)
        if workflow is not None:
            workflow.close()
            return
        self._agent.close()
