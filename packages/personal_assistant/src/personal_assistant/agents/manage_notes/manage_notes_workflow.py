from __future__ import annotations

from structlog import get_logger

from agentic.prompts import QwenPromptBuilder
from agentic.tools import Toolsets
from agentic.workflow import WorkflowBuilder
from agentic.workflow._workflow import AgenticWorkflow
from registry import Prompts

from agentic.workflow.messages import Message, UserCommand, UserMessage
from agentic_runtime.execution import WorkflowExecution
from agentic_runtime.reactor import LLMReactor
from agentic_runtime.routing import make_llm_routing
from agentic_runtime.workflow_runner import run_workflow

from personal_assistant.deciders import make_manage_notes_decider
from personal_assistant.messaging.events import CreatedNote, NoteUpdated
from personal_assistant.settings import settings

from .manage_notes_agent import ManageNotesAgent
from .tools import toolset as manage_notes_toolset

logger = get_logger(__name__)


class ManageNotesWorkflow(AgenticWorkflow):
    def __init__(self, *args, **kwargs) -> None:
        tracer = kwargs.pop("tracer", None)
        self.context = kwargs.pop("context", None)
        self.inputs = kwargs.pop("inputs", [])
        self._agent = ManageNotesAgent(
            model_id=settings.model_name,
            prompt_builder=QwenPromptBuilder(external_prompt_name=Prompts.MANAGE_NOTES),
            toolsets=Toolsets([manage_notes_toolset]),
            tracer=tracer,
            **kwargs,
        )
        self._reactor = LLMReactor(agent=self._agent)
        self._routing = make_llm_routing(self._reactor)
        self._decider = make_manage_notes_decider(
            toolsets=self._agent._agent.toolsets,
            resolve_note_path=self._agent._resolve_note_path,
            agent_name=self._agent.description.agent_name,
        )

        def _emit_events(response) -> tuple[Message, ...]:  # noqa: ANN001
            if not response.has_tool_calls:
                return ()

            tool_call = response.tool_calls[0]
            command = self._agent._agent.toolsets.parse_tool(tool_call)
            if command is None:
                return ()

            from .commands import AddNoteCommand, EditNoteCommand

            match command:
                case AddNoteCommand(note_name=name, note=content):
                    return (
                        CreatedNote(
                            runtime_id=response.runtime_id,
                            turn_id=response.turn_id,
                            source=self._agent.description.agent_name,
                            note_name=name,
                            note_content=content,
                        ),
                        NoteUpdated(
                            runtime_id=response.runtime_id,
                            turn_id=response.turn_id,
                            source=self._agent.description.agent_name,
                            note_name=name,
                            note_path=self._agent._resolve_note_path(name),
                        ),
                    )
                case EditNoteCommand(note_name=name):
                    return (
                        NoteUpdated(
                            runtime_id=response.runtime_id,
                            turn_id=response.turn_id,
                            source=self._agent.description.agent_name,
                            note_name=name,
                            note_path=self._agent._resolve_note_path(name),
                        ),
                    )
                case _:
                    return ()

        self._workflow = (
            WorkflowBuilder(self._agent.description.agent_name)
            .agent(self._agent)
            .inputs(*self.inputs)
            .emit_events(_emit_events)
            .build()
        )
        self.description = self._workflow.description

    def handle(self, message: Message) -> WorkflowExecution | str:
        workflow = getattr(self, "_workflow", None)
        if workflow is not None:
            return workflow.handle(message)
        if isinstance(message, UserCommand):
            if message.name == "start":
                return self._agent.start()
            if message.name == "reset":
                self._agent.reset()
            return ""
        if not isinstance(message, UserMessage):
            return ""

        return run_workflow(
            message=message,
            decider=self._decider,
            routing_fn=self._routing,
        )

    def close(self) -> None:
        workflow = getattr(self, "_workflow", None)
        if workflow is not None:
            workflow.close()
            return
        self._agent.close()
