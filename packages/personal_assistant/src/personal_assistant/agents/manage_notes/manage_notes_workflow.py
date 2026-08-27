from __future__ import annotations

from structlog import get_logger

from agentic.prompts import QwenPromptBuilder
from agentic.tools import Toolsets
from agentic.workflow import WorkflowBuilder
from agentic.workflow._workflow import AgenticWorkflow
from agentic.workflow.messages import Message
from agentic_runtime.execution import WorkflowExecution
from agentic_runtime.reactor import LLMReactor
from agentic_runtime.routing import make_llm_routing
from personal_assistant.deciders import (
    build_successful_note_tool_events,
    make_manage_notes_decider,
)
from personal_assistant.settings import settings
from registry import Prompts

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

        def _emit_events(response) -> tuple[Message, ...]:
            return build_successful_note_tool_events(
                response,
                toolsets=self._agent._agent.toolsets,
                resolve_note_path=self._agent._resolve_note_path,
                agent_name=self._agent.description.agent_name,
            )

        self._workflow = (
            WorkflowBuilder(self._agent.description.agent_name)
            .agent(self._agent)
            .inputs(*self.inputs)
            .emit_events(_emit_events)
            .build()
        )
        self.description = self._workflow.description

    def handle(self, message: Message) -> WorkflowExecution | str:
        return self._workflow.handle(message)

    def close(self) -> None:
        self._workflow.close()
