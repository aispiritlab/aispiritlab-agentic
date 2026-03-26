from __future__ import annotations

from agentic.prompts import QwenPromptBuilder
from agentic.tools import Toolsets
from agentic.workflow import WorkflowBuilder
from agentic.workflow._workflow import AgenticWorkflow
from registry import Prompts

from agentic.workflow.messages import Message, UserCommand, UserMessage
from agentic_runtime.execution import WorkflowExecution

from personal_assistant.settings import settings

from .detective_agent import DiscoveryNotesAgent
from .tools import toolset as discovery_notes_toolset


class DiscoveryNotesWorkflow(AgenticWorkflow):
    def __init__(self, *args, **kwargs) -> None:
        tracer = kwargs.pop("tracer", None)
        self.context = kwargs.pop("context", None)
        self.inputs = kwargs.pop("inputs", [])
        self._agent = DiscoveryNotesAgent(
            model_id=settings.model_name,
            prompt_builder=QwenPromptBuilder(external_prompt_name=Prompts.DISCOVERY_NOTES),
            toolsets=Toolsets([discovery_notes_toolset]),
            tracer=tracer,
            **kwargs,
        )
        self._workflow = (
            WorkflowBuilder(self._agent.description.agent_name)
            .agent(self._agent)
            .inputs(*self.inputs)
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
        if isinstance(message, UserMessage):
            return self._agent.respond(message.text).output
        return ""

    def close(self) -> None:
        workflow = getattr(self, "_workflow", None)
        if workflow is not None:
            workflow.close()
            return
        self._agent.close()
