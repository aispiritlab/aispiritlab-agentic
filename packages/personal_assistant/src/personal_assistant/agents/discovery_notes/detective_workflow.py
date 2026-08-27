from __future__ import annotations

from agentic.prompts import QwenPromptBuilder
from agentic.tools import Toolsets
from agentic.workflow import WorkflowBuilder
from agentic.workflow._workflow import AgenticWorkflow
from agentic.workflow.messages import Message
from agentic_runtime.execution import WorkflowExecution
from personal_assistant.settings import settings
from registry import Prompts

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
        return self._workflow.handle(message)

    def close(self) -> None:
        self._workflow.close()
