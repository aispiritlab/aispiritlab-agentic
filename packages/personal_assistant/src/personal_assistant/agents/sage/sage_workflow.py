from __future__ import annotations

from agentic.prompts import QwenPromptBuilder
from agentic.workflow import WorkflowBuilder
from agentic.workflow._workflow import AgenticWorkflow
from agentic.workflow.messages import Message
from agentic_runtime.execution import WorkflowExecution
from personal_assistant.settings import settings
from registry import Prompts

from .sage_agent import SageAgent


def _strip_thinking(text: str) -> str:
    return text.replace("<think>", "").replace("</think>", "").strip()


class SageWorkflow(AgenticWorkflow):
    def __init__(self, *args, **kwargs) -> None:
        tracer = kwargs.pop("tracer", None)
        self.context = kwargs.pop("context", None)
        self.inputs = kwargs.pop("inputs", [])
        self._agent = SageAgent(
            model_id=settings.thinking_model,
            prompt_builder=QwenPromptBuilder(external_prompt_name=Prompts.SAGE),
            tracer=tracer,
            **kwargs,
        )
        self._workflow = (
            WorkflowBuilder(self._agent.description.agent_name)
            .agent(self._agent)
            .inputs(*self.inputs)
            .reactor("multiturn_llm", post_process=_strip_thinking)
            .build()
        )
        self.description = self._workflow.description

    def handle(self, message: Message) -> WorkflowExecution | str:
        return self._workflow.handle(message)

    def close(self) -> None:
        self._workflow.close()
