from __future__ import annotations

from agentic.core_agent import CoreAgentic
from agentic.models import ModelConfig
from agentic.prompts import QwenPromptBuilder
from agentic.workflow import (
    LLMReactor,
    UserMessage,
    dispatch_output_handlers,
    make_llm_routing,
    run_workflow,
    workflow_output_handler,
)

from workshops.tui import LabApp

from .deciders import critic_decider, writer_decider
from .events import WriterCompleted
from .prompts import CRITIC_SYSTEM_PROMPT, WRITER_SYSTEM_PROMPT

MODEL_ID = "Qwen/Qwen3.5-2B"


class Lab1App(LabApp):
    lab_title = "Lab 1 — Writer & Critic Workflow"
    lab_subtitle = "Event-driven two-agent flow"
    lab_info = [
        f"Model: {MODEL_ID} (local, MLX)",
        "",
        "Flow:",
        "  You → Writer agent",
        "       → WriterCompleted event",
        "            → Critic agent",
    ]

    def __init__(
        self,
        *,
        writer: CoreAgentic,
        critic: CoreAgentic,
    ) -> None:
        super().__init__()
        self._writer = writer
        self._critic = critic

        self._writer_routing = make_llm_routing(LLMReactor(agent=writer))
        self._critic_routing = make_llm_routing(LLMReactor(agent=critic))

        def _handle_writer_completed(message: WriterCompleted) -> str | None:
            self.write_activity("Event", "WriterCompleted → Critic", style="#ff9e64")
            self.set_status("Critic reviewing...")
            execution = run_workflow(
                message=UserMessage(
                    text=f"Review this text and give brief feedback:\n{message.writer_output}",
                    runtime_id=message.runtime_id,
                    turn_id=message.turn_id,
                    source="writer",
                ),
                decider=critic_decider,
                routing_fn=self._critic_routing,
            )
            return execution.text

        self._critic_handler = workflow_output_handler(
            can_handle=(WriterCompleted,),
            each_message=_handle_writer_completed,
            name="critic",
        )

    def handle_input(self, message: str, *, attached_file: str | None = None) -> None:
        self.set_status("Writer generating...")
        self.write_activity("Writer", "Generating text...", style="#9ece6a")

        writer_result = run_workflow(
            message=UserMessage(text=message, source="user"),
            decider=writer_decider,
            routing_fn=self._writer_routing,
        )
        self.write_agent("Writer", writer_result.text or "", accent="#9ece6a")
        self.write_activity("Writer", "Done", style="#9ece6a")

        self.set_status("Dispatching events...")
        critic_responses = dispatch_output_handlers(
            [self._critic_handler],
            writer_result.emitted_events,
        )

        for response in critic_responses:
            self.write_agent("Critic", response or "", accent="#ff9e64")
            self.write_activity("Critic", "Review complete", style="#9ece6a")

    def cleanup(self) -> None:
        self._writer.close()
        self._critic.close()


def run_lab1() -> None:
    print(f"Loading model {MODEL_ID}...")
    writer = CoreAgentic(
        model_id=MODEL_ID,
        prompt_builder=QwenPromptBuilder(system_prompt=WRITER_SYSTEM_PROMPT),
        model_provider_type="mlx",
        config=ModelConfig(max_tokens=256, generation_mode="nothinking"),
    )
    critic = CoreAgentic(
        model_id=MODEL_ID,
        prompt_builder=QwenPromptBuilder(system_prompt=CRITIC_SYSTEM_PROMPT),
        model_provider_type="mlx",
        config=ModelConfig(max_tokens=256, generation_mode="nothinking"),
    )
    writer.preload_model()
    critic.preload_model()
    print("Models loaded. Starting TUI...")
    Lab1App(writer=writer, critic=critic).run()
