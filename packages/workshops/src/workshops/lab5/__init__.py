from __future__ import annotations

from dataclasses import replace
import os
from typing import ClassVar

from agentic.core_agent import CoreAgentic
from agentic.prompts import ChatPromptBuilder, QwenPromptBuilder
from agentic.workflow import (
    LLMReactor,
    dispatch_output_handlers,
    make_llm_routing,
    message_text,
    reply_metadata,
    run_workflow,
    user_message,
    workflow_output_handler,
)
from agentic.workflow.messages import (
    ConversationData,
    Message,
    RecordedMessageMetadata,
    UserMessage,
)
from agentic.workflow.reactor import LLMResponse
from providers.models import ModelConfig
from workshops.tui import LabApp

from .deciders import art_writer_decider, make_image_decider
from .messages import ImageDescribed, ImageMessage
from .prompts import ART_WRITER_PROMPT, IMAGE_SUMMARY_PROMPT

TEXT_MODEL_ID = "Qwen/Qwen3.5-2B"
VLM_MODEL_ID = "mlx-community/nanoLLaVA"


class VLMReactor:
    """Reactor that passes both text and image to a VLM agent."""

    def __init__(self, agent: CoreAgentic) -> None:
        self._agent = agent

    def can_handle(self, command: Message) -> bool:
        return bool(message_text(command))

    def invoke(self, command: Message) -> Message:
        text = message_text(command)
        images: str | list[str] | None = None
        if isinstance(command, ImageMessage) and command.image_path:
            images = [command.image_path]
            if "<image>" not in text:
                text = f"<image>\n{text}"

        response = self._agent.respond(text, images=images)

        metadata = reply_metadata(command, source="vlm")
        return LLMResponse(
            data=ConversationData(role="assistant", text=response.output),
            metadata=replace(metadata, agent_run_id=response.result.run_id),
        )


class Lab5App(LabApp):
    lab_title = "Lab 5 — Vision: Image Summary → Art"
    lab_subtitle = "Multimodal event-driven flow"
    lab_info: ClassVar[list[str]] = [
        f"VLM: {VLM_MODEL_ID}",
        f"LLM: {TEXT_MODEL_ID}",
        "",
        "Flow:",
        "  image path + text → VLM agent",
        "       → ImageDescribed event",
        "            → Art Writer agent",
        "",
        "Input format: <image_path> | <question>",
    ]

    def __init__(
        self,
        *,
        summarizer: CoreAgentic,
        art_writer: CoreAgentic,
    ) -> None:
        super().__init__()
        self._summarizer = summarizer
        self._art_writer = art_writer

        self._vlm_routing = make_llm_routing(VLMReactor(agent=summarizer))
        self._art_routing = make_llm_routing(LLMReactor(agent=art_writer))

        def _handle_image_described(message: Message) -> str | None:
            if not isinstance(message, ImageDescribed):
                return None
            self.write_activity("Event", "ImageDescribed → Art Writer", style="#ff9e64")
            self.set_status("Art Writer creating...")
            execution = run_workflow(
                message=user_message(
                    "Write a creative piece inspired by this image description:\n"
                    f"{message.description}",
                    source="image_summary",
                    reply_to=message,
                ),
                decider=art_writer_decider,
                routing_fn=self._art_routing,
            )
            return execution.text

        self._art_handler = workflow_output_handler(
            can_handle=(ImageDescribed,),
            each_message=_handle_image_described,
            name="art_writer",
        )

    def handle_input(self, message: str, *, attached_file: str | None = None) -> None:
        image_path = attached_file
        text = message

        if not image_path:
            image_path, text = _parse_input(message)

        if image_path and not os.path.isfile(image_path):
            self.write_error(f"File not found: {image_path}")
            return

        if image_path:
            self.write_activity("Input", f"Image: {os.path.basename(image_path)}", style="#7aa2f7")
            msg: UserMessage = ImageMessage(
                data=ConversationData(role="user", text=text),
                metadata=RecordedMessageMetadata(source="user"),
                image_path=image_path,
            )
        else:
            msg = user_message(text, source="user")

        self.set_status("VLM analyzing image...")
        self.write_activity("Summarizer", "Analyzing image...", style="#9ece6a")

        image_decider = make_image_decider(
            image_path=msg.image_path if isinstance(msg, ImageMessage) else "",
        )
        summary_result = run_workflow(
            message=msg,
            decider=image_decider,
            routing_fn=self._vlm_routing,
        )
        self.write_agent("Summarizer", summary_result.text or "", accent="#9ece6a")
        self.write_activity("Summarizer", "Done", style="#9ece6a")

        self.set_status("Dispatching events...")
        art_responses = dispatch_output_handlers(
            [self._art_handler],
            summary_result.emitted_events,
        )

        for response in art_responses:
            self.write_agent("Art Writer", response or "", accent="#bb9af7")
            self.write_activity("Art Writer", "Creation complete", style="#bb9af7")

    def cleanup(self) -> None:
        self._summarizer.close()
        self._art_writer.close()


def _parse_input(message: str) -> tuple[str, str]:
    """Parse input as '<image_path> | <text>' or just '<text>'."""
    if "|" in message:
        parts = message.split("|", maxsplit=1)
        image_path = parts[0].strip()
        text = parts[1].strip() if len(parts) > 1 else "Describe this image in detail."
        return image_path, text
    stripped = message.strip()
    if os.path.isfile(stripped):
        return stripped, "Describe this image in detail."
    return "", stripped


def run_lab5() -> None:
    print(f"Loading VLM model {VLM_MODEL_ID}...")
    summarizer = CoreAgentic(
        model_id=VLM_MODEL_ID,
        prompt_builder=ChatPromptBuilder(system_prompt=IMAGE_SUMMARY_PROMPT),
        model_provider_type="mlx-vlm",
        config=ModelConfig(max_tokens=256, generation_mode="nothinking"),
    )
    print(f"Loading text model {TEXT_MODEL_ID}...")
    art_writer = CoreAgentic(
        model_id=TEXT_MODEL_ID,
        prompt_builder=QwenPromptBuilder(system_prompt=ART_WRITER_PROMPT),
        model_provider_type="mlx",
        config=ModelConfig(max_tokens=256, generation_mode="nothinking"),
    )
    summarizer.preload_model()
    art_writer.preload_model()
    print("Models loaded. Starting TUI...")
    Lab5App(summarizer=summarizer, art_writer=art_writer).run()
