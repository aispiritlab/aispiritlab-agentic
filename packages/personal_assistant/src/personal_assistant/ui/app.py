"""Personal Assistant — Gradio chat UI."""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

from agentic.image_generation_call import ImageGenerationResult
from agentic.models import ModelProvider
from agentic.providers.api.http_client import ModelConnectionError
from agentic.voice import convert_audio, is_empty_transcription
import gradio as gr

from chat import (
    ChatAppConfig,
    ChatHistory,
    ChatMessage,
    MultimodalMessage,
    install_shutdown_handlers,
    launch,
    message_files,
    message_prompt_text,
    restore_shutdown_handlers,
)
from chat.components import add_message, append_voice_response, coerce_text, describe_uploaded_files, extract_uploaded_files

from personal_assistant import (
    Prompts,
    ai_spirit_agent,
    chat_agent,
    clear_chat_history,
    clear_personalization_history,
    generate_image_agent,
    get_initial_greeting,
    get_prompt,
    shutdown_application,
)
from personal_assistant.settings import settings

if TYPE_CHECKING:
    from personal_assistant.agents.manage_notes.evaluation import PromptOptimizationDefinition

PERSONALIZATION_FILE = Path.home() / ".aispiritagent" / "personalization.json"
DEFAULT_IMAGE_PROMPT = "Stwórz obraz na podstawie tego opisu."

model_provider = ModelProvider(name="mlx-community/parakeet-tdt-0.6b-v3", model_provider_type="mlx-audio")

PROMPT_CHOICES = {
    "Manage notes": Prompts.MANAGE_NOTES.value,
    "Discovery notes": Prompts.DISCOVERY_NOTES.value,
    "Sage": Prompts.SAGE.value,
    "Greetings": Prompts.GREETING.value,
}


def _shutdown_chat_application() -> None:
    shutdown_application(model_provider)


def _chat_greeting() -> str:
    return get_initial_greeting()


def _load_personalization_tools():  # noqa: ANN202
    from personal_assistant.agents.personalize.tools import is_personalization_finished, update_personalization

    return is_personalization_finished, update_personalization


def _load_notes_evaluation() -> PromptOptimizationDefinition:
    from personal_assistant.agents.manage_notes.evaluation import NOTES_EVALUATION

    return NOTES_EVALUATION


def _load_prompt_optimization():  # noqa: ANN202
    from evaluation.prompt_optimization import run_prompt_optimization

    return run_prompt_optimization


def _append_voice_response(history: ChatHistory, audio: object) -> tuple[object, object]:
    return append_voice_response(
        history,
        audio,
        voice_model=model_provider.voice_model,
        convert_audio_fn=convert_audio,
        is_empty_fn=is_empty_transcription,
    )


def _build_image_response_message(result: ImageGenerationResult) -> ChatMessage:
    return {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": (
                    f"Wygenerowano obraz. Seed: {result.seed}, "
                    f"rozmiar: {result.width}x{result.height}, kroki: {result.steps}."
                ),
            },
            gr.Image(value=result.image_path, show_label=False),
        ],
    }


def _add_message(
    history: ChatHistory, message: MultimodalMessage | str | None
) -> tuple[ChatHistory, gr.MultimodalTextbox]:
    return add_message(history, message, file_description_prefix="Przesłany plik")


def generate_response(history: ChatHistory, mode: str = "Agenci") -> Generator[ChatHistory, None, None]:
    """Generate bot response and stream it to the chat."""
    if not history:
        yield history
        return

    last_message = history[-1]
    if last_message["role"] != "user":
        yield history
        return

    build_response = message_prompt_text(last_message).strip()
    file_paths = message_files(last_message)

    try:
        if mode == "Chat":
            if not build_response:
                response = "Dodaj tekst, aby użyć trybu Chat."
            else:
                response = chat_agent(build_response)
        elif mode == "Generate image":
            if not build_response:
                response = "Dodaj opis obrazu, aby użyć trybu Generate image."
            else:
                response = generate_image_agent(build_response or DEFAULT_IMAGE_PROMPT)
        else:
            if file_paths and not build_response:
                response = "Przełącz tryb na Generate image, aby analizować przesłane obrazy."
            else:
                response = ai_spirit_agent(build_response)
    except (ModelConnectionError, RuntimeError, ValueError) as exc:
        history.append({"role": "assistant", "content": str(exc)})
        yield history
        return

    if isinstance(response, ImageGenerationResult):
        history.append(_build_image_response_message(response))
        yield history
        return

    response_text = response if isinstance(response, str) else str(response)

    history.append({"role": "assistant", "content": ""})

    for char in response_text:
        history[-1]["content"] += char  # type: ignore[operator]
        yield history


def load_personalization_form() -> tuple[str, str, str]:
    """Load personalization fields from the persisted JSON file."""
    is_personalization_finished, _ = _load_personalization_tools()
    if not is_personalization_finished():
        return "", "", "Status: personalizacja nie jest jeszcze skonfigurowana."

    try:
        with open(PERSONALIZATION_FILE, encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        return "", "", f"Status: nie udało się odczytać personalizacji ({error})."

    name = data.get("name")
    vault_name = data.get("vault_name")
    if not isinstance(vault_name, str) or not vault_name.strip():
        legacy_vault_path = data.get("vault_path")
        if isinstance(legacy_vault_path, str) and legacy_vault_path.strip():
            legacy_name = Path(legacy_vault_path).name.strip()
            vault_name = legacy_name or legacy_vault_path.strip()

    resolved_name = name.strip() if isinstance(name, str) else ""
    resolved_vault_name = vault_name.strip() if isinstance(vault_name, str) else ""
    return (
        resolved_name,
        resolved_vault_name,
        "Status: personalizacja jest skonfigurowana.",
    )


def save_personalization_form(name: str, vault_name: str) -> tuple[str, object]:
    """Persist personalization settings from the UI form."""
    _, update_personalization = _load_personalization_tools()
    resolved_name = name.strip()
    resolved_vault_name = vault_name.strip()

    if not resolved_name or not resolved_vault_name:
        return "Uzupełnij pola: imię i nazwa vaulta.", gr.skip()

    try:
        save_status = update_personalization(name=resolved_name, vault_name=resolved_vault_name)
    except OSError as error:
        return f"Nie udało się zapisać personalizacji: {error}", gr.skip()
    if save_status != "Personalizacja zapisana.":
        return save_status, gr.skip()

    return (
        "Personalizacja zapisana.",
        [{"role": "assistant", "content": _chat_greeting()}],
    )


def load_selected_prompt(prompt_label: str) -> tuple[str, str]:
    """Load selected prompt template from prompt registry."""
    prompt_name = PROMPT_CHOICES.get(prompt_label)
    if prompt_name is None:
        return "Wybierz poprawną opcję promptu.", ""

    try:
        prompt_value = get_prompt(prompt_name)
    except Exception as error:
        return f"Nie udało się pobrać promptu: {error}", ""

    return f"Załadowano prompt: {prompt_label}", prompt_value


def start_training() -> str:
    """Placeholder action for training start."""
    return "Wkrotce"


def create_chat_ui() -> gr.Blocks:
    """Create the Personal Assistant chat UI."""
    distributed_mode = settings.agentic_transport == "redis_streams"
    mode_choices = ["Agenci"] if distributed_mode else ["Agenci", "Chat", "Generate image"]

    with gr.Blocks(fill_height=True, title="AI Spirit Agent") as block:
        gr.Markdown("# AI Spirit Agent")

        with gr.Tabs():
            with gr.Tab("Chat"):
                mode_toggle = gr.Radio(
                    choices=mode_choices,
                    value=mode_choices[0],
                    label="Tryb",
                )
                chatbot = gr.Chatbot(
                    label="AI Spirit Agent",
                    value=[{"role": "assistant", "content": _chat_greeting()}],
                    avatar_images=(
                        None,
                        "https://em-content.zobj.net/source/twitter/53/robot-face_1f916.png",
                    ),
                    height=500,
                )

                with gr.Group():
                    chat_input = gr.MultimodalTextbox(
                        interactive=True,
                        file_count="multiple",
                        placeholder="Wpisz wiadomość lub nagraj głos...",
                        show_label=False,
                        sources=["upload"],
                    )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Powiedz")
                        say_audio_input = gr.Audio(
                            label="",
                            sources=["microphone"],
                            type="numpy",
                            show_label=False,
                            streaming=False,
                        )
                        gr.Button("Powiedz", variant="primary")

                    with gr.Column():
                        gr.Markdown("### Nagraj notatkę")
                        note_audio_input = gr.Audio(
                            label="",
                            sources=["microphone"],
                            type="numpy",
                            show_label=False,
                            streaming=False,
                        )
                        gr.Button("Nagraj notatkę")

                clear_btn = gr.Button("Wyczyść historię")

            if not distributed_mode:
                with gr.Tab("Personalizacja"):
                    gr.Markdown("## Ustawienia personalizacji")
                    name_input = gr.Textbox(label="Imię", placeholder="np. Mateusz")
                    vault_name_input = gr.Textbox(
                        label="Nazwa vaulta",
                        placeholder="np. MyVault",
                    )
                    with gr.Row():
                        save_personalization_btn = gr.Button("Zapisz personalizację", variant="primary")
                        refresh_personalization_btn = gr.Button("Odśwież dane")
                    personalization_status = gr.Markdown()
                    gr.Markdown("## Podgląd promptu")
                    prompt_selector = gr.Dropdown(
                        label="Wybierz prompt",
                        choices=list(PROMPT_CHOICES.keys()),
                        value="Manage notes",
                    )
                    load_prompt_btn = gr.Button("Pobierz prompt")
                    prompt_status = gr.Markdown()
                    prompt_preview = gr.Textbox(
                        label="Treść promptu",
                        lines=14,
                        max_lines=30,
                        interactive=False,
                    )

                with gr.Tab("Trenowanie"):
                    gr.Markdown("## Trenowanie")
                    training_start_btn = gr.Button("Rozpocznij", variant="primary")
                    training_status = gr.Markdown("Wkrotce")

                with gr.Tab("Prompt Optimization"):
                    notes_evaluation = _load_notes_evaluation()
                    gr.Markdown("## Prompt Optimization (MIPROv2)")
                    gr.Markdown(
                        "Podaj prompt do optymalizacji i scenariusze testowe "
                        "(co model powinien zrobić)."
                    )
                    optimization_prompt_input = gr.Textbox(
                        label="Prompt do optymalizacji",
                        lines=16,
                        max_lines=32,
                        placeholder="Wklej prompt, który chcesz zoptymalizować.",
                    )
                    optimization_scenarios_input = gr.Textbox(
                        label="Scenariusze testowe (JSON)",
                        lines=16,
                        max_lines=32,
                        value=notes_evaluation.scenarios_example or "",
                    )
                    with gr.Row():
                        optimization_openrouter_model_input = gr.Textbox(
                            label="Model OpenRouter",
                            value="",
                            placeholder="np. openai/gpt-4o-mini",
                        )
                        optimization_openrouter_api_key_input = gr.Textbox(
                            label="OpenRouter API key",
                            value="",
                            placeholder="sk-or-...",
                            type="password",
                        )
                    with gr.Row():
                        optimization_num_candidates_input = gr.Number(
                            label="Liczba kandydatów (num_candidates)",
                            value=6,
                            precision=0,
                        )
                        optimization_num_trials_input = gr.Number(
                            label="Liczba prób (num_trials)",
                            value=12,
                            precision=0,
                        )
                    with gr.Row():
                        optimization_load_prompt_btn = gr.Button("Wczytaj aktualny NOTE prompt")
                        optimization_run_btn = gr.Button(
                            "Uruchom optymalizację",
                            variant="primary",
                        )
                    optimization_status = gr.Markdown()
                    optimization_result = gr.Textbox(
                        label="Zoptymalizowany prompt",
                        lines=16,
                        max_lines=32,
                        interactive=False,
                    )

        chat_input.submit(
            _add_message,
            inputs=[chatbot, chat_input],
            outputs=[chatbot, chat_input],
            queue=False,
        ).then(
            generate_response,
            inputs=[chatbot, mode_toggle],
            outputs=[chatbot],
        ).then(
            lambda: gr.MultimodalTextbox(interactive=True),
            outputs=[chat_input],
        )
        for event in (say_audio_input.stop_recording,):
            event(
                _append_voice_response,
                inputs=[chatbot, say_audio_input],
                outputs=[chatbot, say_audio_input],
                queue=False,
            ).then(
                generate_response,
                inputs=[chatbot, mode_toggle],
                outputs=[chatbot],
            )

        for event in (note_audio_input.stop_recording,):
            event(
                _append_voice_response,
                inputs=[chat_input, note_audio_input],
                outputs=[chat_input, note_audio_input],
                queue=False,
            )

        def clear_chat() -> ChatHistory:
            """Clear UI and backend agent history."""
            clear_personalization_history()
            clear_chat_history()
            return [{"role": "assistant", "content": _chat_greeting()}]

        clear_btn.click(
            clear_chat,
            outputs=[chatbot],
        )

        if not distributed_mode:
            block.load(
                load_personalization_form,
                outputs=[name_input, vault_name_input, personalization_status],
            )

            refresh_personalization_btn.click(
                load_personalization_form,
                outputs=[name_input, vault_name_input, personalization_status],
            )

            save_personalization_btn.click(
                save_personalization_form,
                inputs=[name_input, vault_name_input],
                outputs=[personalization_status, chatbot],
            )

            load_prompt_btn.click(
                load_selected_prompt,
                inputs=[prompt_selector],
                outputs=[prompt_status, prompt_preview],
            )

            training_start_btn.click(
                start_training,
                outputs=[training_status],
            )

            optimization_load_prompt_btn.click(
                lambda: load_selected_prompt("Manage notes"),
                outputs=[optimization_status, optimization_prompt_input],
            )

            optimization_run_btn.click(
                lambda prompt, scenarios, model, api_key, num_candidates, num_trials: (
                    _load_prompt_optimization()(
                        definition=_load_notes_evaluation(),
                        prompt_to_optimize=prompt,
                        scenarios_json=scenarios,
                        openrouter_model=model,
                        openrouter_api_key=api_key,
                        num_candidates=num_candidates,
                        num_trials=num_trials,
                    )
                ),
                inputs=[
                    optimization_prompt_input,
                    optimization_scenarios_input,
                    optimization_openrouter_model_input,
                    optimization_openrouter_api_key_input,
                    optimization_num_candidates_input,
                    optimization_num_trials_input,
                ],
                outputs=[optimization_status, optimization_result],
            )

    return block


def launch_app() -> None:
    """Launch the Personal Assistant chat UI."""
    previous_handlers = install_shutdown_handlers(_shutdown_chat_application)
    try:
        ui = create_chat_ui()
        config = ChatAppConfig(
            title="AI Spirit Agent",
            server_name=settings.chat_server_name,
            server_port=settings.chat_server_port,
            allowed_paths=[settings.image_output_dir] if settings.image_output_dir else None,
        )
        launch(ui, config)
    finally:
        _shutdown_chat_application()
        restore_shutdown_handlers(previous_handlers)
