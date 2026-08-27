"""Personal Assistant — Gradio chat UI."""

from __future__ import annotations

from collections.abc import Generator, Iterator
import json
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr
from structlog import get_logger

from agentic.voice import convert_audio, is_empty_transcription
from agentic_runtime.slugs import InvalidSlugError
from agentic_runtime.users import (
    create_user as _create_user_profile,
)
from agentic_runtime.users import (
    default_user_slug,
    list_users,
    personalization_path,
)
from agentic_runtime.users import (
    delete_user as _delete_user_profile,
)
from agentic_runtime.workspaces import (
    list_workspaces,
    set_active_workspace,
)
from chat import (
    ChatAppConfig,
    ChatHistory,
    ChatMessage,
    MultimodalMessage,
    install_shutdown_handlers,
    launch,
    message_files,
    message_prompt_text,
    parse_auth,
    restore_shutdown_handlers,
)
from chat.components import (
    add_message,
    append_voice_response,
)
from chat.styles import GLOBAL_CSS
from chat.theme import SPIRIT_THEME
from personal_assistant import (
    Prompts,
    ai_spirit_agent,
    chat_agent,
    clear_chat_history,
    clear_personalization_history,
    drop_runtime_sessions,
    generate_image_agent,
    get_initial_greeting,
    get_prompt,
    shutdown_application,
    switch_user,
)
from personal_assistant.settings import settings
from providers.api.http_client import ModelConnectionError
from providers.image.mflux import ImageGenerationResult
from providers.orchestrator import ModelProvider

if TYPE_CHECKING:
    from evaluation.contracts import EvaluationDefinition

logger = get_logger(__name__)

PERSONALIZATION_FILE = Path.home() / ".aispiritagent" / "personalization.json"  # legacy fallback
DEFAULT_IMAGE_PROMPT = "Stwórz obraz na podstawie tego opisu."


def _get_user_choices() -> list[str]:
    return [u.name for u in list_users()]


def _get_user_slug_map() -> dict[str, str]:
    return {u.name: u.slug for u in list_users()}


def _get_default_user_name() -> str:
    users = list_users()
    return users[0].name if users else "Default"


def _get_workspace_choices() -> list[str]:
    return [w.name for w in list_workspaces()]


def _get_default_workspace() -> str:
    workspaces = list_workspaces()
    return workspaces[0].name if workspaces else "Default"


def _get_default_workspace_slug() -> str:
    workspaces = list_workspaces()
    return workspaces[0].slug if workspaces else "default"


def _get_workspace_slug_map() -> dict[str, str]:
    return {w.name: w.slug for w in list_workspaces()}


def _resolve_user_slug(user_name: str) -> str | None:
    """Map a display name from the client to a registered user slug.

    Returns ``None`` for anything unregistered. Never echoes the client value
    back as a slug: slugs become filesystem paths downstream.
    """
    slug_map = _get_user_slug_map()
    if user_name in slug_map:
        return slug_map[user_name]
    if any(slug == user_name for slug in slug_map.values()):
        return user_name
    return None


def _resolve_workspace_slug(workspace_name: str) -> str | None:
    """Map a display name from the client to a registered workspace slug."""
    slug_map = _get_workspace_slug_map()
    if workspace_name in slug_map:
        return slug_map[workspace_name]
    if any(slug == workspace_name for slug in slug_map.values()):
        return workspace_name
    return None


def _personalization_file_for_user(user_slug: str) -> Path:
    try:
        path = personalization_path(user_slug)
    except InvalidSlugError:
        return PERSONALIZATION_FILE
    if path.exists():
        return path
    return PERSONALIZATION_FILE  # legacy fallback


model_provider = ModelProvider(
    name="mlx-community/parakeet-tdt-0.6b-v3", model_provider_type="mlx-audio"
)

PROMPT_CHOICES = {
    "Manage notes": Prompts.MANAGE_NOTES.value,
    "Discovery notes": Prompts.DISCOVERY_NOTES.value,
    "Sage": Prompts.SAGE.value,
    "Greetings": Prompts.GREETING.value,
}


def _shutdown_chat_application() -> None:
    shutdown_application(model_provider)


def _chat_greeting(user: str = "default", workspace: str | None = None) -> str:
    return get_initial_greeting(user=user, workspace=workspace)


def _load_personalization_tools():
    from personal_assistant.agents.personalize.tools import (
        is_personalization_finished,
        update_personalization,
    )

    return is_personalization_finished, update_personalization


def _load_notes_evaluation() -> EvaluationDefinition:
    from personal_assistant.agents.manage_notes.evaluation import NOTES_EVALUATION

    return NOTES_EVALUATION


def _load_prompt_optimization():
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


#: Characters revealed per streamed update. Yielding per character re-serialises
#: the whole chat history thousands of times for one answer.
_STREAM_CHUNK_CHARS = 24

_ERROR_MESSAGES: dict[type[Exception], str] = {
    ModelConnectionError: (
        "Nie mogę się teraz połączyć z modelem. Sprawdź, czy serwer LLM działa, "
        "i spróbuj ponownie."
    ),
    ValueError: "Nie zrozumiałem tej wiadomości. Spróbuj sformułować ją inaczej.",
}
_DEFAULT_ERROR_MESSAGE = "Coś poszło nie tak po mojej stronie. Spróbuj ponownie za chwilę."


def _user_facing_error(error: Exception) -> str:
    """Map an exception to a safe message.

    Exception text can carry endpoint URLs and local paths, so it goes to the log
    rather than to the chat window.
    """
    for error_type, message in _ERROR_MESSAGES.items():
        if isinstance(error, error_type):
            return message
    return _DEFAULT_ERROR_MESSAGE


def _typing_chunks(text: str, chunk_size: int = _STREAM_CHUNK_CHARS) -> Iterator[str]:
    """Yield progressively longer prefixes of ``text`` for a typing effect."""
    if not text:
        yield ""
        return
    for end in range(chunk_size, len(text), chunk_size):
        yield text[:end]
    yield text


def generate_response(
    history: ChatHistory,
    mode: str = "Agenci",
    active_user: str = "default",
    active_workspace: str = "default",
) -> Generator[ChatHistory]:
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
                response = chat_agent(
                    build_response,
                    user=active_user,
                    workspace=active_workspace,
                )
        elif mode == "Generate image":
            if not build_response:
                response = "Dodaj opis obrazu, aby użyć trybu Generate image."
            else:
                response = generate_image_agent(
                    build_response or DEFAULT_IMAGE_PROMPT,
                    user=active_user,
                    workspace=active_workspace,
                )
        else:
            if file_paths and not build_response:
                response = "Przełącz tryb na Generate image, aby analizować przesłane obrazy."
            else:
                response = ai_spirit_agent(
                    build_response,
                    user=active_user,
                    workspace=active_workspace,
                )
    except (ModelConnectionError, RuntimeError, ValueError) as error:
        logger.warning(
            "chat_turn_failed",
            mode=mode,
            user=active_user,
            workspace=active_workspace,
            error_type=type(error).__name__,
            error=str(error),
        )
        history.append({"role": "assistant", "content": _user_facing_error(error)})
        yield history
        return

    if isinstance(response, ImageGenerationResult):
        history.append(_build_image_response_message(response))
        yield history
        return

    response_text = response if isinstance(response, str) else str(response)

    history.append({"role": "assistant", "content": ""})
    for partial in _typing_chunks(response_text):
        history[-1]["content"] = partial
        yield history


def load_personalization_form(active_user: str = "default") -> tuple[str, str, str]:
    """Load personalization fields from the persisted JSON file."""
    user_slug = _resolve_user_slug(active_user)
    if user_slug is None:
        return "", "", "Status: nie znaleziono tego użytkownika."
    pfile = _personalization_file_for_user(user_slug)

    if not pfile.exists():
        return "", "", "Status: personalizacja nie jest jeszcze skonfigurowana."

    try:
        with open(pfile, encoding="utf-8") as file:
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


def save_personalization_form(
    name: str,
    vault_name: str,
    active_user: str,
    active_workspace: str,
) -> tuple[str, object]:
    """Persist personalization settings from the UI form."""
    _, update_personalization = _load_personalization_tools()
    resolved_name = name.strip()
    resolved_vault_name = vault_name.strip()

    if not resolved_name or not resolved_vault_name:
        return "Uzupełnij pola: imię i nazwa vaulta.", gr.skip()

    try:
        save_status = update_personalization(
            name=resolved_name,
            vault_name=resolved_vault_name,
            user=active_user,
        )
    except OSError as error:
        return f"Nie udało się zapisać personalizacji: {error}", gr.skip()
    if save_status != "Personalizacja zapisana.":
        return save_status, gr.skip()

    return (
        "Personalizacja zapisana.",
        [{"role": "assistant", "content": _chat_greeting(active_user, active_workspace)}],
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


def _on_user_switch(
    user_name: str,
    active_workspace: str,
) -> tuple[str, list[dict[str, str]], str, str, str]:
    """Handle user switch: set context, return greeting + personalization."""
    user_slug = _resolve_user_slug(user_name)
    if user_slug is None:
        return (
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            "Status: nie znaleziono tego użytkownika.",
        )
    workspace_slug = _resolve_workspace_slug(active_workspace) or _get_default_workspace_slug()
    greeting = switch_user(user_slug, workspace=workspace_slug)
    name, vault, status = load_personalization_form(user_slug)
    return user_slug, [{"role": "assistant", "content": greeting}], name, vault, status


def _on_create_user(
    new_name: str,
    active_workspace: str,
) -> tuple[gr.Dropdown, str, list[dict[str, str]]]:
    """Create user, switch to them, return updated dropdown + greeting."""
    new_name = new_name.strip()
    if not new_name:
        return gr.skip(), "", gr.skip()
    profile = _create_user_profile(new_name)
    greeting = switch_user(profile.slug, workspace=active_workspace)
    choices = _get_user_choices()
    return (
        gr.Dropdown(choices=choices, value=profile.name),
        profile.slug,
        [{"role": "assistant", "content": greeting}],
    )


def _on_delete_user(
    user_name: str,
    active_workspace: str,
) -> tuple[gr.Dropdown, str, list[dict[str, str]]]:
    """Delete user, switch to first remaining."""
    user_slug = _resolve_user_slug(user_name)
    if user_slug is None:
        return gr.skip(), gr.skip(), gr.skip()
    try:
        _delete_user_profile(user_slug)
    except ValueError, InvalidSlugError:
        return gr.skip(), user_slug, gr.skip()
    drop_runtime_sessions(user=user_slug)
    users = list_users()
    first = users[0] if users else None
    if first is None:
        return gr.skip(), "", gr.skip()
    workspace_slug = _resolve_workspace_slug(active_workspace) or _get_default_workspace_slug()
    greeting = switch_user(first.slug, workspace=workspace_slug)
    choices = _get_user_choices()
    return (
        gr.Dropdown(choices=choices, value=first.name),
        first.slug,
        [{"role": "assistant", "content": greeting}],
    )


def create_chat_ui() -> gr.Blocks:
    """Create the Personal Assistant chat UI."""
    distributed_mode = settings.agentic_transport == "redis_streams"
    mode_choices = ["Agenci"] if distributed_mode else ["Agenci", "Chat", "Generate image"]
    default_user = _get_default_user_name()
    default_slug = default_user_slug()
    default_workspace = _get_default_workspace()
    default_workspace_slug = _get_default_workspace_slug()

    with gr.Blocks(
        fill_height=True,
        title="AI Spirit Agent",
        theme=SPIRIT_THEME,
        css=GLOBAL_CSS,
    ) as block:
        gr.HTML('<div class="spirit-header"><h1>AI Spirit Agent</h1></div>')

        active_user_state = gr.State(value=default_slug)
        active_workspace_state = gr.State(value=default_workspace_slug)

        with gr.Row(elem_classes=["context-bar"]):
            user_selector = gr.Dropdown(
                label="User",
                choices=_get_user_choices(),
                value=default_user,
                interactive=not distributed_mode,
                scale=2,
                min_width=140,
            )
            workspace_selector = gr.Dropdown(
                label="Workspace",
                choices=_get_workspace_choices(),
                value=default_workspace,
                interactive=not distributed_mode,
                scale=2,
                min_width=140,
            )
            new_user_input = gr.Textbox(
                label="New user",
                placeholder="Name",
                interactive=not distributed_mode,
                scale=2,
                min_width=100,
            )
            create_user_btn = gr.Button(
                "Create", size="sm", scale=1, interactive=not distributed_mode
            )
            delete_user_btn = gr.Button(
                "Delete",
                variant="stop",
                size="sm",
                scale=1,
                interactive=not distributed_mode,
            )

        with gr.Tabs():
            with gr.Tab("Chat", id="tab-chat"):
                mode_toggle = gr.Radio(
                    choices=mode_choices,
                    value=mode_choices[0],
                    label="Tryb",
                )
                chatbot = gr.Chatbot(
                    label="AI Spirit Agent",
                    value=[
                        {
                            "role": "assistant",
                            "content": _chat_greeting(default_slug, default_workspace_slug),
                        }
                    ],
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
                with gr.Tab("Settings", id="tab-settings"):
                    gr.Markdown("## Ustawienia personalizacji")
                    name_input = gr.Textbox(label="Imię", placeholder="np. Mateusz")
                    vault_name_input = gr.Textbox(
                        label="Nazwa vaulta",
                        placeholder="np. MyVault",
                    )
                    with gr.Row():
                        save_personalization_btn = gr.Button(
                            "Zapisz personalizację", variant="primary"
                        )
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

                with gr.Tab("Editor Agents", id="tab-editor"):
                    from agentic_graph import build_agent_builder_tab

                    build_agent_builder_tab(active_workspace_state=active_workspace_state)

                with gr.Tab("Training", id="tab-training"):
                    gr.Markdown("## Trenowanie")
                    training_start_btn = gr.Button("Rozpocznij", variant="primary")
                    training_status = gr.Markdown("Wkrotce")

                with gr.Tab("Optimization", id="tab-optimization"):
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

        # --- Workspace switch ---
        def _on_workspace_switch(ws_name: str, active_user: str) -> tuple[str, ChatHistory]:
            ws_slug = _resolve_workspace_slug(ws_name)
            if ws_slug is None:
                return gr.skip(), gr.skip()
            set_active_workspace(ws_slug)
            return ws_slug, [
                {"role": "assistant", "content": _chat_greeting(active_user, ws_slug)}
            ]

        workspace_selector.change(
            _on_workspace_switch,
            inputs=[workspace_selector, active_user_state],
            outputs=[active_workspace_state, chatbot],
        )

        # --- User management events ---
        if not distributed_mode:
            user_selector.change(
                _on_user_switch,
                inputs=[user_selector, active_workspace_state],
                outputs=[
                    active_user_state,
                    chatbot,
                    name_input,
                    vault_name_input,
                    personalization_status,
                ],
            )
            create_user_btn.click(
                _on_create_user,
                inputs=[new_user_input, active_workspace_state],
                outputs=[user_selector, active_user_state, chatbot],
            )
            delete_user_btn.click(
                _on_delete_user,
                inputs=[user_selector, active_workspace_state],
                outputs=[user_selector, active_user_state, chatbot],
            )

        # --- Chat events ---
        chat_input.submit(
            _add_message,
            inputs=[chatbot, chat_input],
            outputs=[chatbot, chat_input],
            queue=False,
        ).then(
            generate_response,
            inputs=[chatbot, mode_toggle, active_user_state, active_workspace_state],
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
                inputs=[chatbot, mode_toggle, active_user_state, active_workspace_state],
                outputs=[chatbot],
            )

        for event in (note_audio_input.stop_recording,):
            event(
                _append_voice_response,
                inputs=[chat_input, note_audio_input],
                outputs=[chat_input, note_audio_input],
                queue=False,
            )

        def clear_chat(active_user: str, active_workspace: str) -> ChatHistory:
            """Clear UI and backend agent history."""
            clear_personalization_history(user=active_user, workspace=active_workspace)
            clear_chat_history(user=active_user, workspace=active_workspace)
            return [
                {"role": "assistant", "content": _chat_greeting(active_user, active_workspace)}
            ]

        clear_btn.click(
            clear_chat,
            inputs=[active_user_state, active_workspace_state],
            outputs=[chatbot],
        )

        if not distributed_mode:
            block.load(
                load_personalization_form,
                inputs=[active_user_state],
                outputs=[name_input, vault_name_input, personalization_status],
            )

            refresh_personalization_btn.click(
                load_personalization_form,
                inputs=[active_user_state],
                outputs=[name_input, vault_name_input, personalization_status],
            )

            save_personalization_btn.click(
                save_personalization_form,
                inputs=[name_input, vault_name_input, active_user_state, active_workspace_state],
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
            auth=parse_auth(settings.chat_auth),
            auth_message="Zaloguj się, aby korzystać z AI Spirit Agent.",
        )
        launch(ui, config)
    finally:
        _shutdown_chat_application()
        restore_shutdown_handlers(previous_handlers)
