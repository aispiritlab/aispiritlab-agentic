"""Reusable Gradio chat UI building blocks."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

type ChatMessage = dict[str, object]
type ChatHistory = list[ChatMessage]
type MultimodalMessage = dict[str, object]


def coerce_text(value: object) -> str:
    """Coerce a value to a string, treating None as empty."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def extract_uploaded_files(message: MultimodalMessage | str | None) -> list[str]:
    """Extract file paths from a Gradio multimodal message."""
    if not isinstance(message, dict):
        return []

    files = message.get("files")
    if not isinstance(files, list):
        return []

    file_paths: list[str] = []
    for file in files:
        if isinstance(file, str):
            if file:
                file_paths.append(file)
            continue
        if isinstance(file, Path):
            file_paths.append(str(file))
            continue
        if isinstance(file, dict):
            path = file.get("path") or file.get("name")
            if isinstance(path, str) and path:
                file_paths.append(path)
            continue

        for attr in ("path", "name"):
            path = getattr(file, attr, None)
            if isinstance(path, str) and path:
                file_paths.append(path)
                break

    return file_paths


def describe_uploaded_files(file_paths: list[str], *, prefix: str = "Uploaded file") -> str:
    """Build a human-readable description of uploaded files."""
    names = [Path(path).name for path in file_paths]
    if not names:
        return ""
    if len(names) == 1:
        return f"{prefix}: {names[0]}"
    return f"{prefix}s: {', '.join(names)}"


def message_prompt_text(message: ChatMessage) -> str:
    """Extract the prompt text from a chat message (stored in metadata or content)."""
    metadata = message.get("metadata")
    if isinstance(metadata, dict):
        prompt_text = metadata.get("prompt_text")
        if isinstance(prompt_text, str):
            return prompt_text

    prompt_text = message.get("prompt_text")
    if isinstance(prompt_text, str):
        return prompt_text
    return coerce_text(message.get("content"))


def message_files(message: ChatMessage) -> list[str]:
    """Extract file paths from a chat message (stored in metadata)."""
    metadata = message.get("metadata")
    if isinstance(metadata, dict):
        files = metadata.get("files")
        if isinstance(files, list):
            return [str(path) for path in files if isinstance(path, str)]

    files = message.get("files")
    if isinstance(files, list):
        return [str(path) for path in files if isinstance(path, str)]
    return []


def add_message(
    history: ChatHistory,
    message: MultimodalMessage | str | None,
    *,
    file_description_prefix: str = "Uploaded file",
) -> tuple[ChatHistory, gr.MultimodalTextbox]:
    """Add a user message to chat history and clear the input box."""
    if isinstance(message, dict):
        text = coerce_text(message.get("text")).strip()
        file_paths = extract_uploaded_files(message)
    else:
        text = coerce_text(message).strip()
        file_paths = []

    display_parts = [
        part
        for part in [text, describe_uploaded_files(file_paths, prefix=file_description_prefix)]
        if part
    ]
    if display_parts:
        history.append(
            {
                "role": "user",
                "content": "\n".join(display_parts),
                "metadata": {
                    "prompt_text": text,
                    "files": file_paths,
                },
            }
        )

    return history, gr.MultimodalTextbox(value=None, interactive=False)


def append_voice_response(
    history: ChatHistory,
    audio: object,
    *,
    voice_model: object,
    convert_audio_fn: object,
    is_empty_fn: object,
) -> tuple[object, object]:
    """Handle audio input: transcribe and append to history.

    Args:
        history: Current chat history.
        audio: Raw audio data from Gradio.
        voice_model: The voice model instance.
        convert_audio_fn: Callable(voice_model, audio) -> (transcription, error).
        is_empty_fn: Callable(transcription) -> bool.
    """
    if voice_model is None:
        print("Voice model is not available.")
        return history, None

    transcription, response_error = convert_audio_fn(voice_model, audio)  # type: ignore[operator]
    if response_error is not None and is_empty_fn(transcription):  # type: ignore[operator]
        print(response_error)

    if transcription is None:
        print("Voice transcription failed.")
        return history, None

    transcribed_text = str(getattr(transcription, "text", ""))

    if isinstance(history, list):
        history.append({"role": "user", "content": transcribed_text})
        return history, None

    if isinstance(history, dict):
        updated_history = dict(history)
        updated_history["text"] = transcribed_text
        updated_history.setdefault("files", [])
        return updated_history, None

    if isinstance(history, str):
        return transcribed_text, None

    return history, None
