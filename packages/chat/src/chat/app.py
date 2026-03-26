"""Reusable chat application scaffolding."""

from __future__ import annotations

import atexit
import signal
from collections.abc import Callable
from dataclasses import dataclass, field

import gradio as gr


@dataclass(frozen=True)
class ChatAppConfig:
    """Configuration for launching a chat application."""

    title: str = "Chat Agent"
    server_name: str = "127.0.0.1"
    server_port: int = 7860
    allowed_paths: list[str] | None = None
    pwa: bool = True
    share: bool = False


def install_shutdown_handlers(
    shutdown_fn: Callable[[], None],
) -> list[tuple[int, object]]:
    """Install signal and atexit handlers for graceful shutdown.

    Returns the previous signal handlers so they can be restored later.
    """
    previous_handlers: list[tuple[int, object]] = []

    def _handle_signal(signum: int, _frame: object) -> None:
        shutdown_fn()
        raise SystemExit(0)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers.append((signum, signal.getsignal(signum)))
        signal.signal(signum, _handle_signal)

    atexit.register(shutdown_fn)
    return previous_handlers


def restore_shutdown_handlers(previous_handlers: list[tuple[int, object]]) -> None:
    """Restore previously saved signal handlers."""
    for signum, handler in previous_handlers:
        signal.signal(signum, handler)


def launch(blocks: gr.Blocks, config: ChatAppConfig) -> None:
    """Launch a Gradio Blocks app with the given configuration."""
    blocks.queue()
    blocks.launch(
        pwa=config.pwa,
        share=config.share,
        allowed_paths=config.allowed_paths,
        server_name=config.server_name,
        server_port=config.server_port,
    )
