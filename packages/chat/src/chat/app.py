"""Reusable chat application scaffolding."""

from __future__ import annotations

import atexit
from collections.abc import Callable
from dataclasses import dataclass
import signal

import gradio as gr

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class InsecureExposureError(RuntimeError):
    """Raised when an app would be exposed off-host without authentication."""


@dataclass(frozen=True)
class ChatAppConfig:
    """Configuration for launching a chat application."""

    title: str = "Chat Agent"
    server_name: str = "127.0.0.1"
    server_port: int = 7860
    allowed_paths: list[str] | None = None
    pwa: bool = True
    share: bool = False
    auth: tuple[str, str] | list[tuple[str, str]] | Callable[[str, str], bool] | None = None
    auth_message: str | None = None

    @property
    def is_loopback_only(self) -> bool:
        return self.server_name in LOOPBACK_HOSTS


def parse_auth(raw: str) -> list[tuple[str, str]] | None:
    """Parse a ``"user:password,user2:password2"`` string into credentials.

    Returns ``None`` for an empty value so it can be passed straight to
    :class:`ChatAppConfig.auth`.
    """
    credentials: list[tuple[str, str]] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        user, separator, password = entry.partition(":")
        if not separator or not user.strip() or not password:
            raise ValueError(f"Invalid credential entry {entry!r}: expected 'user:password'.")
        credentials.append((user.strip(), password))
    return credentials or None


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
    """Launch a Gradio Blocks app with the given configuration.

    Refuses to bind off-loopback without authentication: the app switches user
    identity from the UI and reads per-user vaults, so an unauthenticated
    non-local bind hands every profile to anyone who can reach the port.
    """
    if config.auth is None and not (config.is_loopback_only and not config.share):
        exposure = "a public share link" if config.share else f"host {config.server_name!r}"
        raise InsecureExposureError(
            f"Refusing to serve {exposure} without authentication. "
            "Set CHAT_AUTH (user:password, comma-separated for several accounts) "
            "or bind to 127.0.0.1."
        )

    blocks.queue()
    blocks.launch(
        pwa=config.pwa,
        share=config.share,
        allowed_paths=config.allowed_paths,
        server_name=config.server_name,
        server_port=config.server_port,
        auth=config.auth,
        auth_message=config.auth_message,
    )
