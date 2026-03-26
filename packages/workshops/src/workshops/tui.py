"""Shared Textual TUI base for workshop labs.

Each lab subclasses ``LabApp`` and overrides ``handle_input`` to wire up
its own agents.  The base provides a consistent layout with a conversation
transcript, an activity/events sidebar, and a composer input area.
"""

from __future__ import annotations

import threading
from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

from rich import box
from rich.markdown import Markdown as RichMarkdown
from rich.panel import Panel
from rich.text import Text
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Footer, Header, RichLog, Static, TextArea


# ---------------------------------------------------------------------------
# Dark colour palette
# ---------------------------------------------------------------------------
_BG = "#1a1b26"
_FG = "#c0caf5"
_HEADER_BG = "#24283b"
_HEADER_FG = "#7aa2f7"
_CARD_BG = "#1f2335"
_SIDEBAR_BG = "#1f2335"
_BORDER = "#3b4261"
_SIDEBAR_BORDER = "#3b4261"
_MUTED = "#565f89"
_SECTION = "#7aa2f7"
_ACCENT_USER = "#7dcfff"
_ACCENT_AGENT = "#9ece6a"
_ACCENT_TOOL = "#bb9af7"
_ACCENT_EVENT = "#ff9e64"
_ACCENT_OK = "#9ece6a"
_ACCENT_ERR = "#f7768e"
_ACCENT_SYSTEM = "#565f89"


class FileBrowserScreen(ModalScreen[str | None]):
    """Modal file browser. Returns the selected file path or None on cancel."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = f"""
    FileBrowserScreen {{
        align: center middle;
    }}
    #file-browser-container {{
        width: 80;
        height: 30;
        background: {_CARD_BG};
        border: round {_BORDER};
        padding: 1 2;
    }}
    #file-browser-title {{
        color: {_SECTION};
        text-style: bold;
        margin: 0 0 1 0;
    }}
    #file-tree {{
        height: 1fr;
        background: {_HEADER_BG};
        border: round {_BORDER};
    }}
    #file-browser-path {{
        height: 3;
        margin: 1 0 0 0;
        background: {_HEADER_BG};
        color: {_FG};
        border: round {_BORDER};
    }}
    #file-browser-actions {{
        height: auto;
        margin: 1 0 0 0;
    }}
    #file-select-btn {{
        width: 12;
        margin: 0 1 0 0;
        background: #7aa2f7;
        color: {_BG};
        text-style: bold;
    }}
    #file-cancel-btn {{
        width: 12;
    }}
    """

    def __init__(self, start_path: str = "~") -> None:
        super().__init__()
        self._start_path = str(Path(start_path).expanduser())
        self._selected_path: str = ""

    def compose(self) -> ComposeResult:
        with Vertical(id="file-browser-container"):
            yield Static("Select a file", id="file-browser-title")
            yield DirectoryTree(self._start_path, id="file-tree")
            yield TextArea("", id="file-browser-path")
            with Horizontal(id="file-browser-actions"):
                yield Button("Select", id="file-select-btn")
                yield Button("Cancel", id="file-cancel-btn")

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        self._selected_path = str(event.path)
        self.query_one("#file-browser-path", TextArea).text = self._selected_path

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "file-select-btn":
            path = self.query_one("#file-browser-path", TextArea).text.strip()
            self.dismiss(path if path else None)
        elif event.button.id == "file-cancel-btn":
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class LabApp(App[None]):
    """Base TUI for workshop labs.

    Subclasses must set ``lab_title``, ``lab_subtitle``, and ``lab_info``
    class variables and implement :meth:`handle_input` (called in a worker
    thread) and optionally :meth:`cleanup`.
    """

    ENABLE_COMMAND_PALETTE = False

    # -- Subclass configuration ------------------------------------------------
    lab_title: str = "Workshop Lab"
    lab_subtitle: str = ""
    lab_info: Sequence[str] = ()

    CSS = f"""
    Screen {{
        background: {_BG};
        color: {_FG};
    }}
    Header {{
        background: {_HEADER_BG};
        color: {_HEADER_FG};
    }}
    Footer {{
        background: {_HEADER_BG};
        color: {_HEADER_FG};
    }}
    #shell {{
        layout: vertical;
        height: 1fr;
        padding: 1 2;
    }}
    #workspace {{
        layout: horizontal;
        height: 1fr;
    }}
    #transcript-pane {{
        width: 3fr;
        height: 1fr;
        margin: 0 1 0 0;
        padding: 1;
        background: {_CARD_BG};
        border: round {_BORDER};
    }}
    #sidebar {{
        width: 38;
        height: 1fr;
        padding: 1;
        background: {_SIDEBAR_BG};
        border: round {_SIDEBAR_BORDER};
    }}
    .section-title {{
        color: {_SECTION};
        text-style: bold;
        margin: 0 0 1 0;
    }}
    #lab-info {{
        height: auto;
        margin: 0 0 1 0;
        color: {_MUTED};
    }}
    #status-card {{
        min-height: 3;
        height: auto;
        margin: 0 0 1 0;
        padding: 1;
        background: {_HEADER_BG};
        border: round {_BORDER};
        color: {_FG};
    }}
    #activity-log {{
        height: 1fr;
        background: {_CARD_BG};
        border: round {_BORDER};
    }}
    #composer-pane {{
        height: auto;
        margin: 1 0 0 0;
        padding: 1;
        background: {_CARD_BG};
        border: round {_BORDER};
    }}
    #composer-title {{
        color: {_SECTION};
        text-style: bold;
        margin: 0 0 1 0;
    }}
    #composer {{
        height: 6;
        min-height: 4;
        max-height: 10;
        background: {_HEADER_BG};
        color: {_FG};
        border: round {_BORDER};
    }}
    #composer-actions {{
        height: auto;
        margin: 1 0 0 0;
    }}
    #composer-hint {{
        width: 1fr;
        color: {_MUTED};
        content-align: left middle;
    }}
    #send {{
        width: 12;
        margin: 0 1 0 0;
        background: #7aa2f7;
        color: {_BG};
        text-style: bold;
    }}
    #quit-btn {{
        width: 10;
    }}
    #attach-btn {{
        width: 12;
        margin: 0 1 0 0;
    }}
    #attached-file {{
        width: 1fr;
        color: {_ACCENT_TOOL};
        content-align: left middle;
        margin: 0 0 0 0;
    }}
    """

    BINDINGS = [
        Binding("ctrl+enter", "send", "Send"),
        Binding("escape", "focus_composer", "Composer"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._busy = False
        self._attached_file: str | None = None

    # -- Layout ----------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="shell"):
            with Horizontal(id="workspace"):
                with Vertical(id="transcript-pane"):
                    yield Static("Conversation", classes="section-title")
                    yield RichLog(
                        id="transcript",
                        wrap=True,
                        markup=True,
                        highlight=True,
                        auto_scroll=True,
                    )
                with Vertical(id="sidebar"):
                    yield Static(self.lab_title, classes="section-title")
                    yield Static(
                        "\n".join(self.lab_info) if self.lab_info else "",
                        id="lab-info",
                    )
                    yield Static("Ready.", id="status-card")
                    yield Static("Activity", classes="section-title")
                    yield RichLog(
                        id="activity-log",
                        wrap=True,
                        markup=True,
                        highlight=True,
                        auto_scroll=True,
                    )
            with Vertical(id="composer-pane"):
                yield Static("Compose", id="composer-title")
                yield TextArea("", id="composer")
                yield Static("", id="attached-file")
                with Horizontal(id="composer-actions"):
                    yield Static(
                        "Ctrl+Enter sends  |  'quit' exits  |  Shift+click to select & copy",
                        id="composer-hint",
                    )
                    yield Button("Attach", id="attach-btn")
                    yield Button("Send", id="send")
                    yield Button("Quit", id="quit-btn")
        yield Footer()

    def on_mount(self) -> None:
        self.dark = True
        self.title = self.lab_title
        self.sub_title = self.lab_subtitle
        self._write_system(f"{self.lab_title} ready.")
        self._composer.focus()

    # -- Widget accessors ------------------------------------------------------

    @property
    def _transcript(self) -> RichLog:
        return self.query_one("#transcript", RichLog)

    @property
    def _activity(self) -> RichLog:
        return self.query_one("#activity-log", RichLog)

    @property
    def _composer(self) -> TextArea:
        return self.query_one("#composer", TextArea)

    # -- Thread dispatch -------------------------------------------------------

    def _on_main_thread(self) -> bool:
        return self._thread_id == threading.get_ident()

    def _call_safe(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Call *callback* on the main thread, regardless of caller thread."""
        if self._on_main_thread():
            callback(*args, **kwargs)
        else:
            self.call_from_thread(callback, *args, **kwargs)

    # -- Busy state ------------------------------------------------------------

    def _set_busy(self, busy: bool, *, status: str | None = None) -> None:
        self._busy = busy
        self._composer.read_only = busy
        for btn_id in ("send", "quit-btn"):
            self.query_one(f"#{btn_id}", Button).disabled = busy
        if status is not None:
            self._set_status(status)

    # -- Status card -----------------------------------------------------------

    def _set_status(self, text: str) -> None:
        self.query_one("#status-card", Static).update(text)

    def set_status(self, text: str) -> None:
        """Thread-safe status update."""
        self._call_safe(self._set_status, text)

    # -- Transcript helpers (thread-safe) --------------------------------------

    def _append_panel(
        self,
        *,
        title: str,
        subtitle: str,
        body: Text | RichMarkdown,
        border_style: str,
    ) -> None:
        panel = Panel(
            body,
            title=title,
            title_align="left",
            subtitle=subtitle,
            subtitle_align="right",
            padding=(0, 1),
            border_style=border_style,
            box=box.ROUNDED,
        )
        self._transcript.write(panel, expand=True, shrink=False)

    def write_user(self, message: str) -> None:
        """Show a user message in the transcript. Thread-safe."""
        self._call_safe(
            self._append_panel,
            title="You",
            subtitle="",
            body=Text(message.rstrip(), style=_ACCENT_USER),
            border_style=_ACCENT_USER,
        )

    def write_agent(self, name: str, message: str, *, accent: str = _ACCENT_AGENT) -> None:
        """Show an agent response in the transcript. Thread-safe."""
        self._call_safe(
            self._append_panel,
            title=name,
            subtitle="agent",
            body=RichMarkdown(message),
            border_style=accent,
        )

    def write_tool_call(self, name: str, params: str = "") -> None:
        """Show a tool call in the transcript. Thread-safe."""
        detail = f"`{name}({params})`" if params else f"`{name}()`"
        self._call_safe(
            self._append_panel,
            title="Tool Call",
            subtitle=name,
            body=RichMarkdown(detail),
            border_style=_ACCENT_TOOL,
        )

    def write_tool_result(self, name: str, result: str) -> None:
        """Show a tool result in the transcript. Thread-safe."""
        self._call_safe(
            self._append_panel,
            title="Tool Result",
            subtitle=name,
            body=Text(result, style=_ACCENT_OK),
            border_style=_ACCENT_OK,
        )

    def _write_system(self, message: str) -> None:
        self._append_panel(
            title="System",
            subtitle="",
            body=Text(message, style=_MUTED),
            border_style=_ACCENT_SYSTEM,
        )

    def write_system(self, message: str) -> None:
        """Show a system notice in the transcript. Thread-safe."""
        self._call_safe(self._write_system, message)

    def write_error(self, message: str) -> None:
        """Show an error in the transcript. Thread-safe."""
        self._call_safe(
            self._append_panel,
            title="Error",
            subtitle="",
            body=Text(message, style=_ACCENT_ERR),
            border_style=_ACCENT_ERR,
        )

    # -- Activity log helpers (thread-safe) ------------------------------------

    def _write_activity_line(self, label: str, detail: str, style: str) -> None:
        line = Text()
        line.append(f"{label}: ", style=f"bold {style}")
        line.append(detail, style=_FG)
        self._activity.write(line)

    def write_activity(self, label: str, detail: str, *, style: str = _ACCENT_OK) -> None:
        """Add a line to the activity feed. Thread-safe."""
        self._call_safe(self._write_activity_line, label, detail, style)

    # -- Actions ---------------------------------------------------------------

    def action_focus_composer(self) -> None:
        self._composer.focus()

    def action_send(self) -> None:
        if self._busy:
            return
        prompt = self._composer.text.strip()
        if not prompt and not self._attached_file:
            return
        if prompt.lower() in ("quit", "exit"):
            self._do_quit()
            return
        self._composer.clear()

        attached = self._attached_file
        self._attached_file = None
        self.query_one("#attached-file", Static).update("")

        display_text = prompt or ""
        if attached:
            display_text = f"{prompt}\n[file: {Path(attached).name}]" if prompt else f"[file: {Path(attached).name}]"

        self._append_panel(
            title="You",
            subtitle="",
            body=Text(display_text.rstrip(), style=_ACCENT_USER),
            border_style=_ACCENT_USER,
        )
        self._write_activity_line("Sent", display_text[:60], _ACCENT_USER)
        self._set_busy(True, status="Processing...")

        final_prompt = prompt if prompt else "Describe this image in detail."
        self.run_worker(
            lambda: self._run_input(final_prompt, attached_file=attached),
            thread=True,
            exclusive=True,
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send":
            self.action_send()
        elif event.button.id == "attach-btn":
            self._open_file_browser()
        elif event.button.id == "quit-btn":
            self._do_quit()

    def _open_file_browser(self) -> None:
        """Open the file browser modal to select a file."""
        self.push_screen(FileBrowserScreen(), callback=self._on_file_selected)

    def _on_file_selected(self, path: str | None) -> None:
        """Callback when a file is selected (or cancelled) from the browser."""
        self._attached_file = path
        indicator = self.query_one("#attached-file", Static)
        if path:
            name = Path(path).name
            indicator.update(f"Attached: {name}  (click Attach to change)")
        else:
            indicator.update("")
        self._composer.focus()

    def _do_quit(self) -> None:
        self.cleanup()
        self.exit()

    def _run_input(self, prompt: str, *, attached_file: str | None = None) -> None:
        try:
            self.handle_input(prompt, attached_file=attached_file)
        except Exception as exc:
            self.write_error(str(exc))
            self.write_activity("Error", str(exc), style=_ACCENT_ERR)
        finally:
            self._call_safe(self._set_busy, False, status="Ready.")
            self._call_safe(self._composer.focus)

    # -- Subclass interface ----------------------------------------------------

    @abstractmethod
    def handle_input(self, message: str, *, attached_file: str | None = None) -> None:
        """Process *message* from the user.

        Called in a worker thread.  Use the ``write_*`` helpers to push
        content to the UI — they are all thread-safe.

        *attached_file* is the path to a file selected via the Attach button,
        or ``None`` if no file was attached.
        """

    def cleanup(self) -> None:
        """Release resources (models, runtimes).  Called on exit."""
