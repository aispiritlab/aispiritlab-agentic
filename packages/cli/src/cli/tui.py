from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
import json
from threading import current_thread, main_thread

from rich import box
from rich.markdown import Markdown as RichMarkdown
from rich.panel import Panel
from rich.text import Text

from agentic.image_generation_call import ImageGenerationResult
from personal_assistant import (
    ai_spirit_agent,
    chat_agent,
    clear_personalization_history,
    generate_image_agent,
    get_initial_greeting,
    get_runtime,
)
from agentic.workflow.messages import Message
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Footer, Header, RichLog, Static, TextArea


class ChatMode(StrEnum):
    AGENTS = "agents"
    DIRECT = "direct"
    IMAGE = "image"

    @property
    def label(self) -> str:
        return {
            ChatMode.AGENTS: "Agents",
            ChatMode.DIRECT: "Direct Chat",
            ChatMode.IMAGE: "Image",
        }[self]

    @property
    def description(self) -> str:
        return {
            ChatMode.AGENTS: "Router and workflows decide which specialist should answer.",
            ChatMode.DIRECT: "Bypass routing and talk to the configured chat model directly.",
            ChatMode.IMAGE: "Generate an image and return the saved file path with metadata.",
        }[self]

    @property
    def placeholder(self) -> str:
        return {
            ChatMode.AGENTS: "Ask the agent to help with notes, discovery, planning, or general tasks...",
            ChatMode.DIRECT: "Talk directly to the chat model...",
            ChatMode.IMAGE: "Describe the image you want to generate...",
        }[self]

    @property
    def accent(self) -> str:
        return {
            ChatMode.AGENTS: "#0f766e",
            ChatMode.DIRECT: "#9a3412",
            ChatMode.IMAGE: "#1d4ed8",
        }[self]

    @property
    def status_text(self) -> str:
        return {
            ChatMode.AGENTS: "Routing your message through the agent runtime...",
            ChatMode.DIRECT: "Sending the message straight to the chat model...",
            ChatMode.IMAGE: "Generating an image from the prompt...",
        }[self]


_MODE_BUTTON_IDS = {
    ChatMode.AGENTS: "mode-agents",
    ChatMode.DIRECT: "mode-direct",
    ChatMode.IMAGE: "mode-image",
}


def _format_tool_parameters(parameters: object) -> str:
    if not isinstance(parameters, dict) or not parameters:
        return ""
    rendered = json.dumps(parameters, ensure_ascii=False, sort_keys=True)
    if len(rendered) > 140:
        rendered = f"{rendered[:137]}..."
    return rendered


def _format_response(response: str | ImageGenerationResult) -> str:
    if isinstance(response, ImageGenerationResult):
        return "\n".join(
            [
                "## Image generated",
                f"- Path: `{response.image_path}`",
                f"- Seed: `{response.seed}`",
                f"- Size: `{response.width}x{response.height}`",
                f"- Steps: `{response.steps}`",
            ]
        )
    return response


class ChatTerminalApp(App[None]):
    CSS = """
    Screen {
        background: #f4efe6;
        color: #172123;
    }

    Header {
        background: #17383d;
        color: #f8f4ec;
    }

    Footer {
        background: #17383d;
        color: #f8f4ec;
    }

    #shell {
        layout: vertical;
        height: 1fr;
        padding: 1 2;
    }

    #workspace {
        layout: horizontal;
        height: 1fr;
    }

    #transcript-pane {
        width: 3fr;
        height: 1fr;
        margin: 0 1 0 0;
        padding: 1;
        background: #fffaf2;
        border: round #7b8f8c;
    }

    #sidebar {
        width: 36;
        height: 1fr;
        padding: 1;
        background: #e8dcc7;
        border: round #8a7552;
    }

    .section-title {
        color: #5f4d2f;
        text-style: bold;
        margin: 0 0 1 0;
    }

    #mode-buttons {
        height: auto;
        margin: 0 0 1 0;
    }

    .mode-switch {
        width: 1fr;
        margin: 0 1 0 0;
    }

    .mode-switch.active {
        background: #17383d;
        color: #f8f4ec;
        text-style: bold;
    }

    #mode-copy {
        height: 4;
        margin: 0 0 1 0;
        color: #423722;
    }

    #status-card {
        min-height: 5;
        height: auto;
        margin: 0 0 1 0;
        padding: 1;
        background: #f8f2e6;
        border: round #9b886a;
        color: #17383d;
    }

    #activity-log {
        height: 1fr;
        background: #fffaf2;
        border: round #9b886a;
    }

    #composer-pane {
        height: auto;
        margin: 1 0 0 0;
        padding: 1;
        background: #fffaf2;
        border: round #7b8f8c;
    }

    #composer-title {
        color: #17383d;
        text-style: bold;
        margin: 0 0 1 0;
    }

    #composer {
        height: 8;
        min-height: 6;
        max-height: 12;
        background: #fffdf8;
        color: #172123;
        border: round #7b8f8c;
    }

    #composer-actions {
        height: auto;
        margin: 1 0 0 0;
    }

    #composer-hint {
        width: 1fr;
        color: #5c6770;
        content-align: left middle;
    }

    #send {
        width: 12;
        margin: 0 1 0 0;
        background: #17383d;
        color: #f8f4ec;
        text-style: bold;
    }

    #clear {
        width: 14;
    }
    """

    TITLE = "AI Spirit Agent"
    SUB_TITLE = "Textual + Rich terminal chat"
    BINDINGS = [
        Binding("ctrl+enter", "send", "Send"),
        Binding("ctrl+l", "clear_chat", "New Chat"),
        Binding("1", "mode_agents", "Agents"),
        Binding("2", "mode_direct", "Direct"),
        Binding("3", "mode_image", "Image"),
        Binding("escape", "focus_composer", "Composer"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._mode = ChatMode.AGENTS
        self._busy = False
        self._runtime = get_runtime()
        self._runtime_subscribed = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="shell"):
            with Horizontal(id="workspace"):
                with Vertical(id="transcript-pane"):
                    yield Static("Conversation", classes="section-title")
                    yield RichLog(id="transcript", wrap=True, markup=False, highlight=True)
                with Vertical(id="sidebar"):
                    yield Static("Mode", classes="section-title")
                    with Horizontal(id="mode-buttons"):
                        yield Button("Agents", id="mode-agents", classes="mode-switch")
                        yield Button("Direct", id="mode-direct", classes="mode-switch")
                        yield Button("Image", id="mode-image", classes="mode-switch")
                    yield Static("", id="mode-copy")
                    yield Static("", id="status-card")
                    yield Static("Activity", classes="section-title")
                    yield RichLog(id="activity-log", wrap=True, markup=False, highlight=True)
            with Vertical(id="composer-pane"):
                yield Static("Compose", id="composer-title")
                yield TextArea("", id="composer")
                with Horizontal(id="composer-actions"):
                    yield Static("Ctrl+Enter sends. Enter adds a new line.", id="composer-hint")
                    yield Button("Send", id="send")
                    yield Button("New Chat", id="clear")
        yield Footer()

    def on_mount(self) -> None:
        if not self._runtime_subscribed:
            self._runtime.bus.subscribe(self._handle_runtime_message)
            self._runtime_subscribed = True
        self._sync_mode_ui()
        self._set_status("Ready for the next message.")
        self._append_notice(
            title="Terminal chat upgraded",
            body=(
                "You now have a persistent transcript, runtime activity feed, mode switching, "
                "and markdown-friendly assistant rendering."
            ),
            border_style="#8a7552",
        )
        self._append_assistant_message(get_initial_greeting(), ChatMode.AGENTS)
        self._composer.focus()

    @property
    def _transcript(self) -> RichLog:
        return self.query_one("#transcript", RichLog)

    @property
    def _activity_log(self) -> RichLog:
        return self.query_one("#activity-log", RichLog)

    @property
    def _composer(self) -> TextArea:
        return self.query_one("#composer", TextArea)

    def _set_status(self, text: str) -> None:
        self.query_one("#status-card", Static).update(text)

    def _sync_mode_ui(self) -> None:
        self.query_one("#mode-copy", Static).update(
            f"{self._mode.label}\n{self._mode.description}"
        )
        self._composer.placeholder = self._mode.placeholder
        for mode, button_id in _MODE_BUTTON_IDS.items():
            self.query_one(f"#{button_id}", Button).set_class(mode == self._mode, "active")

    def _set_busy(self, busy: bool, *, status: str | None = None) -> None:
        self._busy = busy
        self._composer.read_only = busy
        for button_id in (*_MODE_BUTTON_IDS.values(), "send", "clear"):
            self.query_one(f"#{button_id}", Button).disabled = busy
        if status is not None:
            self._set_status(status)

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

    def _append_notice(self, *, title: str, body: str, border_style: str) -> None:
        self._append_panel(
            title=title,
            subtitle="System",
            body=Text(body, style="#423722"),
            border_style=border_style,
        )

    def _append_user_message(self, message: str, mode: ChatMode) -> None:
        self._append_panel(
            title="You",
            subtitle=mode.label,
            body=Text(message.rstrip(), style="#17383d"),
            border_style=mode.accent,
        )

    def _append_assistant_message(
        self,
        message: str,
        mode: ChatMode,
        *,
        title: str = "Assistant",
    ) -> None:
        self._append_panel(
            title=title,
            subtitle=mode.label,
            body=RichMarkdown(message),
            border_style=mode.accent,
        )

    def _write_activity(self, label: str, detail: str, *, style: str) -> None:
        line = Text()
        line.append(f"{label}: ", style=f"bold {style}")
        line.append(detail, style="#172123")
        self._activity_log.write(line)

    def _set_mode(self, mode: ChatMode) -> None:
        if self._busy or mode == self._mode:
            return
        self._mode = mode
        self._sync_mode_ui()
        self._set_status(f"{mode.label} ready.")
        self._write_activity("Mode", f"Switched to {mode.label}.", style=mode.accent)
        self._composer.focus()

    def action_mode_agents(self) -> None:
        self._set_mode(ChatMode.AGENTS)

    def action_mode_direct(self) -> None:
        self._set_mode(ChatMode.DIRECT)

    def action_mode_image(self) -> None:
        self._set_mode(ChatMode.IMAGE)

    def action_focus_composer(self) -> None:
        self._composer.focus()

    def action_clear_chat(self) -> None:
        if self._busy:
            return
        clear_personalization_history()
        self._transcript.clear()
        self._activity_log.clear()
        self._set_status("Fresh session ready.")
        self._append_notice(
            title="New session",
            body="Conversation state was cleared across agent, direct chat, and personalization flows.",
            border_style="#8a7552",
        )
        if self._mode == ChatMode.AGENTS:
            self._append_assistant_message(get_initial_greeting(), ChatMode.AGENTS)
        self._composer.focus()

    def action_send(self) -> None:
        if self._busy:
            return

        prompt = self._composer.text.strip()
        if not prompt:
            self._set_status("Write a message first.")
            self._composer.focus()
            return

        mode = self._mode
        self._composer.clear()
        self._append_user_message(prompt, mode)
        self._write_activity("Queued", f"Submitting message in {mode.label}.", style=mode.accent)
        self._set_busy(True, status=mode.status_text)
        self.run_worker(
            lambda: self._submit_message(prompt, mode),
            thread=True,
            exclusive=True,
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "send":
            self.action_send()
        elif button_id == "clear":
            self.action_clear_chat()
        elif button_id == "mode-agents":
            self.action_mode_agents()
        elif button_id == "mode-direct":
            self.action_mode_direct()
        elif button_id == "mode-image":
            self.action_mode_image()

    def _submit_message(self, prompt: str, mode: ChatMode) -> None:
        try:
            response = self._dispatch_prompt(prompt, mode)
        except Exception as error:
            self.call_from_thread(self._handle_request_error, mode, error)
        else:
            self.call_from_thread(self._handle_request_success, mode, response)

    def _dispatch_prompt(self, prompt: str, mode: ChatMode) -> str | ImageGenerationResult:
        handler: Callable[[str], str | ImageGenerationResult]
        if mode == ChatMode.AGENTS:
            handler = ai_spirit_agent
        elif mode == ChatMode.DIRECT:
            handler = chat_agent
        else:
            handler = generate_image_agent
        return handler(prompt)

    def _handle_request_success(
        self,
        mode: ChatMode,
        response: str | ImageGenerationResult,
    ) -> None:
        self._append_assistant_message(_format_response(response), mode)
        self._set_busy(False, status=f"{mode.label} answered.")
        self._write_activity("Complete", f"{mode.label} finished the request.", style=mode.accent)
        self._composer.focus()

    def _handle_request_error(self, mode: ChatMode, error: Exception) -> None:
        self._append_notice(
            title="Request failed",
            body=str(error),
            border_style="#b42318",
        )
        self._set_busy(False, status="Request failed.")
        self._write_activity("Error", str(error), style="#b42318")
        self._composer.focus()

    def _handle_runtime_message(self, message: Message) -> None:
        if message.runtime_id != self._runtime.runtime_id:
            return

        if current_thread() is main_thread():
            self._process_runtime_message(message)
            return

        try:
            self.call_from_thread(self._process_runtime_message, message)
        except RuntimeError:
            return

    def _process_runtime_message(self, message: Message) -> None:
        if message.kind == "turn_started":
            workflow = "agent"
            if isinstance(message.payload, dict):
                workflow = str(message.payload.get("workflow") or workflow)
            self._set_status(f"Running {workflow}...")
            self._write_activity("Turn", f"Started in {workflow}.", style="#0f766e")
            return

        if message.kind == "event" and message.name == "workflow_selected":
            workflow = "agent"
            if isinstance(message.payload, dict):
                workflow = str(message.payload.get("workflow") or workflow)
            self._set_status(f"Routed to {workflow}.")
            self._write_activity("Route", f"Router selected {workflow}.", style="#9a3412")
            return

        if message.kind == "tool_call":
            tool_name = message.source or "tool"
            if isinstance(message.payload, dict):
                tool_name = str(message.payload.get("name") or tool_name)
            parameters = _format_tool_parameters(
                message.payload.get("parameters") if isinstance(message.payload, dict) else None
            )
            detail = tool_name if not parameters else f"{tool_name} {parameters}"
            self._set_status(f"Running {tool_name}...")
            self._write_activity("Tool", detail, style="#1d4ed8")
            return

        if message.kind == "tool_result":
            tool_name = message.name or message.source or "tool"
            self._write_activity("Tool result", f"{tool_name} completed.", style="#15803d")
            return

        if message.kind == "turn_completed":
            status = message.status or "success"
            if status == "success" and self._busy:
                self._set_status("Finalizing response...")
            elif status != "success":
                self._set_status("Request ended with an error.")
            self._write_activity("Turn", f"Completed with status={status}.", style="#15803d")


def run_textual_chat() -> None:
    ChatTerminalApp().run()
