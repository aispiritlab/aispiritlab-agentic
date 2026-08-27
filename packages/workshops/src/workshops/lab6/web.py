"""Browser chat for the distributed Lab 6 pipeline.

This is the entry point the ``chat`` container in
``containers/docker-compose.lab6.yml`` runs: a thin Gradio front-end that
publishes a question to the planner over Redis Streams and waits for the
summary agent's reply.
"""

from __future__ import annotations

from collections.abc import Generator

import gradio as gr

from agentic_runtime.distributed import DistributedAgenticRuntime
from agentic_runtime.settings import settings
from chat import (
    ChatAppConfig,
    ChatHistory,
    add_message,
    install_shutdown_handlers,
    launch,
    message_prompt_text,
    parse_auth,
    restore_shutdown_handlers,
)

GREETING = (
    "Cześć! To rozproszony tryb Lab 6: **planner → search → summary** przez Redis Streams.\n\n"
    "Wpisz `agents`, aby zobaczyć aktywne usługi, albo zadaj pytanie."
)


def _render_agents(runtime: DistributedAgenticRuntime) -> str:
    agents = runtime.live_agents()
    if not agents:
        return "Brak zarejestrowanych usług rozproszonych."

    lines = ["**Aktywne usługi**", ""]
    for index, agent in enumerate(agents, start=1):
        capabilities = ", ".join(agent.capabilities) or "—"
        lines.append(
            f"{index}. `{agent.agent_name}` — status: {agent.status}, zdolności: {capabilities}"
        )
    return "\n".join(lines)


def build_ui(runtime: DistributedAgenticRuntime) -> gr.Blocks:
    """Build the Lab 6 chat interface bound to ``runtime``."""

    def respond(history: ChatHistory) -> Generator[ChatHistory]:
        if not history or history[-1]["role"] != "user":
            yield history
            return

        prompt = message_prompt_text(history[-1]).strip()
        if not prompt:
            yield history
            return

        if prompt.lower() in {"agents", "status"}:
            history.append({"role": "assistant", "content": _render_agents(runtime)})
            yield history
            return

        try:
            answer = runtime.run(prompt)
        except TimeoutError:
            answer = (
                "Nie doczekałem się odpowiedzi od potoku rozproszonego. "
                "Sprawdź, czy usługi planner/search/summary działają."
            )
        except ConnectionError, RuntimeError, ValueError:
            answer = "Potok rozproszony jest w tej chwili niedostępny. Spróbuj ponownie za chwilę."

        history.append({"role": "assistant", "content": answer})
        yield history

    with gr.Blocks(title="Lab 6 — Distributed Chat", fill_height=True) as block:
        gr.Markdown("# Lab 6 — rozproszony potok agentów")
        chatbot = gr.Chatbot(
            value=[{"role": "assistant", "content": GREETING}],
            height=560,
            show_label=False,
        )
        chat_input = gr.MultimodalTextbox(
            interactive=True,
            placeholder="Zadaj pytanie potokowi…",
            show_label=False,
            sources=[],
        )

        submitted = chat_input.submit(
            lambda history, message: add_message(history, message),
            inputs=[chatbot, chat_input],
            outputs=[chatbot, chat_input],
        )
        submitted.then(respond, inputs=[chatbot], outputs=[chatbot]).then(
            lambda: gr.MultimodalTextbox(interactive=True),
            outputs=[chat_input],
        )

    return block


def main() -> None:
    """Run the Lab 6 browser chat."""
    runtime = DistributedAgenticRuntime.from_settings()
    stopped = False

    def _stop_once() -> None:
        nonlocal stopped
        if stopped:
            return
        stopped = True
        runtime.stop()

    previous_handlers = install_shutdown_handlers(_stop_once)
    try:
        config = ChatAppConfig(
            title="Lab 6 — Distributed Chat",
            server_name=settings.chat_server_name,
            server_port=settings.chat_server_port,
            auth=parse_auth(settings.chat_auth),
        )
        launch(build_ui(runtime), config)
    finally:
        _stop_once()
        restore_shutdown_handlers(previous_handlers)


if __name__ == "__main__":
    main()
