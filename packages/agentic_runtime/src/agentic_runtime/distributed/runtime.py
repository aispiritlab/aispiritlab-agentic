from __future__ import annotations

import uuid

from agentic_runtime.distributed.client import DistributedChatClient
from agentic_runtime.distributed.discovery import AgenticServiceDiscovery
from agentic_runtime.distributed.registry import AgentSnapshot
from agentic_runtime.messaging.messages import Message, UserMessage


class DistributedAgenticRuntime:
    """Runtime backed by Redis Streams — satisfies ``RuntimeProtocol``.

    The chat layer and ``get_runtime()`` can return this transparently;
    callers never know whether the runtime is local or distributed.
    """

    def __init__(
        self,
        discovery: AgenticServiceDiscovery,
        *,
        entry_agent: str = "planner",
        source: str = "chat",
        timeout_seconds: float = 60.0,
        domain: str = "lab6",
    ) -> None:
        self._discovery = discovery
        self._client = discovery.create_client(
            entry_agent=entry_agent,
            source=source,
            timeout_seconds=timeout_seconds,
            domain=domain,
        )
        self._runtime_id = str(uuid.uuid7())

    @classmethod
    def from_settings(cls) -> DistributedAgenticRuntime:
        from agentic_runtime.settings import settings

        discovery = AgenticServiceDiscovery.from_settings()
        return cls(
            discovery,
            entry_agent=settings.chat_entry_agent,
            source=settings.distributed_chat_source,
            timeout_seconds=settings.distributed_chat_timeout_seconds,
        )

    # ------------------------------------------------------------------
    # RuntimeProtocol
    # ------------------------------------------------------------------

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    @property
    def discovery(self) -> AgenticServiceDiscovery:
        return self._discovery

    def run(self, text: str) -> str:
        return self._client.ask(text)

    def handle(self, message: Message | object) -> str:
        if isinstance(message, UserMessage):
            return self.run(message.text)
        raise NotImplementedError(
            "Distributed runtime only supports UserMessage. "
            f"Got {type(message).__name__}."
        )

    def start(self) -> str:
        return (
            "Cześć! To rozproszony tryb: planner -> search -> summary. "
            "Zadaj pytanie, a odpowiedź wróci przez Redis Streams."
        )

    def stop(self) -> None:
        self._client.close()
        self._discovery.close()

    def run_chat(self, text: str) -> str:
        return "Tryb Chat nie jest dostępny w trybie rozproszonym."

    def run_generate_image(
        self,
        text: str,
        images: str | list[str] | None = None,
    ) -> str:
        return "Generowanie obrazów nie jest dostępne w trybie rozproszonym."

    def reset_chat(self) -> None:
        pass

    def clear_personalization_history(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Extra — distributed-specific
    # ------------------------------------------------------------------

    def live_agents(self) -> list[AgentSnapshot]:
        return self._discovery.live_agents()
