from __future__ import annotations

from __future__ import annotations

from collections.abc import Callable, Sequence
import os
import socket
import threading
from typing import TYPE_CHECKING

from structlog import get_logger

from agentic_runtime.distributed.contracts import AgentHeartbeat, AgentRegistration
from agentic_runtime.messaging.messages import (
    AssistantMessage,
    ConversationData,
    Message,
    RecordedMessageMetadata,
    TurnCompleted,
)

if TYPE_CHECKING:
    from agentic_runtime.distributed.discovery import AgenticServiceDiscovery

logger = get_logger(__name__)

type MessageHandler = Callable[[Message, "AgenticServiceDiscovery"], Sequence[Message]]
type CloseHook = Callable[[], None]


class DistributedService:
    def __init__(
        self,
        *,
        agent_name: str,
        capabilities: tuple[str, ...],
        discovery: AgenticServiceDiscovery,
        handler: MessageHandler,
        role: str = "worker",
        heartbeat_seconds: float = 5.0,
        close_hook: CloseHook | None = None,
        min_idle_ms: int = 5_000,
    ) -> None:
        self._agent_name = agent_name
        self._capabilities = capabilities
        self._discovery = discovery
        self._transport = discovery.transport
        self._registry = discovery.registry
        self._handler = handler
        self._role = role
        self._heartbeat_seconds = heartbeat_seconds
        self._close_hook = close_hook
        self._min_idle_ms = min_idle_ms
        self._group = agent_name
        self._consumer_name = f"{socket.gethostname()}-{os.getpid()}"
        self._stop_event = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    def run_forever(self) -> None:
        self._registry.register(
            AgentRegistration(
                agent_name=self._agent_name,
                capabilities=self._capabilities,
                role=self._role,
                consumer_group=self._group,
            )
        )
        self._registry.heartbeat(AgentHeartbeat(agent_name=self._agent_name, status="ready"))
        self._transport.ensure_consumer_group(self._agent_name, self._group)

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"{self._agent_name}-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

        self._drain_pending()

        while not self._stop_event.is_set():
            records = self._transport.consume_target(
                self._agent_name,
                group=self._group,
                consumer=self._consumer_name,
                block_ms=1_000,
                count=10,
            )
            if not records:
                continue

            for record in records:
                self._handle_record(record.stream, record.entry_id, record.record)

    def close(self) -> None:
        self._stop_event.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=1.0)
        if self._close_hook is not None:
            self._close_hook()

    def _drain_pending(self) -> None:
        """Re-process messages left in the PEL from a prior crash."""
        while not self._stop_event.is_set():
            records = self._transport.autoclaim_pending(
                self._agent_name,
                group=self._group,
                consumer=self._consumer_name,
                min_idle_ms=self._min_idle_ms,
                count=10,
            )
            if not records:
                break
            logger.info(
                "drain_pending_messages",
                agent_name=self._agent_name,
                count=len(records),
            )
            for record in records:
                self._handle_record(record.stream, record.entry_id, record.record)

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self._heartbeat_seconds):
            self._registry.heartbeat(
                AgentHeartbeat(agent_name=self._agent_name, status="alive")
            )

    def _handle_record(self, stream: str, entry_id: str, record: object) -> None:
        if not isinstance(record, Message):
            self._transport.ack(stream, self._group, entry_id)
            return

        try:
            responses = tuple(self._handler(record, self._discovery))
            for response in responses:
                self._transport.publish_message(response)
            self._transport.ack(stream, self._group, entry_id)
        except Exception as error:
            logger.warning(
                "distributed_service_handler_failed",
                agent_name=self._agent_name,
                error_type=type(error).__name__,
                error_message=str(error),
            )
            for response in self._error_messages(record, error):
                self._transport.publish_message(response)
            self._transport.ack(stream, self._group, entry_id)

    def _error_messages(self, message: Message, error: Exception) -> tuple[Message, ...]:
        reply_target = self._resolve_reply_target(message)
        return (
            AssistantMessage(
                data=ConversationData(
                    role="assistant",
                    text=f"{self._agent_name} failed: {error}",
                ),
                metadata=RecordedMessageMetadata(
                    runtime_id=message.metadata.runtime_id,
                    session_id=message.metadata.session_id,
                    turn_id=message.metadata.turn_id,
                    domain=message.metadata.domain or "lab6",
                    source=self._agent_name,
                    target=reply_target,
                    status="error",
                    trace=message.metadata.trace,
                ),
            ),
            TurnCompleted(
                data={
                    "workflow": self._agent_name,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                metadata=RecordedMessageMetadata(
                    runtime_id=message.metadata.runtime_id,
                    session_id=message.metadata.session_id,
                    turn_id=message.metadata.turn_id,
                    domain=message.metadata.domain or "lab6",
                    source=self._agent_name,
                    target=reply_target,
                    status="error",
                    trace=message.metadata.trace,
                ),
            ),
        )

    @staticmethod
    def _resolve_reply_target(message: Message) -> str:
        payload = message.data if isinstance(message.data, dict) else {}
        reply_target = payload.get("reply_target")
        if isinstance(reply_target, str) and reply_target:
            return reply_target
        if message.metadata.source:
            return message.metadata.source
        return "chat"
