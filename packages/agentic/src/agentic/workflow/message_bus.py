from __future__ import annotations

from typing import Any, Callable, Protocol
import uuid

from agentic.workflow.messages import Message
from agentic.workflow.output_handler import OutputHandlerDispatcher, WorkflowOutputHandler
from agentic.workflow.streaming import hash_text


type MessageHandler = Callable[[Message], str | None]


class MessageStore(Protocol):
    def enqueue(self, message: Message) -> None: ...

    def close(self) -> None: ...


class InMemoryMessageBus:
    def __init__(self, store: MessageStore | None = None) -> None:
        self.messages: list[Message] = []
        self._subscribers: list[MessageHandler] = []
        self._output_dispatcher = OutputHandlerDispatcher()
        self._store = store
        self._sequence_no = 0

    def _normalize_message(self, message: Message) -> Message:
        updates: dict[str, Any] = {}
        event_id = getattr(message.metadata, "event_id", "")
        message_id = getattr(message.metadata, "message_id", "")
        sequence_no = getattr(message.metadata, "sequence_no", None)
        content_sha256 = getattr(message.metadata, "content_sha256", None)
        text = message.data.text if hasattr(message.data, "text") else None
        if not event_id:
            updates["event_id"] = str(uuid.uuid4())
        if not message_id:
            updates["message_id"] = str(uuid.uuid4())
        if not message.metadata.turn_id:
            updates["turn_id"] = message.metadata.runtime_id
        if sequence_no is None:
            self._sequence_no += 1
            updates["sequence_no"] = self._sequence_no
        if text and not content_sha256:
            updates["content_sha256"] = hash_text(text)
        if updates:
            return message.with_metadata(**updates)
        return message

    def subscribe(self, handler: MessageHandler) -> None:
        self._subscribers.append(handler)

    def register_output_handler(self, handler: WorkflowOutputHandler) -> None:
        self._output_dispatcher.register(handler)

    def publish(self, message: Message) -> list[str]:
        normalized = self._normalize_message(message)
        self.messages.append(normalized)
        if self._store is not None:
            self._store.enqueue(normalized)
        results: list[str] = []
        for handler in self._subscribers:
            result = handler(normalized)
            if result is not None:
                results.append(result)
        results.extend(self._output_dispatcher.dispatch(normalized))
        return results

    def publish_many(self, messages: list[Message]) -> list[str]:
        results: list[str] = []
        for message in messages:
            results.extend(self.publish(message))
        return results

    def flush_output_handlers(self) -> list[str]:
        return self._output_dispatcher.flush()

    def clear(self) -> None:
        self.messages.clear()
        self._output_dispatcher.clear()
        self._sequence_no = 0

    def close(self) -> None:
        self.flush_output_handlers()
        if self._store is not None:
            self._store.close()
