from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from agentic.workflow.messages import Message


@dataclass(frozen=True, slots=True)
class EachMessageHandler:
    handle: Callable[[Message], str | None]


@dataclass(frozen=True, slots=True)
class EachBatchHandler:
    handle: Callable[[list[Message]], list[str | None]]
    batch_size: int = 10


type OutputStrategy = EachMessageHandler | EachBatchHandler


@dataclass(frozen=True, slots=True)
class WorkflowOutputHandler:
    can_handle: tuple[type[Message], ...]
    strategy: OutputStrategy
    name: str = ""


def _collect_batch_results(batch_results: list[str | None]) -> list[str]:
    return [result for result in batch_results if result is not None]


class OutputHandlerDispatcher:
    def __init__(self, handlers: Sequence[WorkflowOutputHandler] | None = None) -> None:
        self._handlers: list[WorkflowOutputHandler] = list(handlers or [])
        self._pending_batches: dict[int, list[Message]] = {}

    def register(self, handler: WorkflowOutputHandler) -> None:
        self._handlers.append(handler)

    def dispatch(self, message: Message) -> list[str]:
        """Run matching handlers and buffer batched handlers until they are ready."""
        results: list[str] = []
        for index, handler in enumerate(self._handlers):
            if not isinstance(message, handler.can_handle):
                continue
            match handler.strategy:
                case EachMessageHandler(handle=handle):
                    result = handle(message)
                    if result is not None:
                        results.append(result)
                case EachBatchHandler(handle=handle, batch_size=batch_size):
                    pending_batch = self._pending_batches.setdefault(index, [])
                    pending_batch.append(message)
                    if len(pending_batch) < batch_size:
                        continue
                    results.extend(_collect_batch_results(handle(pending_batch.copy())))
                    pending_batch.clear()
        return results

    def flush(self) -> list[str]:
        """Emit trailing partial batches for all registered batch handlers."""
        results: list[str] = []
        for index, handler in enumerate(self._handlers):
            match handler.strategy:
                case EachBatchHandler(handle=handle):
                    pending_batch = self._pending_batches.get(index)
                    if not pending_batch:
                        continue
                    results.extend(_collect_batch_results(handle(pending_batch.copy())))
                    pending_batch.clear()
        return results

    def clear(self) -> None:
        self._pending_batches.clear()


def workflow_output_handler(
    *,
    can_handle: tuple[type[Message], ...],
    each_message: Callable[[Message], str | None] | None = None,
    each_batch: Callable[[list[Message]], list[str | None]] | None = None,
    batch_size: int = 10,
    name: str = "",
) -> WorkflowOutputHandler:
    """Factory that enforces exactly one strategy."""
    if each_message is not None and each_batch is not None:
        raise ValueError("Specify either each_message or each_batch, not both")
    if each_message is None and each_batch is None:
        raise ValueError("Specify either each_message or each_batch")

    if each_message is not None:
        strategy: OutputStrategy = EachMessageHandler(handle=each_message)
    else:
        assert each_batch is not None
        strategy = EachBatchHandler(handle=each_batch, batch_size=batch_size)

    return WorkflowOutputHandler(
        can_handle=can_handle,
        strategy=strategy,
        name=name,
    )


def dispatch_output_handlers(
    handlers: Sequence[WorkflowOutputHandler],
    messages: Message | Sequence[Message],
) -> list[str]:
    """One-shot helper that dispatches a message sequence and flushes trailing batches."""
    dispatcher = OutputHandlerDispatcher(handlers)
    results: list[str] = []
    messages_to_dispatch = [messages] if isinstance(messages, Message) else list(messages)
    for message in messages_to_dispatch:
        results.extend(dispatcher.dispatch(message))
    results.extend(dispatcher.flush())
    return results
