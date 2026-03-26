from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from agentic.observability import LLMTracer, NoopLLMTracer

from agentic.workflow.message_stream import MessageStream
from agentic.workflow.messages import Message
from agentic.workflow.reactor import Decider, Reactor, TechnicalRoutingFn


def _default_is_retryable(error: Exception) -> bool:
    return isinstance(error, (TimeoutError, ConnectionError, OSError))


@dataclass(frozen=True, slots=True)
class ConsumerConfig:
    max_steps: int = 100
    max_retries: int = 2
    is_retryable: Callable[[Exception], bool] = field(
        default_factory=lambda: _default_is_retryable
    )


class StepLimitExceeded(RuntimeError):
    def __init__(self, max_steps: int) -> None:
        super().__init__(f"Consumer reached max_steps={max_steps}.")
        self.max_steps = max_steps


class MessageConsumer:
    """Consumer: polluje stream, przekazuje do processorow (Decider, Reactor)."""

    def __init__(
        self,
        config: ConsumerConfig | None = None,
        tracer: LLMTracer | None = None,
    ) -> None:
        self._config = config or ConsumerConfig()
        self._tracer = tracer or NoopLLMTracer()

    def consume(
        self,
        stream: MessageStream,
        decider: Decider,
        routing_fn: TechnicalRoutingFn,
    ) -> None:
        """Poll stream az pusty. Wiadomosc -> decider -> routing -> reactor -> output -> stream."""
        step = 0

        while (msg := stream.read_next()) is not None:
            step += 1
            if step > self._config.max_steps:
                raise StepLimitExceeded(max_steps=self._config.max_steps)

            commands = decider(msg)

            for command in commands:
                reactor = routing_fn(command)
                if reactor is None:
                    # No reactor — command goes to stream as-is (e.g., domain events)
                    stream.append(command)
                    continue

                # Reactor handles the command — only reactor output goes to stream
                with self._tracer.step(
                    name=f"reactor.{type(command).__name__}",
                    span_type="TOOL",
                    attributes={"step": step},
                ) as span:
                    output = self._invoke_with_retry(reactor, command)
                    span.update(output={"message_type": type(output).__name__})

                stream.append(output)

    def _invoke_with_retry(self, reactor: Reactor, command: Message) -> Message:
        last_error: Exception | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                return reactor.invoke(command)
            except Exception as error:
                last_error = error
                if (
                    attempt >= self._config.max_retries
                    or not self._config.is_retryable(error)
                ):
                    raise
        assert last_error is not None
        raise last_error
