from __future__ import annotations

from typing import Sequence

import pytest

from agentic_runtime.messaging.consumer import ConsumerConfig, MessageConsumer, StepLimitExceeded
from agentic_runtime.messaging.message_stream import InMemoryMessageStream
from agentic_runtime.messaging.messages import AssistantMessage, Event, Message, UserMessage
from agentic_runtime.reactor import Reactor


# --- Fakes ---


class FakeReactor:
    """Reactor that returns a canned AssistantMessage."""

    def __init__(self, response_text: str = "done") -> None:
        self._response_text = response_text
        self.invocations: list[Message] = []

    def can_handle(self, command: Message) -> bool:
        return True

    def invoke(self, command: Message) -> Message:
        self.invocations.append(command)
        return AssistantMessage(
            text=self._response_text,
            domain=command.domain,
            source="fake-reactor",
        )


class FailingReactor:
    """Reactor that fails N times, then succeeds."""

    def __init__(self, failures: int, error_type: type[Exception] = ConnectionError) -> None:
        self._failures = failures
        self._error_type = error_type
        self._call_count = 0

    def can_handle(self, command: Message) -> bool:
        return True

    def invoke(self, command: Message) -> Message:
        self._call_count += 1
        if self._call_count <= self._failures:
            raise self._error_type(f"fail #{self._call_count}")
        return AssistantMessage(text="recovered", source="fake")


# --- Helpers ---


def _user(text: str) -> UserMessage:
    return UserMessage(text=text)


def _event(name: str) -> Event:
    return Event(name=name)


def _noop_decider(msg: Message) -> Sequence[Message]:
    """Decider that produces no commands — terminates immediately."""
    return []


def _echo_decider(msg: Message) -> Sequence[Message]:
    """Decider that echoes UserMessage as an Event command, ignores the rest."""
    if isinstance(msg, UserMessage):
        return [Event(name="process", text=msg.text, domain=msg.domain)]
    return []


def _noop_routing(command: Message) -> Reactor | None:
    return None


# --- Tests ---


class TestMessageConsumerBasicFlow:
    def test_empty_stream_does_nothing(self) -> None:
        stream = InMemoryMessageStream()
        consumer = MessageConsumer()
        consumer.consume(stream, _noop_decider, _noop_routing)

        assert stream.is_empty()
        assert len(stream.all_messages()) == 0

    def test_single_message_no_commands(self) -> None:
        stream = InMemoryMessageStream()
        stream.append(_user("hello"))
        consumer = MessageConsumer()

        consumer.consume(stream, _noop_decider, _noop_routing)

        assert stream.is_empty()
        assert len(stream.all_messages()) == 1  # only the original

    def test_decider_produces_command_no_reactor(self) -> None:
        stream = InMemoryMessageStream()
        stream.append(_user("hello"))
        consumer = MessageConsumer()

        consumer.consume(stream, _echo_decider, _noop_routing)

        messages = stream.all_messages()
        # original UserMessage + Event command (no reactor output since routing returns None)
        # But the Event goes back to stream → decider sees it → returns [] → done
        assert len(messages) == 2
        assert isinstance(messages[0], UserMessage)
        assert isinstance(messages[1], Event)

    def test_full_cycle_decider_routing_reactor(self) -> None:
        reactor = FakeReactor(response_text="result")
        stream = InMemoryMessageStream()
        stream.append(_user("input"))

        def routing(command: Message) -> Reactor | None:
            if isinstance(command, Event):
                return reactor
            return None

        consumer = MessageConsumer()
        consumer.consume(stream, _echo_decider, routing)

        messages = stream.all_messages()
        # 1. UserMessage("input") — initial
        # 2. AssistantMessage("result") — reactor output (Event command NOT in stream)
        # Decider sees AssistantMessage → [] → done
        assert len(messages) == 2
        assert isinstance(messages[0], UserMessage)
        assert isinstance(messages[1], AssistantMessage)
        assert messages[1].text == "result"

    def test_reactor_receives_correct_command(self) -> None:
        reactor = FakeReactor()
        stream = InMemoryMessageStream()
        stream.append(_user("data"))

        def routing(command: Message) -> Reactor | None:
            return reactor if isinstance(command, Event) else None

        consumer = MessageConsumer()
        consumer.consume(stream, _echo_decider, routing)

        assert len(reactor.invocations) == 1
        assert reactor.invocations[0].text == "data"


class TestMessageConsumerStepLimit:
    def test_raises_on_step_limit(self) -> None:
        """Decider that always produces a command creates infinite loop → step limit."""

        def infinite_decider(msg: Message) -> Sequence[Message]:
            return [Event(name="loop")]

        config = ConsumerConfig(max_steps=5)
        consumer = MessageConsumer(config=config)
        stream = InMemoryMessageStream()
        stream.append(_user("start"))

        with pytest.raises(StepLimitExceeded) as exc_info:
            consumer.consume(stream, infinite_decider, _noop_routing)

        assert exc_info.value.max_steps == 5


class TestMessageConsumerRetry:
    def test_retries_on_retryable_error(self) -> None:
        reactor = FailingReactor(failures=1, error_type=ConnectionError)
        stream = InMemoryMessageStream()
        stream.append(_user("test"))

        def routing(command: Message) -> Reactor | None:
            return reactor if isinstance(command, Event) else None

        config = ConsumerConfig(max_retries=2)
        consumer = MessageConsumer(config=config)
        consumer.consume(stream, _echo_decider, routing)

        messages = stream.all_messages()
        assert any(m.text == "recovered" for m in messages)

    def test_raises_non_retryable_error(self) -> None:
        reactor = FailingReactor(failures=1, error_type=ValueError)
        stream = InMemoryMessageStream()
        stream.append(_user("test"))

        def routing(command: Message) -> Reactor | None:
            return reactor if isinstance(command, Event) else None

        consumer = MessageConsumer()

        with pytest.raises(ValueError, match="fail #1"):
            consumer.consume(stream, _echo_decider, routing)

    def test_raises_after_max_retries_exceeded(self) -> None:
        reactor = FailingReactor(failures=5, error_type=ConnectionError)
        stream = InMemoryMessageStream()
        stream.append(_user("test"))

        def routing(command: Message) -> Reactor | None:
            return reactor if isinstance(command, Event) else None

        config = ConsumerConfig(max_retries=2)
        consumer = MessageConsumer(config=config)

        with pytest.raises(ConnectionError):
            consumer.consume(stream, _echo_decider, routing)


class TestMessageConsumerMultiStep:
    def test_multi_step_chain(self) -> None:
        """Decider produces a chain: step1 → step2 → done."""
        step_count = 0

        def chain_decider(msg: Message) -> Sequence[Message]:
            nonlocal step_count
            if isinstance(msg, UserMessage):
                return [Event(name="step1")]
            if isinstance(msg, Event) and msg.name == "step1":
                return [Event(name="step2")]
            if isinstance(msg, Event) and msg.name == "step2":
                return [Event(name="done")]
            return []

        stream = InMemoryMessageStream()
        stream.append(_user("start"))

        consumer = MessageConsumer()
        consumer.consume(stream, chain_decider, _noop_routing)

        messages = stream.all_messages()
        names = [m.name for m in messages if isinstance(m, Event)]
        assert names == ["step1", "step2", "done"]

    def test_decider_fans_out_multiple_commands(self) -> None:
        """Decider produces multiple commands from one message."""
        reactor = FakeReactor(response_text="processed")

        def fan_out_decider(msg: Message) -> Sequence[Message]:
            if isinstance(msg, UserMessage):
                return [
                    Event(name="task_a", text="a"),
                    Event(name="task_b", text="b"),
                ]
            return []

        def routing(command: Message) -> Reactor | None:
            return reactor if isinstance(command, Event) else None

        stream = InMemoryMessageStream()
        stream.append(_user("go"))

        consumer = MessageConsumer()
        consumer.consume(stream, fan_out_decider, routing)

        messages = stream.all_messages()
        # Events routed to reactor → NOT in stream (only reactor outputs are)
        results = [m for m in messages if isinstance(m, AssistantMessage)]
        assert len(results) == 2
        assert reactor.invocations[0].name == "task_a"
        assert reactor.invocations[1].name == "task_b"
