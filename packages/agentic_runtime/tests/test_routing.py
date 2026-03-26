from __future__ import annotations

from agentic_runtime.messaging.messages import AssistantMessage, Event, Message, UserMessage
from agentic_runtime.reactor import LLMResponse
from agentic_runtime.routing import make_llm_routing


class FakeReactor:
    def can_handle(self, command: Message) -> bool:
        return True

    def invoke(self, command: Message) -> Message:
        return AssistantMessage(text="ok")


class TestMakeLLMRouting:
    def test_user_message_routes_to_reactor(self) -> None:
        reactor = FakeReactor()
        routing = make_llm_routing(reactor)
        result = routing(UserMessage(text="hello"))
        assert result is reactor

    def test_assistant_message_returns_none(self) -> None:
        routing = make_llm_routing(FakeReactor())
        assert routing(AssistantMessage(text="hi")) is None

    def test_event_returns_none(self) -> None:
        routing = make_llm_routing(FakeReactor())
        assert routing(Event(name="something")) is None

    def test_llm_response_returns_none(self) -> None:
        routing = make_llm_routing(FakeReactor())
        assert routing(LLMResponse(text="done")) is None

    def test_base_message_returns_none(self) -> None:
        routing = make_llm_routing(FakeReactor())
        assert routing(Message()) is None
