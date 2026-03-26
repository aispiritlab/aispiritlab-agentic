"""Workflow runner — bridges Consumer + Decider + Reactor to WorkflowExecution.

Runs a workflow through the MessageConsumer and builds a WorkflowExecution
from the resulting stream, preserving backward compatibility with the existing
runtime bus publishing.
"""
from __future__ import annotations

from agentic.workflow.consumer import ConsumerConfig, MessageConsumer
from agentic.workflow.execution import ExecutionTurnRecord, WorkflowExecution
from agentic.workflow.message_stream import InMemoryMessageStream
from agentic.workflow.messages import Message, UserMessage
from agentic.workflow.reactor import Decider, LLMResponse, TechnicalRoutingFn


def run_workflow(
    message: UserMessage,
    decider: Decider,
    routing_fn: TechnicalRoutingFn,
    *,
    config: ConsumerConfig | None = None,
) -> WorkflowExecution:
    """Run a workflow through the consumer and build WorkflowExecution from the stream.

    1. Creates an InMemoryMessageStream
    2. Appends the initial message
    3. Runs the consumer (decider → routing → reactor cycle)
    4. Builds WorkflowExecution from stream messages
    """
    stream = InMemoryMessageStream()
    consumer = MessageConsumer(config=config)

    stream.append(message)
    consumer.consume(stream, decider, routing_fn)

    return _build_execution(stream.all_messages())


def _build_execution(messages: list[Message] | tuple[Message, ...]) -> WorkflowExecution:
    """Build WorkflowExecution from stream messages."""
    llm_responses: list[LLMResponse] = []
    domain_events: list[Message] = []

    for msg in messages:
        if isinstance(msg, LLMResponse):
            llm_responses.append(msg)
        elif isinstance(msg, UserMessage):
            continue  # skip input messages
        else:
            domain_events.append(msg)

    if not llm_responses:
        return WorkflowExecution(text="")

    last_response = llm_responses[-1]

    recorded_turns = tuple(
        ExecutionTurnRecord(
            agent_result=resp._agent_result,
            tool_results=resp._tool_results,
        )
        for resp in llm_responses
        if resp._agent_result is not None
    )

    return WorkflowExecution(
        text=last_response.text or "",
        agent_result=last_response._agent_result,
        tool_results=last_response._tool_results,
        emitted_events=tuple(domain_events),
        recorded_turns=recorded_turns,
    )
