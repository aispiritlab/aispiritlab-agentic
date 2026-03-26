from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

from agentic.observability import LLMTracer

from agentic.workflow.execution import ExecutionTurnRecord, WorkflowExecution
from agentic.workflow.message_bus import InMemoryMessageBus
from agentic.workflow.messages import Event, Message, TurnCompleted, TurnStarted, UserMessage
from agentic.workflow.streaming import (
    build_assistant_messages,
    build_prompt_snapshot_message,
    build_tool_messages,
)


type TurnHandler = Callable[[UserMessage], Any]


@dataclass(frozen=True, slots=True)
class TurnPlan:
    incoming: UserMessage
    handler: TurnHandler
    trace_name: str
    lifecycle_domain: str
    lifecycle_target: str | None
    lifecycle_workflow_name: str
    output_agent_name: str
    selected_workflow: str | None = None
    pre_messages: tuple[Message, ...] = ()


def coerce_execution(reply: Any) -> WorkflowExecution:
    if isinstance(reply, WorkflowExecution):
        return reply
    if isinstance(reply, str):
        return WorkflowExecution(text=reply)
    if reply is None:
        return WorkflowExecution(text="")
    if isinstance(reply, tuple) and len(reply) > 0:
        return WorkflowExecution(text=coerce_execution(reply[0]).text)
    return WorkflowExecution(text=str(reply))


def coerce_reply_text(reply: Any) -> str:
    return coerce_execution(reply).text


class TurnExecutor:
    def __init__(
        self,
        *,
        bus: InMemoryMessageBus,
        tracer: LLMTracer,
        max_inline_bytes: int = 4096,
        chunk_bytes: int = 4096,
    ) -> None:
        self.bus = bus
        self._tracer = tracer
        self._max_inline_bytes = max_inline_bytes
        self._chunk_bytes = chunk_bytes

    @staticmethod
    def _bind_to_turn(message: Message, turn_id: str, runtime_id: str) -> Message:
        updates: dict[str, Any] = {}
        if not message.runtime_id:
            updates["runtime_id"] = runtime_id
        if not message.turn_id:
            updates["turn_id"] = turn_id
        if updates:
            return replace(message, **updates)
        return message

    def _publish_turn_started(
        self,
        *,
        runtime_id: str,
        turn_id: str,
        domain: str,
        target: str | None,
        workflow_name: str,
        trace_id: str | None = None,
    ) -> None:
        self.bus.publish(
            TurnStarted(
                runtime_id=runtime_id,
                turn_id=turn_id,
                domain=domain,
                source="runtime",
                target=target,
                payload={"workflow": workflow_name},
                trace_id=trace_id,
            )
        )

    def _publish_turn_completed(
        self,
        *,
        runtime_id: str,
        turn_id: str,
        domain: str,
        target: str | None,
        workflow_name: str,
        status: str,
        final_message_id: str | None = None,
        error: Exception | None = None,
        trace_id: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {"workflow": workflow_name}
        if final_message_id is not None:
            payload["final_message_id"] = final_message_id
        if error is not None:
            payload["error_type"] = type(error).__name__
            payload["error_message"] = str(error)
        self.bus.publish(
            TurnCompleted(
                runtime_id=runtime_id,
                turn_id=turn_id,
                domain=domain,
                source="runtime",
                target=target,
                status=status,
                payload=payload,
                trace_id=trace_id,
            )
        )

    def publish_workflow_execution(
        self,
        *,
        incoming: UserMessage,
        workflow_name: str,
        execution: WorkflowExecution,
    ) -> str | None:
        reply_to = incoming.message_id or None

        turns = execution.recorded_turns
        if not turns and execution.agent_result is not None:
            turns = (
                ExecutionTurnRecord(
                    agent_result=execution.agent_result,
                    tool_results=execution.tool_results,
                ),
            )

        for turn in turns:
            snapshot_message = build_prompt_snapshot_message(
                incoming=incoming,
                agent_name=workflow_name,
                snapshot=turn.agent_result.prompt_snapshot,
                agent_run_id=turn.agent_result.run_id,
            )
            if snapshot_message is not None:
                self.bus.publish(snapshot_message)

            tool_messages, reply_to = build_tool_messages(
                incoming=incoming,
                agent_name=workflow_name,
                agent_result=turn.agent_result,
                tool_results=turn.tool_results,
                reply_to_message_id=reply_to,
            )
            self.bus.publish_many(tool_messages)

        if execution.emitted_events:
            bound_events = [
                self._bind_to_turn(event, incoming.turn_id, incoming.runtime_id)
                for event in execution.emitted_events
            ]
            self.bus.publish_many(bound_events)

        final_agent_result = turns[-1].agent_result if turns else execution.agent_result
        assistant_messages, final_message_id = build_assistant_messages(
            incoming=incoming,
            agent_name=workflow_name,
            text=execution.text,
            reply_to_message_id=reply_to,
            agent_run_id=final_agent_result.run_id if final_agent_result else None,
            max_inline_bytes=self._max_inline_bytes,
            chunk_bytes=self._chunk_bytes,
        )
        self.bus.publish_many(assistant_messages)
        return final_message_id

    def execute(self, plan: TurnPlan) -> str:
        if plan.pre_messages:
            self.bus.publish_many(list(plan.pre_messages))

        with self._tracer.workflow(
            name=plan.trace_name,
            session_id=plan.incoming.runtime_id,
            input=plan.incoming.text,
            metadata={"turn_id": plan.incoming.turn_id},
        ) as span:
            trace_id = self._tracer.current_trace_id
            incoming = replace(plan.incoming, trace_id=trace_id)

            self._publish_turn_started(
                runtime_id=incoming.runtime_id,
                turn_id=incoming.turn_id,
                domain=plan.lifecycle_domain,
                target=plan.lifecycle_target,
                workflow_name=plan.lifecycle_workflow_name,
                trace_id=trace_id,
            )
            if plan.selected_workflow is not None:
                self.bus.publish(
                    Event(
                        runtime_id=incoming.runtime_id,
                        turn_id=incoming.turn_id,
                        domain="routing",
                        source="router",
                        target=plan.selected_workflow,
                        scope="transport",
                        name="workflow_selected",
                        payload={"workflow": plan.selected_workflow},
                        trace_id=trace_id,
                    )
                )
            self.bus.publish(incoming)

            try:
                reply = plan.handler(incoming)
                execution = coerce_execution(reply)
                final_message_id = self.publish_workflow_execution(
                    incoming=incoming,
                    workflow_name=plan.output_agent_name,
                    execution=execution,
                )
            except Exception as error:
                span.update(level="ERROR", output={"error": str(error)})
                self._publish_turn_completed(
                    runtime_id=incoming.runtime_id,
                    turn_id=incoming.turn_id,
                    domain=plan.lifecycle_domain,
                    target=plan.lifecycle_target,
                    workflow_name=plan.lifecycle_workflow_name,
                    status="error",
                    error=error,
                    trace_id=trace_id,
                )
                raise

            span.update(output={"status": "success", "final_message_id": final_message_id})
            self._publish_turn_completed(
                runtime_id=incoming.runtime_id,
                turn_id=incoming.turn_id,
                domain=plan.lifecycle_domain,
                target=plan.lifecycle_target,
                workflow_name=plan.lifecycle_workflow_name,
                status="success",
                final_message_id=final_message_id,
                trace_id=trace_id,
            )
            return execution.text
