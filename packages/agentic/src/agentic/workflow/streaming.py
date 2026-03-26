from __future__ import annotations

import hashlib
from typing import Iterable
from uuid import uuid4

from agentic.agent import AgentResult, PromptSnapshot as AgentPromptSnapshot
from agentic.tools import ToolRunResult

from agentic.workflow.messages import (
    AssistantMessage,
    Message,
    MessageChunk,
    MessageCompleted,
    MessageStarted,
    PromptSnapshot,
    ToolCallEvent,
    ToolResultMessage,
    UserMessage,
)


def new_message_id() -> str:
    return str(uuid4())


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utf8_chunks(text: str, chunk_bytes: int) -> list[str]:
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if not text:
        return [""]

    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for character in text:
        encoded = character.encode("utf-8")
        size = len(encoded)
        if current and current_size + size > chunk_bytes:
            chunks.append("".join(current))
            current = []
            current_size = 0
        current.append(character)
        current_size += size
    if current:
        chunks.append("".join(current))
    return chunks


def build_prompt_snapshot_message(
    *,
    incoming: UserMessage,
    agent_name: str,
    snapshot: AgentPromptSnapshot | None,
    agent_run_id: str | None,
) -> PromptSnapshot | None:
    if snapshot is None:
        return None
    return PromptSnapshot(
        runtime_id=incoming.runtime_id,
        turn_id=incoming.turn_id,
        domain=incoming.domain,
        source=agent_name,
        target=incoming.source,
        text=snapshot.text,
        payload={"tool_schema": list(snapshot.tool_schema)},
        prompt_name=snapshot.prompt_name,
        prompt_hash=snapshot.prompt_hash,
        agent_run_id=agent_run_id,
    )


def build_tool_messages(
    *,
    incoming: UserMessage,
    agent_name: str,
    agent_result: AgentResult | None,
    tool_results: Iterable[ToolRunResult],
    reply_to_message_id: str | None,
) -> tuple[list[Message], str | None]:
    if agent_result is None:
        return [], reply_to_message_id

    published: list[Message] = []
    reply_to = reply_to_message_id
    tool_results_list = list(tool_results)
    for index, tool_call in enumerate(agent_result.tool_calls):
        tool_name, parameters = tool_call
        tool_call_id = str(uuid4())
        tool_message_id = new_message_id()
        published.append(
            ToolCallEvent(
                runtime_id=incoming.runtime_id,
                turn_id=incoming.turn_id,
                message_id=tool_message_id,
                reply_to_message_id=reply_to,
                domain=incoming.domain,
                source=agent_name,
                target=incoming.source,
                payload={"name": tool_name, "parameters": parameters},
                tool_call_id=tool_call_id,
                agent_run_id=agent_result.run_id,
            )
        )
        reply_to = tool_message_id

        if index >= len(tool_results_list):
            continue
        tool_result = tool_results_list[index]
        tool_result_message_id = new_message_id()
        published.append(
            ToolResultMessage(
                runtime_id=incoming.runtime_id,
                turn_id=incoming.turn_id,
                message_id=tool_result_message_id,
                reply_to_message_id=reply_to,
                domain=incoming.domain,
                source=tool_name,
                target=agent_name,
                name=tool_name,
                text=tool_result.output,
                payload={
                    "name": tool_name,
                    "parameters": dict(parameters),
                },
                tool_call_id=tool_call_id,
                agent_run_id=agent_result.run_id,
                content_sha256=hash_text(tool_result.output),
            )
        )
        reply_to = tool_result_message_id

    return published, reply_to


def build_assistant_messages(
    *,
    incoming: UserMessage,
    agent_name: str,
    text: str,
    reply_to_message_id: str | None,
    agent_run_id: str | None,
    max_inline_bytes: int,
    chunk_bytes: int,
) -> tuple[list[Message], str | None]:
    if not text:
        return [], None

    message_id = new_message_id()
    encoded_length = len(text.encode("utf-8"))
    content_sha = hash_text(text)
    published: list[Message] = []

    if encoded_length > max_inline_bytes:
        chunks = utf8_chunks(text, chunk_bytes)
        published.append(
            MessageStarted(
                runtime_id=incoming.runtime_id,
                turn_id=incoming.turn_id,
                message_id=message_id,
                reply_to_message_id=reply_to_message_id,
                domain=incoming.domain,
                source=agent_name,
                target=incoming.source,
                role="assistant",
                payload={
                    "logical_kind": "assistant_message",
                    "chunk_count": len(chunks),
                    "total_bytes": encoded_length,
                },
                agent_run_id=agent_run_id,
                content_sha256=content_sha,
            )
        )
        for index, chunk in enumerate(chunks):
            published.append(
                MessageChunk(
                    runtime_id=incoming.runtime_id,
                    turn_id=incoming.turn_id,
                    message_id=message_id,
                    reply_to_message_id=reply_to_message_id,
                    domain=incoming.domain,
                    source=agent_name,
                    target=incoming.source,
                    role="assistant",
                    text=chunk,
                    chunk_index=index,
                    chunk_count=len(chunks),
                    agent_run_id=agent_run_id,
                )
            )
        published.append(
            MessageCompleted(
                runtime_id=incoming.runtime_id,
                turn_id=incoming.turn_id,
                message_id=message_id,
                reply_to_message_id=reply_to_message_id,
                domain=incoming.domain,
                source=agent_name,
                target=incoming.source,
                role="assistant",
                payload={
                    "chunk_count": len(chunks),
                    "total_bytes": encoded_length,
                },
                agent_run_id=agent_run_id,
                content_sha256=content_sha,
            )
        )
    else:
        published.append(
            AssistantMessage(
                runtime_id=incoming.runtime_id,
                turn_id=incoming.turn_id,
                message_id=message_id,
                reply_to_message_id=reply_to_message_id,
                domain=incoming.domain,
                source=agent_name,
                target=incoming.source,
                role="assistant",
                scope="transport",
                text=text,
                agent_run_id=agent_run_id,
                content_sha256=content_sha,
            )
        )

    published.append(
        AssistantMessage(
            runtime_id=incoming.runtime_id,
            turn_id=incoming.turn_id,
            message_id=message_id,
            reply_to_message_id=reply_to_message_id,
            domain=incoming.domain,
            source=agent_name,
            target=incoming.source,
            text=text,
            agent_run_id=agent_run_id,
            content_sha256=content_sha,
        )
    )
    return published, message_id
