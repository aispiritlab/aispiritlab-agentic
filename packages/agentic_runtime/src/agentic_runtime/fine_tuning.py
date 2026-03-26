from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sqlite3
from typing import Any

import orjson


def _connect(path: str | Path) -> sqlite3.Connection:
    return sqlite3.connect(Path(path).expanduser())


def _rows_to_messages(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for row in rows:
        role = row["role"]
        if role == "assistant" and row["kind"] == "tool_call":
            payload = orjson.loads(row["payload_json"]) if row["payload_json"] else {}
            tool_name = payload.get("name") if isinstance(payload, dict) else None
            parameters = payload.get("parameters") if isinstance(payload, dict) else None
            tool_call = None
            if isinstance(tool_name, str):
                tool_call = {
                    "id": row["tool_call_id"] or "",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": orjson.dumps(parameters or {}).decode("utf-8"),
                    },
                }
            messages.append(
                {
                    "role": "assistant",
                    "content": row["text"] or "",
                    "tool_calls": [tool_call] if tool_call is not None else [],
                }
            )
            continue

        entry: dict[str, Any] = {
            "role": role,
            "content": row["text"] or "",
        }
        if role == "tool" and row["tool_call_id"]:
            entry["tool_call_id"] = row["tool_call_id"]
        if role == "tool" and row["name"]:
            entry["name"] = row["name"]
        messages.append(entry)
    return messages


def _successful_turns(connection: sqlite3.Connection, runtime_id: str | None) -> list[sqlite3.Row]:
    query = """
        SELECT turn_id, runtime_id, domain, created_at_ns
        FROM message_stream
        WHERE kind = 'turn_completed'
          AND status = 'success'
    """
    parameters: list[Any] = []
    if runtime_id is not None:
        query += " AND runtime_id = ?"
        parameters.append(runtime_id)
    query += """
        AND turn_id NOT IN (
            SELECT DISTINCT ms.turn_id
            FROM message_stream ms
            LEFT JOIN conversation_records cr ON cr.message_id = ms.message_id
            WHERE ms.role = 'assistant'
              AND ms.scope = 'transport'
              AND cr.message_id IS NULL
        )
        AND turn_id NOT IN (
            SELECT DISTINCT turn_id
            FROM conversation_records
            WHERE role = 'tool'
              AND (
                    text LIKE 'Błąd:%'
                 OR text LIKE 'Error:%'
              )
        )
        ORDER BY created_at_ns
    """
    return connection.execute(query, parameters).fetchall()


def _turn_domain(turn_rows: list[sqlite3.Row], default_domain: str) -> str:
    return next(
        (
            row["domain"]
            for row in turn_rows
            if row["role"] != "system" and row["domain"] != "routing"
        ),
        default_domain,
    )


def _turn_rows_for_domain(turn_rows: list[sqlite3.Row], domain: str) -> tuple[list[sqlite3.Row], list[sqlite3.Row]]:
    system_rows = [
        row for row in turn_rows if row["role"] == "system" and row["domain"] == domain
    ]
    body_rows = [
        row for row in turn_rows if row["role"] != "system" and row["domain"] == domain
    ]
    return system_rows, body_rows


def export_agent_fine_tuning_rows(
    path: str | Path,
    *,
    runtime_id: str | None = None,
    max_history_messages: int = 20,
) -> list[dict[str, Any]]:
    with _connect(path) as connection:
        connection.row_factory = sqlite3.Row
        turn_rows = _successful_turns(connection, runtime_id)
        conversation_rows = connection.execute(
            """
            SELECT *
            FROM conversation_records
            {where_clause}
            ORDER BY created_at_ns, COALESCE(sequence_no, 0), message_id
            """.format(
                where_clause="WHERE runtime_id = ?" if runtime_id is not None else ""
            ),
            [runtime_id] if runtime_id is not None else [],
        ).fetchall()

    rows_by_turn: dict[str, list[sqlite3.Row]] = defaultdict(list)
    history_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in conversation_rows:
        rows_by_turn[row["turn_id"]].append(row)

    exported: list[dict[str, Any]] = []
    for turn in turn_rows:
        turn_id = turn["turn_id"]
        current_rows = rows_by_turn.get(turn_id, [])
        if not current_rows:
            continue

        domain = _turn_domain(current_rows, turn["domain"])
        system_rows, body_rows = _turn_rows_for_domain(current_rows, domain)
        if not any(row["role"] == "user" for row in body_rows):
            continue

        prompt_row = system_rows[0] if system_rows else None
        tools = []
        if prompt_row is not None and prompt_row["payload_json"]:
            payload = orjson.loads(prompt_row["payload_json"])
            if isinstance(payload, dict):
                tools = list(payload.get("tool_schema", []))

        prior_context = history_by_domain[domain][-max_history_messages:]
        messages: list[dict[str, Any]] = []
        if prompt_row is not None:
            messages.append({"role": "system", "content": prompt_row["text"] or ""})
        messages.extend(prior_context)
        messages.extend(_rows_to_messages(body_rows))

        exported.append(
            {
                "messages": messages,
                "tools": tools,
                "metadata": {
                    "runtime_id": turn["runtime_id"],
                    "turn_id": turn_id,
                    "domain": domain,
                    "prompt_name": prompt_row["prompt_name"] if prompt_row is not None else None,
                    "prompt_hash": prompt_row["prompt_hash"] if prompt_row is not None else None,
                },
            }
        )

        history_by_domain[domain].extend(_rows_to_messages(body_rows))

    return exported


def export_router_fine_tuning_rows(
    path: str | Path,
    *,
    runtime_id: str | None = None,
) -> list[dict[str, Any]]:
    with _connect(path) as connection:
        connection.row_factory = sqlite3.Row
        routing_events = connection.execute(
            """
            SELECT turn_id, runtime_id, payload_json
            FROM message_stream
            WHERE kind = 'event'
              AND name = 'workflow_selected'
              {runtime_clause}
            ORDER BY created_at_ns
            """.format(
                runtime_clause="AND runtime_id = ?" if runtime_id is not None else ""
            ),
            [runtime_id] if runtime_id is not None else [],
        ).fetchall()
        conversation_rows = connection.execute(
            """
            SELECT *
            FROM conversation_records
            {where_clause}
            ORDER BY created_at_ns, COALESCE(sequence_no, 0), message_id
            """.format(
                where_clause="WHERE runtime_id = ?" if runtime_id is not None else ""
            ),
            [runtime_id] if runtime_id is not None else [],
        ).fetchall()

    rows_by_turn: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in conversation_rows:
        rows_by_turn[row["turn_id"]].append(row)

    exported: list[dict[str, Any]] = []
    for event in routing_events:
        turn_rows = rows_by_turn.get(event["turn_id"], [])
        router_prompt = next(
            (row for row in turn_rows if row["domain"] == "routing" and row["role"] == "system"),
            None,
        )
        user_row = next((row for row in turn_rows if row["role"] == "user"), None)
        if user_row is None:
            continue

        payload = orjson.loads(event["payload_json"]) if event["payload_json"] else {}
        workflow = payload.get("workflow")
        exported.append(
            {
                "messages": [
                    {"role": "system", "content": router_prompt["text"] if router_prompt is not None else ""},
                    {"role": "user", "content": user_row["text"] or ""},
                    {"role": "assistant", "content": str(workflow or "")},
                ],
                "metadata": {
                    "runtime_id": event["runtime_id"],
                    "turn_id": event["turn_id"],
                    "domain": "routing",
                    "expected_workflow": workflow,
                    "prompt_name": router_prompt["prompt_name"] if router_prompt is not None else None,
                    "prompt_hash": router_prompt["prompt_hash"] if router_prompt is not None else None,
                },
            }
        )
    return exported


def write_jsonl(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(row))
            handle.write(b"\n")
    return destination
