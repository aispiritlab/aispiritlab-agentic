"""Events tab — view runtime events for the current preview session."""

from __future__ import annotations

from collections.abc import Sequence

import gradio as gr


def _event_arrow(record: dict[str, object]) -> str:
    source = str(record.get("source", "") or "")
    target = str(record.get("target", "") or "")
    return f"{source} → {target}" if target else source


def _normalize_record(record: object) -> dict[str, object]:
    if isinstance(record, dict):
        return record
    return {}


def _filter_event_records(
    records: Sequence[object],
    agent_filter: str,
    type_filter: str,
) -> tuple[list[dict[str, object]], list[list[str]], str]:
    normalized = [_normalize_record(record) for record in records]
    filtered: list[dict[str, object]] = []
    rows: list[list[str]] = []

    for record in normalized:
        source = str(record.get("source", "") or "")
        target = str(record.get("target", "") or "")
        type_name = str(record.get("type_name", "") or "")
        if agent_filter and agent_filter != "All" and agent_filter not in {source, target}:
            continue
        if type_filter and type_filter != "All" and type_name != type_filter:
            continue
        filtered.append(record)
        rows.append(
            [
                str(record.get("index", "")),
                type_name,
                _event_arrow(record),
                str(record.get("status", "") or record.get("kind", "") or ""),
                str(record.get("text", "") or ""),
            ]
        )

    count_text = f"Showing {len(filtered)} of {len(normalized)} events"
    return filtered, rows, count_text


def _refresh_events(
    records: Sequence[object],
    agent_filter: str,
    type_filter: str,
) -> tuple[object, object, list[list[str]], str, list[dict[str, object]], str]:
    normalized = [_normalize_record(record) for record in records]
    agent_choices = sorted(
        {
            value
            for record in normalized
            for value in (
                str(record.get("source", "") or ""),
                str(record.get("target", "") or ""),
            )
            if value
        }
    )
    type_choices = sorted(
        {
            str(record.get("type_name", "") or "")
            for record in normalized
            if record.get("type_name")
        }
    )
    next_agent_filter = agent_filter if agent_filter in {"All", *agent_choices} else "All"
    next_type_filter = type_filter if type_filter in {"All", *type_choices} else "All"
    filtered, rows, count_text = _filter_event_records(
        normalized,
        next_agent_filter,
        next_type_filter,
    )
    detail = "Select an event from the table above." if filtered else "No events captured yet."
    return (
        gr.Dropdown(choices=["All", *agent_choices], value=next_agent_filter),
        gr.Dropdown(choices=["All", *type_choices], value=next_type_filter),
        rows,
        count_text,
        filtered,
        detail,
    )


def _select_event_detail(
    row_index: int,
    filtered_records: Sequence[object],
) -> str:
    normalized = [_normalize_record(record) for record in filtered_records]
    if row_index < 0 or row_index >= len(normalized):
        return ""
    return str(normalized[row_index].get("detail_markdown", "") or "")


def _select_event(evt: gr.SelectData, filtered_records: Sequence[object]) -> str:
    row_index = evt.index[0] if evt.index else -1
    return _select_event_detail(row_index, filtered_records)


def _clear_events() -> tuple[
    list[dict[str, object]],
    object,
    object,
    list[list[str]],
    str,
    list[dict[str, object]],
    str,
]:
    return (
        [],
        gr.Dropdown(choices=["All"], value="All"),
        gr.Dropdown(choices=["All"], value="All"),
        [],
        "Cleared",
        [],
        "Select an event from the table above.",
    )


def build_events_tab(preview_events_state: gr.State | None = None) -> None:
    """Build the Events tab UI."""
    preview_events_state = preview_events_state or gr.State(value=[])

    with gr.Row():
        agent_filter = gr.Dropdown(
            label="Agent",
            choices=["All"],
            value="All",
            scale=2,
        )
        type_filter = gr.Dropdown(
            label="Type",
            choices=["All"],
            value="All",
            scale=2,
        )
        refresh_btn = gr.Button("Refresh", size="sm", scale=1)
        clear_btn = gr.Button("Clear", size="sm", variant="stop", scale=1)

    event_count = gr.Markdown("No events yet")
    filtered_events_state = gr.State(value=[])

    event_table = gr.Dataframe(
        headers=["#", "Type", "Source → Target", "Status", "Text"],
        column_count=(5, "fixed"),
        interactive=False,
        wrap=True,
    )

    gr.Markdown("### Event Detail")
    event_detail = gr.Markdown("Select an event from the table above.")

    refresh_outputs = [
        agent_filter,
        type_filter,
        event_table,
        event_count,
        filtered_events_state,
        event_detail,
    ]

    refresh_btn.click(
        _refresh_events,
        inputs=[preview_events_state, agent_filter, type_filter],
        outputs=refresh_outputs,
    )
    agent_filter.change(
        _refresh_events,
        inputs=[preview_events_state, agent_filter, type_filter],
        outputs=refresh_outputs,
    )
    type_filter.change(
        _refresh_events,
        inputs=[preview_events_state, agent_filter, type_filter],
        outputs=refresh_outputs,
    )
    preview_events_state.change(
        _refresh_events,
        inputs=[preview_events_state, agent_filter, type_filter],
        outputs=refresh_outputs,
    )

    event_table.select(
        _select_event,
        inputs=[filtered_events_state],
        outputs=[event_detail],
    )

    clear_btn.click(
        _clear_events,
        outputs=[
            preview_events_state,
            agent_filter,
            type_filter,
            event_table,
            event_count,
            filtered_events_state,
            event_detail,
        ],
    )
