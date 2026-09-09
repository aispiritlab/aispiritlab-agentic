"""What happens to a trace key on an event that overrides `with_metadata`.

An event with its own `__init__` cannot `replace` itself, so it overrides
`with_metadata` — and each such module used to carry its own copy of the fold
that turns `span_id=…` into a `TraceSnapshot`. A copy that omits it raises only
once a tracer is attached, which is why the copies stayed wrong.
"""

from __future__ import annotations

from agentic.specialized_agents.events import TaskCompleted, TaskDelegated
from agentic.workflow.messages import RecordedMessageMetadata, metadata_with_updates


def test_a_trace_key_is_folded_into_the_snapshot_rather_than_set_as_a_field() -> None:
    updated = metadata_with_updates(
        RecordedMessageMetadata(), trace_id="trace-1", span_id="span-1"
    )

    assert updated.trace_id == "trace-1"
    assert updated.span_id == "span-1"


def test_an_ordinary_field_is_replaced_and_the_rest_is_kept() -> None:
    updated = metadata_with_updates(RecordedMessageMetadata(turn_id="t-1"), source="planner")

    assert updated.source == "planner"
    assert updated.turn_id == "t-1"


def test_an_event_with_its_own_init_survives_a_tracer_setting_a_span() -> None:
    for event in (TaskDelegated(target_agent="researcher"), TaskCompleted(result="done")):
        traced = event.with_metadata(span_id="span-1", span_name="delegate")

        assert traced.metadata.span_id == "span-1"
        assert traced.metadata.span_name == "delegate"


def test_a_traced_event_keeps_its_data() -> None:
    traced = TaskDelegated(target_agent="researcher").with_metadata(span_id="span-1")

    assert traced.data["target_agent"] == "researcher"
