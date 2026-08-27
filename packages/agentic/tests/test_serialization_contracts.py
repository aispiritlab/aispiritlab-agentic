from __future__ import annotations

from dataclasses import dataclass, field
import json

from agentic.workflow.messages import Event, RecordedMessageMetadata, normalize_recorded_message
from agentic.workflow.serialization import (
    deserialize_record,
    register_payload_upcaster,
    register_record_contract,
    serialize_record,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class RenamedOrderEvent(Event):
    kind: str = "event"
    type: str = "order.renamed"
    data: dict[str, object] = field(default_factory=dict)


def test_normalized_envelope_has_identity_lineage_contract_and_time() -> None:
    message = normalize_recorded_message(
        Event(
            type="order.created",
            metadata=RecordedMessageMetadata(
                runtime_id="run-1",
                reply_to_message_id="command-1",
            ),
        ),
        stream_name="order:o-1",
        stream_position=0,
        global_position=4,
        recorded_at_ns=123,
    )

    assert message.metadata.message_id
    assert message.metadata.event_id
    assert message.metadata.correlation_id == "run-1"
    assert message.metadata.causation_id == "command-1"
    assert message.metadata.idempotency_key == message.metadata.message_id
    assert message.metadata.contract_name == "order.created"
    assert message.metadata.occurred_at_ns > 0
    assert message.metadata.recorded_at_ns == 123
    assert message.metadata.stream_name == "order:o-1"
    assert message.metadata.stream_position == 0
    assert message.metadata.global_position == 4


def test_stable_contract_survives_python_type_name_and_upcasts_payload() -> None:
    contract = "tests.order-renamed.v1"
    register_record_contract(
        RenamedOrderEvent,
        contract,
        aliases=("legacy.module:OrderRenamed",),
    )
    serialized = serialize_record(
        RenamedOrderEvent(
            data={"customer": "Ada"},
            metadata=RecordedMessageMetadata(
                message_id="message-1",
                event_id="event-1",
                contract_name=contract,
                schema_version=1,
            ),
        )
    )
    raw = json.loads(serialized)
    raw["__type__"] = "moved.module:OrderRenamed"
    register_payload_upcaster(
        contract,
        1,
        2,
        lambda payload: {
            **payload,
            "data": {
                "customer_name": payload["data"]["customer"],
            },
        },
    )

    restored = deserialize_record(json.dumps(raw))

    assert isinstance(restored, RenamedOrderEvent)
    assert restored.data == {"customer_name": "Ada"}
    assert restored.metadata.schema_version == 2
