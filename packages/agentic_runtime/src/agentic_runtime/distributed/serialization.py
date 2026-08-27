"""Shared wire serialization for workflow and distributed records.

The distributed runtime intentionally delegates to ``agentic.workflow`` so a
message has one contract registry and one upcasting path in every transport.
"""

from agentic.workflow.serialization import (
    deserialize_record,
    register_payload_upcaster,
    register_record_contract,
    register_record_types,
    serialize_record,
)
from agentic_runtime.distributed.contracts import AgentHeartbeat, AgentRegistration

register_record_types(AgentHeartbeat, AgentRegistration)

__all__ = [
    "deserialize_record",
    "register_payload_upcaster",
    "register_record_contract",
    "register_record_types",
    "serialize_record",
]
