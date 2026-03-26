from agentic_runtime.messaging.messages import Event, UserCommand


def test_user_command_uses_emmett_style_type_data_and_metadata_without_target() -> None:
    command = UserCommand(
        runtime_id="runtime-1",
        turn_id="turn-1",
        message_id="message-1",
        domain="manage_notes",
        source="runtime",
        target="manage_notes",
        name="reset",
        payload={"hard": True},
        sequence_no=7,
    )

    assert command.type == "reset"
    assert command.data == {"hard": True}
    assert command.metadata == {
        "runtime_id": "runtime-1",
        "turn_id": "turn-1",
        "message_id": "message-1",
        "domain": "manage_notes",
        "source": "runtime",
        "scope": "canonical",
        "sequence_no": 7,
    }


def test_event_uses_emmett_style_type_data_and_metadata() -> None:
    event = Event(
        runtime_id="runtime-1",
        turn_id="turn-1",
        message_id="message-1",
        reply_to_message_id="message-0",
        domain="routing",
        source="router",
        target="manage_notes",
        role="assistant",
        scope="transport",
        name="workflow_selected",
        payload={"workflow": "manage_notes"},
        sequence_no=3,
        tool_call_id="call-1",
        agent_run_id="run-1",
        prompt_name="router",
        prompt_hash="hash",
        status="success",
    )

    assert event.type == "workflow_selected"
    assert event.data == {"workflow": "manage_notes"}
    assert event.metadata == {
        "runtime_id": "runtime-1",
        "turn_id": "turn-1",
        "message_id": "message-1",
        "reply_to_message_id": "message-0",
        "domain": "routing",
        "source": "router",
        "target": "manage_notes",
        "role": "assistant",
        "scope": "transport",
        "sequence_no": 3,
        "tool_call_id": "call-1",
        "agent_run_id": "run-1",
        "prompt_name": "router",
        "prompt_hash": "hash",
        "status": "success",
    }
