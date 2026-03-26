from __future__ import annotations

import json

from personal_assistant.agents.manage_notes.evaluation import NOTES_EVALUATION, NOTES_TOOL_SCENARIOS
from evaluation import build_conversation_examples, build_conversation_scenarios


def test_notes_tool_scenarios_have_unique_names_and_user_messages() -> None:
    names = [scenario.name for scenario in NOTES_TOOL_SCENARIOS]
    user_messages = [scenario.user_message_pl for scenario in NOTES_TOOL_SCENARIOS]

    assert len(names) == len(set(names))
    assert len(user_messages) == len(set(user_messages))


def test_conversation_scenarios_have_unique_names_and_known_steps() -> None:
    conversations = build_conversation_scenarios(NOTES_EVALUATION)
    scenario_names = {scenario.name for scenario in NOTES_TOOL_SCENARIOS}
    conversation_names = [conversation.name for conversation in conversations]

    assert conversations
    assert len(conversation_names) == len(set(conversation_names))
    assert all(conversation.steps for conversation in conversations)
    assert all(
        step.source_scenario_name in scenario_names
        for conversation in conversations
        for step in conversation.steps
    )


def test_prefill_expansion_for_edit_last_note_context_case() -> None:
    conversations = build_conversation_scenarios(NOTES_EVALUATION)
    target = next(
        conversation
        for conversation in conversations
        if conversation.name == "single::edit_last_note_from_context"
    )

    assert [step.source_scenario_name for step in target.steps] == [
        "add_note_short_form",
        "read_note_after_add",
        "edit_last_note_from_context",
    ]


def test_conversation_examples_are_json_serializable_with_minimal_schema() -> None:
    examples = build_conversation_examples(NOTES_EVALUATION)

    payload = json.dumps(examples, ensure_ascii=False)
    assert payload
    assert examples

    first = examples[0]
    assert set(first.keys()) >= {"name", "steps", "messages"}
    assert isinstance(first["steps"], list)
    assert isinstance(first["messages"], list)
    assert first["messages"][0]["role"] == "assistant"
