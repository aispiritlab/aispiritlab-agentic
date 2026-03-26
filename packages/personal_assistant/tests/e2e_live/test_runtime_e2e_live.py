from __future__ import annotations

import json

import pytest
from agentic_runtime.messaging.messages import (
    AssistantMessage,
    CreatedNote,
    Event,
    NoteUpdated,
    ToolCallEvent,
    ToolResultMessage,
    TurnCompleted,
    TurnStarted,
)

pytestmark = pytest.mark.agent_e2e_live


def _workflow_selected(turn_messages) -> str:
    event = next(
        message
        for message in turn_messages
        if isinstance(message, Event) and message.name == "workflow_selected"
    )
    return str(event.payload["workflow"])


def _turn_started(turn_messages) -> TurnStarted:
    return next(message for message in turn_messages if isinstance(message, TurnStarted))


def _turn_completed(turn_messages) -> TurnCompleted:
    return next(message for message in turn_messages if isinstance(message, TurnCompleted))


def _tool_calls(turn_messages) -> list[ToolCallEvent]:
    return [message for message in turn_messages if isinstance(message, ToolCallEvent)]


def _tool_results(turn_messages) -> list[ToolResultMessage]:
    return [message for message in turn_messages if isinstance(message, ToolResultMessage)]


def _final_assistant_text(turn_messages) -> str:
    canonical_messages = [
        message.text
        for message in turn_messages
        if isinstance(message, AssistantMessage) and message.scope == "canonical"
    ]
    assert canonical_messages
    return canonical_messages[-1]


def _contains_event(turn_messages, event_type) -> bool:
    return any(isinstance(message, event_type) for message in turn_messages)


def _complete_personalization(live_runtime) -> list[tuple[str, str]]:
    transcript: list[tuple[str, str]] = []
    pending_message = "Mam na imię E2E Tester."

    for _ in range(6):
        reply, _turn_messages = live_runtime.run_turn(pending_message)
        transcript.append((pending_message, reply))
        if live_runtime.personalization_file.exists():
            return transcript

        normalized = reply.lower()
        if "jak mam się do ciebie zwracać" in normalized or "jak mam sie do ciebie zwracac" in normalized:
            pending_message = "Mam na imię E2E Tester."
            continue
        if "podaj nazwę vaulta" in normalized or "podaj nazwe vaulta" in normalized:
            pending_message = f"Mój vault w Obsidianie nazywa się {live_runtime.vault_name}."
            continue
        if "czy potwierdzasz" in normalized or "tak/nie" in normalized:
            pending_message = "Tak."
            continue
        if "co poprawić" in normalized or "co poprawic" in normalized:
            pending_message = "nazwę vaulta"
            continue

        pending_message = f"Mój vault w Obsidianie nazywa się {live_runtime.vault_name}."

    raise AssertionError(f"Personalization did not finish. Transcript: {transcript}")


@pytest.mark.parametrize(
    ("message", "expected_route"),
    (
        ("Mam na imię Ewa.", "personalize"),
        ("Dodaj notatkę o nazwie Zakupy z treścią Mleko i chleb", "manage_notes"),
        ("Wyświetl listę moich notatek", "manage_notes"),
        ("Nie wiem czy kupić dom czy mieszkanie, pomóż podjąć decyzję.", "sage"),
    ),
)
def test_router_exact_match_live(live_runtime, message, expected_route):
    route = live_runtime.runtime.router.route(
        message,
        live_runtime.runtime._available_workflows_summary(),
    )

    assert route == expected_route


def test_personalize_then_manage_notes_via_runtime(live_runtime):
    transcript = _complete_personalization(live_runtime)

    saved = json.loads(live_runtime.personalization_file.read_text(encoding="utf-8"))
    assert saved["name"] == "E2E Tester"
    assert saved["vault_name"] == live_runtime.vault_name

    add_reply, add_turn = live_runtime.run_turn(
        "Dodaj notatkę o nazwie Zakupy z treścią Mleko i chleb"
    )

    assert transcript
    assert _workflow_selected(add_turn) == "manage_notes"
    assert _turn_started(add_turn).domain == "manage_notes"
    assert _turn_completed(add_turn).status == "success"
    assert "Zakupy" in add_reply
    assert live_runtime.note_path("Zakupy").exists()


def test_manage_notes_roundtrip_live(live_runtime):
    live_runtime.seed_personalization()

    add_reply, add_turn = live_runtime.run_turn(
        "Dodaj notatkę o nazwie Zakupy z treścią Mleko i chleb"
    )
    read_reply, read_turn = live_runtime.run_turn("Odczytaj notatkę Zakupy")
    edit_reply, edit_turn = live_runtime.run_turn("Edytuj notatkę Zakupy, dopisz Jajka")
    list_reply, list_turn = live_runtime.run_turn("Wyświetl listę moich notatek")

    assert _workflow_selected(add_turn) == "manage_notes"
    assert _workflow_selected(read_turn) == "manage_notes"
    assert _workflow_selected(edit_turn) == "manage_notes"
    assert _workflow_selected(list_turn) == "manage_notes"

    add_tool_call = _tool_calls(add_turn)
    read_tool_call = _tool_calls(read_turn)
    edit_tool_call = _tool_calls(edit_turn)
    list_tool_call = _tool_calls(list_turn)

    assert len(add_tool_call) == 1
    assert add_tool_call[0].payload == {
        "name": "add_note",
        "parameters": {"note_name": "Zakupy", "note": "Mleko i chleb"},
    }
    assert len(read_tool_call) == 1
    assert read_tool_call[0].payload == {
        "name": "get_note",
        "parameters": {"note_name": "Zakupy"},
    }
    assert len(edit_tool_call) == 1
    assert edit_tool_call[0].payload == {
        "name": "edit_note",
        "parameters": {"note_name": "Zakupy", "note": "Jajka"},
    }
    assert len(list_tool_call) == 1
    assert list_tool_call[0].payload == {
        "name": "list_notes",
        "parameters": {},
    }

    assert _contains_event(add_turn, CreatedNote)
    assert _contains_event(add_turn, NoteUpdated)
    assert _contains_event(edit_turn, NoteUpdated)
    assert _tool_results(add_turn)
    assert _tool_results(read_turn)
    assert _tool_results(edit_turn)
    assert _tool_results(list_turn)

    note_content = live_runtime.note_path("Zakupy").read_text(encoding="utf-8")

    assert "Zakupy" in add_reply
    assert "Mleko i chleb" in read_reply
    assert "Jajka" in edit_reply
    assert "Zakupy" in list_reply
    assert "Mleko i chleb" in note_content
    assert "Jajka" in note_content


def test_manage_notes_triggers_organizer_live(live_runtime):
    live_runtime.seed_personalization()

    reply, turn_messages = live_runtime.run_turn(
        "Dodaj notatkę o nazwie Projekt Atlas z treścią Plan sprintu, milestone i lista zadań do wdrożenia MVP w tym miesiącu"
    )

    note_content = live_runtime.note_path("Projekt Atlas").read_text(encoding="utf-8")

    assert _workflow_selected(turn_messages) == "manage_notes"
    assert _contains_event(turn_messages, CreatedNote)
    assert "Projekt Atlas" in reply
    assert "tags:" in note_content
    assert "para/project" in note_content
    assert "Plan sprintu" in note_content


def test_router_to_sage_live(live_runtime):
    live_runtime.seed_personalization()

    reply, turn_messages = live_runtime.run_turn(
        "Nie wiem czy kupić dom czy mieszkanie, pomóż podjąć decyzję."
    )

    assert _workflow_selected(turn_messages) == "sage"
    assert _turn_started(turn_messages).domain == "sage"
    assert _turn_completed(turn_messages).status == "success"
    assert _tool_calls(turn_messages) == []
    assert _final_assistant_text(turn_messages) == reply
    assert reply.strip()
    assert "krok" in reply.lower()
