from __future__ import annotations

from evaluation import Flow, Flows


DEFAULT_NOTE_FLOWS = Flows(
    Flow(
        name="flow::zakupy_add_read_edit_read_list",
        steps=(
            "add_note",
            "read_note",
            "edit_note_explicit",
            "read_note",
            "list_notes",
        ),
    ),
    Flow(
        name="flow::zakupy_no_diacritics_variants",
        steps=(
            "add_note",
            "list_notes_no_diacritics",
            "edit_note_no_diacritics",
            "read_note",
        ),
    ),
    Flow(
        name="flow::super_context_resolution",
        steps=(
            "add_note_short_form",
            "read_note_after_add",
            "read_this_note_from_context",
            "edit_this_note_from_context",
            "read_last_note_from_context",
            "edit_last_note_from_context",
            "read_note_after_add",
        ),
    ),
    Flow(
        name="flow::masterox_quoted_content_roundtrip",
        steps=(
            "add_note_z_nazwa_o_tresci_embedded_quotes",
            "read_note_display_variant_masterox",
            "read_last_note_from_context_diacritics_masterox",
        ),
    ),
    Flow(
        name="flow::family_and_weekend_notes_then_list",
        steps=(
            "add_note_o_tresci_multiword_name",
            "add_note_dodaj_o_tresci_variant",
            "list_notes_typo",
            "read_note_simple",
        ),
    ),
    Flow(
        name="flow::mix_read_missing_then_create_then_read",
        steps=(
            "read_note_simple",
            "add_note_o_tresci_simple_name",
            "read_note_no_diacritics",
            "list_notes",
        ),
    ),
)


__all__ = ["DEFAULT_NOTE_FLOWS", "Flow", "Flows"]
