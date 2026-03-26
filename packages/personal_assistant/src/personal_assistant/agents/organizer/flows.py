from __future__ import annotations

from evaluation import Flow, Flows


DEFAULT_ORGANIZER_FLOWS = Flows(
    Flow(
        name="flow::organizer_classify_all_para_buckets",
        steps=(
            "classify_project_note",
            "classify_area_note",
            "classify_resource_note",
            "classify_archive_note",
        ),
    ),
    Flow(
        name="flow::organizer_fallback_then_project",
        steps=(
            "classify_ambiguous_note",
            "classify_project_note",
        ),
    ),
)


__all__ = ["DEFAULT_ORGANIZER_FLOWS", "Flow", "Flows"]
