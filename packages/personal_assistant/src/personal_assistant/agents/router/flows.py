from __future__ import annotations

from evaluation import Flow, Flows


DEFAULT_ROUTER_FLOWS = Flows(
    Flow(
        name="flow::router_personalize_then_manage_notes",
        steps=(
            "route_personalize_name",
            "route_personalize_vault",
            "route_manage_notes_add",
        ),
    ),
    Flow(
        name="flow::router_discovery_then_manage_notes",
        steps=(
            "route_discovery_search",
            "route_manage_notes_read",
            "route_manage_notes_list",
        ),
    ),
    Flow(
        name="flow::router_sage_decision",
        steps=("route_sage_decision",),
    ),
    Flow(
        name="flow::router_sage_decision_variants",
        steps=(
            "route_sage_house_purchase",
            "route_sage_house_vs_flat",
            "route_sage_compare_options",
        ),
    ),
)


__all__ = ["DEFAULT_ROUTER_FLOWS", "Flow", "Flows"]
