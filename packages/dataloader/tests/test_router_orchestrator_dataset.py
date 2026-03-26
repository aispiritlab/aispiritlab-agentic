from __future__ import annotations

import json

from dataloader.router_orchestrator_dataset import (
    ROUTES,
    build_history_aware_decision_prompt,
    build_seed_scenarios,
    load_agent_specs,
    write_assets,
    write_configs,
)


def test_prompt_mentions_history_previous_decision_routes_and_future_agents() -> None:
    specs = load_agent_specs(catalog_mode="expanded")
    prompt = build_history_aware_decision_prompt(agent_specs=specs)

    assert "ostatniej decyzji routera" in prompt
    assert "Agent może być nowy lub przyszły" in prompt
    assert "<start_of_turn>user" in prompt
    assert "<start_of_turn>model" in prompt
    assert "Przykłady:" in prompt
    assert "calendar" in prompt
    for route in ROUTES:
        assert route in prompt


def test_build_seed_scenarios_contains_reference_current_and_expanded_patterns() -> None:
    scenarios = build_seed_scenarios(count_per_template=1, catalog_mode="expanded")

    assert scenarios
    names = [scenario.name for scenario in scenarios]
    assert len(names) == len(set(names))
    assert any(name.startswith("reference::") for name in names)
    assert any(name.startswith("synthetic::direct_request::calendar::") for name in names)
    assert any(scenario.expected_agent == "calendar" for scenario in scenarios)
    assert any(
        "<start_of_turn>model\nmanage_notes\n<end_of_turn>"
        in scenario.render_user_input()
        for scenario in scenarios
        if scenario.history
    )


def test_write_assets_creates_prompt_scenarios_jsonl_and_catalog(tmp_path) -> None:
    written = write_assets(tmp_path, count_per_template=1, catalog_mode="expanded")

    assert set(written) == {"prompt", "scenarios", "jsonl", "catalog"}

    prompt_text = written["prompt"].read_text(encoding="utf-8")
    assert "Wybierz 1 agenta" in prompt_text
    assert "calendar" in prompt_text

    catalog_payload = json.loads(written["catalog"].read_text(encoding="utf-8"))
    assert catalog_payload
    assert any(agent["name"] == "calendar" for agent in catalog_payload)

    scenarios_payload = json.loads(written["scenarios"].read_text(encoding="utf-8"))
    assert scenarios_payload
    assert "rendered_input" in scenarios_payload[0]
    assert any(item["expected_agent"] == "calendar" for item in scenarios_payload)

    rows = [
        json.loads(line)
        for line in written["jsonl"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    first = rows[0]
    assert [message["role"] for message in first["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert first["metadata"]["source"] == "evaluation-inspired-synthetic-router"
    assert "allowed_agents" in first["metadata"]
    assert "calendar" in first["metadata"]["allowed_agents"]


def test_write_configs_creates_basic_and_simple_reasoning_yaml(tmp_path) -> None:
    written = write_configs(
        tmp_path,
        count_per_template=1,
        catalog_mode="expanded",
        basic_samples=123,
        reasoning_samples=45,
    )

    assert set(written) == {"basic", "reasoning"}

    basic_config = written["basic"].read_text(encoding="utf-8")
    reasoning_config = written["reasoning"].read_text(encoding="utf-8")

    assert "save_as:" in basic_config
    assert "router_orchestrator_basic.jsonl" in basic_config
    assert "type: basic" in basic_config
    assert "calendar" in basic_config
    assert "num_samples: 123" in basic_config

    assert "router_orchestrator_simple_reasoning.jsonl" in reasoning_config
    assert "reasoning_style: freetext" in reasoning_config
    assert "type: cot" in reasoning_config
    assert "num_samples: 45" in reasoning_config
