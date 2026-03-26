from pathlib import Path

import pytest

from dataloader.deepfabric_proxy import main as deepfabric_proxy_main

flow_php_repo_dataset = pytest.importorskip(
    "dataloader.flow_php_repo_dataset",
    reason="dataloader.flow_php_repo_dataset module not yet implemented",
)

CONFIG_MODES = flow_php_repo_dataset.CONFIG_MODES
CURATED_FLOW_FILES = flow_php_repo_dataset.CURATED_FLOW_FILES
DATASET_PROFILES = flow_php_repo_dataset.DATASET_PROFILES
DATA_DIR = flow_php_repo_dataset.DATA_DIR
build_deepfabric_config = flow_php_repo_dataset.build_deepfabric_config
build_scenario_seed_files = flow_php_repo_dataset.build_scenario_seed_files
build_settings_for_profile = flow_php_repo_dataset.build_settings_for_profile
normalize_row = flow_php_repo_dataset.normalize_row
normalize_row_with_reason = flow_php_repo_dataset.normalize_row_with_reason
validate_assistant_content = flow_php_repo_dataset.validate_assistant_content
validate_agent_rows = flow_php_repo_dataset.validate_agent_rows


def test_build_scenario_seed_files_reads_curated_files_and_synthetic_claude(tmp_path: Path) -> None:
    repo_root = tmp_path / "flow"
    required_files = list(CURATED_FLOW_FILES)

    for relative_path in required_files:
        file_path = repo_root / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(f"contents for {relative_path}\n", encoding="utf-8")

    seed_files = build_scenario_seed_files(repo_root)

    assert "CLAUDE.md" in seed_files
    assert "MAINTAINER_TASKS.md" in seed_files
    assert "composer test" in seed_files["CLAUDE.md"]
    assert "Describe the Proposal" in seed_files["MAINTAINER_TASKS.md"]
    assert seed_files["README.md"] == "contents for README.md\n"
    assert "src/core/etl/src/Flow/ETL/FlowContext.php" in seed_files
    assert "src/adapter/etl-adapter-text/README.md" in seed_files
    assert "web/landing/tests/Flow/Website/Tests/Integration/ExamplesTest.php" in seed_files
    assert "src/core/etl/src/Flow/ETL/Analyze.php" in seed_files
    assert "src/core/etl/src/Flow/ETL/Function/Cast.php" in seed_files
    assert "src/lib/types/src/Flow/Types/Type/Logical/TimeZoneType.php" in seed_files


def test_build_scenario_seed_files_rejects_oversized_files(tmp_path: Path) -> None:
    repo_root = tmp_path / "flow"
    required_files = list(CURATED_FLOW_FILES)

    for relative_path in required_files:
        file_path = repo_root / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        content = "x" * 8

        if relative_path == "composer.json":
            content = "x" * 128

        file_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="Seed file is larger than"):
        build_scenario_seed_files(repo_root, max_file_bytes=64)


def test_build_deepfabric_config_uses_mode_specific_tools_and_model_slot(tmp_path: Path) -> None:
    seed_files = {
        "CLAUDE.md": "guidance",
        "MAINTAINER_TASKS.md": "Describe the Proposal\n# Context",
        "README.md": "repo readme",
    }
    output_dir = tmp_path / "artifacts"

    read_only_config = build_deepfabric_config(
        config_mode=CONFIG_MODES["read_only"],
        repo_root=Path("/tmp/flow"),
        seed_files=seed_files,
        output_dir=output_dir,
    )
    approved_edit_config = build_deepfabric_config(
        config_mode=CONFIG_MODES["approved_edit"],
        repo_root=Path("/tmp/flow"),
        seed_files=seed_files,
        output_dir=output_dir,
    )

    assert 'provider: "ollama"' in read_only_config
    assert 'model: "qwen3.5:4b"' in read_only_config
    assert 'provider: "openrouter"' in read_only_config
    assert 'model: "minimax/minimax-m2.5"' in read_only_config
    assert 'base_url: "https://openrouter.ai/api/v1"' in read_only_config
    assert 'model: "google/gemini-2.5-flash"' in read_only_config
    assert "`read_file` requires a concrete `file_path` argument" in read_only_config
    assert "The final assistant turn must contain the completed answer" in read_only_config
    assert 'package_area=<one of: core, adapter, bridge, lib, cli, repo_root>' in read_only_config
    assert "MAINTAINER_TASKS.md" in read_only_config
    assert "web/landing/tests/Flow/Website/Tests/Integration/ExamplesTest.php" in read_only_config
    assert "src/adapter/etl-adapter-doctrine/README.md" in read_only_config
    assert "src/core/etl/src/Flow/ETL/Analyze.php" in read_only_config
    assert "src/core/etl/src/Flow/ETL/Function/Cast.php" in read_only_config
    assert "src/lib/types/src/Flow/Types/Type/Logical/TimeZoneType.php" in read_only_config
    assert "Describe the Proposal" in read_only_config
    assert "The synthetic `MAINTAINER_TASKS.md` in the VFS is an authoritative style reference" in read_only_config
    assert "some tasks should be written like maintainer-authored proposals" in read_only_config
    assert "- read_file" in read_only_config
    assert "- write_file" not in read_only_config
    assert "    strict: true" in read_only_config
    assert "  depth: 4" in read_only_config
    assert "  degree: 4" in read_only_config
    assert "    max_agent_steps: 8" in read_only_config
    assert "  max_concurrent: 8" in read_only_config
    assert "    max_tokens: 1400" in read_only_config
    assert "  sample_retries: 3" in read_only_config
    assert '  num_samples: "200%"' in read_only_config
    assert "  checkpoint:" in read_only_config
    assert "    interval: 100" in read_only_config
    assert "    retry_failed: true" in read_only_config
    assert f'save_as: "{output_dir / "flow-php-repo-read-only.raw.jsonl"}"' in read_only_config
    assert read_only_config.count('model: "minimax/minimax-m2.5"') == 1
    assert read_only_config.count('base_url: "https://openrouter.ai/api/v1"') == 2
    assert read_only_config.count('provider: "openrouter"') == 2
    assert read_only_config.count('model: "google/gemini-2.5-flash"') == 1
    assert "- write_file" in approved_edit_config
    assert (
        f'save_as: "{output_dir / "flow-php-repo-approved-edit.raw.jsonl"}"'
        in approved_edit_config
    )


def test_data_dir_defaults_to_package_artifact_directory() -> None:
    assert DATA_DIR.name == "flow_php_repo"
    assert DATA_DIR.parent.name == "data"


def test_build_settings_for_profile_uses_profile_defaults_and_explicit_overrides() -> None:
    baseline_settings = build_settings_for_profile("baseline")
    xl_settings = build_settings_for_profile("xl", num_samples="500%", batch_size=8)

    assert baseline_settings.topics_depth == DATASET_PROFILES["baseline"].topics_depth
    assert baseline_settings.topics_provider == "openrouter"
    assert baseline_settings.topics_model == "minimax/minimax-m2.5"
    assert baseline_settings.generation_provider == "openrouter"
    assert baseline_settings.generation_model == "google/gemini-2.5-flash"
    assert baseline_settings.generation_base_url == "https://openrouter.ai/api/v1"
    assert baseline_settings.num_samples == "auto"
    assert xl_settings.topics_degree == DATASET_PROFILES["xl"].topics_degree
    assert xl_settings.num_samples == "500%"
    assert xl_settings.batch_size == 8


def test_build_settings_for_profile_legacy_question_overrides_apply_to_topics_and_generation() -> None:
    settings = build_settings_for_profile(
        "baseline",
        question_provider="openai",
        question_model="MiniMax-M2.5",
        question_base_url="https://api.minimax.io/v1",
    )

    assert settings.topics_provider == "openai"
    assert settings.topics_model == "MiniMax-M2.5"
    assert settings.topics_base_url == "https://api.minimax.io/v1"
    assert settings.generation_provider == "openai"
    assert settings.generation_model == "MiniMax-M2.5"
    assert settings.generation_base_url == "https://api.minimax.io/v1"


def test_normalize_row_strips_header_and_preserves_successful_agent_trace() -> None:
    raw_row = {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {
                "role": "user",
                "content": (
                    "[FLOW_DATASET]\n"
                    "scenario_type=approved_edit\n"
                    "package_area=adapter\n"
                    "approval_mode=approved_edit\n"
                    "source_files=README.md,src/adapter/etl-adapter-csv/src/Flow/ETL/Adapter/CSV/functions.php\n"
                    "[/FLOW_DATASET]\n\n"
                    "Update the adapter test and explain the change."
                ),
            },
            {"role": "assistant", "content": ""},
            {"role": "tool", "content": "read_file: opened requested file"},
            {
                "role": "assistant",
                "content": (
                    "```php\n"
                    "<?php\n\n"
                    "declare(strict_types=1);\n"
                    "```\n"
                ),
            },
        ],
        "tools": [{"name": "read_file"}],
    }

    normalized_row = normalize_row(raw_row, approval_mode="approved_edit", repo_root=Path("/tmp/flow"))

    assert normalized_row is not None
    assert normalized_row.chat_row["metadata"] == {
        "repo_path": "/tmp/flow",
        "scenario_type": "approved_edit",
        "package_area": "adapter",
        "approval_mode": "approved_edit",
        "source_files": [
            "README.md",
            "src/adapter/etl-adapter-csv/src/Flow/ETL/Adapter/CSV/functions.php",
        ],
    }
    assert normalized_row.chat_row["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "Update the adapter test and explain the change."},
        {
            "role": "assistant",
            "content": "```php\n<?php\n\ndeclare(strict_types=1);\n```",
        },
    ]
    assert normalized_row.agent_row is not None
    assert [message["role"] for message in normalized_row.agent_row["messages"]] == [
        "system",
        "user",
        "tool",
        "assistant",
    ]


def test_normalize_row_preserves_tool_calls_tool_call_ids_and_reasoning() -> None:
    raw_row = {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {
                "role": "user",
                "content": (
                    "[FLOW_DATASET]\n"
                    "scenario_type=review\n"
                    "package_area=repo_root\n"
                    "approval_mode=read_only\n"
                    "source_files=README.md\n"
                    "[/FLOW_DATASET]\n\n"
                    "Review the README guidance."
                ),
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{\"file_path\":\"README.md\"}"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "Error (InvalidArguments): Missing required argument: file_path",
                "tool_call_id": "call_1",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{\"file_path\":\"README.md\"}"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "# README\n",
                "tool_call_id": "call_2",
            },
            {
                "role": "assistant",
                "content": "The README is consistent with the seeded repository guidance.",
            },
        ],
        "tools": [{"name": "read_file"}],
        "reasoning": {
            "style": "agent",
            "content": [{"step_number": 1, "thought": "Inspect README", "action": "read_file"}],
        },
    }

    normalized_row = normalize_row(raw_row, approval_mode="read_only", repo_root=Path("/tmp/flow"))

    assert normalized_row is not None
    assert normalized_row.agent_row is not None
    assert normalized_row.agent_row["reasoning"] == raw_row["reasoning"]
    assert normalized_row.agent_row["messages"][2]["tool_calls"][0]["id"] == "call_1"
    assert normalized_row.agent_row["messages"][3]["tool_call_id"] == "call_1"
    assert normalized_row.agent_row["messages"][-1]["content"] == (
        "The README is consistent with the seeded repository guidance."
    )


def test_normalize_row_drops_failed_tool_trace() -> None:
    raw_row = {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "Inspect the repo."},
            {"role": "tool", "content": "Error: Tool 'list_tools' not found in registry"},
            {"role": "assistant", "content": "No answer"},
        ]
    }

    assert normalize_row(raw_row, approval_mode="read_only") is None


def test_normalize_row_handles_inline_dataset_header() -> None:
    raw_row = {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {
                "role": "user",
                "content": (
                    "[FLOW_DATASET] "
                    "scenario_type=approved_edit, package_area=repo_root, approval_mode=approved_edit, "
                    "source_files=composer.json,phpunit.xml.dist,CLAUDE.md "
                    "[/FLOW_DATASET] "
                    "Set up mutation testing for the CSV adapter tests."
                ),
            },
            {"role": "tool", "content": "[\"composer.json\"]"},
            {"role": "assistant", "content": "Review the current mutation setup first."},
        ],
        "tools": [{"name": "list_files"}],
    }

    normalized_row = normalize_row(raw_row, approval_mode="approved_edit", repo_root=Path("/tmp/flow"))

    assert normalized_row is not None
    assert normalized_row.chat_row["metadata"]["scenario_type"] == "approved_edit"
    assert normalized_row.chat_row["metadata"]["package_area"] == "repo_root"
    assert normalized_row.chat_row["messages"][1]["content"] == (
        "Set up mutation testing for the CSV adapter tests."
    )


def test_normalize_row_drops_truncated_dataset_header_without_task_text() -> None:
    raw_row = {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "[FLOW_DATASET]"},
            {"role": "tool", "content": "[\"README.md\"]"},
            {"role": "assistant", "content": "tool_calls"},
        ]
    }

    assert normalize_row(raw_row, approval_mode="read_only") is None


def test_normalize_row_drops_invalid_arguments_tool_error() -> None:
    raw_row = {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "Inspect phpunit config."},
            {"role": "tool", "content": "Error (InvalidArguments): Missing required argument: file_path"},
            {"role": "assistant", "content": "I need to retry with the correct arguments."},
        ]
    }

    assert normalize_row(raw_row, approval_mode="approved_edit") is None


def test_normalize_row_drops_planning_only_final_answer() -> None:
    raw_row = {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "Review the package boundaries."},
            {"role": "tool", "content": "[\"composer.json\"]"},
            {"role": "assistant", "content": "I need to read the composer.json files from both packages."},
        ],
        "tools": [{"name": "list_files"}],
    }

    assert normalize_row(raw_row, approval_mode="read_only") is None


def test_normalize_row_drops_php_snippet_without_strict_types() -> None:
    raw_row = {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "Review the ObjectMother fixture guidance."},
            {
                "role": "assistant",
                "content": (
                    "Use a fixture mother like this:\n\n"
                    "```php\n"
                    "<?php\n\n"
                    "final class ResourceMother\n"
                    "{\n"
                    "}\n"
                    "```"
                ),
            },
        ]
    }

    normalized_row, drop_reason = normalize_row_with_reason(raw_row, approval_mode="read_only")

    assert normalized_row is None
    assert drop_reason == "assistant_php_missing_strict_types"


def test_validate_assistant_content_rejects_planning_only_text() -> None:
    with pytest.raises(ValueError, match="completed answer"):
        validate_assistant_content("Based on the repository exploration, here's what I found:")


def test_validate_assistant_content_rejects_placeholder_text() -> None:
    with pytest.raises(ValueError, match="placeholder"):
        validate_assistant_content("No answer")


def test_validate_agent_rows_rejects_write_file_in_read_only_mode() -> None:
    agent_row = {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "Review the repo."},
            {"role": "assistant", "content": "Calling write_file now."},
            {"role": "tool", "content": "write_file: updated file"},
            {"role": "assistant", "content": "Finished."},
        ],
        "tools": [{"name": "write_file"}],
        "metadata": {
            "repo_path": "/tmp/flow",
            "scenario_type": "review",
            "package_area": "repo_root",
            "approval_mode": "read_only",
            "source_files": ["README.md"],
        },
    }

    with pytest.raises(ValueError, match="Read-only rows cannot mention `write_file`"):
        validate_agent_rows([agent_row], repo_root=Path("/tmp/flow"))


def test_validate_agent_rows_allows_reasoning_and_tool_errors() -> None:
    agent_row = {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "Review the repo."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{\"file_path\":\"README.md\"}"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "Error (InvalidArguments): Missing required argument: file_path",
                "tool_call_id": "call_1",
            },
            {"role": "assistant", "content": "The invalid tool call should be corrected on the next step."},
        ],
        "tools": [{"name": "read_file"}],
        "metadata": {
            "repo_path": "/tmp/flow",
            "scenario_type": "review",
            "package_area": "repo_root",
            "approval_mode": "read_only",
            "source_files": ["README.md"],
        },
        "reasoning": {"style": "agent", "content": [{"step_number": 1}]},
    }

    validate_agent_rows([agent_row], repo_root=Path("/tmp/flow"))


def test_deepfabric_proxy_forwards_arguments_to_uv_tool_run(monkeypatch) -> None:
    recorded: dict[str, object] = {}

    class CompletedProcess:
        returncode = 0

    def fake_which(binary: str) -> str:
        assert binary == "uv"
        return "/usr/local/bin/uv"

    def fake_run(command: list[str], check: bool, env: dict[str, str]) -> CompletedProcess:
        recorded["command"] = command
        recorded["check"] = check
        recorded["env"] = env
        return CompletedProcess()

    monkeypatch.setattr("dataloader.deepfabric_proxy.shutil.which", fake_which)
    monkeypatch.setattr("dataloader.deepfabric_proxy.subprocess.run", fake_run)
    monkeypatch.setattr("sys.argv", ["deepfabric", "generate", "config.yaml"])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")

    with pytest.raises(SystemExit) as exit_info:
        deepfabric_proxy_main()

    assert exit_info.value.code == 0
    assert recorded["check"] is False
    assert recorded["command"] == [
        "/usr/local/bin/uv",
        "tool",
        "run",
        "--from",
        "deepfabric",
        "deepfabric",
        "generate",
        "config.yaml",
    ]
    assert recorded["env"]["OPENAI_API_KEY"] == "router-key"


def test_deepfabric_proxy_uses_minimax_api_key_when_openai_key_is_missing(monkeypatch) -> None:
    recorded: dict[str, object] = {}

    class CompletedProcess:
        returncode = 0

    def fake_which(binary: str) -> str:
        assert binary == "uv"
        return "/usr/local/bin/uv"

    def fake_run(command: list[str], check: bool, env: dict[str, str]) -> CompletedProcess:
        recorded["command"] = command
        recorded["check"] = check
        recorded["env"] = env
        return CompletedProcess()

    monkeypatch.setattr("dataloader.deepfabric_proxy.shutil.which", fake_which)
    monkeypatch.setattr("dataloader.deepfabric_proxy.subprocess.run", fake_run)
    monkeypatch.setattr("sys.argv", ["deepfabric", "generate", "config.yaml"])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("MINIMAX_API_KEY", "minimax-key")

    with pytest.raises(SystemExit) as exit_info:
        deepfabric_proxy_main()

    assert exit_info.value.code == 0
    assert recorded["check"] is False
    assert recorded["env"]["OPENAI_API_KEY"] == "minimax-key"


def test_deepfabric_proxy_prefers_openrouter_key_over_existing_openai_key(monkeypatch) -> None:
    recorded: dict[str, object] = {}

    class CompletedProcess:
        returncode = 0

    def fake_which(binary: str) -> str:
        assert binary == "uv"
        return "/usr/local/bin/uv"

    def fake_run(command: list[str], check: bool, env: dict[str, str]) -> CompletedProcess:
        recorded["command"] = command
        recorded["check"] = check
        recorded["env"] = env
        return CompletedProcess()

    monkeypatch.setattr("dataloader.deepfabric_proxy.shutil.which", fake_which)
    monkeypatch.setattr("dataloader.deepfabric_proxy.subprocess.run", fake_run)
    monkeypatch.setattr("sys.argv", ["deepfabric", "generate", "config.yaml"])
    monkeypatch.setenv("OPENAI_API_KEY", "stale-openai-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")

    with pytest.raises(SystemExit) as exit_info:
        deepfabric_proxy_main()

    assert exit_info.value.code == 0
    assert recorded["check"] is False
    assert recorded["env"]["OPENAI_API_KEY"] == "router-key"
