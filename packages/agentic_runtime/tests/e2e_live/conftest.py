from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from agentic.models import ModelProvider
from agentic.observability import NoopLLMTracer
from registry.prompts import (
    DECISION_PROMPT,
    GREETING_PROMPT,
    MANAGE_NOTES_PROMPT,
    ORGANIZER_PROMPT,
    SAGE_PROMPT,
    Prompts,
)


_PROMPT_CONSTANTS: dict[str, str] = {
    Prompts.GREETING: GREETING_PROMPT,
    Prompts.MANAGE_NOTES: MANAGE_NOTES_PROMPT,
    Prompts.ORGANIZER: ORGANIZER_PROMPT,
    Prompts.SAGE: SAGE_PROMPT,
    Prompts.DECISION: DECISION_PROMPT,
}


def _registry_prompt_loader(name: str) -> str:
    return _PROMPT_CONSTANTS[name]


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes"}


def _ensure_model_available(model_name: str) -> None:
    provider = ModelProvider(model_name)
    with provider.session("model") as model:
        if model is not None:
            return
    error = provider.get_load_error("model") or "unknown load error"
    pytest.skip(f"Live end-to-end model '{model_name}' is not available: {error}")


def _write_fake_obsidian_cli(script_path: Path) -> None:
    script_path.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


def _mapping() -> dict[str, str]:
    raw = Path(sys.argv[0]).with_suffix(".json").read_text(encoding="utf-8")
    loaded = json.loads(raw)
    return {str(key): str(value) for key, value in loaded.items()}


def _vault_root(vault_name: str) -> Path:
    mapping = _mapping()
    resolved = mapping.get(vault_name)
    if resolved is None:
        raise SystemExit(f"Vault '{vault_name}' not found.")
    root = Path(resolved)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _parse_params(arguments: list[str]) -> tuple[set[str], dict[str, str]]:
    flags: set[str] = set()
    params: dict[str, str] = {}
    for argument in arguments:
        if "=" not in argument:
            flags.add(argument)
            continue
        key, value = argument.split("=", 1)
        params[key] = value
    return flags, params


def _note_path(root: Path, note_name: str) -> Path:
    return root / f"{note_name}.md"


def _read_frontmatter_tags(content: str) -> list[str]:
    if not content.startswith("---\\n"):
        return []
    _, frontmatter, _ = content.split("---", 2)
    loaded = yaml.safe_load(frontmatter) or {}
    if not isinstance(loaded, dict):
        return []
    tags = loaded.get("tags")
    if isinstance(tags, list):
        return [str(tag).lstrip("#") for tag in tags]
    if isinstance(tags, str):
        return [tags.lstrip("#")]
    return []


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print("Missing command.", file=sys.stderr)
        return 1

    if args[0] == "vault":
        if len(args) < 2:
            print("Missing vault name.", file=sys.stderr)
            return 1
        _vault_root(args[1])
        print(args[1])
        return 0

    if not args[0].startswith("vault="):
        print("Missing vault selector.", file=sys.stderr)
        return 1

    vault_name = args[0].split("=", 1)[1]
    if len(args) < 2:
        print("Missing vault command.", file=sys.stderr)
        return 1

    root = _vault_root(vault_name)
    command = args[1]
    flags, params = _parse_params(args[2:])

    if command == "read":
        note_name = params.get("file", "")
        path = _note_path(root, note_name)
        if not path.exists():
            print(f"Notatka {note_name} nie istnieje.", file=sys.stderr)
            return 1
        sys.stdout.write(path.read_text(encoding="utf-8"))
        return 0

    if command == "append":
        note_name = params.get("file", "")
        path = _note_path(root, note_name)
        if not path.exists():
            print(f"Notatka {note_name} nie istnieje.", file=sys.stderr)
            return 1
        with path.open("a", encoding="utf-8") as handle:
            handle.write(params.get("content", ""))
        return 0

    if command == "create":
        note_name = params.get("name", "")
        path = _note_path(root, note_name)
        if path.exists() and "overwrite" not in flags:
            print(f"Notatka {note_name} already exists.", file=sys.stderr)
            return 1
        path.write_text(params.get("content", ""), encoding="utf-8")
        return 0

    if command == "files":
        for note_path in sorted(root.glob("*.md")):
            print(note_path)
        return 0

    if command == "tags":
        note_name = params.get("file", "")
        path = _note_path(root, note_name)
        if not path.exists():
            print("[]")
            return 0
        content = path.read_text(encoding="utf-8")
        print(json.dumps(_read_frontmatter_tags(content)))
        return 0

    print(f"Unsupported command: {command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )
    script_path.chmod(0o755)


@dataclass
class LiveRuntimeHarness:
    runtime: object
    vault_name: str
    vault_path: Path
    personalization_file: Path

    def seed_personalization(self, name: str = "E2E Tester") -> None:
        self.personalization_file.parent.mkdir(parents=True, exist_ok=True)
        self.personalization_file.write_text(
            json.dumps({"name": name, "vault_name": self.vault_name}),
            encoding="utf-8",
        )

    def note_path(self, note_name: str) -> Path:
        return self.vault_path / f"{note_name}.md"

    def run_turn(self, text: str):
        before_count = len(self.runtime.bus.messages)
        reply = self.runtime.handle_message(text)
        turn_messages = self.runtime.bus.messages[before_count:]
        turn_ids = {message.turn_id for message in turn_messages if message.turn_id}
        assert len(turn_ids) == 1
        return reply, turn_messages


@pytest.fixture(autouse=True, scope="session")
def agent_e2e_live_enabled():
    if _truthy_env("RUN_AGENT_E2E_LIVE"):
        return
    pytest.skip(
        "Set RUN_AGENT_E2E_LIVE=1 to run runtime-first live end-to-end tests.",
    )


@pytest.fixture(autouse=True, scope="session")
def prompt_loader():
    import agentic.prompts as prompts_module
    import registry.prompts as registry_prompts_module

    originals = (
        getattr(registry_prompts_module, "get_prompt"),
        getattr(prompts_module, "get_prompt"),
    )

    registry_prompts_module.get_prompt = _registry_prompt_loader
    prompts_module.get_prompt = _registry_prompt_loader

    yield

    registry_prompts_module.get_prompt = originals[0]
    prompts_module.get_prompt = originals[1]


@pytest.fixture(autouse=True, scope="session")
def agent_e2e_live_model_ready(agent_e2e_live_enabled):
    from agentic_runtime.settings import settings

    for model_name in {
        settings.model_name,
        settings.orchestration_model_name,
        settings.thinkink_model,
    }:
        _ensure_model_available(model_name)


@pytest.fixture()
def live_runtime(tmp_path, monkeypatch):
    import agentic_runtime.output_handlers as output_handlers_module
    import agentic_runtime.personalize.tools as personalize_tools
    import agentic_runtime.router.router_agent as router_agent_module
    import agentic_runtime.runtime as runtime_module
    from agentic_runtime.manage_notes import tools as note_tools
    from agentic_runtime.runtime import AgenticRuntime
    from agentic_runtime.settings import settings

    vault_name = "TestVault"
    vault_path = tmp_path / "vaults" / vault_name
    vault_path.mkdir(parents=True)

    cli_path = tmp_path / "fake_obsidian.py"
    _write_fake_obsidian_cli(cli_path)
    cli_path.with_suffix(".json").write_text(
        json.dumps({vault_name: str(vault_path)}),
        encoding="utf-8",
    )

    temp_home = tmp_path / "home"
    personalization_file = temp_home / ".aispiritagent" / "personalization.json"
    personalization_file.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(runtime_module, "init_tracing", lambda: "agent-e2e-live")
    monkeypatch.setattr(
        runtime_module,
        "create_tracer",
        lambda enabled=True: NoopLLMTracer(),
    )
    monkeypatch.setattr(runtime_module, "resync_notes", lambda: None)
    monkeypatch.setattr(
        router_agent_module,
        "create_tracer",
        lambda enabled=True: NoopLLMTracer(),
    )
    monkeypatch.setattr(output_handlers_module, "update_note_in_kb", lambda _path: None)
    monkeypatch.setattr(output_handlers_module, "delete_note_from_kb", lambda _path: None)

    monkeypatch.setattr(personalize_tools, "HOME", temp_home)
    monkeypatch.setattr(personalize_tools, "OBSIDIAN_CLI_BIN", str(cli_path))
    monkeypatch.setattr(personalize_tools, "load_vault_markdown_dataset", lambda: [])
    monkeypatch.setattr(personalize_tools, "initial_rag", lambda _documents: None)
    monkeypatch.setattr(
        personalize_tools.git_tracer,
        "initial_tracking_project",
        lambda _path: None,
    )
    monkeypatch.setattr(note_tools, "PERSONALIZATION_FILE", personalization_file)
    monkeypatch.setattr(note_tools, "OBSIDIAN_CLI_BIN", str(cli_path))

    sqlite_path = tmp_path / "runtime.sqlite3"
    monkeypatch.setattr(settings, "message_store_path", str(sqlite_path))
    monkeypatch.setattr(settings, "message_stream_inline_bytes", 65536)
    monkeypatch.setattr(settings, "message_stream_chunk_bytes", 65536)

    runtime = AgenticRuntime()
    harness = LiveRuntimeHarness(
        runtime=runtime,
        vault_name=vault_name,
        vault_path=vault_path,
        personalization_file=personalization_file,
    )
    try:
        yield harness
    finally:
        runtime.stop()
