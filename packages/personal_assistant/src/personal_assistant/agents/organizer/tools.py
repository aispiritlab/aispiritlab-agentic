from __future__ import annotations

from pathlib import Path
import re

from agentic import git_tracer
from agentic.tools import Toolset
import orjson
import yaml

from ..manage_notes.tools import (
    _get_vault_config,
    _get_vault_path,
    _missing_vault_configuration_message,
    _normalize_note_name,
    _read_note_using_filesystem,
    _read_note_using_obsidian,
    _run_obsidian_command,
)

_FRONTMATTER_PATTERN = re.compile(
    r"\A---\n(?P<frontmatter>.*?)\n---\n?(?P<body>.*)\Z",
    re.DOTALL,
)


def _normalize_tag(tag: str) -> str:
    return tag.strip().lstrip("#")


def _coerce_tags(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = _normalize_tag(value)
        return [normalized] if normalized else []
    if isinstance(value, list):
        tags: list[str] = []
        for item in value:
            if isinstance(item, str):
                normalized = _normalize_tag(item)
                if normalized:
                    tags.append(normalized)
        return tags
    return []


def _split_frontmatter(content: str) -> tuple[dict[str, object], str]:
    match = _FRONTMATTER_PATTERN.match(content)
    if match is None:
        return {}, content

    parsed = yaml.safe_load(match.group("frontmatter")) or {}
    if not isinstance(parsed, dict):
        parsed = {}
    return parsed, match.group("body")


def _render_note(frontmatter: dict[str, object], body: str) -> str:
    rendered_frontmatter = yaml.safe_dump(
        frontmatter,
        allow_unicode=False,
        default_flow_style=False,
        sort_keys=False,
    )
    return f"---\n{rendered_frontmatter}---\n{body}"


def _read_note_content(note_name: str) -> tuple[bool, str]:
    resolved_note_name = _normalize_note_name(note_name)
    vault = _get_vault_config()
    if vault.vault_name:
        return _read_note_using_obsidian(vault.vault_name, resolved_note_name)
    if vault.vault_path:
        return _read_note_using_filesystem(vault.vault_path, resolved_note_name)
    return False, _missing_vault_configuration_message()


def _overwrite_note_content_using_obsidian(
    vault_name: str,
    note_name: str,
    content: str,
) -> tuple[bool, str]:
    return _run_obsidian_command(
        vault_name,
        "create",
        f"name={note_name}",
        f"content={content}",
        "overwrite",
    )


def _overwrite_note_content_using_filesystem(
    vault_path: Path,
    note_name: str,
    content: str,
) -> tuple[bool, str]:
    if not vault_path.exists():
        return False, f"Vault path {vault_path} does not exist."

    note_path = vault_path / f"{note_name}.md"
    with open(note_path, "w", encoding="utf-8") as file:
        file.write(content)
    return True, ""


def _overwrite_note_content(note_name: str, content: str) -> tuple[bool, str]:
    resolved_note_name = _normalize_note_name(note_name)
    vault = _get_vault_config()
    if vault.vault_name:
        return _overwrite_note_content_using_obsidian(
            vault.vault_name,
            resolved_note_name,
            content,
        )
    if vault.vault_path:
        return _overwrite_note_content_using_filesystem(
            vault.vault_path,
            resolved_note_name,
            content,
        )
    return False, _missing_vault_configuration_message()


def _load_existing_tags_with_obsidian(vault_name: str, note_name: str) -> list[str]:
    success, output = _run_obsidian_command(
        vault_name,
        "tags",
        f"file={note_name}",
        "format=json",
    )
    if not success or not output:
        return []

    try:
        payload = orjson.loads(output)
    except orjson.JSONDecodeError:
        return []

    if isinstance(payload, list):
        return [
            normalized
            for item in payload
            if isinstance(item, str) and (normalized := _normalize_tag(item))
        ]
    return []


@git_tracer.commit(lambda: _get_vault_path())
def tag_note(note_name: str, tag: str) -> str:
    """Add a PARA tag to a note.

    Args:
        note_name: The note name.
        tag: The tag to apply.
    """
    resolved_note_name = _normalize_note_name(note_name)
    resolved_tag = _normalize_tag(tag)

    if not resolved_note_name:
        return "Brak nazwy notatki."
    if not resolved_tag:
        return "Brak tagu do dodania."

    success, content = _read_note_content(resolved_note_name)
    if not success:
        return content

    frontmatter, body = _split_frontmatter(content)
    existing_tags = set(_coerce_tags(frontmatter.get("tags")))

    vault = _get_vault_config()
    if vault.vault_name:
        existing_tags.update(
            _load_existing_tags_with_obsidian(vault.vault_name, resolved_note_name)
        )

    if resolved_tag in existing_tags:
        return f"Notatka {resolved_note_name} ma już tag {resolved_tag}."

    frontmatter["tags"] = sorted(existing_tags | {resolved_tag})
    updated_content = _render_note(frontmatter, body)
    write_success, output = _overwrite_note_content(resolved_note_name, updated_content)
    if not write_success:
        return output
    return f"Notatka {resolved_note_name} otagowana jako {resolved_tag}."


toolset = Toolset([tag_note])
