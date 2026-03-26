from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess

from structlog import get_logger

from agentic import git_tracer
from agentic.tools import Tool, Toolset
import orjson

from .commands import AddNoteCommand, EditNoteCommand, GetNoteCommand, ListNotesCommand

HOME = Path.home()
PERSONALIZATION_FILE = HOME / ".aispiritagent" / "personalization.json"
logger = get_logger(__name__)
_CHUNK_SUFFIX_PATTERN = re.compile(r"\s+chunk=\d+\s*$", re.IGNORECASE)
_NOTE_MISSING_HINTS = (
    "does not exist",
    "not found",
    "cannot find",
    "can't find",
    "nie istnieje",
    "nie znaleziono",
    "brak pliku",
)
_NOTE_EXISTS_HINTS = (
    "already exists",
    "istnieje",
)
OBSIDIAN_CLI_BIN = os.getenv("OBSIDIAN_CLI_BIN", "obsidian")


@dataclass(frozen=True)
class VaultConfig:
    vault_name: str | None
    vault_path: Path | None


def _normalize_note_name(note_name: str) -> str:
    """Normalize note identifiers coming from RAG search results."""
    normalized = note_name.strip()
    normalized = _CHUNK_SUFFIX_PATTERN.sub("", normalized)
    if normalized.lower().endswith(".md"):
        normalized = normalized[:-3]
    return normalized


def _load_personalization_data() -> dict[str, object]:
    if not PERSONALIZATION_FILE.exists():
        return {}

    with open(PERSONALIZATION_FILE, "rb") as file:
        payload = orjson.loads(file.read())
    if not isinstance(payload, dict):
        return {}
    return payload


def _resolve_vault_path(raw_path: str) -> Path:
    resolved = Path(raw_path).expanduser()
    if not resolved.is_absolute():
        resolved = (PERSONALIZATION_FILE.parent / resolved).resolve()
    return resolved


def _get_vault_config() -> VaultConfig:
    data = _load_personalization_data()
    vault_name = data.get("vault_name")
    vault_path = data.get("vault_path")

    resolved_vault_name = vault_name.strip() if isinstance(vault_name, str) else ""
    resolved_vault_path = vault_path.strip() if isinstance(vault_path, str) else ""

    return VaultConfig(
        vault_name=resolved_vault_name or None,
        vault_path=_resolve_vault_path(resolved_vault_path) if resolved_vault_path else None,
    )


def _get_vault_path() -> Path | None:
    # Keep git-tracer compatibility for local-vault mode.
    return _get_vault_config().vault_path


def _missing_vault_configuration_message() -> str:
    return (
        "Brak skonfigurowanego vaulta. "
        "Ustaw przynajmniej nazwę vaulta Obsidian (vault_name)."
    )


def _looks_like_missing_note_error(error_message: str) -> bool:
    normalized = error_message.lower()
    return any(marker in normalized for marker in _NOTE_MISSING_HINTS)


def _looks_like_existing_note_error(error_message: str) -> bool:
    normalized = error_message.lower()
    return any(marker in normalized for marker in _NOTE_EXISTS_HINTS)


def _run_obsidian_command(vault_name: str, command: str, *arguments: str) -> tuple[bool, str]:
    cli_args = [OBSIDIAN_CLI_BIN, f"vault={vault_name}", command, *arguments]
    try:
        process = subprocess.run(
            cli_args,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return (
            False,
            "Nie znaleziono polecenia 'obsidian'. "
            "Zainstaluj Obsidian CLI albo ustaw zmienną OBSIDIAN_CLI_BIN.",
        )

    stdout = process.stdout.strip()
    stderr = process.stderr.strip()
    if process.returncode != 0:
        if stderr:
            return False, stderr
        if stdout:
            return False, stdout
        return False, f"Polecenie Obsidian CLI zakończyło się kodem {process.returncode}."
    return True, stdout


def _append_or_create_note_using_obsidian(
    vault_name: str, note_name: str, note: str
) -> tuple[bool | None, str | None]:
    read_success, read_output = _run_obsidian_command(
        vault_name,
        "read",
        f"file={note_name}",
    )
    if read_success:
        prefix = ""
        if read_output and not read_output.endswith("\n") and not note.startswith("\n"):
            prefix = "\n"

        append_success, append_output = _run_obsidian_command(
            vault_name,
            "append",
            f"file={note_name}",
            f"content={prefix}{note}",
        )
        if not append_success:
            return None, append_output
        return True, None

    if not _looks_like_missing_note_error(read_output):
        return None, read_output

    create_success, create_output = _run_obsidian_command(
        vault_name,
        "create",
        f"name={note_name}",
        f"content={note}",
    )
    if create_success:
        return False, None

    if not _looks_like_existing_note_error(create_output):
        return None, create_output

    append_payload = note if note.startswith("\n") else f"\n{note}"
    append_success, append_output = _run_obsidian_command(
        vault_name,
        "append",
        f"file={note_name}",
        f"content={append_payload}",
    )
    if not append_success:
        return None, append_output
    return True, None


def _append_or_create_note_using_filesystem(
    vault_path: Path, note_name: str, note: str
) -> tuple[bool | None, str | None]:
    if not vault_path.exists():
        return None, f"Vault path {vault_path} does not exist."

    note_path = vault_path / f"{note_name}.md"
    existed = note_path.exists()
    if existed:
        prefix = ""
        if note_path.stat().st_size > 0 and not note.startswith("\n"):
            prefix = "\n"
        with open(note_path, "a", encoding="utf-8") as file:
            file.write(prefix)
            file.write(note)
    else:
        with open(note_path, "w", encoding="utf-8") as file:
            file.write(note)

    return existed, None


def _read_note_using_obsidian(vault_name: str, note_name: str) -> tuple[bool, str]:
    return _run_obsidian_command(vault_name, "read", f"file={note_name}")


def _read_note_using_filesystem(vault_path: Path, note_name: str) -> tuple[bool, str]:
    if not vault_path.exists():
        return False, f"Vault path {vault_path} does not exist."

    note_path = vault_path / f"{note_name}.md"
    if not note_path.exists():
        return False, f"Notatka {note_name} nie istnieje."

    with open(note_path, "r", encoding="utf-8") as file:
        content = file.read()

    return True, content


def _extract_note_name_from_path(text: str) -> str | None:
    candidate = text.strip().strip("'\"")
    if not candidate:
        return None

    if candidate.lower().endswith(".md"):
        return Path(candidate).stem

    md_match = re.search(r"([^\n]*?\.md)", candidate, flags=re.IGNORECASE)
    if md_match is None:
        return None

    resolved = md_match.group(1).strip().strip("'\"")
    if not resolved:
        return None
    return Path(resolved).stem


def _list_notes_using_obsidian(vault_name: str) -> tuple[bool, str]:
    success, output = _run_obsidian_command(vault_name, "files", "ext=md")
    if not success:
        return False, output

    notes: set[str] = set()
    for line in output.splitlines():
        note_name = _extract_note_name_from_path(line)
        if note_name:
            notes.add(note_name)

    if not notes:
        return True, "Brak notatek."
    return True, "Notatki:\n" + "\n".join(f"- {name}" for name in sorted(notes))


def _list_notes_using_filesystem(vault_path: Path) -> tuple[bool, str]:
    if not vault_path.exists():
        return False, f"Vault path {vault_path} does not exist."

    notes = sorted(note.stem for note in vault_path.glob("*.md") if note.is_file())
    if not notes:
        return True, "Brak notatek."
    return True, "Notatki:\n" + "\n".join(f"- {name}" for name in notes)


def _append_or_create_note(note_name: str, note: str) -> tuple[bool | None, str | None]:
    """Append to an existing note or create it when missing.

    Returns:
        Tuple of (existed_before_write, error_message).
    """
    resolved_note_name = _normalize_note_name(note_name)
    vault = _get_vault_config()
    if vault.vault_name:
        return _append_or_create_note_using_obsidian(vault.vault_name, resolved_note_name, note)
    if vault.vault_path:
        return _append_or_create_note_using_filesystem(vault.vault_path, resolved_note_name, note)
    return None, _missing_vault_configuration_message()


@git_tracer.commit(lambda: _get_vault_path())
def add_note(note_name: str, note: str) -> str:
    """Add a note to the Obsidian vault.
    Args:
        note_name: The name of the note.
        note: The content of the note.
    """
    existed, error = _append_or_create_note(note_name, note)
    if error is not None:
        return error
    resolved_note_name = _normalize_note_name(note_name)
    if existed:
        return f"Notatka {resolved_note_name} zaktualizowana."
    return f"Notatka {resolved_note_name} dodana."


@git_tracer.commit(lambda: _get_vault_path())
def edit_note(note_name: str, note: str) -> str:
    """Edit a note by appending content or creating it when missing.
    Args:
        note_name: The name of the note.
        note: The content fragment to append.
    """
    existed, error = _append_or_create_note(note_name, note)
    if error is not None:
        return error
    resolved_note_name = _normalize_note_name(note_name)
    if existed:
        return f"Notatka {resolved_note_name} zaktualizowana."
    return f"Notatka {resolved_note_name} dodana."


def get_note(note_name: str) -> str:
    """Read a note from the Obsidian vault.
    Args:
        note_name: The name of the note.
    """
    resolved_note_name = _normalize_note_name(note_name)
    vault = _get_vault_config()
    if vault.vault_name:
        success, output = _read_note_using_obsidian(vault.vault_name, resolved_note_name)
        if not success:
            if _looks_like_missing_note_error(output):
                return f"Notatka {resolved_note_name} nie istnieje."
            return output
        content = output
    elif vault.vault_path:
        success, output = _read_note_using_filesystem(vault.vault_path, resolved_note_name)
        if not success:
            return output
        content = output
    else:
        return _missing_vault_configuration_message()

    return f"Notatka {resolved_note_name}:\n{content}"


def list_notes() -> str:
    """List all notes from the Obsidian vault."""
    logger.info("Listing notes from Obsidian vault")
    vault = _get_vault_config()
    if vault.vault_name:
        success, output = _list_notes_using_obsidian(vault.vault_name)
        if not success:
            return output
        return output
    if vault.vault_path:
        success, output = _list_notes_using_filesystem(vault.vault_path)
        if not success:
            return output
        return output
    return _missing_vault_configuration_message()


toolset = Toolset([
    Tool(add_note, command=AddNoteCommand),
    Tool(edit_note, command=EditNoteCommand),
    Tool(get_note, command=GetNoteCommand),
    Tool(list_notes, command=ListNotesCommand),
])
