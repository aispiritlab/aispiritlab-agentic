import os
from pathlib import Path
import subprocess

from agentic import git_tracer
from agentic.tools import Toolset
import orjson
import structlog

HOME = Path.home()
logger = structlog.get_logger(__name__)
OBSIDIAN_CLI_BIN = os.getenv("OBSIDIAN_CLI_BIN", "obsidian")


def is_personalization_finished() -> bool:
    return (HOME / ".aispiritagent" / "personalization.json").exists()


def update_personalization(
    name: str,
    vault_name: str,
    vault_path: str | None = None,
) -> str:
    """Update personalization settings
    Args:
        name: The username.
        vault_name: The Obsidian vault name used by Obsidian CLI (vault=<name>).
        vault_path: Optional absolute path kept for backward compatibility/local fallback.
    """
    resolved_name = name.strip()
    resolved_vault_name = vault_name.strip()
    if not resolved_name:
        return "Brak imienia użytkownika."
    if not resolved_vault_name:
        return "Brak nazwy vaulta Obsidian."

    verify_ok, verify_error = _verify_vault_name(resolved_vault_name)
    if not verify_ok:
        return verify_error

    logger.info(
        "updating_personalization",
        name=resolved_name,
        vault_name=resolved_vault_name,
        vault_path=vault_path,
    )
    os.makedirs(HOME / ".aispiritagent", exist_ok=True)
    payload: dict[str, str] = {"name": resolved_name, "vault_name": resolved_vault_name}
    if vault_path is not None and vault_path.strip():
        payload["vault_path"] = vault_path.strip()

    with open(HOME / ".aispiritagent" / "personalization.json", "wb") as f:
        f.write(orjson.dumps(payload))

    git_tracer.initial_tracking_project(HOME / ".aispiritagent")

    # RAG indexing is best-effort: it needs a local vault path. Vault-only setup should still succeed.
    try:
        from knowledge_base.loader import load_vault_markdown_dataset
        from knowledge_base.main import initial_rag

        documents = load_vault_markdown_dataset()
    except (FileNotFoundError, ModuleNotFoundError) as error:
        logger.warning("rag_bootstrap_skipped", error=str(error))
    else:
        initial_rag([
            doc for doc in documents if doc.page_content.strip() != ""
        ])

    return "Personalizacja zapisana."


def _verify_vault_name(vault_name: str) -> tuple[bool, str]:
    command = [OBSIDIAN_CLI_BIN, "vault", vault_name]
    try:
        result = subprocess.run(
            command,
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

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if result.returncode != 0:
        details = stderr or stdout or f"kod wyjścia: {result.returncode}"
        return False, f"Nie udało się zweryfikować vaulta '{vault_name}': {details}"

    if stderr:
        logger.warning("obsidian_cli_warning", warning=stderr)
    return True, ""



toolset = Toolset([update_personalization])
