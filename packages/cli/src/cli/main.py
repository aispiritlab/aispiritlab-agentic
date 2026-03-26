"""Command-line interface for AI Spirit Agent."""

from collections.abc import Callable
import sys
from pathlib import Path

import click

from personal_assistant import (
    Prompts,
    ai_spirit_agent,
    clear_personalization_history,
    get_initial_greeting,
    get_prompt,
    personalize_agent,
    shutdown_application,
)


PROMPT_CHOICES = {
    "Manage notes": Prompts.MANAGE_NOTES.value,
    "Discovery notes": Prompts.DISCOVERY_NOTES.value,
    "Sage": Prompts.SAGE.value,
    "Greetings": Prompts.GREETING.value,
}
DEFAULT_NOTES_EVALUATION_DEFINITION = "personal_assistant.agents.manage_notes.evaluation:NOTES_EVALUATION"


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """AI Spirit Agent CLI - Interact with the agent from the command line."""
    if ctx.invoked_subcommand is None:
        run_interactive_mode()


@cli.command()
@click.argument("query", nargs=-1, required=True)
def chat(query: tuple[str, ...]) -> None:
    """Ask a single question in chat mode.

    Example:
        ai-spirit-cli chat "What is Flow PHP?"
    """
    run_single_query(query, ai_spirit_agent)


@cli.command()
@click.argument("query", nargs=-1, required=True)
def personalizacja(query: tuple[str, ...]) -> None:
    """Run a single personalization step.

    Example:
        ai-spirit-cli personalizacja "Mam na imię Mateusz, nazwa vaulta to MyVault"
    """
    run_single_query(query, personalize_agent)


@cli.command(hidden=True)
@click.argument("query", nargs=-1, required=True)
def ask(query: tuple[str, ...]) -> None:
    """Backward-compatible alias for `chat`."""
    run_single_query(query, ai_spirit_agent)



@cli.command("runtime-interactive")
def runtime_interactive() -> None:
    """Launch the interactive Textual chat client."""
    launch_textual_chat()


def run_single_query(query: tuple[str, ...], agent_fn: Callable[[str], str]) -> None:
    """Execute one prompt in a selected mode and print response."""
    query_text = " ".join(query)

    if not query_text or query_text.strip() == "":
        click.echo(click.style("Error: Query cannot be empty", fg="red"), err=True)
        sys.exit(1)

    try:
        click.echo(click.style("\nAgent: ", fg="magenta", bold=True))
        response = agent_fn(query_text)
        click.echo(response)
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)


@cli.command("get-prompt")
@click.argument("prompt_name", type=click.Choice(list(PROMPT_CHOICES.keys()), case_sensitive=False))
def get_prompt_command(prompt_name: str) -> None:
    """Print selected prompt template from registry.

    Example:
        ai-spirit-cli get-prompt Note
    """
    resolved_prompt_name = PROMPT_CHOICES[prompt_name.title()]
    try:
        click.echo(get_prompt(resolved_prompt_name))
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)


@cli.group("trenowanie")
def training_group() -> None:
    """Training commands."""


@training_group.command("start")
def training_start_command() -> None:
    """Start training flow."""
    click.echo("Wkrotce")


@cli.group("evaluation")
def evaluation_group() -> None:
    """Evaluation commands."""


@evaluation_group.command("optimize-prompt")
@click.option(
    "--definition",
    required=True,
    help=(
        "Evaluation definition in `module:attribute` format, for example "
        "`agentic_runtime.manage_notes.evaluation:NOTES_EVALUATION`."
    ),
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    default=Path("packages/evaluation/src/evaluation/optimized_prompt.txt"),
    show_default=True,
    help="Ścieżka docelowa zoptymalizowanego promptu.",
)
@click.option(
    "--openrouter-model",
    default="openai/gpt-4o-mini",
    show_default=True,
    help="Model OpenRouter do MIPROv2.",
)
@click.option(
    "--openrouter-api-key",
    default="",
    help="Opcjonalne nadpisanie OpenRouter API key.",
)
@click.option(
    "--num-candidates",
    type=int,
    default=6,
    show_default=True,
    help="Liczba kandydatów w algorytmie MIPROv2.",
)
@click.option(
    "--num-trials",
    type=int,
    default=12,
    show_default=True,
    help="Liczba prób w MIPROv2.",
)
@click.option(
    "--runtime-option",
    multiple=True,
    help="Opcja runtime przekazana do callbacku w formacie KEY=VALUE.",
)
def optimize_prompt_command(
    definition: str,
    output_path: Path,
    openrouter_model: str,
    openrouter_api_key: str,
    num_candidates: int,
    num_trials: int,
    runtime_option: tuple[str, ...],
) -> None:
    """Optimize a prompt using a generic evaluation definition."""
    _run_evaluation_prompt_optimization(
        definition=definition,
        output_path=output_path,
        openrouter_model=openrouter_model,
        openrouter_api_key=openrouter_api_key,
        num_candidates=num_candidates,
        num_trials=num_trials,
        runtime_option=runtime_option,
    )


@cli.command("optimize-notes-prompt")
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    default=Path("packages/evaluation/src/evaluation/optimized_note_prompt.txt"),
    show_default=True,
    help="Ścieżka docelowa zoptymalizowanego promptu.",
)
@click.option(
    "--vault-path",
    type=click.Path(path_type=Path),
    default=Path("packages/evaluation/src/evaluation/tmp_vault"),
    show_default=True,
    help="Vault używany w trakcie optymalizacji.",
)
@click.option(
    "--openrouter-model",
    default="openai/gpt-4o-mini",
    show_default=True,
    help="Model OpenRouter do MIPROv2.",
)
@click.option(
    "--openrouter-api-key",
    default="",
    help="Opcjonalne nadpisanie OpenRouter API key.",
)
@click.option(
    "--num-candidates",
    type=int,
    default=6,
    show_default=True,
    help="Liczba kandydatów w algorytmie MIPROv2.",
)
@click.option(
    "--num-trials",
    type=int,
    default=12,
    show_default=True,
    help="Liczba prób w MIPROv2.",
)
def optimize_notes_prompt_command(
    output_path: Path,
    vault_path: Path,
    openrouter_model: str,
    openrouter_api_key: str,
    num_candidates: int,
    num_trials: int,
) -> None:
    """Backward-compatible note prompt optimization command."""
    _run_evaluation_prompt_optimization(
        definition=DEFAULT_NOTES_EVALUATION_DEFINITION,
        output_path=output_path,
        openrouter_model=openrouter_model,
        openrouter_api_key=openrouter_api_key,
        num_candidates=num_candidates,
        num_trials=num_trials,
        runtime_option=(f"vault_path={vault_path}",),
    )


def _run_evaluation_prompt_optimization(
    *,
    definition: str,
    output_path: Path,
    openrouter_model: str,
    openrouter_api_key: str,
    num_candidates: int,
    num_trials: int,
    runtime_option: tuple[str, ...],
) -> None:
    try:
        from evaluation import AgentPromptOptimization, load_evaluation_definition
    except Exception as error:
        click.echo(click.style("Nie mozna zaladowac modułu ewaluacji.", fg="red"), err=True)
        click.echo(click.style(f"{error}", fg="red"), err=True)
        sys.exit(1)

    try:
        output = AgentPromptOptimization(
            definition=load_evaluation_definition(definition),
            output_path=output_path,
            openrouter_model=openrouter_model,
            openrouter_api_key=openrouter_api_key or None,
            num_candidates=num_candidates,
            num_trials=num_trials,
            runtime_options=_parse_runtime_options(runtime_option),
        ).run()
    except Exception as error:
        click.echo(click.style(f"Error: {error}", fg="red"), err=True)
        sys.exit(1)

    click.echo(f"Zapisano zoptymalizowany prompt: {output}")


def _parse_runtime_options(items: tuple[str, ...]) -> dict[str, str]:
    runtime_options: dict[str, str] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key.strip():
            raise click.BadParameter(
                f"Niepoprawna opcja runtime: {item}. Użyj formatu KEY=VALUE."
            )
        runtime_options[key.strip()] = value
    return runtime_options


@cli.command("clear-history")
def clear_history() -> None:
    """Clear active conversation history and print fresh greeting."""
    try:
        clear_personalization_history()
        click.echo(click.style("History cleared.", fg="green"))
        click.echo(click.style("\nAgent: ", fg="magenta", bold=True), nl=False)
        click.echo(get_initial_greeting())
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)


def _load_textual_chat_runner() -> Callable[[], None]:
    from .tui import run_textual_chat

    return run_textual_chat


def launch_textual_chat() -> None:
    """Import and run the Textual chat client on demand."""
    try:
        run_textual_chat = _load_textual_chat_runner()
    except ModuleNotFoundError as error:
        if error.name in {"rich", "textual"}:
            raise click.ClickException(
                "Interactive chat requires `textual` and `rich`. Run `uv sync` and try again."
            ) from error
        raise

    run_textual_chat()


def run_interactive_mode() -> None:
    """Run the CLI in the Textual interactive chat mode."""
    launch_textual_chat()


def main() -> None:
    """Main entry point for the CLI application."""
    try:
        cli()
    finally:
        shutdown_application()


if __name__ == "__main__":
    main()
