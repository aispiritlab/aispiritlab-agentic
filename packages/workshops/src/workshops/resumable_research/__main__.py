from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from agentic.integrations.search_provider import LangSearchProvider
from agentic.llm_call import LLMCall
from agentic.workflow import SQLiteEventStore
from agentic_runtime.settings import settings as runtime_settings
from workshops.lab6.runtime import Lab6Planner, Lab6Summary, configure_lab6_providers
from workshops.settings import settings as workshop_settings

from .agent import ResearchRunResult, ResumableResearchAgent
from .storage import ResearchFileRepository

_PLANNER_PROMPT = (
    "You plan web research. Rewrite the question into 1 to 5 focused search queries. "
    'Return strict JSON only: {"queries": ["..."]}.'
)
_SUMMARY_PROMPT = (
    "Jesteś agentem badawczym. Odpowiadaj po polsku wyłącznie na podstawie "
    "dostarczonych wyników. Oddziel fakty od niepewności i zakończ sekcją Źródła z URL."
)


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--workspace",
        type=Path,
        default=Path("data/resumable-research"),
        help="Directory containing the event database and research files.",
    )

    parser = argparse.ArgumentParser(description="Crash-resumable web research agent")
    commands = parser.add_subparsers(dest="command", required=True)

    start = commands.add_parser("start", parents=[common], help="Start new research")
    start.add_argument("question", nargs="+", help="Question to research")
    start.add_argument("--id", dest="research_id")
    start.add_argument("--max-steps", type=int)

    resume = commands.add_parser("resume", parents=[common], help="Resume existing research")
    resume.add_argument("research_id")
    resume.add_argument("--max-steps", type=int)

    show = commands.add_parser("show", parents=[common], help="Show the durable research file")
    show.add_argument("research_id")
    return parser


def _build_agent(workspace: Path) -> ResumableResearchAgent:
    configure_lab6_providers()
    event_store = SQLiteEventStore(workspace / "research-events.sqlite3")
    files = ResearchFileRepository(workspace / "files")
    planner = Lab6Planner(
        llm=LLMCall(
            model_name=runtime_settings.orchestration_model_name,
            system_prompt=_PLANNER_PROMPT,
            max_tokens=384,
        )
    )
    summarizer = Lab6Summary(
        llm=LLMCall(
            model_name=runtime_settings.model_name,
            system_prompt=_SUMMARY_PROMPT,
            max_tokens=1024,
        )
    )
    search_provider = LangSearchProvider(
        api_key=workshop_settings.langsearch_api_key or "",
        base_url=workshop_settings.langsearch_base_url,
        timeout=workshop_settings.langsearch_timeout,
    )
    return ResumableResearchAgent(
        event_store=event_store,
        files=files,
        planner=planner,
        search_provider=search_provider,
        summarizer=summarizer,
        results_per_query=workshop_settings.lab6_search_results_per_query,
    )


def _print_result(result: ResearchRunResult) -> None:
    print(
        json.dumps(
            {
                "research_id": result.research_id,
                "status": result.status,
                "file": str(result.file_path),
                "stream_version": result.stream_version,
                "file_revision": result.file_revision,
                "completed_queries": result.completed_queries,
                "total_queries": result.total_queries,
                "last_error": result.last_error,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if result.summary:
        print("\n" + result.summary)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    workspace = args.workspace.expanduser().resolve()

    if args.command == "show":
        repository = ResearchFileRepository(workspace / "files")
        document = repository.load(args.research_id)
        print(json.dumps(document.to_dict(), ensure_ascii=False, indent=2))
        return 0

    agent: ResumableResearchAgent | None = None
    try:
        agent = _build_agent(workspace)
        if args.command == "start":
            result = agent.start(
                " ".join(args.question),
                research_id=args.research_id,
                max_steps=args.max_steps,
            )
        else:
            result = agent.resume(args.research_id, max_steps=args.max_steps)
        _print_result(result)
        return 0 if result.last_error is None else 2
    except Exception as error:
        print(f"research agent failed: {error}", file=sys.stderr)
        return 1
    finally:
        if agent is not None:
            agent.close()


if __name__ == "__main__":
    raise SystemExit(main())
