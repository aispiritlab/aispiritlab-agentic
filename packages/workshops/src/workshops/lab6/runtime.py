from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
import json
import re

from agentic.integrations.search_provider import LangSearchProvider, SearchProvider, normalize_results
from agentic.llm_call import LLMCall
from agentic.providers.api import OpenAIProvider
from agentic_runtime.distributed import AgenticServiceDiscovery
from agentic_runtime.distributed.service import DistributedService
from agentic_runtime.messaging.messages import AssistantMessage, Message, TurnCompleted, UserMessage
from agentic_runtime.settings import settings

from .messages import SearchPlanned, SummaryRequested

_PLANNER_PROMPT = (
    "You are the planner agent in a distributed research system.\n"
    "Rewrite the user's request into 1 to 3 focused web-search queries.\n"
    "Return strict JSON only in the format {\"queries\": [\"...\"]}.\n"
    "Do not include commentary, markdown, or code fences."
)
_SUMMARY_PROMPT = (
    "Jesteś agentem podsumowującym w rozproszonym systemie research.\n"
    "Odpowiadaj po polsku, używaj wyłącznie dostarczonych wyników wyszukiwania,\n"
    "zaznaczaj brak pewności gdy źródła są niepełne i na końcu wypisz sekcję 'Źródła'\n"
    "z numerowaną listą użytych adresów URL."
)


def configure_lab6_providers() -> None:
    OpenAIProvider.configure(
        base_url=settings.api_base_url,
        api_key=settings.api_key,
        timeout=settings.api_timeout,
    )


def _truncate_text(text: str, limit: int) -> str:
    stripped = text.strip()
    if limit <= 0:
        return ""
    if len(stripped) <= limit:
        return stripped
    if limit <= 3:
        return stripped[:limit]
    return stripped[: limit - 3].rstrip() + "..."


def _compact_results_for_summary(
    results: Sequence[dict[str, object]],
    *,
    max_results: int,
    snippet_chars: int,
    total_chars: int,
) -> tuple[dict[str, str], ...]:
    compacted: list[dict[str, str]] = []
    remaining_chars = max(total_chars, 0)

    for raw_result in results:
        if len(compacted) >= max_results or remaining_chars <= 0:
            break

        title = _truncate_text(str(raw_result.get("title", "")), 160)
        url = _truncate_text(str(raw_result.get("url", "")), 300)
        published_at = _truncate_text(str(raw_result.get("published_at", "")), 64)

        reserved_chars = len(title) + len(url) + len(published_at) + 64
        available_for_snippet = min(snippet_chars, max(remaining_chars - reserved_chars, 0))
        if available_for_snippet <= 0:
            break

        snippet = _truncate_text(str(raw_result.get("snippet", "")), available_for_snippet)
        compacted_result = {
            "title": title,
            "url": url,
            "snippet": snippet,
            "published_at": published_at,
        }
        compacted.append(compacted_result)
        remaining_chars -= sum(len(value) for value in compacted_result.values())

    return tuple(compacted)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped.strip()


def _parse_queries(text: str, fallback: str) -> tuple[str, ...]:
    stripped = _strip_code_fences(text)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        queries = payload.get("queries")
        if isinstance(queries, list):
            resolved = tuple(
                query.strip()
                for query in (str(item) for item in queries)
                if query.strip()
            )
            if resolved:
                return resolved[:3]

    line_candidates = []
    for line in stripped.splitlines():
        candidate = re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
        if candidate:
            line_candidates.append(candidate)
    if line_candidates:
        return tuple(line_candidates[:3])
    return (fallback.strip(),)


@dataclass(slots=True)
class Lab6Planner:
    llm: LLMCall

    def plan(self, question: str) -> tuple[str, ...]:
        prompt = (
            "Rewrite the following question into focused web-search queries.\n\n"
            f"Question: {question}\n"
        )
        self.llm.reset()
        response = self.llm.call(prompt)
        return _parse_queries(response, question)

    def close(self) -> None:
        self.llm.close()


@dataclass(slots=True)
class Lab6Summary:
    llm: LLMCall

    def summarize(self, question: str, results: Sequence[dict[str, object]]) -> str:
        compacted_results = _compact_results_for_summary(
            results,
            max_results=settings.lab6_summary_max_results,
            snippet_chars=settings.lab6_summary_snippet_chars,
            total_chars=settings.lab6_summary_total_chars,
        )
        if not compacted_results:
            return "Nie znalazłem wystarczających wyników, aby przygotować odpowiedź."

        parts = [f"Pytanie użytkownika: {question}", "", "Wyniki wyszukiwania:"]
        for index, result in enumerate(compacted_results, start=1):
            title = str(result.get("title", "")).strip()
            url = str(result.get("url", "")).strip()
            snippet = str(result.get("snippet", "")).strip()
            published_at = str(result.get("published_at", "")).strip()
            parts.extend(
                [
                    f"[{index}] {title}",
                    f"URL: {url}",
                    f"Opis: {snippet}",
                ]
            )
            if published_at:
                parts.append(f"Data publikacji: {published_at}")
            parts.append("")

        self.llm.reset()
        return self.llm.call("\n".join(parts)).strip()

    def close(self) -> None:
        self.llm.close()


@dataclass(slots=True)
class PlannerHandler:
    planner: Lab6Planner

    def __call__(
        self,
        message: Message,
        discovery: AgenticServiceDiscovery,
    ) -> Sequence[Message]:
        if not isinstance(message, UserMessage):
            return ()

        search_agent = discovery.find("web-search")
        queries = self.planner.plan(message.text)
        return (
            SearchPlanned(
                runtime_id=message.runtime_id,
                turn_id=message.turn_id,
                domain=message.domain or "lab6",
                source="planner",
                target=search_agent.agent_name,
                text=message.text,
                question=message.text,
                queries=queries,
                reply_target=message.source or "chat",
            ),
        )

    def close(self) -> None:
        self.planner.close()


@dataclass(slots=True)
class SearchHandler:
    search_provider: SearchProvider
    results_per_query: int

    def __call__(
        self,
        message: Message,
        discovery: AgenticServiceDiscovery,
    ) -> Sequence[Message]:
        if not isinstance(message, SearchPlanned):
            return ()

        summary_agent = discovery.find("summarize")

        deduped: OrderedDict[str, dict[str, object]] = OrderedDict()
        for query in message.queries:
            results = self.search_provider.search(query, count=self.results_per_query)
            for result in normalize_results(results):
                url = str(result.get("url", "")).strip()
                if not url or url in deduped:
                    continue
                deduped[url] = dict(result)

        normalized = tuple(deduped.values())
        compacted = _compact_results_for_summary(
            normalized,
            max_results=settings.lab6_summary_max_results,
            snippet_chars=settings.lab6_summary_snippet_chars,
            total_chars=settings.lab6_summary_total_chars,
        )
        return (
            SummaryRequested(
                runtime_id=message.runtime_id,
                turn_id=message.turn_id,
                domain=message.domain,
                source="search",
                target=summary_agent.agent_name,
                question=message.question,
                queries=message.queries,
                results=compacted,
                reply_target=message.reply_target,
            ),
        )

    def close(self) -> None:
        self.search_provider.close()


@dataclass(slots=True)
class SummaryHandler:
    summary: Lab6Summary

    def __call__(
        self,
        message: Message,
        discovery: AgenticServiceDiscovery,
    ) -> Sequence[Message]:
        del discovery
        if not isinstance(message, SummaryRequested):
            return ()

        final_text = self.summary.summarize(message.question, message.results)
        return (
            AssistantMessage(
                runtime_id=message.runtime_id,
                turn_id=message.turn_id,
                domain=message.domain,
                source="summary",
                target=message.reply_target,
                text=final_text,
            ),
            TurnCompleted(
                runtime_id=message.runtime_id,
                turn_id=message.turn_id,
                domain=message.domain,
                source="summary",
                target=message.reply_target,
                status="success",
                payload={"workflow": "summary"},
            ),
        )

    def close(self) -> None:
        self.summary.close()


def build_lab6_service(
    agent_name: str,
    *,
    discovery: AgenticServiceDiscovery,
) -> DistributedService:
    resolved_name = agent_name.strip().lower()
    if resolved_name == "planner":
        planner = Lab6Planner(
            llm=LLMCall(
                model_name=settings.orchestration_model_name,
                system_prompt=_PLANNER_PROMPT,
                max_tokens=256,
            )
        )
        handler = PlannerHandler(planner=planner)
        capabilities = ("plan", "query-planning")
        close_hook = handler.close
    elif resolved_name == "search":
        provider = LangSearchProvider(
            api_key=settings.langsearch_api_key or "",
            base_url=settings.langsearch_base_url,
            timeout=settings.langsearch_timeout,
        )
        handler = SearchHandler(
            search_provider=provider,
            results_per_query=settings.lab6_search_results_per_query,
        )
        capabilities = ("web-search", "langsearch")
        close_hook = handler.close
    elif resolved_name == "summary":
        summary = Lab6Summary(
            llm=LLMCall(
                model_name=settings.model_name,
                system_prompt=_SUMMARY_PROMPT,
                max_tokens=768,
            )
        )
        handler = SummaryHandler(summary=summary)
        capabilities = ("summarize", "answer-synthesis")
        close_hook = handler.close
    else:
        raise ValueError(f"Unsupported lab6 agent service: {agent_name}")

    return discovery.create_service(
        name=resolved_name,
        capabilities=capabilities,
        handler=handler,
        role="lab6-worker",
        heartbeat_seconds=settings.agent_heartbeat_seconds,
        close_hook=close_hook,
    )
