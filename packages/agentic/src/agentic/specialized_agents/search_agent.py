from __future__ import annotations

import json
from typing import TYPE_CHECKING

from agentic.core_agent import CoreAgentic
from agentic.integrations.search_provider import SearchProvider, normalize_results
from agentic.message import ToolMessage
from agentic.metadata import Description
from providers.models import ModelConfig
from providers.orchestrator import ModelProviderType
from agentic.tools import Toolset, Toolsets

from agentic.specialized_agents._prompt_builders import build_specialized_prompt_builder

if TYPE_CHECKING:
    from knowledge_base.documents import Document
    from knowledge_base.store import QdrantKnowledgeBase


_MAX_TOOL_RESULTS = 5
_MAX_TOOL_SNIPPET_CHARS = 320
_MAX_TOOL_CONTENT_CHARS = 320
_MAX_QUERY_LENGTH = 1000


def _truncate_text(value: str, *, limit: int) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def _compact_web_results(results: list[object], *, max_results: int = _MAX_TOOL_RESULTS) -> tuple[dict[str, str | None], ...]:
    normalized = normalize_results(results[:max_results])
    return tuple(
        {
            "title": _truncate_text(str(item.get("title") or ""), limit=160),
            "url": _truncate_text(str(item.get("url") or ""), limit=240),
            "snippet": _truncate_text(str(item.get("snippet") or ""), limit=_MAX_TOOL_SNIPPET_CHARS),
            "published_at": (
                _truncate_text(str(item.get("published_at") or ""), limit=64) or None
            ),
        }
        for item in normalized
    )


def _compact_knowledge_entries(entries: list[dict[str, str]], *, max_results: int = _MAX_TOOL_RESULTS) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "content": _truncate_text(str(entry.get("content") or ""), limit=_MAX_TOOL_CONTENT_CHARS),
            "url": _truncate_text(str(entry.get("url") or ""), limit=240),
            "keywords": _truncate_text(str(entry.get("keywords") or ""), limit=160),
        }
        for entry in entries[:max_results]
    )

def _build_search_system_prompt(*, has_knowledge_base: bool) -> str:
    strategy_lines = [
        "You are a search agent. Use the available tools to find information.",
        "",
        "Available tools:",
        "{tools}",
        "",
        "Tool calling rules:",
    ]
    if has_knowledge_base:
        strategy_lines.extend(
            [
                '- Always include the required "query" parameter when calling knowledge_search or web_search.',
                '- Example web search call:',
                '<tool_call>{"name":"web_search","parameters":{"query":"latest AI news","count":5}}</tool_call>',
                '- Example knowledge-base call:',
                '<tool_call>{"name":"knowledge_search","parameters":{"query":"latest AI news","k":5}}</tool_call>',
                "",
                "Strategy:",
                "1. First check the knowledge base for previously indexed results with knowledge_search.",
                "2. If the knowledge base lacks relevant results, use web_search to find new information.",
                "3. Synthesize your findings into a clear, sourced answer.",
            ]
        )
    else:
        strategy_lines.extend(
            [
                '- Always include the required "query" parameter when calling web_search.',
                '- Example web search call:',
                '<tool_call>{"name":"web_search","parameters":{"query":"latest AI news","count":5}}</tool_call>',
                "",
                "Strategy:",
                "1. Use web_search to gather relevant results.",
                "2. Synthesize your findings into a clear, sourced answer.",
            ]
        )
    strategy_lines.extend(
        [
            "",
            "Format tool calls as:",
            "<tool_call>",
            '{{"name":"TOOL_NAME","parameters":{{...}}}}',
            "</tool_call>",
            "",
            "When you have enough information, respond with your final answer without calling any tool.",
        ]
    )
    return "\n".join(strategy_lines)


class SearchAgent(CoreAgentic):
    """Searches the web and knowledge base to find and index information."""

    description = Description(
        agent_name="searcher",
        description="Searches the web and knowledge base to find and index information.",
        capabilities=("search", "web", "knowledge-base", "indexing"),
    )

    def __init__(
        self,
        model_id: str,
        search_provider: SearchProvider,
        *,
        knowledge_base: QdrantKnowledgeBase | None = None,
        model_provider_type: ModelProviderType = "mlx",
        config: ModelConfig | None = None,
        max_search_turns: int = 5,
    ) -> None:
        self._search_provider = search_provider
        self._knowledge_base = knowledge_base
        self._max_search_turns = max_search_turns

        def web_search(query: str, count: int = 5) -> str:
            """Search the web for information on a topic.

            Args:
                query: The search query.
                count: Maximum number of results to return.
            """
            query = query[:_MAX_QUERY_LENGTH]
            safe_count = max(1, min(count, _MAX_TOOL_RESULTS))
            results = self._search_provider.search(query, count=safe_count)

            if self._knowledge_base is not None and results:
                self._index_results(results)

            return json.dumps(_compact_web_results(results), ensure_ascii=False, indent=2)

        def knowledge_search(query: str, k: int = 5) -> str:
            """Search the knowledge base for previously indexed information.

            Args:
                query: The search query.
                k: Maximum number of results to return.
            """
            if self._knowledge_base is None:
                return "Knowledge base not configured."

            docs = self._knowledge_base.similarity_search(query, k=k)
            if not docs:
                return "No relevant documents found in knowledge base."

            entries: list[dict[str, str]] = []
            for doc in docs:
                meta = doc.metadata
                entries.append({
                    "content": doc.page_content,
                    "url": meta.get("url", ""),
                    "keywords": meta.get("keywords", ""),
                })
            return json.dumps(_compact_knowledge_entries(entries), ensure_ascii=False, indent=2)

        tools = [web_search]
        if self._knowledge_base is not None:
            tools.append(knowledge_search)

        super().__init__(
            model_id=model_id,
            prompt_builder=build_specialized_prompt_builder(
                system_prompt=_build_search_system_prompt(
                    has_knowledge_base=self._knowledge_base is not None
                ),
                model_provider_type=model_provider_type,
            ),
            toolsets=Toolsets([Toolset(tools)]),
            config=config or ModelConfig(max_tokens=512, generation_mode="nothinking"),
            model_provider_type=model_provider_type,
        )

    def _index_results(self, results: list[object]) -> None:
        """Index search results into the Qdrant knowledge base."""
        if self._knowledge_base is None:
            return

        from knowledge_base.documents import Document

        documents: list[Document] = []
        for result in results:
            title = getattr(result, "title", "")
            url = getattr(result, "url", "")
            snippet = getattr(result, "snippet", "")
            if not snippet:
                continue
            documents.append(
                Document(
                    page_content=snippet,
                    metadata={
                        "url": url,
                        "keywords": title,
                        "content": snippet,
                        "source": "web_search",
                    },
                )
            )

        if documents:
            self._knowledge_base.add(documents)

    def search(self, message: str) -> str:
        """Search for information using web search and knowledge base.

        Runs a multi-turn tool-calling loop to gather information and
        produce a synthesized answer.

        Args:
            message: The search query or question.

        Returns:
            A synthesized answer with sources.
        """
        self._agent.clear_history()
        response = self.respond(message)
        response.result.loop_iteration = 0

        turn = 0
        while response.tool_results and turn < self._max_search_turns:
            turn += 1

            parts: list[str] = []
            for tool_result in response.tool_results:
                tool_name, tool_args = tool_result.tool_call
                parts.append(
                    "\n".join([
                        f"Tool: {tool_name}",
                        f"Arguments: {json.dumps(tool_args, ensure_ascii=False)}",
                        "Output:",
                        tool_result.output,
                    ])
                )
            tool_msg = ToolMessage("\n\n".join(parts))
            response = self.respond(tool_msg)
            response.result.loop_iteration = turn

        return response.output

    def close(self) -> None:
        """Close the search provider and release resources."""
        self._search_provider.close()
