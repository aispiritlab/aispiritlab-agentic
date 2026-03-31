from __future__ import annotations

import json
from typing import TYPE_CHECKING

from agentic.core_agent import CoreAgentic
from agentic.integrations.search_provider import SearchProvider, normalize_results
from agentic.message import ToolMessage
from agentic.metadata import Description
from agentic.models import ModelConfig
from agentic.prompts import QwenPromptBuilder
from agentic.providers.provider import ModelProviderType
from agentic.tools import Toolset, Toolsets

if TYPE_CHECKING:
    from knowledge_base.documents import Document
    from knowledge_base.store import QdrantKnowledgeBase

_SEARCH_SYSTEM_PROMPT = """You are a search agent. Use the available tools to find information.

Available tools:
{tools}

Strategy:
1. First check the knowledge base for previously indexed results with knowledge_search.
2. If the knowledge base lacks relevant results, use web_search to find new information.
3. Synthesize your findings into a clear, sourced answer.

Format tool calls as:
<tool_call>
{{"name":"TOOL_NAME","parameters":{{...}}}}
</tool_call>

When you have enough information, respond with your final answer without calling any tool."""


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
            results = self._search_provider.search(query, count=count)

            if self._knowledge_base is not None and results:
                self._index_results(results)

            normalized = normalize_results(results)
            return json.dumps(normalized, ensure_ascii=False, indent=2)

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

            entries = []
            for doc in docs:
                meta = doc.metadata
                entries.append({
                    "content": doc.page_content,
                    "url": meta.get("url", ""),
                    "keywords": meta.get("keywords", ""),
                })
            return json.dumps(entries, ensure_ascii=False, indent=2)

        super().__init__(
            model_id=model_id,
            prompt_builder=QwenPromptBuilder(system_prompt=_SEARCH_SYSTEM_PROMPT),
            toolsets=Toolsets([Toolset([web_search, knowledge_search])]),
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

        return response.output

    def close(self) -> None:
        """Close the search provider and release resources."""
        self._search_provider.close()
