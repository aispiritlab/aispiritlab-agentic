from __future__ import annotations

from agentic.integrations.search_provider import SearchResult
from agentic.specialized_agents.search_agent import (
    SearchAgent,
    _compact_knowledge_entries,
    _compact_web_results,
)


class _StubSearchProvider:
    def search(self, query: str, count: int = 5) -> list[object]:
        del query, count
        return []

    def close(self) -> None:
        return None


class _StubKnowledgeBase:
    def similarity_search(self, query: str, k: int = 5) -> list[object]:
        del query, k
        return []

    def add(self, documents: list[object]) -> None:
        del documents


def test_search_agent_without_knowledge_base_exposes_only_web_search() -> None:
    agent = SearchAgent(
        model_id="test-model",
        search_provider=_StubSearchProvider(),
        knowledge_base=None,
    )

    assert agent._agent.toolsets._tool_names == ("web_search",)
    prompt = agent._agent.system_prompt.system_prompt or ""
    assert "knowledge_search" not in prompt
    assert '"query":"latest AI news"' in prompt


def test_search_agent_with_knowledge_base_exposes_both_tools() -> None:
    agent = SearchAgent(
        model_id="test-model",
        search_provider=_StubSearchProvider(),
        knowledge_base=_StubKnowledgeBase(),
    )

    assert agent._agent.toolsets._tool_names == ("web_search", "knowledge_search")
    prompt = agent._agent.system_prompt.system_prompt or ""
    assert "knowledge_search" in prompt
    assert '"k":5' in prompt


def test_compact_web_results_clamps_result_count_and_snippets() -> None:
    results = [
        SearchResult(
            title="Title",
            url="https://example.com",
            snippet="A" * 1000,
        )
        for _ in range(8)
    ]

    compacted = _compact_web_results(results)

    assert len(compacted) == 5
    assert all(len(str(item["snippet"])) <= 320 for item in compacted)


def test_compact_knowledge_entries_clamps_result_count_and_content() -> None:
    entries = [
        {
            "content": "B" * 1000,
            "url": "https://example.com",
            "keywords": "kw" * 100,
        }
        for _ in range(8)
    ]

    compacted = _compact_knowledge_entries(entries)

    assert len(compacted) == 5
    assert all(len(item["content"]) <= 320 for item in compacted)
