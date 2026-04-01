from __future__ import annotations

from agentic.integrations.search_provider import SearchResult
from agentic.prompts import ChatPromptBuilder, QwenPromptBuilder
from agentic.specialized_agents.planner_agent import PlannerAgent
from agentic.specialized_agents.router_agent import RouterAgent
from agentic.specialized_agents.search_agent import SearchAgent
from agentic.specialized_agents.summarization_agent import SummarizationAgent


class _StubSearchProvider:
    def search(self, query: str, *, count: int = 5) -> list[SearchResult]:
        del query, count
        return []

    def close(self) -> None:
        return None


def test_router_agent_uses_chat_prompts_for_openai() -> None:
    agent = RouterAgent(model_id="test-model", model_provider_type="openai")
    assert isinstance(agent._agent.system_prompt, ChatPromptBuilder)
    agent.close()


def test_router_agent_keeps_qwen_prompts_for_native_backends() -> None:
    agent = RouterAgent(model_id="test-model", model_provider_type="mlx")
    assert isinstance(agent._agent.system_prompt, QwenPromptBuilder)
    agent.close()


def test_planner_agent_uses_chat_prompts_for_openai() -> None:
    agent = PlannerAgent(
        model_id="test-model",
        agent_names=["searcher"],
        model_provider_type="openai",
    )
    assert isinstance(agent._agent.system_prompt, ChatPromptBuilder)
    agent.close()


def test_search_agent_uses_chat_prompts_for_openai() -> None:
    agent = SearchAgent(
        model_id="test-model",
        search_provider=_StubSearchProvider(),
        model_provider_type="openai",
    )
    assert isinstance(agent._agent.system_prompt, ChatPromptBuilder)
    agent.close()


def test_summarization_agent_uses_chat_prompts_for_openai() -> None:
    agent = SummarizationAgent(model_id="test-model", model_provider_type="openai")
    assert isinstance(agent._agent.system_prompt, ChatPromptBuilder)
    agent.close()
