"""Static registry of prepared blocks available in the agent graph builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BlockKind = Literal["agent", "integration", "structural_output", "provider"]


@dataclass(frozen=True, slots=True)
class BlockSpec:
    agent_name: str
    display_name: str
    description: str
    capabilities: tuple[str, ...]
    block_kind: BlockKind
    config_defaults: tuple[tuple[str, str], ...] = ()


PERSONAL_ASSISTANT_BLOCKS: tuple[BlockSpec, ...] = (
    BlockSpec(
        agent_name="router",
        display_name="PA Router",
        description="Routes user messages to the appropriate specialist agent.",
        capabilities=("routing", "classification"),
        block_kind="agent",
    ),
    BlockSpec(
        agent_name="personalize",
        display_name="Personalize",
        description="Handles user onboarding and personalization setup.",
        capabilities=("personalization", "onboarding", "profile"),
        block_kind="agent",
    ),
    BlockSpec(
        agent_name="manage_notes",
        display_name="Manage Notes",
        description="Creates, edits, deletes, and lists user notes.",
        capabilities=("notes", "crud", "knowledge-management"),
        block_kind="agent",
    ),
    BlockSpec(
        agent_name="discovery_notes",
        display_name="Discovery Notes",
        description="Searches notes using RAG-based retrieval.",
        capabilities=("search", "rag", "discovery"),
        block_kind="agent",
    ),
    BlockSpec(
        agent_name="sage",
        display_name="Sage",
        description="Provides structured decision-making support using a 7-step methodology.",
        capabilities=("decision-making", "analysis", "sage"),
        block_kind="agent",
    ),
    BlockSpec(
        agent_name="organizer",
        display_name="Organizer",
        description="Classifies content using the PARA method.",
        capabilities=("classification", "organization", "para"),
        block_kind="agent",
    ),
)

WORKSHOP_AGENT_BLOCKS: tuple[BlockSpec, ...] = (
    BlockSpec(
        agent_name="llm_chat",
        display_name="LLM Chat",
        description="General-purpose chat agent using an OpenAI-compatible API.",
        capabilities=("chat", "general", "fallback"),
        block_kind="agent",
    ),
    BlockSpec(
        agent_name="api_router",
        display_name="API Router",
        description="Routes messages to the best agent using an OpenAI-compatible API.",
        capabilities=("routing", "classification"),
        block_kind="agent",
    ),
    BlockSpec(
        agent_name="planner",
        display_name="Planner",
        description="Breaks requests into steps and delegates to connected agent blocks.",
        capabilities=("planning", "delegation"),
        block_kind="agent",
    ),
    BlockSpec(
        agent_name="summarizer",
        display_name="Summarizer",
        description="Summarizes text into concise outputs.",
        capabilities=("summarization", "text-processing"),
        block_kind="agent",
    ),
    BlockSpec(
        agent_name="searcher",
        display_name="Searcher",
        description="Searches via a connected search integration and can store findings.",
        capabilities=("search", "web", "knowledge-base", "indexing"),
        block_kind="agent",
    ),
)

PROVIDER_BLOCKS: tuple[BlockSpec, ...] = (
    BlockSpec(
        agent_name="model_provider",
        display_name="Model Provider",
        description="Shared model provider binding reused by connected agent blocks.",
        capabilities=("model", "provider", "shared"),
        block_kind="provider",
        config_defaults=(
            ("provider_type", "openai"),
            ("model_id", "qwen3.5-4b"),
        ),
    ),
)

INTEGRATION_BLOCKS: tuple[BlockSpec, ...] = (
    BlockSpec(
        agent_name="tavily_search",
        display_name="Tavily Search",
        description="Tavily web search integration providing real-time search results.",
        capabilities=("search", "web", "tavily"),
        block_kind="integration",
        config_defaults=(("api_key_env", "TAVILY_API_KEY"),),
    ),
    BlockSpec(
        agent_name="valyu_search",
        display_name="Valyu Search",
        description="Valyu AI search integration for web and proprietary data.",
        capabilities=("search", "web", "valyu"),
        block_kind="integration",
        config_defaults=(("api_key_env", "VALYU_API_KEY"),),
    ),
    BlockSpec(
        agent_name="knowledge_base",
        display_name="Knowledge Base",
        description="Qdrant-backed knowledge base for indexing and retrieval.",
        capabilities=("knowledge-base", "qdrant", "vector-search"),
        block_kind="integration",
        config_defaults=(("path", "data/knowledge_base"),),
    ),
)

STRUCTURAL_OUTPUT_BLOCKS: tuple[BlockSpec, ...] = (
    BlockSpec(
        agent_name="markdown_output",
        display_name="Markdown Output",
        description="Writes connected agent results to a markdown file.",
        capabilities=("output", "markdown", "persistence"),
        block_kind="structural_output",
        config_defaults=(("path", "outputs/agentic_graph.md"),),
    ),
)

KNOWN_BLOCKS: tuple[BlockSpec, ...] = (
    PERSONAL_ASSISTANT_BLOCKS
    + WORKSHOP_AGENT_BLOCKS
    + PROVIDER_BLOCKS
    + INTEGRATION_BLOCKS
    + STRUCTURAL_OUTPUT_BLOCKS
)


def get_block_by_name(name: str) -> BlockSpec | None:
    for block in KNOWN_BLOCKS:
        if block.agent_name == name:
            return block
    return None


def get_blocks_by_kind(kind: BlockKind) -> tuple[BlockSpec, ...]:
    return tuple(block for block in KNOWN_BLOCKS if block.block_kind == kind)


def is_agent_block(name: str) -> bool:
    block = get_block_by_name(name)
    return block is not None and block.block_kind == "agent"


def is_integration_block(name: str) -> bool:
    block = get_block_by_name(name)
    return block is not None and block.block_kind == "integration"


def is_structural_output_block(name: str) -> bool:
    block = get_block_by_name(name)
    return block is not None and block.block_kind == "structural_output"
