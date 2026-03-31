from __future__ import annotations

from typing import Any

import httpx
import structlog

from agentic.integrations.search_provider import SearchResult

logger = structlog.get_logger(__name__)


class TavilySearchProvider:
    """Search provider backed by the Tavily web-search API.

    API docs: https://docs.tavily.com/documentation/api-reference/endpoint/search
    """

    def __init__(
        self,
        api_key: str,
        *,
        search_depth: str = "basic",
        topic: str = "general",
        timeout: float = 20.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("TAVILY_API_KEY is required for the Tavily provider")

        self._api_key = api_key
        self._search_depth = search_depth
        self._topic = topic
        self._client = httpx.Client(
            base_url="https://api.tavily.com",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            timeout=timeout,
        )

    def search(self, query: str, *, count: int = 5) -> list[SearchResult]:
        logger.info(
            "search_api_request",
            provider="tavily",
            query=query,
            count=count,
            search_depth=self._search_depth,
            topic=self._topic,
        )
        try:
            response = self._client.post(
                "/search",
                json={
                    "query": query,
                    "max_results": count,
                    "search_depth": self._search_depth,
                    "topic": self._topic,
                },
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
        except httpx.HTTPError as error:
            logger.warning(
                "search_api_failed",
                provider="tavily",
                query=query,
                count=count,
                status_code=getattr(getattr(error, "response", None), "status_code", None),
                error=str(error),
            )
            raise

        raw_results = payload.get("results", [])
        if not isinstance(raw_results, list):
            logger.warning(
                "search_api_invalid_payload",
                provider="tavily",
                query=query,
                count=count,
                status_code=response.status_code,
                payload_type=type(raw_results).__name__,
            )
            return []

        results: list[SearchResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            results.append(
                SearchResult(
                    title=str(item.get("title", "")).strip(),
                    url=str(item.get("url", "")).strip(),
                    snippet=str(item.get("content", "")).strip(),
                    published_at=None,
                )
            )
        logger.info(
            "search_api_success",
            provider="tavily",
            query=query,
            count=count,
            status_code=response.status_code,
            result_count=len(results),
        )
        return results

    def close(self) -> None:
        self._client.close()
