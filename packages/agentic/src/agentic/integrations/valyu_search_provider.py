from __future__ import annotations

from typing import Any

import httpx
import structlog

from agentic.integrations.search_provider import SearchResult

logger = structlog.get_logger(__name__)


class ValyuSearchProvider:
    """Search provider backed by the Valyu AI search API.

    API docs: https://docs.valyu.ai/api-reference/endpoint/deepsearch
    """

    def __init__(
        self,
        api_key: str,
        *,
        search_type: str = "all",
        timeout: float = 20.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("VALYU_API_KEY is required for the Valyu provider")

        self._search_type = search_type
        self._client = httpx.Client(
            base_url="https://api.valyu.ai/v1",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
            },
            timeout=timeout,
        )

    def search(self, query: str, *, count: int = 5) -> list[SearchResult]:
        logger.info(
            "search_api_request",
            provider="valyu",
            query=query,
            count=count,
            search_type=self._search_type,
        )
        try:
            response = self._client.post(
                "/search",
                json={
                    "query": query,
                    "search_type": self._search_type,
                    "max_num_results": count,
                },
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
        except httpx.HTTPError as error:
            logger.warning(
                "search_api_failed",
                provider="valyu",
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
                provider="valyu",
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
            published = item.get("publication_date")
            results.append(
                SearchResult(
                    title=str(item.get("title", "")).strip(),
                    url=str(item.get("url", "")).strip(),
                    snippet=str(
                        item.get("content")
                        or item.get("description")
                        or ""
                    ).strip(),
                    published_at=(
                        str(published).strip() if published is not None else None
                    ),
                )
            )
        logger.info(
            "search_api_success",
            provider="valyu",
            query=query,
            count=count,
            status_code=response.status_code,
            result_count=len(results),
        )
        return results

    def close(self) -> None:
        self._client.close()
