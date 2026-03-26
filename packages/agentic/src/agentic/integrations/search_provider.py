from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from agentic.providers.api.http_client import HttpClient


@dataclass(frozen=True, slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    published_at: str | None = None


class SearchProvider(Protocol):
    def search(self, query: str, *, count: int = 5) -> list[SearchResult]: ...

    def close(self) -> None: ...


class LangSearchProvider:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.langsearch.com",
        timeout: float = 20.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("LANGSEARCH_API_KEY is required for the LangSearch provider")

        self._client = HttpClient(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

    def search(self, query: str, *, count: int = 5) -> list[SearchResult]:
        payload = self._client.post(
            "/v1/web-search",
            data={
                "query": query,
                "count": count,
                "summary": True,
                "freshness": "noLimit",
            },
        )

        if not isinstance(payload, dict):
            raise RuntimeError("LangSearch returned an unexpected payload shape")

        code = payload.get("code")
        if code not in (None, 200):
            raise RuntimeError(f"LangSearch returned non-success code: {code}")

        pages = payload.get("data", {}).get("webPages", {}).get("value", [])
        if not isinstance(pages, list):
            return []

        results: list[SearchResult] = []
        for page in pages:
            if not isinstance(page, dict):
                continue
            results.append(
                SearchResult(
                    title=str(page.get("name", "")).strip(),
                    url=str(page.get("url", "")).strip(),
                    snippet=str(
                        page.get("summary")
                        or page.get("snippet")
                        or page.get("description")
                        or ""
                    ).strip(),
                    published_at=(
                        str(page["datePublished"]).strip()
                        if page.get("datePublished") is not None
                        else None
                    ),
                )
            )
        return results

    def close(self) -> None:
        self._client.close()


def normalize_results(results: Sequence[SearchResult]) -> tuple[dict[str, str | None], ...]:
    return tuple(
        {
            "title": result.title,
            "url": result.url,
            "snippet": result.snippet,
            "published_at": result.published_at,
        }
        for result in results
    )
