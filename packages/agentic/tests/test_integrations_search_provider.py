from __future__ import annotations

from agentic.integrations.search_provider import LangSearchProvider, SearchResult, normalize_results


class _HttpClientStub:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict[str, object] | None]] = []
        self.closed = False

    def post(self, path: str, data: dict[str, object] | None = None) -> object:
        self.calls.append((path, data))
        return self.payload

    def close(self) -> None:
        self.closed = True


def test_langsearch_provider_maps_payload_to_search_results(monkeypatch) -> None:
    payload = {
        "code": 200,
        "data": {
            "webPages": {
                "value": [
                    {
                        "name": "Example",
                        "url": "https://example.com",
                        "summary": "Summary text",
                        "datePublished": "2026-03-26",
                    }
                ]
            }
        },
    }
    client = _HttpClientStub(payload)
    monkeypatch.setattr(
        "agentic.integrations.search_provider.HttpClient",
        lambda base_url, api_key, timeout: client,
    )

    provider = LangSearchProvider(
        api_key="test-key",
        base_url="https://api.langsearch.com",
        timeout=12.0,
    )

    results = provider.search("llm", count=3)

    assert results == [
        SearchResult(
            title="Example",
            url="https://example.com",
            snippet="Summary text",
            published_at="2026-03-26",
        )
    ]
    assert client.calls == [
        (
            "/v1/web-search",
            {
                "query": "llm",
                "count": 3,
                "summary": True,
                "freshness": "noLimit",
            },
        )
    ]
    provider.close()
    assert client.closed is True


def test_normalize_results_preserves_expected_fields() -> None:
    results = [
        SearchResult(
            title="Title",
            url="https://example.com",
            snippet="Snippet",
            published_at=None,
        )
    ]

    assert normalize_results(results) == (
        {
            "title": "Title",
            "url": "https://example.com",
            "snippet": "Snippet",
            "published_at": None,
        },
    )
