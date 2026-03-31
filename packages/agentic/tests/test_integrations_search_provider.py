from __future__ import annotations

from agentic.integrations.search_provider import (
    LangSearchProvider,
    SearchResult,
    normalize_results,
)
from agentic.integrations.tavily_search_provider import TavilySearchProvider
from agentic.integrations.valyu_search_provider import ValyuSearchProvider
import httpx
import pytest


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


class _ResponseStub:
    def __init__(self, payload: object, *, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


class _HttpxClientStub:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object]]] = []

    def post(self, path: str, json: dict[str, object]) -> object:
        self.calls.append((path, json))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def close(self) -> None:
        return None


class _LoggerStub:
    def __init__(self) -> None:
        self.info_calls: list[tuple[str, dict[str, object]]] = []
        self.warning_calls: list[tuple[str, dict[str, object]]] = []

    def info(self, event: str, **kwargs: object) -> None:
        self.info_calls.append((event, kwargs))

    def warning(self, event: str, **kwargs: object) -> None:
        self.warning_calls.append((event, kwargs))


def test_tavily_provider_logs_success(monkeypatch) -> None:
    client = _HttpxClientStub(
        _ResponseStub(
            {
                "results": [
                    {"title": "DDD", "url": "https://example.com/ddd", "content": "Domain-driven design"},
                ]
            },
            status_code=200,
        )
    )
    logger = _LoggerStub()
    monkeypatch.setattr(
        "agentic.integrations.tavily_search_provider.httpx.Client",
        lambda *args, **kwargs: client,
    )
    monkeypatch.setattr("agentic.integrations.tavily_search_provider.logger", logger)

    provider = TavilySearchProvider(api_key="test-key")

    results = provider.search("DDD", count=2)

    assert len(results) == 1
    assert logger.info_calls[0][0] == "search_api_request"
    assert logger.info_calls[1][0] == "search_api_success"
    assert logger.info_calls[1][1]["status_code"] == 200
    assert logger.info_calls[1][1]["result_count"] == 1


def test_valyu_provider_logs_http_failure(monkeypatch) -> None:
    request = httpx.Request("POST", "https://api.valyu.ai/v1/search")
    response = httpx.Response(401, request=request)
    client = _HttpxClientStub(
        httpx.HTTPStatusError("Unauthorized", request=request, response=response)
    )
    logger = _LoggerStub()
    monkeypatch.setattr(
        "agentic.integrations.valyu_search_provider.httpx.Client",
        lambda *args, **kwargs: client,
    )
    monkeypatch.setattr("agentic.integrations.valyu_search_provider.logger", logger)

    provider = ValyuSearchProvider(api_key="test-key")

    with pytest.raises(httpx.HTTPStatusError):
        provider.search("DDD", count=3)

    assert logger.info_calls[0][0] == "search_api_request"
    assert logger.warning_calls[0][0] == "search_api_failed"
    assert logger.warning_calls[0][1]["status_code"] == 401
