"""Tests for the httpx-based HTTP client."""

from __future__ import annotations

import httpx
import pytest
import respx

from providers.client._http_client import HttpClient, ModelConnectionError


@pytest.fixture
def base_url() -> str:
    return "http://test-server:8080"


class TestHttpClient:
    def test_post_sends_json(self, base_url: str) -> None:
        with respx.mock:
            route = respx.post(f"{base_url}/v1/chat/completions").mock(
                return_value=httpx.Response(200, json={"id": "123", "choices": []})
            )
            client = HttpClient(base_url)
            result = client.post("/v1/chat/completions", data={"model": "test"})
            client.close()

        assert result == {"id": "123", "choices": []}
        assert route.called

    def test_get_sends_request(self, base_url: str) -> None:
        with respx.mock:
            route = respx.get(f"{base_url}/health").mock(
                return_value=httpx.Response(200, json={"status": "ok"})
            )
            client = HttpClient(base_url)
            result = client.get("/health")
            client.close()

        assert result == {"status": "ok"}
        assert route.called

    def test_post_connection_error_raises(self, base_url: str) -> None:
        with respx.mock:
            respx.post(f"{base_url}/v1/chat/completions").mock(
                side_effect=httpx.ConnectError("connection refused")
            )
            client = HttpClient(base_url)
            with pytest.raises(ModelConnectionError, match="Could not connect"):
                client.post("/v1/chat/completions", data={})
            client.close()

    def test_get_connection_error_raises(self, base_url: str) -> None:
        with respx.mock:
            respx.get(f"{base_url}/health").mock(
                side_effect=httpx.ConnectError("connection refused")
            )
            client = HttpClient(base_url)
            with pytest.raises(ModelConnectionError, match="Could not connect"):
                client.get("/health")
            client.close()

    def test_auth_header_set_when_api_key_provided(self, base_url: str) -> None:
        client = HttpClient(base_url, api_key="test-key")
        assert client._client.headers["authorization"] == "Bearer test-key"
        client.close()

    def test_no_auth_header_without_api_key(self, base_url: str) -> None:
        client = HttpClient(base_url)
        assert "authorization" not in client._client.headers
        client.close()
