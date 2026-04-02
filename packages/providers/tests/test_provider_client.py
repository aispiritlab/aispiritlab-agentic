"""Tests for the ProviderClient factory."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from providers.client import ProviderClient
from providers.schema.config import InferenceConfig
from providers.schema.response import InferenceResponse


def _make_mock_openai_response() -> SimpleNamespace:
    return SimpleNamespace(
        id="req-1",
        model="test-model",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Hello from openai"),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


class TestProviderClientOpenAI:
    def test_chat_uses_openai_backend(self) -> None:
        mock_response = _make_mock_openai_response()

        with patch("providers.client._openai_client.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_cls.return_value = mock_client

            client = ProviderClient("http://localhost:8080", backend="openai")
            result = client.chat("Hello")

        assert isinstance(result, InferenceResponse)
        assert result.text == "Hello from openai"

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported backend"):
            ProviderClient("http://localhost:8080", backend="bogus")  # type: ignore[arg-type]


class TestProviderClientHttpx:
    def test_chat_uses_httpx_backend(self) -> None:
        response_data = {
            "id": "req-2",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from httpx"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

        with respx.mock:
            respx.post("http://localhost:8080/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=response_data)
            )
            client = ProviderClient("http://localhost:8080", backend="httpx")
            result = client.chat("Hello")
            client.close()

        assert isinstance(result, InferenceResponse)
        assert result.text == "Hello from httpx"

    def test_achat_raises_for_httpx_backend(self) -> None:
        client = ProviderClient("http://localhost:8080", backend="httpx")
        with pytest.raises(NotImplementedError, match="only supported with the 'openai' backend"):
            import asyncio
            asyncio.run(client.achat("Hello"))
        client.close()
