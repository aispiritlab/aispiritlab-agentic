"""Tests for the OpenAI library inference client."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from providers.client._openai_client import OpenAIInferenceClient
from providers.schema.config import InferenceConfig
from providers.schema.response import InferenceResponse


def _make_mock_response(
    text: str = "Hello!",
    model: str = "test-model",
    request_id: str = "req-1",
    finish_reason: str = "stop",
    prompt_tokens: int = 5,
    completion_tokens: int = 3,
    total_tokens: int = 8,
) -> SimpleNamespace:
    choice = SimpleNamespace(
        message=SimpleNamespace(content=text),
        finish_reason=finish_reason,
    )
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
    return SimpleNamespace(
        id=request_id,
        model=model,
        choices=[choice],
        usage=usage,
    )


class TestOpenAIInferenceClient:
    def test_chat_returns_inference_response(self) -> None:
        mock_response = _make_mock_response()

        with patch("providers.client._openai_client.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            client = OpenAIInferenceClient(
                "http://localhost:8080",
                model="test-model",
                config=InferenceConfig(max_tokens=256),
            )
            result = client.chat("Hi there")

        assert isinstance(result, InferenceResponse)
        assert result.text == "Hello!"
        assert result.model == "test-model"
        assert result.request_id == "req-1"
        assert result.finish_reason == "stop"
        assert result.prompt_tokens == 5
        assert result.completion_tokens == 3
        assert result.total_tokens == 8
        assert result.latency_ms > 0

    def test_chat_with_message_list(self) -> None:
        mock_response = _make_mock_response(text="Response")

        with patch("providers.client._openai_client.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            client = OpenAIInferenceClient("http://localhost:8080")
            messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ]
            result = client.chat(messages)

        assert result.text == "Response"
        call_args = mock_client.chat.completions.create.call_args
        assert len(call_args.kwargs["messages"]) == 2

    def test_chat_empty_choices(self) -> None:
        mock_response = SimpleNamespace(
            id="req-2", model="test", choices=[], usage=None,
        )

        with patch("providers.client._openai_client.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            client = OpenAIInferenceClient("http://localhost:8080")
            result = client.chat("Hello")

        assert result.text == ""
        assert result.finish_reason == ""

    def test_chat_passes_config_params(self) -> None:
        mock_response = _make_mock_response()
        config = InferenceConfig(max_tokens=1024, temperature=0.5, top_p=0.9, presence_penalty=0.1)

        with patch("providers.client._openai_client.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            client = OpenAIInferenceClient("http://localhost:8080", config=config)
            client.chat("Test")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["max_tokens"] == 1024
        assert call_kwargs["presence_penalty"] == 0.1

    def test_aclose_awaits_async_client_close(self) -> None:
        mock_response = _make_mock_response()

        with (
            patch("providers.client._openai_client.OpenAI") as mock_openai_cls,
            patch("providers.client._openai_client.AsyncOpenAI") as mock_async_openai_cls,
        ):
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client

            mock_async_client = MagicMock()
            mock_async_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_async_client.close = AsyncMock(return_value=None)
            mock_async_openai_cls.return_value = mock_async_client

            client = OpenAIInferenceClient("http://localhost:8080")
            asyncio.run(client.achat("Hello"))
            asyncio.run(client.aclose())

        mock_client.close.assert_called_once()
        mock_async_client.close.assert_called_once()
