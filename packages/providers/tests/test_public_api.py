from __future__ import annotations

from providers import ModelConfig, ModelProvider, ModelResponse, ProviderProto


def test_public_api_exports_expected_symbols() -> None:
    config = ModelConfig(max_tokens=64)
    response = ModelResponse(text="ok")
    provider = ModelProvider("demo-model", supported_providers={})

    assert config.max_tokens == 64
    assert response.text == "ok"
    assert provider.get_load_error() is None
    assert ProviderProto is not None
