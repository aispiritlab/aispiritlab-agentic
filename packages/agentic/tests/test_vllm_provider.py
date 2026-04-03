from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from providers.models import ModelConfig
from providers.models.response import ModelResponse
from providers.orchestrator import ModelProvider
from providers.vllm import VLLMProvider


# ── Stubs ─────────────────────────────────────────────────────────────────────


@dataclass
class _StubCompletionOutput:
    text: str = "hello world"
    token_ids: tuple[int, ...] = (1, 2, 3)
    finish_reason: str = "stop"


@dataclass
class _StubRequestOutput:
    request_id: str = "req-001"
    prompt_token_ids: tuple[int, ...] = (10, 11)
    outputs: list[_StubCompletionOutput] = field(
        default_factory=lambda: [_StubCompletionOutput()]
    )


class StubLLM:
    """Mimics vllm.LLM for native inference."""

    def generate(self, prompts: list[str], sampling_params: object) -> list[_StubRequestOutput]:
        return [_StubRequestOutput()]

    def chat(
        self, messages: list[dict[str, str]], sampling_params: object
    ) -> list[_StubRequestOutput]:
        return [_StubRequestOutput()]


class _StubCompletions:
    def create(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            id="chatcmpl-001",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="hello from openai"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=5,
                completion_tokens=3,
                total_tokens=8,
            ),
        )


class _StubChat:
    completions = _StubCompletions()


class StubOpenAIClient:
    """Mimics openai.OpenAI for openai_compatible inference."""

    chat = _StubChat()


class StubRayProcessor:
    """Mimics a Ray Data processor."""

    def __call__(self, ds: object) -> SimpleNamespace:
        return SimpleNamespace(
            take_all=lambda: [{"generated_text": "hello from ray"}]
        )


class StubNativeProvider:
    """Provider that returns StubLLM as backend."""

    @classmethod
    def load_backend(cls, model_name: str) -> StubLLM:
        return StubLLM()

    @classmethod
    def build_model(cls, backend: object, model_name: str, config: ModelConfig, **kwargs: object) -> object:
        from providers.vllm.vllm_model import VllmNativeModel

        return VllmNativeModel(backend, model_name, config=config)

    @classmethod
    def close_backend(cls, backend: object) -> None:
        pass


class StubOpenAIProvider:
    """Provider that returns StubOpenAIClient as backend."""

    @classmethod
    def load_backend(cls, model_name: str) -> StubOpenAIClient:
        return StubOpenAIClient()

    @classmethod
    def build_model(cls, backend: object, model_name: str, config: ModelConfig, **kwargs: object) -> object:
        from providers.vllm.vllm_model import VllmOpenAIModel

        return VllmOpenAIModel(model_name, backend, config=config)

    @classmethod
    def close_backend(cls, backend: object) -> None:
        pass


# ── Tests: VLLMProvider configure ─────────────────────────────────────────────


def test_vllm_provider_configure_sets_strategy() -> None:
    VLLMProvider.configure(inference_strategy="openai_compatible")
    assert VLLMProvider._config.inference_strategy == "openai_compatible"

    VLLMProvider.configure(inference_strategy="native")
    assert VLLMProvider._config.inference_strategy == "native"


def test_vllm_provider_configure_sets_all_fields() -> None:
    VLLMProvider.configure(
        inference_strategy="ray",
        base_url="http://custom:9000/v1",
        api_key="my-key",
        tensor_parallel_size=4,
        gpu_memory_utilization=0.8,
    )
    assert VLLMProvider._config.inference_strategy == "ray"
    assert VLLMProvider._config.base_url == "http://custom:9000/v1"
    assert VLLMProvider._config.api_key == "my-key"
    assert VLLMProvider._config.tensor_parallel_size == 4
    assert VLLMProvider._config.gpu_memory_utilization == 0.8

    # Reset to defaults
    VLLMProvider.configure()


def test_vllm_cli_raises_not_implemented() -> None:
    VLLMProvider.configure(inference_strategy="cli")
    with pytest.raises(NotImplementedError, match="CLI inference is in progress"):
        VLLMProvider.load_backend("some-model")
    VLLMProvider.configure()


# ── Tests: Native model via ModelProvider ─────────────────────────────────────


def test_vllm_native_provider_loads_via_model_provider() -> None:
    provider = ModelProvider(
        "facebook/opt-125m",
        model_provider_type="vllm",
        supported_providers={"vllm": StubNativeProvider},
    )
    model = provider.model
    assert model is not None
    provider.close()


class _StubSamplingParams:
    """Stand-in for vllm.SamplingParams when vllm is not installed."""
    pass


def test_vllm_native_model_response() -> None:
    from providers.vllm.vllm_model import VllmNativeModel

    model = VllmNativeModel(StubLLM(), "facebook/opt-125m")
    resp = model.response("Hello", sampling_params=_StubSamplingParams())
    assert isinstance(resp, ModelResponse)
    assert resp.text == "hello world"
    assert resp.model == "facebook/opt-125m"
    assert resp.request_id == "req-001"
    assert resp.prompt_tokens == 2
    assert resp.completion_tokens == 3


def test_vllm_native_model_response_with_messages() -> None:
    from providers.vllm.vllm_model import VllmNativeModel

    model = VllmNativeModel(StubLLM(), "facebook/opt-125m")
    resp = model.response(
        [{"role": "user", "content": "Hello"}],
        sampling_params=_StubSamplingParams(),
    )
    assert resp.text == "hello world"


def test_vllm_native_model_raises_when_closed() -> None:
    from providers.vllm.vllm_model import VllmNativeModel

    model = VllmNativeModel(StubLLM(), "facebook/opt-125m")
    model.close()
    with pytest.raises(RuntimeError, match="Model has been closed"):
        model.response("Hello")


# ── Tests: OpenAI-compatible model ───────────────────────────────────────────


def test_vllm_openai_provider_loads_via_model_provider() -> None:
    provider = ModelProvider(
        "NousResearch/Meta-Llama-3-8B-Instruct",
        model_provider_type="vllm",
        supported_providers={"vllm": StubOpenAIProvider},
    )
    model = provider.model
    assert model is not None
    provider.close()


def test_vllm_openai_model_response() -> None:
    from providers.vllm.vllm_model import VllmOpenAIModel

    model = VllmOpenAIModel("test-model", StubOpenAIClient())
    resp = model.response("Hello!")
    assert isinstance(resp, ModelResponse)
    assert resp.text == "hello from openai"
    assert resp.request_id == "chatcmpl-001"
    assert resp.finish_reason == "stop"
    assert resp.prompt_tokens == 5
    assert resp.completion_tokens == 3
    assert resp.total_tokens == 8


def test_vllm_openai_model_response_with_messages() -> None:
    from providers.vllm.vllm_model import VllmOpenAIModel

    model = VllmOpenAIModel("test-model", StubOpenAIClient())
    resp = model.response([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ])
    assert resp.text == "hello from openai"


def test_vllm_openai_model_raises_when_closed() -> None:
    from providers.vllm.vllm_model import VllmOpenAIModel

    model = VllmOpenAIModel("test-model", StubOpenAIClient())
    model.close()
    with pytest.raises(RuntimeError, match="Model client has been closed"):
        model.response("Hello")


# ── Tests: Ray model ─────────────────────────────────────────────────────────


def test_vllm_ray_model_response() -> None:
    from providers.vllm.vllm_model import VllmRayModel

    def fake_dataset_factory(items: list[dict[str, str]]) -> SimpleNamespace:
        return SimpleNamespace()

    processor = StubRayProcessor()
    model = VllmRayModel(
        processor, "test-model", dataset_factory=fake_dataset_factory
    )
    resp = model.response("Hello")
    assert isinstance(resp, ModelResponse)
    assert resp.text == "hello from ray"
    assert resp.model == "test-model"


def test_vllm_ray_model_raises_when_closed() -> None:
    from providers.vllm.vllm_model import VllmRayModel

    model = VllmRayModel(StubRayProcessor(), "test-model")
    model.close()
    with pytest.raises(RuntimeError, match="Ray processor has been closed"):
        model.response("Hello")
