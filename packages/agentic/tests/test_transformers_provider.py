from types import SimpleNamespace

import pytest

from agentic.models import ModelConfig, ModelProvider
from agentic.models.response import ModelResponse


# ── Stubs ─────────────────────────────────────────────────────────────────────


class StubTensor:
    def __init__(self, data: list[list[int]]) -> None:
        self._data = data
        self.shape = (len(data), len(data[0]))

    def __getitem__(self, key: object) -> "StubTensor":
        return StubTensor([[10, 20, 30]])


class StubTokenizerOutput(dict[str, StubTensor]):
    """Dict-like object that also supports attribute access, mimicking transformers BatchEncoding."""

    def __init__(self) -> None:
        super().__init__(
            input_ids=StubTensor([[1, 2, 3]]),
            attention_mask=StubTensor([[1, 1, 1]]),
        )

    @property
    def input_ids(self) -> StubTensor:
        return self["input_ids"]

    def to(self, device: str) -> "StubTokenizerOutput":
        return self


class StubTokenizer:
    def __init__(self) -> None:
        self.pad_token: str | None = "<pad>"
        self.eos_token: str = "</s>"

    def __call__(
        self, texts: list[str], return_tensors: str = "pt", padding: bool = True
    ) -> StubTokenizerOutput:
        return StubTokenizerOutput()

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> str:
        return "<|user|> Hello <|assistant|>"

    def batch_decode(
        self, token_ids: object, skip_special_tokens: bool = True
    ) -> list[str]:
        return ["generated text output"]


class StubModel:
    device = "cpu"

    def generate(self, **kwargs: object) -> StubTensor:
        return StubTensor([[1, 2, 3, 10, 20, 30]])


class StubTransformersProvider:
    """Provider that returns stub model/tokenizer as backend."""

    @classmethod
    def load_backend(cls, model_name: str) -> tuple[StubModel, StubTokenizer]:
        return StubModel(), StubTokenizer()

    @classmethod
    def build_model(cls, backend: object, model_name: str, config: ModelConfig, **kwargs: object) -> object:
        from agentic.providers.transformers.transformers_model import TransformersModel

        return TransformersModel(backend, model_name, config=config)

    @classmethod
    def close_backend(cls, backend: object) -> None:
        pass


# ── Tests: TransformersProvider via ModelProvider ─────────────────────────────


def test_transformers_provider_loads_via_model_provider() -> None:
    provider = ModelProvider(
        "mistralai/Mistral-7B-v0.1",
        model_provider_type="transformers",
        supported_providers={"transformers": StubTransformersProvider},
    )
    model = provider.model
    assert model is not None
    provider.close()


# ── Tests: TransformersModel ─────────────────────────────────────────────────


def test_transformers_model_response_with_string_prompt() -> None:
    from agentic.providers.transformers.transformers_model import TransformersModel

    backend = (StubModel(), StubTokenizer())
    model = TransformersModel(backend, "test-model")
    resp = model.response("Hello world")
    assert isinstance(resp, ModelResponse)
    assert resp.text == "generated text output"
    assert resp.model == "test-model"
    assert resp.latency_ms > 0


def test_transformers_model_response_with_messages() -> None:
    from agentic.providers.transformers.transformers_model import TransformersModel

    backend = (StubModel(), StubTokenizer())
    model = TransformersModel(backend, "test-model")
    resp = model.response([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ])
    assert isinstance(resp, ModelResponse)
    assert resp.text == "generated text output"


def test_transformers_model_raises_when_closed() -> None:
    from agentic.providers.transformers.transformers_model import TransformersModel

    backend = (StubModel(), StubTokenizer())
    model = TransformersModel(backend, "test-model")
    model.close()
    with pytest.raises(RuntimeError, match="Model has been closed"):
        model.response("Hello")


def test_transformers_model_tracks_token_counts() -> None:
    from agentic.providers.transformers.transformers_model import TransformersModel

    backend = (StubModel(), StubTokenizer())
    model = TransformersModel(backend, "test-model")
    resp = model.response("Hello")
    assert resp.prompt_tokens > 0
    assert resp.completion_tokens > 0
    assert resp.total_tokens == resp.prompt_tokens + resp.completion_tokens
