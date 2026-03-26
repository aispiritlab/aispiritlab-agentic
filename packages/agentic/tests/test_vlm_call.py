from contextlib import contextmanager

from agentic.models.response import ModelResponse
import agentic.vlm_call as vlm_call_module
from agentic.vlm_call import VLMCall


def test_vlm_call_passes_images_and_preserves_history(monkeypatch) -> None:
    captured_prompts: list[list[dict[str, str]]] = []
    captured_images: list[str | list[str] | None] = []

    class FakeModel:
        def response(self, prompt, **kwargs) -> ModelResponse:  # noqa: ANN001
            captured_prompts.append(prompt)
            captured_images.append(kwargs.get("image"))
            return ModelResponse(text="opis obrazu")

    class FakeProvider:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            del args, kwargs
            self._model_name = "fake-vlm"
            self._model = FakeModel()

        @contextmanager
        def session(self, name: str = "model"):
            assert name == "model"
            yield self._model

        def get_load_error(self, name: str = "model") -> str | None:
            assert name == "model"
            return None

    monkeypatch.setattr(vlm_call_module, "ModelProvider", FakeProvider)

    call = VLMCall("fake-vlm", system_prompt="SYSTEM")

    first_reply = call.call("Co widzisz?", images=["/tmp/cat.png"])
    second_reply = call.call("A jaki ma kolor?")

    assert first_reply == "opis obrazu"
    assert second_reply == "opis obrazu"
    assert captured_images == [["/tmp/cat.png"], None]
    assert captured_prompts[0] == [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "Co widzisz?"},
    ]
    assert captured_prompts[1] == [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "Co widzisz?"},
        {"role": "assistant", "content": "opis obrazu"},
        {"role": "user", "content": "A jaki ma kolor?"},
    ]


def test_vlm_call_reset_clears_history(monkeypatch) -> None:
    captured_prompts: list[list[dict[str, str]]] = []

    class FakeModel:
        def response(self, prompt, **kwargs) -> ModelResponse:  # noqa: ANN001
            del kwargs
            captured_prompts.append(prompt)
            return ModelResponse(text="ok")

    class FakeProvider:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            del args, kwargs
            self._model_name = "fake-vlm"
            self._model = FakeModel()

        @contextmanager
        def session(self, name: str = "model"):
            assert name == "model"
            yield self._model

        def get_load_error(self, name: str = "model") -> str | None:
            assert name == "model"
            return None

    monkeypatch.setattr(vlm_call_module, "ModelProvider", FakeProvider)

    call = VLMCall("fake-vlm", system_prompt="SYSTEM")
    call.call("Pierwsza wiadomość")
    call.reset()
    call.call("Druga wiadomość")

    assert captured_prompts[-1] == [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "Druga wiadomość"},
    ]
