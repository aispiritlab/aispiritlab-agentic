from pathlib import Path

import agentic.image_generation_call as image_generation_call_module
from agentic.image_generation_call import MfluxImageCall


def test_mflux_image_call_generates_file_and_metadata(monkeypatch, tmp_path: Path) -> None:
    saved_paths: list[Path] = []

    class FakeImage:
        def save(self, path: str | Path) -> None:
            saved_path = Path(path)
            saved_path.write_bytes(b"png")
            saved_paths.append(saved_path)

    class FakeModel:
        def generate_image(self, **kwargs) -> FakeImage:  # noqa: ANN003
            assert kwargs["prompt"] == "Stwórz obraz ptaka"
            assert kwargs["seed"] == 123
            assert kwargs["width"] == 640
            assert kwargs["height"] == 480
            assert kwargs["num_inference_steps"] == 6
            return FakeImage()

    call = MfluxImageCall(
        model_name="z-image-turbo",
        quantize=8,
        width=640,
        height=480,
        steps=6,
        output_dir=tmp_path,
    )
    monkeypatch.setattr(call, "_build_model", lambda: FakeModel())

    result = call.generate_image("Stwórz obraz ptaka", seed=123)

    assert result.prompt == "Stwórz obraz ptaka"
    assert result.seed == 123
    assert result.width == 640
    assert result.height == 480
    assert result.steps == 6
    assert saved_paths == [Path(result.image_path)]
    assert Path(result.image_path).exists()


def test_mflux_image_call_rejects_empty_prompt(tmp_path: Path) -> None:
    call = MfluxImageCall(output_dir=tmp_path)

    try:
        call.generate_image("   ")
    except ValueError as error:
        assert str(error) == "Prompt for image generation cannot be empty."
    else:
        raise AssertionError("Expected ValueError for an empty image prompt")


def test_mflux_image_call_reports_missing_dependency(monkeypatch, tmp_path: Path) -> None:
    call = MfluxImageCall(output_dir=tmp_path)

    def _raise_missing() -> object:
        raise RuntimeError(
            "MFLUX is not installed. Install it with `uv add --prerelease=allow mflux`."
        )

    monkeypatch.setattr(call, "_build_model", _raise_missing)

    try:
        call.generate_image("bird")
    except RuntimeError as error:
        assert "MFLUX is not installed" in str(error)
    else:
        raise AssertionError("Expected RuntimeError when the MFLUX dependency is unavailable")


def test_mflux_image_call_close_is_idempotent_and_reloads_model(monkeypatch, tmp_path: Path) -> None:
    build_calls: list[str] = []
    cache_clear_calls: list[str] = []

    class FakeImage:
        def save(self, path: str | Path) -> None:
            Path(path).write_bytes(b"png")

    class FakeModel:
        def __init__(self, label: str) -> None:
            self._label = label

        def generate_image(self, **kwargs) -> FakeImage:  # noqa: ANN003
            assert kwargs["prompt"] == "ptak"
            return FakeImage()

    call = MfluxImageCall(output_dir=tmp_path)

    def _build_model() -> FakeModel:
        label = f"model-{len(build_calls) + 1}"
        build_calls.append(label)
        return FakeModel(label)

    monkeypatch.setattr(call, "_build_model", _build_model)
    monkeypatch.setattr(
        image_generation_call_module,
        "clear_mlx_cache",
        lambda: cache_clear_calls.append("clear"),
    )

    call.generate_image("ptak")
    call.close()
    call.close()
    call.generate_image("ptak")

    assert build_calls == ["model-1", "model-2"]
    assert cache_clear_calls == ["clear", "clear"]
