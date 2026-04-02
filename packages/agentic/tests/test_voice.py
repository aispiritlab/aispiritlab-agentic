from __future__ import annotations

from types import ModuleType, SimpleNamespace

import numpy as np

import agentic.voice as voice_module


class _FakeMx:
    float32 = "float32"

    @staticmethod
    def array(value: np.ndarray) -> np.ndarray:
        return value


def test_ensure_target_sample_rate_passthrough_for_16khz() -> None:
    audio = np.array([0.1, -0.1], dtype=np.float32)

    result = voice_module._ensure_target_sample_rate(audio, voice_module.TARGET_SAMPLE_RATE)

    assert result is audio


def test_ensure_target_sample_rate_uses_mlx_audio_resampler(
    monkeypatch,
) -> None:
    calls: list[tuple[np.ndarray, int, int]] = []

    utils_module = ModuleType("mlx_audio.stt.utils")

    def fake_resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        calls.append((audio, source_rate, target_rate))
        return np.array([0.25, -0.25], dtype=np.float32)

    utils_module.resample_audio = fake_resample_audio
    monkeypatch.setitem(__import__("sys").modules, "mlx_audio", ModuleType("mlx_audio"))
    monkeypatch.setitem(__import__("sys").modules, "mlx_audio.stt", ModuleType("mlx_audio.stt"))
    monkeypatch.setitem(__import__("sys").modules, "mlx_audio.stt.utils", utils_module)

    audio = np.array([1.0, -1.0], dtype=np.float32)
    result = voice_module._ensure_target_sample_rate(audio, 48_000)

    assert np.array_equal(result, np.array([0.25, -0.25], dtype=np.float32))
    assert len(calls) == 1
    assert calls[0][1:] == (48_000, voice_module.TARGET_SAMPLE_RATE)


def test_convert_audio_rejects_invalid_payload(monkeypatch) -> None:
    monkeypatch.setattr(voice_module, "mx", _FakeMx)

    result, error = voice_module.convert_audio(SimpleNamespace(), "bad-payload")

    assert result is None
    assert error == "Invalid audio payload"


def test_convert_audio_normalizes_16khz_integer_audio(monkeypatch) -> None:
    monkeypatch.setattr(voice_module, "mx", _FakeMx)

    class FakeVoiceModel:
        def __init__(self) -> None:
            self.calls: list[tuple[np.ndarray, object]] = []

        def response(self, payload: np.ndarray, *, dtype: object) -> SimpleNamespace:
            self.calls.append((payload, dtype))
            return SimpleNamespace(text="got it", sentences=["got it"])

    voice_model = FakeVoiceModel()

    audio = (
        16_000,
        np.array([0, voice_module.INT16_MAX, -voice_module.INT16_MAX], dtype=np.int16),
    )
    result, error = voice_module.convert_audio(voice_model, audio)

    assert error is None
    assert result is not None
    assert len(voice_model.calls) == 1
    payload, dtype = voice_model.calls[0]
    assert dtype == _FakeMx.float32
    assert np.allclose(payload, np.array([0.0, 1.0, -1.0], dtype=np.float32))


def test_convert_audio_reports_empty_transcription(monkeypatch) -> None:
    monkeypatch.setattr(voice_module, "mx", _FakeMx)

    class FakeVoiceModel:
        def response(self, payload: np.ndarray, *, dtype: object) -> SimpleNamespace:
            del payload, dtype
            return SimpleNamespace(text="", sentences=[])

    audio = (16_000, np.array([0.0, 0.0], dtype=np.float32))
    result, error = voice_module.convert_audio(FakeVoiceModel(), audio)

    assert result is not None
    assert error == "Model returned empty transcription"
