from __future__ import annotations

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError:  # pragma: no cover - exercised on Linux/Docker only
    mx = None


TARGET_SAMPLE_RATE = 16000
INT16_MAX = np.iinfo(np.int16).max


def _to_mono_array(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        if audio.shape[0] == 1:
            return audio[0]
        if audio.shape[1] == 1:
            return audio[:, 0]
        if audio.shape[-1] == 2:
            return audio.mean(axis=-1)
        if audio.shape[0] == 2:
            return audio.mean(axis=0)
    return audio.reshape(-1)


def _parse_audio_tuple(audio: object) -> tuple[np.ndarray, int, bool] | None:
    if not isinstance(audio, (tuple, list)) or len(audio) != 2:
        return None

    sample_rate, payload = audio
    if not isinstance(sample_rate, (int, float)) or payload is None:
        return None

    sample_rate_int = int(sample_rate)
    if sample_rate_int <= 0:
        return None

    payload_array = np.asarray(payload)
    if not payload_array.size or not np.issubdtype(payload_array.dtype, np.number):
        return None

    payload_array = _to_mono_array(payload_array).reshape(-1)
    return payload_array, sample_rate_int, np.issubdtype(payload_array.dtype, np.integer)


def _ensure_target_sample_rate(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == TARGET_SAMPLE_RATE:
        return audio

    from mlx_audio.stt.utils import resample_audio

    return resample_audio(audio.astype(np.float32), sample_rate, TARGET_SAMPLE_RATE)


def is_empty_transcription(result: object) -> bool:
    if result is None:
        return True
    return not getattr(result, "text", "") and not getattr(result, "sentences", None)


def convert_numpy_audio(audio: object) -> object | None:
    if mx is None:
        return None

    parsed = _parse_audio_tuple(audio)
    if parsed is None:
        return None

    arr, sample_rate, is_integer = parsed
    arr = _ensure_target_sample_rate(arr, sample_rate)
    if is_integer:
        arr = arr.astype(np.float32) / INT16_MAX

    return mx.array(np.clip(arr.astype(np.float32), -1.0, 1.0))


def convert_audio(voice_model: object, audio_payload: object) -> tuple[object | None, str | None]:
    if mx is None:
        return None, "Voice support requires MLX and is unavailable in this environment"

    payload = convert_numpy_audio(audio_payload)
    if payload is None:
        return None, "Invalid audio payload"

    try:
        transcription = voice_model.response(payload, dtype=mx.float32)
        if is_empty_transcription(transcription):
            return transcription, "Model returned empty transcription"
        return transcription, None
    except Exception as error:
        return None, f"voice model failed: {error}"
