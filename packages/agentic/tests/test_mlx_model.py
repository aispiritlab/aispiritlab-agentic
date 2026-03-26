from agentic.models import (
    ModelConfig,
    QWEN_DEFAULT_PRESET_BY_MODE,
    QWEN_SAMPLING_PRESETS,
    resolve_sampling_profile,
)


def test_resolve_sampling_profile_uses_mode_default_preset() -> None:
    profile = resolve_sampling_profile("thinking")

    expected = QWEN_SAMPLING_PRESETS[QWEN_DEFAULT_PRESET_BY_MODE["thinking"]]
    assert profile == expected


def test_sampling_profile_exposes_mlx_kwargs() -> None:
    profile = resolve_sampling_profile("nothinking")

    assert profile.mlx_sampler_kwargs() == {
        "temp": profile.temperature,
        "top_p": profile.top_p,
        "top_k": profile.top_k,
        "min_p": profile.min_p,
    }
    assert profile.mlx_logits_processor_kwargs() == {
        "repetition_penalty": profile.repetition_penalty,
    }


def test_model_config_resolves_sampling_profile_for_mode() -> None:
    config = ModelConfig(max_tokens=128, generation_mode="nothinking")

    expected = QWEN_SAMPLING_PRESETS[QWEN_DEFAULT_PRESET_BY_MODE["nothinking"]]
    assert config.max_tokens == 128
    assert config.sampling_profile == expected
