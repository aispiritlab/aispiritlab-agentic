"""Generation configuration for model providers."""

from __future__ import annotations

from dataclasses import dataclass, field

from providers.sampling.qwen import (
    GenerationMode,
    SamplingPresetName,
    SamplingProfile,
    resolve_sampling_profile,
)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Immutable generation settings.

    Frozen so it is safe as a shared default argument — a mutable instance would
    be created once at import and silently shared by every provider.
    """

    max_tokens: int = 512
    generation_mode: GenerationMode = "nothinking"
    sampling_preset: SamplingPresetName | None = None
    sampling_profile: SamplingProfile = field(init=False)

    def __init__(
        self,
        max_tokens: int = 512,
        *,
        generation_mode: GenerationMode = "nothinking",
        sampling_preset: SamplingPresetName | None = None,
        sampling_profile: SamplingProfile | None = None,
    ) -> None:
        object.__setattr__(self, "max_tokens", max_tokens)
        object.__setattr__(self, "generation_mode", generation_mode)
        object.__setattr__(self, "sampling_preset", sampling_preset)
        object.__setattr__(
            self,
            "sampling_profile",
            sampling_profile or resolve_sampling_profile(generation_mode, sampling_preset),
        )


#: Shared default. Safe to reuse because ModelConfig is frozen.
DEFAULT_MODEL_CONFIG = ModelConfig()
