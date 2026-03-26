from agentic.providers.mlx.models.qwen import (
    GenerationMode,
    SamplingPresetName,
    SamplingProfile,
    resolve_sampling_profile,
)


class ModelConfig:
    def __init__(
        self,
        max_tokens: int = 512,
        *,
        generation_mode: GenerationMode = "nothinking",
        sampling_preset: SamplingPresetName | None = None,
        sampling_profile: SamplingProfile | None = None,
    ):
        self.max_tokens = max_tokens
        self.generation_mode = generation_mode
        self.sampling_profile = sampling_profile or resolve_sampling_profile(
            generation_mode,
            sampling_preset,
        )
