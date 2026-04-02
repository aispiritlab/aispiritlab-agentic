from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

GenerationMode = Literal["thinking", "nothinking", "orchestration"]
SamplingPresetName = Literal[
    "thinking_general",
    "thinking_coding",
    "nothinking_general",
    "nothinking_reasoning",
    "orchestration",
]


@dataclass(frozen=True, slots=True)
class SamplingProfile:
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    repetition_penalty: float
    # Metadata for cross-backend parity. It is tracked here but not applied in local MLX.
    presence_penalty: float = 0.0

    def mlx_sampler_kwargs(self) -> dict[str, float | int]:
        return {
            "temp": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
        }

    def mlx_logits_processor_kwargs(self) -> dict[str, float]:
        return {"repetition_penalty": self.repetition_penalty}


QWEN_SAMPLING_PRESETS: Final[dict[SamplingPresetName, SamplingProfile]] = {
    "thinking_general": SamplingProfile(
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=1.5,
    ),
    "thinking_coding": SamplingProfile(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
    ),
    "nothinking_general": SamplingProfile(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=1.5,
    ),
    "nothinking_reasoning": SamplingProfile(
        temperature=1.0,
        top_p=1.0,
        top_k=40,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=2.0,
    ),
    "orchestration": SamplingProfile(
        temperature=0.0,  # deterministic routing
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
    ),
}

QWEN_DEFAULT_PRESET_BY_MODE: Final[dict[GenerationMode, SamplingPresetName]] = {
    "thinking": "thinking_general",
    "nothinking": "nothinking_general",
    "orchestration": "orchestration",
}


def resolve_sampling_profile(
    mode: GenerationMode,
    preset: SamplingPresetName | None = None,
) -> SamplingProfile:
    resolved_preset = preset or QWEN_DEFAULT_PRESET_BY_MODE[mode]
    return QWEN_SAMPLING_PRESETS[resolved_preset]
