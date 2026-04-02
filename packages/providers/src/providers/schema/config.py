"""Inference configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    presence_penalty: float = 0.0
