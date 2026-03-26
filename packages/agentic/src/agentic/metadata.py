from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Description:
    agent_name: str
    description: str
    capabilities: tuple[str, ...]

