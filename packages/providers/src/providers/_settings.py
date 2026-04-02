"""Provider settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _default_bin_dir() -> Path:
    return Path.home() / ".cache" / "ai-spirit" / "providers" / "bin"


@dataclass(frozen=True, slots=True)
class ProviderSettings:
    bin_dir: Path = field(default_factory=_default_bin_dir)
    default_port: int = 8080
    health_check_timeout: float = 30.0
    health_check_interval: float = 0.5
    server_startup_timeout: float = 60.0
