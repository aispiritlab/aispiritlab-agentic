"""Loader layer — download and manage provider binaries."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from providers._settings import ProviderSettings
from providers.loader.llama_cpp import LlamaCppLoader

__all__ = [
    "LOADER_REGISTRY",
    "LlamaCppLoader",
    "LoaderProvider",
]


class LoaderProvider(Protocol):
    name: str
    repo: str

    def download(self, settings: ProviderSettings, *, release_tag: str | None = None) -> Path: ...
    def is_downloaded(self, settings: ProviderSettings) -> bool: ...
    def get_binary_path(self, settings: ProviderSettings) -> Path | None: ...


LOADER_REGISTRY: dict[str, LoaderProvider] = {
    "llama-cpp": LlamaCppLoader(),
}
