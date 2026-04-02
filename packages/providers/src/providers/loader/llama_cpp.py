"""LlamaCpp loader — downloads llama.cpp binaries from GitHub releases."""

from __future__ import annotations

from pathlib import Path

from structlog import get_logger

from providers._platform import PlatformInfo, detect_platform
from providers._settings import ProviderSettings
from providers.loader._downloader import download_asset
from providers.loader._github_release import (
    Release,
    fetch_latest_release,
    fetch_release,
)

logger = get_logger(__name__)

_ASSET_PATTERNS: dict[tuple[str, str], str] = {
    ("macos", "arm64"): "bin-macos-arm64",
    ("macos", "x64"): "bin-macos-x64",
    ("ubuntu", "x64"): "bin-ubuntu-x64",
    ("ubuntu", "arm64"): "bin-ubuntu-arm64",
    ("win", "x64"): "bin-win-x64",
    ("win", "arm64"): "bin-win-arm64",
}

_SERVER_BINARY = "llama-server"


class LlamaCppLoader:
    name = "llama-cpp"
    repo = "ggml-org/llama.cpp"

    def _resolve_release(self, release_tag: str | None = None) -> Release:
        if release_tag:
            return fetch_release(self.repo, release_tag)
        return fetch_latest_release(self.repo)

    def _resolve_asset_pattern(self, platform: PlatformInfo) -> str:
        key = (platform.os, platform.arch)
        pattern = _ASSET_PATTERNS.get(key)
        if pattern is None:
            raise RuntimeError(
                f"No llama.cpp binary available for {platform.os}/{platform.arch}"
            )
        return pattern

    def _find_server_binary(self, extract_dir: Path) -> Path:
        for path in extract_dir.rglob("*"):
            if path.is_file() and path.name == _SERVER_BINARY:
                return path
        for path in extract_dir.rglob("*"):
            if path.is_file() and path.name == f"{_SERVER_BINARY}.exe":
                return path
        raise FileNotFoundError(
            f"Could not find {_SERVER_BINARY} binary in extracted files"
        )

    def download(
        self,
        settings: ProviderSettings,
        *,
        release_tag: str | None = None,
        platform_info: PlatformInfo | None = None,
    ) -> Path:
        """Download llama.cpp server binary. Returns path to the llama-server binary."""
        platform = platform_info or detect_platform()
        release = self._resolve_release(release_tag)
        asset_pattern = self._resolve_asset_pattern(platform)

        asset = release.find_asset(asset_pattern)
        if asset is None:
            raise RuntimeError(
                f"No matching asset for pattern '{asset_pattern}' in release {release.tag}. "
                f"Available: {[a.name for a in release.assets]}"
            )

        dest_dir = settings.bin_dir / self.name / release.tag
        logger.info(
            "downloading_llama_cpp",
            release=release.tag,
            asset=asset.name,
            dest=str(dest_dir),
        )
        extract_dir = download_asset(asset, dest_dir)
        return self._find_server_binary(extract_dir)

    def is_downloaded(self, settings: ProviderSettings) -> bool:
        provider_dir = settings.bin_dir / self.name
        if not provider_dir.exists():
            return False
        for tag_dir in provider_dir.iterdir():
            extract_dir = tag_dir / "extracted"
            if extract_dir.exists():
                try:
                    self._find_server_binary(extract_dir)
                    return True
                except FileNotFoundError:
                    continue
        return False

    def get_binary_path(self, settings: ProviderSettings) -> Path | None:
        """Return path to the already-downloaded llama-server binary, or None."""
        provider_dir = settings.bin_dir / self.name
        if not provider_dir.exists():
            return None
        for tag_dir in sorted(provider_dir.iterdir(), reverse=True):
            extract_dir = tag_dir / "extracted"
            if extract_dir.exists():
                try:
                    return self._find_server_binary(extract_dir)
                except FileNotFoundError:
                    continue
        return None
