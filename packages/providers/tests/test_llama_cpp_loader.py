"""Tests for the LlamaCpp loader."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import respx

from providers._platform import PlatformInfo
from providers._settings import ProviderSettings
from providers.loader._github_release import Release, ReleaseAsset
from providers.loader.llama_cpp import LlamaCppLoader


def _create_tar_gz_with_server() -> bytes:
    buf = io.BytesIO()
    content = b"fake-llama-server-binary"
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="build/bin/llama-server")
        info.size = len(content)
        tf.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def _make_release(tag: str = "b8629") -> Release:
    archive_bytes = _create_tar_gz_with_server()
    digest = f"sha256:{hashlib.sha256(archive_bytes).hexdigest()}"
    return Release(
        tag=tag,
        assets=(
            ReleaseAsset(
                name=f"llama-{tag}-bin-macos-arm64.tar.gz",
                download_url=f"https://example.com/llama-{tag}-bin-macos-arm64.tar.gz",
                size=len(archive_bytes),
                digest=digest,
            ),
            ReleaseAsset(
                name=f"llama-{tag}-bin-ubuntu-x64.tar.gz",
                download_url=f"https://example.com/llama-{tag}-bin-ubuntu-x64.tar.gz",
                size=len(archive_bytes),
                digest=digest,
            ),
        ),
    ), archive_bytes


class TestLlamaCppLoader:
    def test_download_fetches_latest_and_extracts(self, tmp_path: Path) -> None:
        release, archive_bytes = _make_release()
        settings = ProviderSettings(bin_dir=tmp_path)
        platform = PlatformInfo(os="macos", arch="arm64")
        loader = LlamaCppLoader()

        with (
            patch.object(loader, "_resolve_release", return_value=release),
            respx.mock,
        ):
            respx.get("https://example.com/llama-b8629-bin-macos-arm64.tar.gz").mock(
                return_value=httpx.Response(200, content=archive_bytes)
            )
            binary_path = loader.download(settings, platform_info=platform)

        assert binary_path.name == "llama-server"
        assert binary_path.exists()

    def test_download_with_pinned_release(self, tmp_path: Path) -> None:
        release, archive_bytes = _make_release(tag="b8628")
        settings = ProviderSettings(bin_dir=tmp_path)
        platform = PlatformInfo(os="ubuntu", arch="x64")
        loader = LlamaCppLoader()

        with (
            patch(
                "providers.loader.llama_cpp.fetch_release", return_value=release
            ),
            respx.mock,
        ):
            respx.get("https://example.com/llama-b8628-bin-ubuntu-x64.tar.gz").mock(
                return_value=httpx.Response(200, content=archive_bytes)
            )
            binary_path = loader.download(settings, release_tag="b8628", platform_info=platform)

        assert binary_path.name == "llama-server"

    def test_unsupported_platform_raises(self, tmp_path: Path) -> None:
        settings = ProviderSettings(bin_dir=tmp_path)
        platform = PlatformInfo(os="freebsd", arch="riscv64")
        loader = LlamaCppLoader()

        release, _ = _make_release()
        with patch.object(loader, "_resolve_release", return_value=release):
            with pytest.raises(RuntimeError, match="No llama.cpp binary available"):
                loader.download(settings, platform_info=platform)

    def test_no_matching_asset_raises(self, tmp_path: Path) -> None:
        release = Release(tag="b8629", assets=())
        settings = ProviderSettings(bin_dir=tmp_path)
        platform = PlatformInfo(os="macos", arch="arm64")
        loader = LlamaCppLoader()

        with patch.object(loader, "_resolve_release", return_value=release):
            with pytest.raises(RuntimeError, match="No matching asset"):
                loader.download(settings, platform_info=platform)

    def test_is_downloaded_false_when_empty(self, tmp_path: Path) -> None:
        settings = ProviderSettings(bin_dir=tmp_path)
        loader = LlamaCppLoader()
        assert not loader.is_downloaded(settings)

    def test_is_downloaded_true_after_download(self, tmp_path: Path) -> None:
        release, archive_bytes = _make_release()
        settings = ProviderSettings(bin_dir=tmp_path)
        platform = PlatformInfo(os="macos", arch="arm64")
        loader = LlamaCppLoader()

        with (
            patch.object(loader, "_resolve_release", return_value=release),
            respx.mock,
        ):
            respx.get("https://example.com/llama-b8629-bin-macos-arm64.tar.gz").mock(
                return_value=httpx.Response(200, content=archive_bytes)
            )
            loader.download(settings, platform_info=platform)

        assert loader.is_downloaded(settings)

    def test_get_binary_path_returns_none_when_not_downloaded(self, tmp_path: Path) -> None:
        settings = ProviderSettings(bin_dir=tmp_path)
        loader = LlamaCppLoader()
        assert loader.get_binary_path(settings) is None

    def test_get_binary_path_returns_path_after_download(self, tmp_path: Path) -> None:
        release, archive_bytes = _make_release()
        settings = ProviderSettings(bin_dir=tmp_path)
        platform = PlatformInfo(os="macos", arch="arm64")
        loader = LlamaCppLoader()

        with (
            patch.object(loader, "_resolve_release", return_value=release),
            respx.mock,
        ):
            respx.get("https://example.com/llama-b8629-bin-macos-arm64.tar.gz").mock(
                return_value=httpx.Response(200, content=archive_bytes)
            )
            loader.download(settings, platform_info=platform)

        binary = loader.get_binary_path(settings)
        assert binary is not None
        assert binary.name == "llama-server"
