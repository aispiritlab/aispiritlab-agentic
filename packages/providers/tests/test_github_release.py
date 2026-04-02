"""Tests for the GitHub Releases API client."""

from __future__ import annotations

import httpx
import pytest
import respx

from providers.loader._github_release import (
    Release,
    ReleaseAsset,
    ReleaseNotFoundError,
    fetch_latest_release,
    fetch_release,
)

_SAMPLE_RELEASE_JSON = {
    "tag_name": "b8629",
    "assets": [
        {
            "name": "llama-b8629-bin-macos-arm64.tar.gz",
            "browser_download_url": "https://github.com/ggml-org/llama.cpp/releases/download/b8629/llama-b8629-bin-macos-arm64.tar.gz",
            "size": 12345678,
            "digest": "sha256:abc123",
        },
        {
            "name": "llama-b8629-bin-ubuntu-x64.tar.gz",
            "browser_download_url": "https://github.com/ggml-org/llama.cpp/releases/download/b8629/llama-b8629-bin-ubuntu-x64.tar.gz",
            "size": 9876543,
            "digest": "sha256:def456",
        },
    ],
}


class TestFetchLatestRelease:
    def test_parses_release_correctly(self) -> None:
        with respx.mock:
            respx.get("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest").mock(
                return_value=httpx.Response(200, json=_SAMPLE_RELEASE_JSON)
            )
            release = fetch_latest_release("ggml-org/llama.cpp")

        assert isinstance(release, Release)
        assert release.tag == "b8629"
        assert len(release.assets) == 2

        macos_asset = release.assets[0]
        assert isinstance(macos_asset, ReleaseAsset)
        assert macos_asset.name == "llama-b8629-bin-macos-arm64.tar.gz"
        assert macos_asset.size == 12345678
        assert macos_asset.digest == "sha256:abc123"

    def test_find_asset_by_pattern(self) -> None:
        with respx.mock:
            respx.get("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest").mock(
                return_value=httpx.Response(200, json=_SAMPLE_RELEASE_JSON)
            )
            release = fetch_latest_release("ggml-org/llama.cpp")

        asset = release.find_asset("bin-macos-arm64")
        assert asset is not None
        assert "macos-arm64" in asset.name

        missing = release.find_asset("bin-windows-arm64")
        assert missing is None

    def test_404_raises_release_not_found(self) -> None:
        with respx.mock:
            respx.get("https://api.github.com/repos/foo/bar/releases/latest").mock(
                return_value=httpx.Response(404)
            )
            with pytest.raises(ReleaseNotFoundError, match="No releases found"):
                fetch_latest_release("foo/bar")


class TestFetchRelease:
    def test_fetches_specific_tag(self) -> None:
        with respx.mock:
            respx.get("https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b8629").mock(
                return_value=httpx.Response(200, json=_SAMPLE_RELEASE_JSON)
            )
            release = fetch_release("ggml-org/llama.cpp", "b8629")

        assert release.tag == "b8629"
        assert len(release.assets) == 2

    def test_404_raises_release_not_found(self) -> None:
        with respx.mock:
            respx.get("https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/v999").mock(
                return_value=httpx.Response(404)
            )
            with pytest.raises(ReleaseNotFoundError, match="Release v999 not found"):
                fetch_release("ggml-org/llama.cpp", "v999")


class TestReleaseAsset:
    def test_is_frozen(self) -> None:
        asset = ReleaseAsset(name="test", download_url="http://example.com", size=0, digest="")
        with pytest.raises(AttributeError):
            asset.name = "changed"  # type: ignore[misc]
