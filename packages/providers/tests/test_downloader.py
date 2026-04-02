"""Tests for the download and extraction logic."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import httpx
import pytest
import respx

from providers.loader._downloader import (
    ChecksumMismatchError,
    download_asset,
)
from providers.loader._github_release import ReleaseAsset


def _create_tar_gz_with_binary(binary_name: str = "llama-server", content: bytes = b"fake-binary") -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name=f"bin/{binary_name}")
        info.size = len(content)
        tf.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TestDownloadAsset:
    def test_downloads_and_extracts(self, tmp_path: Path) -> None:
        archive_bytes = _create_tar_gz_with_binary()
        digest = f"sha256:{_sha256_hex(archive_bytes)}"

        asset = ReleaseAsset(
            name="llama-test.tar.gz",
            download_url="https://example.com/llama-test.tar.gz",
            size=len(archive_bytes),
            digest=digest,
        )

        with respx.mock:
            respx.get("https://example.com/llama-test.tar.gz").mock(
                return_value=httpx.Response(200, content=archive_bytes)
            )
            extract_dir = download_asset(asset, tmp_path)

        assert extract_dir.exists()
        assert (extract_dir / "bin" / "llama-server").exists()

    def test_skips_download_if_archive_exists(self, tmp_path: Path) -> None:
        archive_bytes = _create_tar_gz_with_binary()
        digest = f"sha256:{_sha256_hex(archive_bytes)}"

        asset = ReleaseAsset(
            name="llama-test.tar.gz",
            download_url="https://example.com/llama-test.tar.gz",
            size=len(archive_bytes),
            digest=digest,
        )

        archive_path = tmp_path / asset.name
        archive_path.write_bytes(archive_bytes)

        with respx.mock:
            route = respx.get("https://example.com/llama-test.tar.gz")
            extract_dir = download_asset(asset, tmp_path)

        assert not route.called
        assert extract_dir.exists()

    def test_checksum_mismatch_raises(self, tmp_path: Path) -> None:
        archive_bytes = _create_tar_gz_with_binary()

        asset = ReleaseAsset(
            name="llama-bad.tar.gz",
            download_url="https://example.com/llama-bad.tar.gz",
            size=len(archive_bytes),
            digest="sha256:0000000000000000000000000000000000000000000000000000000000000000",
        )

        with respx.mock:
            respx.get("https://example.com/llama-bad.tar.gz").mock(
                return_value=httpx.Response(200, content=archive_bytes)
            )
            with pytest.raises(ChecksumMismatchError, match="Checksum mismatch"):
                download_asset(asset, tmp_path)

        assert not (tmp_path / asset.name).exists()

    def test_no_digest_skips_verification(self, tmp_path: Path) -> None:
        archive_bytes = _create_tar_gz_with_binary()

        asset = ReleaseAsset(
            name="llama-nodigest.tar.gz",
            download_url="https://example.com/llama-nodigest.tar.gz",
            size=len(archive_bytes),
            digest="",
        )

        with respx.mock:
            respx.get("https://example.com/llama-nodigest.tar.gz").mock(
                return_value=httpx.Response(200, content=archive_bytes)
            )
            extract_dir = download_asset(asset, tmp_path)

        assert extract_dir.exists()
