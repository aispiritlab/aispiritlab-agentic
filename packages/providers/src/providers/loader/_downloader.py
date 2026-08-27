"""Download and extract release archives."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import tarfile
import zipfile

import httpx
from structlog import get_logger

from providers.loader._github_release import ReleaseAsset

logger = get_logger(__name__)


class ChecksumMismatchError(Exception):
    """Raised when downloaded file checksum does not match expected digest."""


def _verify_digest(file_path: Path, expected_digest: str) -> bool:
    if not expected_digest:
        return True
    digest_value = expected_digest
    digest_value = digest_value.removeprefix("sha256:")
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == digest_value


def _extract_archive(archive_path: Path, dest_dir: Path) -> None:
    name = archive_path.name
    if name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(dest_dir, filter="data")
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    else:
        raise ValueError(f"Unsupported archive format: {name}")


def _make_executable(path: Path) -> None:
    current = path.stat().st_mode
    path.chmod(current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def download_asset(asset: ReleaseAsset, dest_dir: Path) -> Path:
    """Download a release asset, verify checksum, extract, and return the dest directory.

    Idempotent: skips download if archive already exists and checksum matches.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / asset.name

    if archive_path.exists() and _verify_digest(archive_path, asset.digest):
        logger.info("asset_already_downloaded", asset=asset.name)
    else:
        logger.info("downloading_asset", url=asset.download_url, size=asset.size)
        with (
            httpx.Client(timeout=300.0, follow_redirects=True) as client,
            client.stream("GET", asset.download_url) as response,
            archive_path.open("wb") as archive,
        ):
            response.raise_for_status()
            archive.writelines(response.iter_bytes(chunk_size=65536))

        if asset.digest and not _verify_digest(archive_path, asset.digest):
            archive_path.unlink()
            raise ChecksumMismatchError(
                f"Checksum mismatch for {asset.name}. Expected: {asset.digest}"
            )

    extract_dir = dest_dir / "extracted"
    if not extract_dir.exists():
        logger.info("extracting_archive", archive=asset.name)
        _extract_archive(archive_path, extract_dir)

    for path in extract_dir.rglob("*"):
        if path.is_file() and not os.access(path, os.X_OK):
            name_lower = path.name.lower()
            if "llama-server" in name_lower or "llama-cli" in name_lower:
                _make_executable(path)

    return extract_dir
