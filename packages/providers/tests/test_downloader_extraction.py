"""Archive extraction containment for downloaded release assets.

``download_asset`` unpacks a third-party archive into the model cache. Neither
format may write outside the destination directory, and the checksum gate must
actually reject a tampered download.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import stat
import tarfile
import zipfile

import pytest

from providers.loader._downloader import _extract_archive, _make_executable, _verify_digest


def _tar_with(members: list[tuple[tarfile.TarInfo, bytes | None]], path: Path) -> Path:
    with tarfile.open(path, "w:gz") as tf:
        for info, payload in members:
            tf.addfile(info, io.BytesIO(payload) if payload is not None else None)
    return path


def _file_info(name: str, payload: bytes) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    return info


# ---------------------------------------------------------------------------
# tar.gz
# ---------------------------------------------------------------------------


def test_tar_extracts_a_well_formed_archive(tmp_path: Path) -> None:
    archive = _tar_with([(_file_info("bin/llama-server", b"ELF"), b"ELF")], tmp_path / "a.tar.gz")
    dest = tmp_path / "dest"

    _extract_archive(archive, dest)

    assert (dest / "bin" / "llama-server").read_bytes() == b"ELF"


@pytest.mark.parametrize("name", ["../escape.txt", "../../escape.txt", "a/../../escape.txt"])
def test_tar_refuses_a_member_that_would_escape(tmp_path: Path, name: str) -> None:
    archive = _tar_with([(_file_info(name, b"pwned"), b"pwned")], tmp_path / "evil.tar.gz")
    dest = tmp_path / "dest"

    with pytest.raises(tarfile.FilterError):
        _extract_archive(archive, dest)

    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path.parent / "escape.txt").exists()


def test_tar_relocates_an_absolute_member_inside_the_destination(tmp_path: Path) -> None:
    # The "data" filter strips the leading separator rather than raising, so an
    # absolute member is rewritten as a relative one. It must still land inside
    # the destination and must not overwrite the real path it names.
    target = tmp_path / "absolute.txt"
    archive = _tar_with([(_file_info(str(target), b"pwned"), b"pwned")], tmp_path / "evil.tar.gz")
    dest = tmp_path / "dest"

    _extract_archive(archive, dest)

    assert not target.exists()
    written = [p for p in dest.rglob("*") if p.is_file()]
    assert written
    for path in written:
        assert path.resolve().is_relative_to(dest.resolve())


def test_tar_refuses_a_symlink_pointing_outside(tmp_path: Path) -> None:
    info = tarfile.TarInfo("link")
    info.type = tarfile.SYMTYPE
    info.linkname = str(tmp_path / "outside.txt")
    archive = _tar_with([(info, None)], tmp_path / "evil.tar.gz")
    dest = tmp_path / "dest"

    with pytest.raises(tarfile.FilterError):
        _extract_archive(archive, dest)

    assert not (dest / "link").exists()


def test_tar_refuses_a_special_file(tmp_path: Path) -> None:
    info = tarfile.TarInfo("dev/null")
    info.type = tarfile.CHRTYPE
    archive = _tar_with([(info, None)], tmp_path / "evil.tar.gz")

    with pytest.raises(tarfile.FilterError):
        _extract_archive(archive, tmp_path / "dest")


def test_tgz_is_treated_as_a_gzipped_tar(tmp_path: Path) -> None:
    archive = _tar_with([(_file_info("f.txt", b"ok"), b"ok")], tmp_path / "a.tgz")
    dest = tmp_path / "dest"

    _extract_archive(archive, dest)

    assert (dest / "f.txt").read_bytes() == b"ok"


# ---------------------------------------------------------------------------
# zip
# ---------------------------------------------------------------------------


def test_zip_extracts_a_well_formed_archive(tmp_path: Path) -> None:
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("bin/llama-cli", "ELF")
    dest = tmp_path / "dest"

    _extract_archive(archive, dest)

    assert (dest / "bin" / "llama-cli").read_text() == "ELF"


@pytest.mark.parametrize(
    "name",
    ["../escape.txt", "../../escape.txt", "/abs/escape.txt", "a/../../escape.txt"],
)
def test_zip_contains_every_member_inside_the_destination(tmp_path: Path, name: str) -> None:
    # CPython's ZipFile strips drive letters, leading separators and ".."
    # components, so a traversal member lands inside the destination instead of
    # escaping. This test exists so a future change of extraction routine
    # cannot silently give that up.
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(name, "pwned")
    dest = tmp_path / "dest"

    _extract_archive(archive, dest)

    written = [p for p in dest.rglob("*") if p.is_file()]
    assert written, "nothing was extracted"
    for path in written:
        assert path.resolve().is_relative_to(dest.resolve())
    assert not (tmp_path / "escape.txt").exists()
    assert not Path("/abs/escape.txt").exists()


def test_zip_does_not_create_symlinks(tmp_path: Path) -> None:
    # A symlink member is written as a regular file holding the target path,
    # so it cannot redirect a later write out of the destination.
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        info = zipfile.ZipInfo("link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "/etc/passwd")
    dest = tmp_path / "dest"

    _extract_archive(archive, dest)

    assert (dest / "link").is_file()
    assert not (dest / "link").is_symlink()


# ---------------------------------------------------------------------------
# Format dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["asset.rar", "asset.7z", "asset", "asset.tar", "asset.gz"])
def test_an_unsupported_format_is_rejected(tmp_path: Path, name: str) -> None:
    archive = tmp_path / name
    archive.write_bytes(b"whatever")

    with pytest.raises(ValueError, match="Unsupported archive format"):
        _extract_archive(archive, tmp_path / "dest")


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------


def test_a_matching_digest_verifies(tmp_path: Path) -> None:
    payload = b"release binary"
    asset = tmp_path / "asset.bin"
    asset.write_bytes(payload)

    assert _verify_digest(asset, hashlib.sha256(payload).hexdigest()) is True


def test_the_sha256_prefix_is_accepted(tmp_path: Path) -> None:
    payload = b"release binary"
    asset = tmp_path / "asset.bin"
    asset.write_bytes(payload)

    assert _verify_digest(asset, f"sha256:{hashlib.sha256(payload).hexdigest()}") is True


def test_a_tampered_file_fails_verification(tmp_path: Path) -> None:
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"release binary")
    expected = hashlib.sha256(b"release binary").hexdigest()
    asset.write_bytes(b"release binary + backdoor")

    assert _verify_digest(asset, expected) is False


def test_an_empty_digest_skips_verification(tmp_path: Path) -> None:
    # GitHub does not publish a digest for every asset.
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"anything")

    assert _verify_digest(asset, "") is True


def test_a_large_file_is_hashed_in_chunks(tmp_path: Path) -> None:
    payload = b"x" * (8192 * 3 + 17)
    asset = tmp_path / "big.bin"
    asset.write_bytes(payload)

    assert _verify_digest(asset, hashlib.sha256(payload).hexdigest()) is True


# ---------------------------------------------------------------------------
# Executable bits
# ---------------------------------------------------------------------------


def test_make_executable_adds_the_execute_bits_without_dropping_others(tmp_path: Path) -> None:
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"ELF")
    binary.chmod(0o644)

    _make_executable(binary)

    mode = binary.stat().st_mode
    assert mode & stat.S_IXUSR
    assert mode & stat.S_IXGRP
    assert mode & stat.S_IXOTH
    assert mode & stat.S_IRUSR
    assert mode & stat.S_IWUSR
