"""Tests for platform detection."""

from __future__ import annotations

from unittest.mock import patch

from providers._platform import PlatformInfo, detect_platform


def test_detect_platform_returns_platform_info() -> None:
    result = detect_platform()
    assert isinstance(result, PlatformInfo)
    assert result.os in ("macos", "ubuntu", "win")
    assert result.arch in ("arm64", "x64")


def test_detect_platform_darwin_arm64() -> None:
    with (
        patch("providers._platform.platform.system", return_value="Darwin"),
        patch("providers._platform.platform.machine", return_value="arm64"),
    ):
        result = detect_platform()
    assert result == PlatformInfo(os="macos", arch="arm64")


def test_detect_platform_linux_x86_64() -> None:
    with (
        patch("providers._platform.platform.system", return_value="Linux"),
        patch("providers._platform.platform.machine", return_value="x86_64"),
    ):
        result = detect_platform()
    assert result == PlatformInfo(os="ubuntu", arch="x64")


def test_detect_platform_windows_amd64() -> None:
    with (
        patch("providers._platform.platform.system", return_value="Windows"),
        patch("providers._platform.platform.machine", return_value="AMD64"),
    ):
        result = detect_platform()
    assert result == PlatformInfo(os="win", arch="x64")


def test_detect_platform_linux_aarch64() -> None:
    with (
        patch("providers._platform.platform.system", return_value="Linux"),
        patch("providers._platform.platform.machine", return_value="aarch64"),
    ):
        result = detect_platform()
    assert result == PlatformInfo(os="ubuntu", arch="arm64")


def test_platform_info_is_frozen() -> None:
    info = PlatformInfo(os="macos", arch="arm64")
    try:
        info.os = "linux"  # type: ignore[misc]
        raise AssertionError("Should have raised FrozenInstanceError")
    except AttributeError:
        pass
