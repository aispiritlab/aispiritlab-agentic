"""Platform detection for binary downloads."""

from __future__ import annotations

from dataclasses import dataclass
import platform


@dataclass(frozen=True, slots=True)
class PlatformInfo:
    os: str
    arch: str


_ARCH_MAP: dict[str, str] = {
    "x86_64": "x64",
    "amd64": "x64",
    "aarch64": "arm64",
    "arm64": "arm64",
}


def detect_platform() -> PlatformInfo:
    raw_os = platform.system().lower()
    if raw_os == "darwin":
        os_name = "macos"
    elif raw_os == "linux":
        os_name = "ubuntu"
    elif raw_os == "windows":
        os_name = "win"
    else:
        os_name = raw_os

    raw_arch = platform.machine().lower()
    arch = _ARCH_MAP.get(raw_arch, raw_arch)

    return PlatformInfo(os=os_name, arch=arch)
