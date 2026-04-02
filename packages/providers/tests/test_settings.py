"""Tests for provider settings."""

from __future__ import annotations

from pathlib import Path

from providers._settings import ProviderSettings


def test_default_settings() -> None:
    settings = ProviderSettings()
    assert settings.bin_dir == Path.home() / ".cache" / "ai-spirit" / "providers" / "bin"
    assert settings.default_port == 8080
    assert settings.health_check_timeout == 30.0
    assert settings.health_check_interval == 0.5
    assert settings.server_startup_timeout == 60.0


def test_custom_bin_dir(tmp_path: Path) -> None:
    settings = ProviderSettings(bin_dir=tmp_path)
    assert settings.bin_dir == tmp_path


def test_settings_is_frozen() -> None:
    settings = ProviderSettings()
    try:
        settings.default_port = 9090  # type: ignore[misc]
        raise AssertionError("Should have raised FrozenInstanceError")
    except AttributeError:
        pass
