"""Tests for the CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from providers._cli import run


class TestCliList:
    def test_list_shows_providers(self, capsys: pytest.CaptureFixture[str]) -> None:
        run(["list"])
        output = capsys.readouterr().out
        assert "llama-cpp" in output
        assert "ggml-org/llama.cpp" in output

    def test_default_command_is_list(self, capsys: pytest.CaptureFixture[str]) -> None:
        run([])
        output = capsys.readouterr().out
        assert "Available providers" in output


class TestCliDownload:
    def test_download_calls_loader(self) -> None:
        mock_loader = MagicMock()
        mock_loader.download.return_value = Path("/tmp/llama-server")
        mock_loader.is_downloaded.return_value = False
        mock_loader.repo = "ggml-org/llama.cpp"

        with patch("providers._cli.LOADER_REGISTRY", {"llama-cpp": mock_loader}):
            run(["llama-cpp", "download"])

        mock_loader.download.assert_called_once()

    def test_download_with_release_tag(self) -> None:
        mock_loader = MagicMock()
        mock_loader.download.return_value = Path("/tmp/llama-server")
        mock_loader.is_downloaded.return_value = False
        mock_loader.repo = "ggml-org/llama.cpp"

        with patch("providers._cli.LOADER_REGISTRY", {"llama-cpp": mock_loader}):
            run(["llama-cpp", "download", "--release", "b8628"])

        call_kwargs = mock_loader.download.call_args
        assert call_kwargs.kwargs["release_tag"] == "b8628"

    def test_download_unknown_provider_exits(self) -> None:
        with pytest.raises(SystemExit):
            run(["unknown-provider", "download"])


class TestCliStatus:
    def test_status_no_server(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        with patch("providers._cli.ProviderSettings", return_value=MagicMock(bin_dir=tmp_path)):
            run(["llama-cpp", "status"])
        output = capsys.readouterr().out
        assert "No running server" in output

    def test_status_with_stale_pid(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        pid_dir = tmp_path / "llama-cpp"
        pid_dir.mkdir()
        (pid_dir / "llama-server.pid").write_text("999999")

        with patch("providers._cli.ProviderSettings", return_value=MagicMock(bin_dir=tmp_path)):
            run(["llama-cpp", "status"])
        output = capsys.readouterr().out
        assert "stale PID" in output or "not found" in output


class TestCliStop:
    def test_stop_no_server_exits(self, tmp_path: Path) -> None:
        with (
            patch("providers._cli.ProviderSettings", return_value=MagicMock(bin_dir=tmp_path)),
            pytest.raises(SystemExit),
        ):
            run(["llama-cpp", "stop"])
