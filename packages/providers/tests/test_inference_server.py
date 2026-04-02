"""Tests for the InferenceServer."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers._settings import ProviderSettings
from providers.server import InferenceServer, ServerConfig


def _make_mock_process(pid: int = 12345) -> MagicMock:
    process = MagicMock()
    process.pid = pid
    process.returncode = None
    process.wait = AsyncMock(return_value=0)
    process.send_signal = MagicMock()
    process.kill = MagicMock()
    return process


@pytest.mark.asyncio(loop_scope="function")
class TestInferenceServer:
    async def test_start_and_stop(self, tmp_path: Path) -> None:
        settings = ProviderSettings(bin_dir=tmp_path, server_startup_timeout=2.0)
        config = ServerConfig(model_path="model.gguf", port=49500)
        binary = tmp_path / "llama-server"
        binary.touch()

        mock_process = _make_mock_process()

        with (
            patch("providers.server._process.asyncio.create_subprocess_exec", return_value=mock_process),
            patch("providers.server.wait_for_healthy", new_callable=AsyncMock),
        ):
            server = InferenceServer(binary, config, settings)
            await server.start()

            assert server.is_running
            assert server.port == 49500
            assert server.base_url == "http://127.0.0.1:49500"

            mock_process.returncode = 0
            await server.stop()
            assert not server.is_running

    async def test_context_manager(self, tmp_path: Path) -> None:
        settings = ProviderSettings(bin_dir=tmp_path, server_startup_timeout=2.0)
        config = ServerConfig(model_path="model.gguf", port=49501)
        binary = tmp_path / "llama-server"
        binary.touch()

        mock_process = _make_mock_process()

        with (
            patch("providers.server._process.asyncio.create_subprocess_exec", return_value=mock_process),
            patch("providers.server.wait_for_healthy", new_callable=AsyncMock),
        ):
            async with InferenceServer(binary, config, settings) as server:
                assert server.is_running
                assert server.port == 49501
                mock_process.returncode = 0

    async def test_auto_port_allocation(self, tmp_path: Path) -> None:
        settings = ProviderSettings(bin_dir=tmp_path, default_port=49510, server_startup_timeout=2.0)
        config = ServerConfig(model_path="model.gguf")
        binary = tmp_path / "llama-server"
        binary.touch()

        mock_process = _make_mock_process()

        with (
            patch("providers.server._process.asyncio.create_subprocess_exec", return_value=mock_process),
            patch("providers.server.wait_for_healthy", new_callable=AsyncMock),
        ):
            server = InferenceServer(binary, config, settings)
            await server.start()
            assert server.port >= 49510
            mock_process.returncode = 0
            await server.stop()

    async def test_pid_file_created_and_removed(self, tmp_path: Path) -> None:
        settings = ProviderSettings(bin_dir=tmp_path, server_startup_timeout=2.0)
        config = ServerConfig(model_path="model.gguf", port=49520)
        binary = tmp_path / "llama-server"
        binary.touch()

        mock_process = _make_mock_process(pid=99999)

        with (
            patch("providers.server._process.asyncio.create_subprocess_exec", return_value=mock_process),
            patch("providers.server.wait_for_healthy", new_callable=AsyncMock),
        ):
            server = InferenceServer(binary, config, settings)
            await server.start()

            pid_path = tmp_path / "llama-cpp" / "llama-server.pid"
            assert pid_path.exists()
            assert pid_path.read_text().strip() == "99999"

            mock_process.returncode = 0
            await server.stop()
            assert not pid_path.exists()

    async def test_failed_start_cleans_up_process_and_pid_file(self, tmp_path: Path) -> None:
        settings = ProviderSettings(bin_dir=tmp_path, server_startup_timeout=2.0)
        config = ServerConfig(model_path="model.gguf", port=49521)
        binary = tmp_path / "llama-server"
        binary.touch()

        mock_process = _make_mock_process(pid=54321)

        with (
            patch("providers.server._process.asyncio.create_subprocess_exec", return_value=mock_process),
            patch("providers.server.wait_for_healthy", new_callable=AsyncMock, side_effect=RuntimeError("boom")),
        ):
            server = InferenceServer(binary, config, settings)
            with pytest.raises(RuntimeError, match="boom"):
                await server.start()

        pid_path = tmp_path / "llama-cpp" / "llama-server.pid"
        mock_process.send_signal.assert_called_once()
        assert not pid_path.exists()
        assert not server.is_running


class TestInferenceServerProperties:
    def test_base_url_raises_before_start(self, tmp_path: Path) -> None:
        config = ServerConfig(model_path="model.gguf")
        server = InferenceServer(tmp_path / "llama-server", config)
        with pytest.raises(RuntimeError, match="not been started"):
            _ = server.base_url

    def test_port_raises_before_start(self, tmp_path: Path) -> None:
        config = ServerConfig(model_path="model.gguf")
        server = InferenceServer(tmp_path / "llama-server", config)
        with pytest.raises(RuntimeError, match="not been started"):
            _ = server.port

    def test_is_running_false_before_start(self, tmp_path: Path) -> None:
        config = ServerConfig(model_path="model.gguf")
        server = InferenceServer(tmp_path / "llama-server", config)
        assert not server.is_running
