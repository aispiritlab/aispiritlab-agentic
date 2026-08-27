"""Inference server management via subprocess."""

from __future__ import annotations

import asyncio
import atexit
from pathlib import Path
from types import TracebackType
from typing import Self

from structlog import get_logger

from providers._settings import ProviderSettings
from providers.server._health import ServerStartupTimeoutError, wait_for_healthy
from providers.server._port import find_free_port
from providers.server._process import (
    ServerConfig,
    remove_pid_file,
    start_process,
    stop_process,
    write_pid_file,
)

logger = get_logger(__name__)

__all__ = [
    "InferenceServer",
    "ServerConfig",
    "ServerStartupTimeoutError",
]


class InferenceServer:
    """Manages a llama-server (or compatible) subprocess."""

    def __init__(
        self,
        binary_path: Path,
        config: ServerConfig,
        settings: ProviderSettings | None = None,
    ) -> None:
        self._binary_path = binary_path
        self._config = config
        self._settings = settings or ProviderSettings()
        self._process: asyncio.subprocess.Process | None = None
        self._port: int | None = config.port
        self._pid_path: Path | None = None

    @property
    def base_url(self) -> str:
        if self._port is None:
            raise RuntimeError("Server has not been started yet")
        return f"http://127.0.0.1:{self._port}"

    @property
    def port(self) -> int:
        if self._port is None:
            raise RuntimeError("Server has not been started yet")
        return self._port

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def start(self) -> None:
        if self._port is None:
            self._port = find_free_port(self._settings.default_port)

        log_file = self._settings.bin_dir / "llama-cpp" / "llama-server.log"
        self._pid_path = self._settings.bin_dir / "llama-cpp" / "llama-server.pid"

        try:
            self._process = await start_process(
                self._binary_path,
                self._config,
                self._port,
                log_file=log_file,
            )

            if self._process.pid is not None:
                write_pid_file(self._process.pid, self._pid_path)

            await wait_for_healthy(
                self.base_url,
                timeout=self._settings.server_startup_timeout,
                interval=self._settings.health_check_interval,
                request_timeout=self._settings.health_check_timeout,
            )
        except BaseException:
            await self._cleanup_failed_start()
            raise

        atexit.register(self._sync_stop)
        logger.info("server_started", port=self._port, pid=self._process.pid)

    async def stop(self, timeout: float = 10.0) -> None:
        if self._process is not None:
            await stop_process(self._process, timeout=timeout)
            self._process = None
        if self._pid_path is not None:
            remove_pid_file(self._pid_path)
        logger.info("server_stopped", port=self._port)

    async def _cleanup_failed_start(self) -> None:
        if self._process is not None:
            await stop_process(self._process, timeout=1.0)
            self._process = None
        if self._pid_path is not None:
            remove_pid_file(self._pid_path)

    def _sync_stop(self) -> None:
        if self._process is not None and self._process.returncode is None:
            import os
            import signal
            import sys

            pid = self._process.pid
            if pid is not None:
                try:
                    if sys.platform == "win32":
                        os.kill(pid, signal.SIGTERM)
                    else:
                        os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        if self._pid_path is not None:
            remove_pid_file(self._pid_path)

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.stop()
