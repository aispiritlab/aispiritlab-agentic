"""Subprocess lifecycle management for inference servers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import signal
import sys
from typing import TextIO

from structlog import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ServerConfig:
    model_path: str
    port: int | None = None
    context_size: int = 4096
    n_gpu_layers: int = -1
    use_hf: bool = False
    extra_args: tuple[str, ...] = ()


def build_command(binary_path: Path, config: ServerConfig, port: int) -> list[str]:
    cmd = [str(binary_path)]

    if config.use_hf:
        cmd.extend(["-hf", config.model_path])
    else:
        cmd.extend(["-m", config.model_path])

    cmd.extend(["--port", str(port)])
    cmd.extend(["-c", str(config.context_size)])
    cmd.extend(["-ngl", str(config.n_gpu_layers)])
    cmd.extend(list(config.extra_args))

    return cmd


def _open_log_file(log_file: Path) -> TextIO:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    return log_file.open("w", encoding="utf-8")


async def start_process(
    binary_path: Path,
    config: ServerConfig,
    port: int,
    log_file: Path | None = None,
) -> asyncio.subprocess.Process:
    cmd = build_command(binary_path, config, port)
    logger.info("starting_server", cmd=" ".join(cmd))

    opened_file = None
    if log_file:
        # Filesystem calls block the event loop, so they run on a worker thread.
        opened_file = await asyncio.to_thread(_open_log_file, log_file)
        stdout_file = opened_file
        stderr_file = stdout_file
    else:
        stdout_file = asyncio.subprocess.DEVNULL  # type: ignore[assignment]
        stderr_file = asyncio.subprocess.DEVNULL  # type: ignore[assignment]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=stdout_file,
            stderr=stderr_file,
        )
    finally:
        if opened_file is not None:
            opened_file.close()
    return process


async def stop_process(process: asyncio.subprocess.Process, timeout: float = 10.0) -> None:
    if process.returncode is not None:
        return

    logger.info("stopping_server", pid=process.pid)

    if sys.platform == "win32":
        process.terminate()
    else:
        process.send_signal(signal.SIGTERM)

    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
        logger.info("server_stopped_gracefully", pid=process.pid)
    except TimeoutError:
        logger.warning("server_force_killing", pid=process.pid)
        process.kill()
        await process.wait()


def write_pid_file(pid: int, pid_path: Path) -> None:
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(pid))


def read_pid_file(pid_path: Path) -> int | None:
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text().strip())
    except ValueError, OSError:
        return None


def remove_pid_file(pid_path: Path) -> None:
    pid_path.unlink(missing_ok=True)
