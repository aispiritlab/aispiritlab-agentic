"""CLI entry point for the providers package."""

from __future__ import annotations

import argparse
import asyncio
import sys

from structlog import get_logger

from providers._settings import ProviderSettings
from providers.loader import LOADER_REGISTRY
from providers.server import InferenceServer, ServerConfig
from providers.server._process import read_pid_file, remove_pid_file

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="providers",
        description="Manage inference provider binaries and servers",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="List available providers")

    provider_parser = subparsers.add_parser("provider", help="Provider-specific commands")
    provider_parser.add_argument("name", help="Provider name (e.g. llama-cpp)")

    action_subparsers = provider_parser.add_subparsers(dest="action")

    download_parser = action_subparsers.add_parser("download", help="Download provider binary")
    download_parser.add_argument("--release", default=None, help="Pin to a specific release tag")

    start_parser = action_subparsers.add_parser("start", help="Start inference server")
    start_parser.add_argument("--model", required=True, help="Model path or HuggingFace repo ID")
    start_parser.add_argument("--port", type=int, default=None, help="Server port")
    start_parser.add_argument("--context-size", type=int, default=4096, help="Context size")
    start_parser.add_argument("--n-gpu-layers", type=int, default=-1, help="GPU layers (-1 = all)")
    start_parser.add_argument("--hf", action="store_true", help="Use HuggingFace model download")

    action_subparsers.add_parser("stop", help="Stop running server")
    action_subparsers.add_parser("status", help="Check server status")

    return parser


def _parse_provider_args(argv: list[str]) -> argparse.Namespace:
    """Parse argv, treating 'providers <name> <action>' as 'providers provider <name> <action>'."""
    if len(argv) >= 2 and argv[0] in LOADER_REGISTRY:
        argv = ["provider"] + argv
    parser = _build_parser()
    return parser.parse_args(argv)


def _cmd_list() -> None:
    print("Available providers:")
    for name, loader in LOADER_REGISTRY.items():
        settings = ProviderSettings()
        status = "downloaded" if loader.is_downloaded(settings) else "not downloaded"
        print(f"  {name} ({loader.repo}) [{status}]")


def _cmd_download(name: str, release_tag: str | None) -> None:
    loader = LOADER_REGISTRY.get(name)
    if loader is None:
        print(f"Unknown provider: {name}")
        print(f"Available: {', '.join(LOADER_REGISTRY)}")
        sys.exit(1)

    settings = ProviderSettings()
    binary_path = loader.download(settings, release_tag=release_tag)
    print(f"Downloaded: {binary_path}")


def _cmd_start(name: str, args: argparse.Namespace) -> None:
    loader = LOADER_REGISTRY.get(name)
    if loader is None:
        print(f"Unknown provider: {name}")
        sys.exit(1)

    settings = ProviderSettings()
    binary_path = loader.get_binary_path(settings)
    if binary_path is None:
        print(f"Provider {name} not downloaded. Run: providers {name} download")
        sys.exit(1)

    config = ServerConfig(
        model_path=args.model,
        port=args.port,
        context_size=args.context_size,
        n_gpu_layers=args.n_gpu_layers,
        use_hf=args.hf,
    )

    async def _run() -> None:
        server = InferenceServer(binary_path, config, settings)
        await server.start()
        print(f"Server running at {server.base_url} (PID: {server._process.pid if server._process else '?'})")
        print("Press Ctrl+C to stop")
        try:
            while server.is_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping server...")
            await server.stop()

    asyncio.run(_run())


def _cmd_stop(name: str) -> None:
    import os
    import signal

    settings = ProviderSettings()
    pid_path = settings.bin_dir / name / "llama-server.pid"
    pid = read_pid_file(pid_path)
    if pid is None:
        print(f"No running server found for {name}")
        sys.exit(1)

    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to PID {pid}")
    except ProcessLookupError:
        print(f"Process {pid} not found (already stopped?)")

    remove_pid_file(pid_path)


def _cmd_status(name: str) -> None:
    import os

    settings = ProviderSettings()
    pid_path = settings.bin_dir / name / "llama-server.pid"
    pid = read_pid_file(pid_path)
    if pid is None:
        print(f"No running server for {name}")
        return

    try:
        os.kill(pid, 0)
        print(f"Server {name} is running (PID: {pid})")
    except ProcessLookupError:
        print(f"Server {name} has stale PID file (PID {pid} not found)")
        remove_pid_file(pid_path)


def run(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    args = _parse_provider_args(argv)

    if args.command == "list" or args.command is None:
        _cmd_list()
    elif args.command == "provider":
        name: str = args.name
        action: str | None = args.action
        if action == "download":
            _cmd_download(name, args.release)
        elif action == "start":
            _cmd_start(name, args)
        elif action == "stop":
            _cmd_stop(name)
        elif action == "status":
            _cmd_status(name)
        else:
            print(f"Unknown action: {action}")
            print("Available actions: download, start, stop, status")
            sys.exit(1)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)
