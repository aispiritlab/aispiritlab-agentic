"""Port allocation utilities."""

from __future__ import annotations

import socket


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


def find_free_port(start: int = 8080, end: int | None = None, host: str = "127.0.0.1") -> int:
    """Find an available TCP port in the given range."""
    if end is None:
        end = start + 100
    for port in range(start, end + 1):
        if not is_port_in_use(port, host):
            return port
    raise RuntimeError(f"No free port found in range {start}-{end}")
