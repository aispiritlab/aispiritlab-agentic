"""Tests for port allocation utilities."""

from __future__ import annotations

import socket

import pytest

from providers.server._port import find_free_port, is_port_in_use


def _bind_tcp_socket(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen(1)
    return sock


def _reserve_consecutive_ports(count: int, start: int = 49152) -> tuple[int, list[socket.socket]]:
    port = start
    while port < 65535 - count:
        sockets: list[socket.socket] = []
        try:
            for offset in range(count):
                sockets.append(_bind_tcp_socket(port + offset))
            return port, sockets
        except OSError:
            for sock in sockets:
                sock.close()
            port += count
    raise RuntimeError("Could not reserve consecutive test ports")


class TestIsPortInUse:
    def test_detects_bound_port(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.listen(1)
            assert is_port_in_use(port)

    def test_detects_free_port(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        assert not is_port_in_use(port)


class TestFindFreePort:
    def test_returns_available_port(self) -> None:
        port = find_free_port(start=49152, end=49200)
        assert 49152 <= port <= 49200
        assert not is_port_in_use(port)

    def test_skips_bound_ports(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            bound_port = sock.getsockname()[1]
            sock.listen(1)

            port = find_free_port(start=bound_port, end=bound_port + 10)
            assert port != bound_port
            assert bound_port + 1 <= port <= bound_port + 10

    def test_raises_when_no_port_available(self) -> None:
        start_port, sockets = _reserve_consecutive_ports(3)
        try:
            with pytest.raises(RuntimeError, match="No free port found"):
                find_free_port(start=start_port, end=start_port + 2)
        finally:
            for s in sockets:
                s.close()
