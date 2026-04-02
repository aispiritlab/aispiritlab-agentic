"""Tests for the health check probe."""

from __future__ import annotations

import httpx
import pytest
import respx

from providers.server._health import ServerStartupTimeoutError, wait_for_healthy


@pytest.mark.asyncio(loop_scope="function")
class TestWaitForHealthy:
    async def test_succeeds_on_200(self) -> None:
        with respx.mock:
            respx.get("http://127.0.0.1:8080/health").mock(
                return_value=httpx.Response(200, json={"status": "ok"})
            )
            await wait_for_healthy("http://127.0.0.1:8080", timeout=2.0, interval=0.1)

    async def test_retries_until_healthy(self) -> None:
        call_count = 0

        def _side_effect(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return httpx.Response(503)
            return httpx.Response(200, json={"status": "ok"})

        with respx.mock:
            respx.get("http://127.0.0.1:8080/health").mock(side_effect=_side_effect)
            await wait_for_healthy("http://127.0.0.1:8080", timeout=5.0, interval=0.1)

        assert call_count == 3

    async def test_timeout_raises(self) -> None:
        with respx.mock:
            respx.get("http://127.0.0.1:8080/health").mock(
                side_effect=httpx.ConnectError("connection refused")
            )
            with pytest.raises(ServerStartupTimeoutError, match="did not become healthy"):
                await wait_for_healthy("http://127.0.0.1:8080", timeout=0.3, interval=0.1)

    async def test_uses_custom_request_timeout(self) -> None:
        captured_timeout: list[float] = []

        def _handler(request: httpx.Request) -> httpx.Response:
            captured_timeout.append(request.extensions["timeout"]["connect"])
            return httpx.Response(200, json={"status": "ok"})

        with respx.mock:
            respx.get("http://127.0.0.1:8080/health").mock(side_effect=_handler)
            await wait_for_healthy(
                "http://127.0.0.1:8080",
                timeout=0.3,
                interval=0.1,
                request_timeout=1.25,
            )

        assert captured_timeout == [1.25]
