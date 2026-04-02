"""Health check probe for inference servers."""

from __future__ import annotations

import asyncio

import httpx
from structlog import get_logger

logger = get_logger(__name__)


class ServerStartupTimeoutError(Exception):
    """Raised when the server does not become healthy in time."""


async def wait_for_healthy(
    base_url: str,
    *,
    timeout: float = 30.0,
    interval: float = 0.5,
    request_timeout: float = 5.0,
) -> None:
    """Poll GET /health until 200 or timeout."""
    deadline = asyncio.get_event_loop().time() + timeout
    url = f"{base_url}/health"

    while True:
        try:
            async with httpx.AsyncClient(timeout=request_timeout) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    logger.info("server_healthy", url=base_url)
                    return
                logger.debug("health_check_not_ready", status=response.status_code)
        except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
            logger.debug("health_check_connection_failed", url=url)

        if asyncio.get_event_loop().time() >= deadline:
            raise ServerStartupTimeoutError(
                f"Server at {base_url} did not become healthy within {timeout}s"
            )
        await asyncio.sleep(interval)
