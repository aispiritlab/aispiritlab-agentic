"""HTTP client for OpenAI-compatible APIs."""

from __future__ import annotations

from typing import Any

import httpx


class ModelConnectionError(Exception):
    """Raised when the model API is unreachable."""


class HttpClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )

    def get(self, path: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        try:
            response = self._client.get(path, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError as exc:
            raise ModelConnectionError(
                f"Could not connect to model API at {self.base_url}. "
                "Is the server running?"
            ) from exc

    def post(self, path: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            response = self._client.post(path, json=data)
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError as exc:
            raise ModelConnectionError(
                f"Could not connect to model API at {self.base_url}. "
                "Is the server running?"
            ) from exc

    def close(self) -> None:
        self._client.close()
