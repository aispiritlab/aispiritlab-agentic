"""Agent protocol shared across packages.

Structural type only: implementations live in ``agentic`` and its dependents, so
low-level packages can type against an agent without importing the SDK.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Agent(Protocol):
    """Anything that can handle a request and return a reply."""

    def handler(self, context: Any) -> Any: ...
