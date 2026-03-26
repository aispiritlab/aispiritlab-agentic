from __future__ import annotations

from collections.abc import Callable, Sequence

from agentic_runtime.distributed.client import DistributedChatClient
from agentic_runtime.distributed.registry import AgentSnapshot, RedisServiceRegistry
from agentic_runtime.distributed.transport import RedisStreamsTransport
from agentic_runtime.messaging.messages import Message

type ServiceHandler = Callable[[Message, "AgenticServiceDiscovery"], Sequence[Message]]
type CloseHook = Callable[[], None]


class AgenticServiceDiscovery:
    """Facade for distributed agent infrastructure.

    Workshop participants use this as the single entry point — never touching
    ``RedisStreamsTransport`` or ``RedisServiceRegistry`` directly.
    """

    def __init__(
        self,
        transport: RedisStreamsTransport,
        registry: RedisServiceRegistry,
        *,
        liveness_ttl_seconds: float = 15.0,
    ) -> None:
        self._transport = transport
        self._registry = registry
        self._liveness_ttl_seconds = liveness_ttl_seconds

    @classmethod
    def from_settings(cls) -> AgenticServiceDiscovery:
        from agentic_runtime.settings import settings

        transport = RedisStreamsTransport(
            settings.redis_url,
            prefix=settings.redis_stream_prefix,
        )
        registry = RedisServiceRegistry(transport)
        return cls(
            transport,
            registry,
            liveness_ttl_seconds=settings.agent_liveness_ttl_seconds,
        )

    # ------------------------------------------------------------------
    # Discovery — the workshop-facing API
    # ------------------------------------------------------------------

    def find(self, capability: str) -> AgentSnapshot:
        """Find a live agent by capability. Raises ``RuntimeError`` if none found."""
        agent = self._registry.find_by_capability(
            capability,
            max_age_seconds=self._liveness_ttl_seconds,
        )
        if agent is None:
            raise RuntimeError(
                f"No live agent with capability '{capability}' is registered."
            )
        return agent

    def find_optional(self, capability: str) -> AgentSnapshot | None:
        """Find a live agent by capability, returning ``None`` if none found."""
        return self._registry.find_by_capability(
            capability,
            max_age_seconds=self._liveness_ttl_seconds,
        )

    def live_agents(self) -> list[AgentSnapshot]:
        """Return all live agents sorted by name."""
        return self._registry.live_agents(max_age_seconds=self._liveness_ttl_seconds)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    def create_service(
        self,
        name: str,
        *,
        capabilities: tuple[str, ...],
        handler: ServiceHandler,
        role: str = "worker",
        heartbeat_seconds: float = 5.0,
        close_hook: CloseHook | None = None,
    ) -> DistributedService:
        """Create a ``DistributedService`` wired to this discovery instance."""
        from agentic_runtime.distributed.service import DistributedService

        return DistributedService(
            agent_name=name,
            capabilities=capabilities,
            discovery=self,
            handler=handler,
            role=role,
            heartbeat_seconds=heartbeat_seconds,
            close_hook=close_hook,
        )

    def create_client(
        self,
        *,
        entry_agent: str = "planner",
        source: str = "chat",
        timeout_seconds: float = 60.0,
        domain: str = "lab6",
    ) -> DistributedChatClient:
        """Create a ``DistributedChatClient`` using the owned transport."""
        return DistributedChatClient(
            self._transport,
            entry_agent=entry_agent,
            source=source,
            timeout_seconds=timeout_seconds,
            domain=domain,
        )

    # ------------------------------------------------------------------
    # Internal — used by DistributedService, not by workshop participants
    # ------------------------------------------------------------------

    @property
    def transport(self) -> RedisStreamsTransport:
        return self._transport

    @property
    def registry(self) -> RedisServiceRegistry:
        return self._registry

    @property
    def liveness_ttl_seconds(self) -> float:
        return self._liveness_ttl_seconds

    def close(self) -> None:
        """Close the underlying transport (Redis connection)."""
        self._transport.close()


# Avoid circular import — DistributedService is imported lazily in create_service.
# This forward reference is only for type-checking.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_runtime.distributed.service import DistributedService
