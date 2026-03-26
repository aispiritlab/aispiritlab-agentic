from __future__ import annotations

from agentic_runtime.distributed import AgenticServiceDiscovery
from agentic_runtime.settings import settings

from .runtime import build_lab6_service, configure_lab6_providers


def main() -> None:
    if not settings.agent_service_name.strip():
        raise SystemExit("AGENT_SERVICE_NAME is required to run a distributed agent service.")

    configure_lab6_providers()

    discovery = AgenticServiceDiscovery.from_settings()
    service = build_lab6_service(settings.agent_service_name, discovery=discovery)
    try:
        service.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.close()
        discovery.close()


if __name__ == "__main__":
    main()
