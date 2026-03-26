from __future__ import annotations

from agentic_runtime.distributed import DistributedAgenticRuntime
from agentic_runtime.settings import settings

from workshops.tui import LabApp

from . import messages as _messages  # noqa: F401


class Lab6App(LabApp):
    lab_title = "Lab 6 — Distributed Planner/Search/Summary"
    lab_subtitle = "Redis Streams showcase"
    lab_info = [
        "Topology:",
        "  chat -> planner -> search -> summary",
        "",
        f"Redis: {settings.redis_url}",
        f"Entry agent: {settings.chat_entry_agent}",
        "",
        "Commands:",
        "  agents  -> show live registered services",
        "  <text>  -> ask the distributed pipeline",
    ]

    def __init__(self, *, runtime: DistributedAgenticRuntime) -> None:
        super().__init__()
        self._runtime = runtime

    def handle_input(self, message: str, *, attached_file: str | None = None) -> None:
        del attached_file
        prompt = message.strip()
        if not prompt:
            return

        if prompt.lower() in {"agents", "status"}:
            agents = self._runtime.live_agents()
            if not agents:
                self.write_system("No live distributed agents are registered.")
                return

            lines = ["### Live Agents\n"]
            for index, agent in enumerate(agents, start=1):
                capabilities = ", ".join(agent.capabilities) or "(none)"
                lines.append(
                    f"{index}. **{agent.agent_name}** — status={agent.status}, "
                    f"capabilities={capabilities}"
                )
            self.write_agent("Registry", "\n".join(lines), accent="#565f89")
            return

        self.set_status("Planner creating distributed turn...")
        self.write_activity("Chat", "Published question to planner", style="#7aa2f7")
        response = self._runtime.run(prompt)
        self.write_activity("Summary", "Received final response", style="#9ece6a")
        self.write_agent("Summary", response, accent="#9ece6a")

    def cleanup(self) -> None:
        self._runtime.stop()


def run_lab6() -> None:
    runtime = DistributedAgenticRuntime.from_settings()
    Lab6App(runtime=runtime).run()
