"""Lab4: Full Runtime — message bus, lifecycle events, workflow registration.

Builds on lab3's event-driven pattern by adding a WorkshopRuntime that provides:
- Central message bus (all messages published and logged)
- Workflow registration by name
- Output handlers registered on bus (auto-triggered on publish)
- Turn lifecycle events (TurnStarted / TurnCompleted)
- Message log for observability
"""
from __future__ import annotations

from collections.abc import Callable

from agentic.core_agent import CoreAgentic
from agentic.models import ModelConfig
from agentic.prompts import QwenPromptBuilder
from agentic.specialized_agents import PlannerAgent, TaskCompleted

from workshops.tui import LabApp

from .handlers import build_worker_output_handler
from .prompts import RESEARCHER_SYSTEM_PROMPT, WRITER_SYSTEM_PROMPT
from .runtime import WorkshopRuntime
from .workflows import make_planner_workflow, make_worker_workflow

MODEL_ID = "Qwen/Qwen3.5-2B"


class Lab4App(LabApp):
    lab_title = "Lab 4 — Full Runtime"
    lab_subtitle = "Message bus + lifecycle events"
    lab_info = [
        f"Model: {MODEL_ID} (local, MLX)",
        "",
        "New vs Lab 3:",
        "  - WorkshopRuntime + message bus",
        "  - Workflows registered by name",
        "  - Output handlers on bus",
        "  - TurnStarted / TurnCompleted",
        "",
        "Type 'log' to see the message bus log.",
    ]

    def __init__(
        self,
        *,
        planner: PlannerAgent,
        agents: dict[str, CoreAgentic],
        runtime: WorkshopRuntime,
    ) -> None:
        super().__init__()
        self._planner = planner
        self._agents = agents
        self._runtime = runtime
        self._completed: list[TaskCompleted] = []

    def on_task_completed(self, task: TaskCompleted) -> None:
        """Callback invoked by the output handler when a worker finishes."""
        self._completed.append(task)
        self.write_activity(
            "Completed", f"{task.target_agent}: {task.task_description[:40]}",
            style="#9ece6a",
        )

    def handle_input(self, message: str, *, attached_file: str | None = None) -> None:
        if message.lower() == "log":
            self._show_message_log()
            return

        self._completed.clear()

        self.set_status("Runtime executing turn...")
        self.write_activity("Turn", "Started", style="#9ece6a")
        plan_text = self._runtime.run(message, "planner")

        self.write_agent("Planner", plan_text or "")
        self.write_activity("Planner", "Plan created", style="#9ece6a")

        for task in self._completed:
            self.write_agent(
                "Worker",
                f"[{task.target_agent.title()}]: {task.result}",
                accent="#bb9af7",
            )

        if self._completed:
            self.set_status("Planner summarising...")
            self.write_activity("Planner", "Summarising...", style="#9ece6a")
            parts = [
                f"Agent: {tc.target_agent}\nTask: {tc.task_description}\nResult: {tc.result}"
                for tc in self._completed
            ]
            summary = self._planner.summarize("\n\n".join(parts))
            self.write_agent("Planner Summary", summary or "", accent="#9ece6a")

        self.write_activity("Turn", "Completed", style="#9ece6a")

    def _show_message_log(self) -> None:
        log = self._runtime.message_log
        if not log:
            self.write_system("Message bus log is empty.")
            return

        lines = ["### Message Bus Log\n"]
        for i, msg in enumerate(log, 1):
            kind = msg.kind
            source = msg.source or "?"
            name = msg.name or ""
            text_preview = (msg.text or "")[:60]
            if name:
                lines.append(f"{i}. `[{kind}]` source=**{source}** name=*{name}*")
            elif text_preview:
                lines.append(f"{i}. `[{kind}]` source=**{source}** text=*{text_preview}...*")
            else:
                lines.append(f"{i}. `[{kind}]` source=**{source}**")
        lines.append(f"\n**Total: {len(log)} messages**")
        self.write_agent("Message Bus", "\n".join(lines), accent="#565f89")

    def cleanup(self) -> None:
        self._runtime.stop()
        self._planner.close()
        for agent in self._agents.values():
            agent.close()


def run_lab4() -> None:
    print(f"Loading model {MODEL_ID}...")
    planner = PlannerAgent(model_id=MODEL_ID, agent_names=["researcher", "writer"])
    researcher = CoreAgentic(
        model_id=MODEL_ID,
        prompt_builder=QwenPromptBuilder(system_prompt=RESEARCHER_SYSTEM_PROMPT),
        config=ModelConfig(max_tokens=256, generation_mode="nothinking"),
    )
    writer = CoreAgentic(
        model_id=MODEL_ID,
        prompt_builder=QwenPromptBuilder(system_prompt=WRITER_SYSTEM_PROMPT),
        config=ModelConfig(max_tokens=256, generation_mode="nothinking"),
    )
    planner.preload_model()
    researcher.preload_model()
    writer.preload_model()
    agents = {"researcher": researcher, "writer": writer}

    # Build app first so we can wire its callback into the runtime
    app = Lab4App(
        planner=planner,
        agents=agents,
        runtime=WorkshopRuntime(),
    )

    # Wire runtime: workflows + output handler with callback bridge
    worker_workflows = {
        name: make_worker_workflow(name, agent)
        for name, agent in [("researcher", researcher), ("writer", writer)]
    }
    app._runtime.register_workflow("planner", make_planner_workflow(planner))
    app._runtime.register_workflow("researcher", worker_workflows["researcher"])
    app._runtime.register_workflow("writer", worker_workflows["writer"])
    app._runtime.register_output_handler(
        build_worker_output_handler(
            worker_workflows,
            on_completed=app.on_task_completed,
        )
    )

    print("Models loaded. Starting TUI...")
    app.run()
