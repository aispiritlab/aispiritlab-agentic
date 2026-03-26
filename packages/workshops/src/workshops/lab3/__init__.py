"""Lab3: Event-driven PlannerAgent — no for loops.

Same goal as lab2 (planner delegates to researcher + writer), but orchestration
uses dispatch_output_handlers instead of a manual for loop. TaskDelegated events
flow through the output handler which routes them to the correct worker agent.
"""
from __future__ import annotations

from collections.abc import Callable

from agentic.core_agent import CoreAgentic
from agentic.models import ModelConfig
from agentic.prompts import QwenPromptBuilder
from agentic.specialized_agents import PlannerAgent, TaskCompleted
from agentic.workflow import (
    LLMReactor,
    TechnicalRoutingFn,
    dispatch_output_handlers,
    make_llm_routing,
)

from workshops.tui import LabApp

from .handlers import build_worker_dispatch_handler
from .prompts import RESEARCHER_SYSTEM_PROMPT, WRITER_SYSTEM_PROMPT

MODEL_ID = "Qwen/Qwen3.5-2B"


class Lab3App(LabApp):
    lab_title = "Lab 3 — Event-Driven Planner"
    lab_subtitle = "No for loops — pure event dispatch"
    lab_info = [
        f"Model: {MODEL_ID} (local, MLX)",
        "",
        "Same agents as Lab 2, but:",
        "  1. Planner emits TaskDelegated events",
        "  2. dispatch_output_handlers routes",
        "  3. Workers produce TaskCompleted",
        "  4. Planner summarises",
    ]

    def __init__(
        self,
        *,
        planner: PlannerAgent,
        agents: dict[str, CoreAgentic],
        worker_routings: dict[str, TechnicalRoutingFn],
    ) -> None:
        super().__init__()
        self._planner = planner
        self._agents = agents
        self._worker_routings = worker_routings
        self._completed: list[TaskCompleted] = []

    def _on_task_completed(self, task: TaskCompleted) -> None:
        self._completed.append(task)
        self.write_activity(
            "Completed", f"{task.target_agent}: {task.task_description[:40]}",
            style="#9ece6a",
        )

    def handle_input(self, message: str, *, attached_file: str | None = None) -> None:
        self.set_status("Planner creating plan...")
        self.write_activity("Planner", "Creating plan...", style="#9ece6a")
        tasks = self._planner.plan(message)

        if not tasks:
            self.write_agent("Planner", "*(no tasks delegated)*")
            return

        plan_lines = ["### Plan\n"]
        for i, task in enumerate(tasks, 1):
            plan_lines.append(f"{i}. **{task.target_agent}** → {task.task_description}")
        self.write_agent("Planner", "\n".join(plan_lines))
        self.write_activity("Plan", f"{len(tasks)} step(s)", style="#ff9e64")

        self.set_status("Dispatching events to workers...")
        self.write_activity("Dispatch", "Routing TaskDelegated events...", style="#bb9af7")
        self._completed.clear()
        handler, _ = build_worker_dispatch_handler(
            self._worker_routings,
            on_completed=self._on_task_completed,
        )
        responses = dispatch_output_handlers([handler], tasks)

        for response in responses:
            self.write_agent("Worker", response or "", accent="#bb9af7")

        if self._completed:
            self.set_status("Planner summarising...")
            self.write_activity("Planner", "Summarising...", style="#9ece6a")
            parts = [
                f"Agent: {tc.target_agent}\nTask: {tc.task_description}\nResult: {tc.result}"
                for tc in self._completed
            ]
            summary = self._planner.summarize("\n\n".join(parts))
            self.write_agent("Planner Summary", summary or "", accent="#9ece6a")
            self.write_activity("Summary", "Complete", style="#9ece6a")

    def cleanup(self) -> None:
        self._planner.close()
        for agent in self._agents.values():
            agent.close()


def run_lab3() -> None:
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
    worker_routings = {
        name: make_llm_routing(LLMReactor(agent=agent))
        for name, agent in agents.items()
    }
    print("Models loaded. Starting TUI...")
    Lab3App(planner=planner, agents=agents, worker_routings=worker_routings).run()
