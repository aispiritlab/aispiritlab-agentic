from __future__ import annotations

from agentic.core_agent import CoreAgentic
from agentic.models import ModelConfig
from agentic.prompts import QwenPromptBuilder
from agentic.specialized_agents import PlannerAgent, TaskCompleted
from agentic.workflow import (
    LLMReactor,
    UserMessage,
    make_llm_routing,
    run_workflow,
)

from workshops.tui import LabApp

from .deciders import worker_decider
from .prompts import RESEARCHER_SYSTEM_PROMPT, WRITER_SYSTEM_PROMPT

MODEL_ID = "Qwen/Qwen3.5-2B"


class Lab2App(LabApp):
    lab_title = "Lab 2 — Planner: Plan & Delegate"
    lab_subtitle = "Multi-agent task delegation"
    lab_info = [
        f"Model: {MODEL_ID} (local, MLX)",
        "",
        "Agents: Planner, Researcher, Writer",
        "",
        "Flow:",
        "  You → Planner creates plan",
        "       → Delegates tasks via events",
        "            → Workers execute",
        "       → Planner summarises results",
    ]

    def __init__(
        self,
        *,
        planner: PlannerAgent,
        agents: dict[str, CoreAgentic],
    ) -> None:
        super().__init__()
        self._planner = planner
        self._agents = agents
        self._worker_routings = {
            name: make_llm_routing(LLMReactor(agent=agent))
            for name, agent in agents.items()
        }

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
            self.write_activity(
                f"Step {i}", f"{task.target_agent}: {task.task_description[:50]}",
                style="#ff9e64",
            )
        self.write_agent("Planner", "\n".join(plan_lines))

        completed: list[TaskCompleted] = []
        for task in tasks:
            agent_name = task.target_agent
            routing = self._worker_routings.get(agent_name)
            if routing is None:
                self.write_error(f"Unknown agent: {agent_name}")
                continue

            self.set_status(f"Running {agent_name}...")
            self.write_activity(agent_name.title(), "Working...", style="#bb9af7")
            worker_result = run_workflow(
                message=UserMessage(text=task.task_description, source="planner"),
                decider=worker_decider,
                routing_fn=routing,
            )
            self.write_agent(agent_name.title(), worker_result.text or "", accent="#bb9af7")
            self.write_activity(agent_name.title(), "Done", style="#9ece6a")

            completed.append(
                TaskCompleted(
                    source=agent_name,
                    target_agent=agent_name,
                    task_description=task.task_description,
                    result=worker_result.text,
                )
            )

        if completed:
            self.set_status("Planner summarising...")
            self.write_activity("Planner", "Summarising results...", style="#9ece6a")
            parts = [
                f"Agent: {tc.target_agent}\nTask: {tc.task_description}\nResult: {tc.result}"
                for tc in completed
            ]
            summary = self._planner.summarize("\n\n".join(parts))
            self.write_agent("Planner Summary", summary or "", accent="#9ece6a")
            self.write_activity("Summary", "Complete", style="#9ece6a")

    def cleanup(self) -> None:
        self._planner.close()
        for agent in self._agents.values():
            agent.close()


def run_lab2() -> None:
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
    print("Models loaded. Starting TUI...")
    Lab2App(planner=planner, agents={"researcher": researcher, "writer": writer}).run()
