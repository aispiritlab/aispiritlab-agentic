from __future__ import annotations

import json

from agentic.core_agent import CoreAgentic
from agentic.message import ToolMessage
from agentic.metadata import Description
from agentic.models import ModelConfig
from agentic.prompts import QwenPromptBuilder
from agentic.providers.provider import ModelProviderType
from agentic.tools import Toolset, Toolsets

from agentic.specialized_agents.events import TaskDelegated

_PLANNER_SYSTEM_PROMPT_TEMPLATE = """You are a planner. Break the user request into steps and delegate each step to an agent.
Available agents: {agent_names}

Available tools:
{{tools}}

For each step of your plan, call delegate_task with the agent name and task description.
Call delegate_task ONE TIME per response. After each delegation, wait for the result.
Use ONLY the agents listed above.

Format:
<tool_call>
{{"name":"delegate_task","parameters":{{"agent_name":"AGENT_NAME","task_description":"WHAT TO DO"}}}}
</tool_call>

When you have delegated all steps, respond with a short confirmation without calling any tool."""


class PlannerAgent(CoreAgentic):
    """Creates execution plans and delegates tasks to other agents via tool calls."""

    description = Description(
        agent_name="planner",
        description="Plans and delegates tasks to other agents.",
        capabilities=("planning", "delegation"),
    )

    def __init__(
        self,
        model_id: str,
        agent_names: list[str],
        *,
        model_provider_type: ModelProviderType = "mlx",
        config: ModelConfig | None = None,
        max_planning_turns: int = 5,
    ) -> None:
        self._pending_tasks: list[TaskDelegated] = []
        self._seen_tasks: set[tuple[str, str]] = set()
        self._agent_names = agent_names
        self._max_planning_turns = max_planning_turns

        def delegate_task(agent_name: str, task_description: str) -> str:
            """Delegate a task to a specific agent. Call this for each step of your plan.

            Args:
                agent_name: Name of the agent to delegate to.
                task_description: What the agent should do.
            """
            key = (agent_name, task_description)
            if key in self._seen_tasks:
                return f"Already delegated to {agent_name}: {task_description}"
            self._seen_tasks.add(key)
            self._pending_tasks.append(
                TaskDelegated(
                    source="planner",
                    target_agent=agent_name,
                    task_description=task_description,
                )
            )
            return f"Delegated to {agent_name}: {task_description}"

        system_prompt = _PLANNER_SYSTEM_PROMPT_TEMPLATE.format(
            agent_names=", ".join(agent_names),
        )

        super().__init__(
            model_id=model_id,
            prompt_builder=QwenPromptBuilder(system_prompt=system_prompt),
            toolsets=Toolsets([Toolset([delegate_task])]),
            config=config or ModelConfig(max_tokens=256, generation_mode="nothinking"),
            model_provider_type=model_provider_type,
        )

    def plan(self, message: str) -> list[TaskDelegated]:
        """Create a plan by running a multi-turn tool-calling loop.

        The LLM calls delegate_task for each step. Each tool call is captured
        in an internal list. The loop stops when the LLM produces no tool call,
        or when it repeats a delegation (dedup guard), or when max turns is hit.

        Returns:
            List of unique TaskDelegated events representing the plan steps.
        """
        self._pending_tasks.clear()
        self._seen_tasks.clear()
        self._agent.clear_history()

        response = self.respond(message)
        prev_count = len(self._pending_tasks)

        turn = 0
        while response.tool_results and turn < self._max_planning_turns:
            turn += 1

            # If no new unique task was added, model is repeating — stop
            if len(self._pending_tasks) == prev_count:
                break
            prev_count = len(self._pending_tasks)

            parts: list[str] = []
            for tool_result in response.tool_results:
                tool_name, tool_args = tool_result.tool_call
                parts.append(
                    "\n".join([
                        f"Tool: {tool_name}",
                        f"Arguments: {json.dumps(tool_args, ensure_ascii=False)}",
                        "Output:",
                        tool_result.output,
                    ])
                )
            tool_msg = ToolMessage("\n\n".join(parts))
            response = self.respond(tool_msg)

        return list(self._pending_tasks)

    def summarize(self, completed_tasks: str) -> str:
        """Summarize completed task results. Preserves history from the plan phase.

        Args:
            completed_tasks: Formatted string of completed task results.

        Returns:
            A summary of all completed work.
        """
        prompt = (
            "All tasks have been completed. Here are the results:\n\n"
            f"{completed_tasks}\n\n"
            "Provide a brief final summary."
        )
        return self.respond(prompt).output
