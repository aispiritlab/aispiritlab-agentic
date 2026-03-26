from agentic.agent import Agent, AgentResult
from agentic.models import ModelConfig, ModelProvider
from agentic.prompts import QwenPromptBuilder
from agentic.tools import Toolset, Toolsets

from workshops.lab0.prompt import SYSTEM_PROMPT
from workshops.lab0.tools import calculate, get_current_time, roll_dice
from workshops.tui import LabApp

MODEL_ID = "Qwen/Qwen3.5-2B"


class Lab0App(LabApp):
    lab_title = "Lab 0 — Simple Agent with Tools"
    lab_subtitle = "Agent + Toolset basics"
    lab_info = [
        f"Model: {MODEL_ID} (local, MLX)",
        "Tools: get_current_time, calculate, roll_dice",
        "",
        "Type 'tools' to inspect registered schemas.",
    ]

    def __init__(
        self, *, agent: Agent, model_provider: ModelProvider,
    ) -> None:
        super().__init__()
        self._agent = agent
        self._model_provider = model_provider

    def handle_input(self, message: str, *, attached_file: str | None = None) -> None:
        if message.lower() == "tools":
            self._show_tool_schemas()
            return

        self.set_status("Thinking...")
        result: AgentResult = self._agent.run(message)
        self._render_result(result)

    def _show_tool_schemas(self) -> None:
        lines: list[str] = ["### Registered Tools\n"]
        for toolset in self._agent.toolsets:
            for tool in toolset.tools:
                lines.append(f"**{tool.name}** — {tool.doc}")
                for arg in tool.args:
                    req = "required" if arg["required"] else "optional"
                    lines.append(f"  - `{arg['name']}`: {arg['type']} ({req})")
                lines.append("")
        self.write_agent("Tools", "\n".join(lines))

    def _render_result(self, result: AgentResult) -> None:
        if result.reasoning:
            self.write_activity(
                "Reasoning", result.reasoning[:80] + "...", style="#ff9e64",
            )

        if result.tool_calls:
            for tool_call in result.tool_calls:
                name, params = tool_call
                self.write_tool_call(name, str(params))
                self.write_activity("Tool", f"Calling {name}", style="#bb9af7")
                self.set_status(f"Running {name}...")
                run_result = self._agent.toolsets.run_tool(tool_call)
                output = run_result.output if run_result is not None else "(no output)"
                self.write_tool_result(name, str(output))
                self.write_activity("Result", f"{name} → {str(output)[:60]}", style="#9ece6a")
        else:
            self.write_agent("Assistant", result.content or "")

        if result.usage:
            latency = result.usage.get("latency_ms", "?")
            tokens = result.usage.get("total_tokens", "?")
            self.write_activity(
                "Usage", f"{latency}ms, {tokens} tokens", style="#565f89",
            )

    def cleanup(self) -> None:
        self._model_provider.close()


def run_lab0() -> None:
    print(f"Loading model {MODEL_ID}... (this may take a moment on first run)")
    model_provider = ModelProvider(
        MODEL_ID,
        model_provider_type="mlx",
        config=ModelConfig(max_tokens=512, generation_mode="nothinking"),
    )
    toolset = Toolset([get_current_time, calculate, roll_dice])
    toolsets = Toolsets([toolset])
    agent = Agent(
        model_provider=model_provider,
        prompt_builder=QwenPromptBuilder(system_prompt=SYSTEM_PROMPT),
        toolsets=toolsets,
    )
    if model_provider.model is None:
        error = model_provider.get_load_error("model") or "unknown load error"
        raise RuntimeError(f"Model is not available for inference: {error}")
    print("Model loaded. Starting TUI...")
    Lab0App(agent=agent, model_provider=model_provider).run()
