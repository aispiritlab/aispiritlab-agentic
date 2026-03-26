from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from deepeval.dataset import Golden
from deepeval.errors import DeepEvalError
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import ExactMatchMetric
from deepeval.optimizer import PromptOptimizer
from deepeval.optimizer.algorithms.miprov2 import MIPROV2
from deepeval.optimizer.types import OptimizationReport
from deepeval.optimizer.utils import build_prompt_config_snapshots
from deepeval.prompt import Prompt

from evaluation.contracts import EvaluationDefinition, Flows, ToolScenario
from evaluation.deepeval_providers.deepeval_openrouter import FixedOpenRouterModel
from evaluation.definition_loader import load_evaluation_definition
from evaluation.eval_dataset import (
    build_goldens_from_flows,
    build_goldens_from_scenarios,
    build_prompt_optimization_goldens,
)


class CompatibleMIPROV2(MIPROV2):
    """Compatibility fix for DeepEval 3.8.x OptimizationReport shape mismatch."""

    def _build_result(self, best):  # type: ignore[override]
        prompt_config_snapshots = build_prompt_config_snapshots(
            self.prompt_configurations_by_id
        )
        report = OptimizationReport(
            optimization_id=self.optimization_id,
            best_id=best.id,
            accepted_iterations=[],
            pareto_scores=self.pareto_score_table,
            parents=self.parents_by_id,
            prompt_configurations=prompt_config_snapshots,
        )
        return best.prompts[self.SINGLE_MODULE_ID], report


class AgentPromptOptimization:
    def __init__(
        self,
        *,
        definition: EvaluationDefinition,
        output_path: Path,
        openrouter_model: str,
        openrouter_api_key: str | None = None,
        num_candidates: int = 6,
        num_trials: int = 12,
        runtime_options: Mapping[str, Any] | None = None,
        scenarios: Sequence[ToolScenario] | None = None,
        flows: Flows | None = None,
        prompt_text: str | None = None,
    ) -> None:
        self._definition = definition
        self._output_path = output_path
        self._openrouter_model = openrouter_model
        self._openrouter_api_key = openrouter_api_key
        self._num_candidates = num_candidates
        self._num_trials = num_trials
        self._runtime_options = dict(runtime_options or {})
        self._scenarios = tuple(scenarios or definition.scenarios)
        self._flows = flows if flows is not None else definition.flows
        self._prompt_text = prompt_text

    def with_scenarios(self, scenarios: Sequence[ToolScenario]) -> "AgentPromptOptimization":
        self._scenarios = tuple(scenarios)
        return self

    def with_flows(self, flows: Flows | None) -> "AgentPromptOptimization":
        self._flows = flows
        return self

    def with_runtime_options(
        self,
        runtime_options: Mapping[str, Any] | None,
    ) -> "AgentPromptOptimization":
        self._runtime_options = dict(runtime_options or {})
        return self

    def with_prompt_text(self, prompt_text: str | None) -> "AgentPromptOptimization":
        self._prompt_text = prompt_text
        return self

    def _build_goldens(self) -> list[Golden]:
        if not self._scenarios:
            raise ValueError("Define at least one scenario before running optimization.")
        if self._flows is None:
            return build_goldens_from_scenarios(self._scenarios)
        return build_goldens_from_flows(self._scenarios, self._flows)

    def run(self) -> Path:
        optimized_prompt = optimize_prompt_text(
            definition=self._definition,
            prompt_text=self._prompt_text or self._definition.resolve_prompt_text(),
            openrouter_model=self._openrouter_model,
            openrouter_api_key=self._openrouter_api_key,
            num_candidates=self._num_candidates,
            num_trials=self._num_trials,
            goldens=self._build_goldens(),
            runtime_options=self._runtime_options,
        )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(optimized_prompt, encoding="utf-8")
        return self._output_path




def optimize_prompt_text(
    *,
    definition: EvaluationDefinition,
    prompt_text: str,
    openrouter_model: str,
    openrouter_api_key: str | None = None,
    num_candidates: int = 6,
    num_trials: int = 12,
    goldens: list[Golden] | None = None,
    runtime_options: Mapping[str, Any] | None = None,
) -> str:
    if goldens is None:
        goldens = build_prompt_optimization_goldens(definition)

    callback = definition.create_agent_callback(runtime_options)

    def model_callback(prompt: Prompt, golden: Golden) -> str:
        callback.reset()
        if hasattr(callback, "prepare"):
            callback.prepare()  # type: ignore[attr-defined]
        metadata = getattr(golden, "additional_metadata", None) or {}
        prefill_messages = metadata.get("prefill_messages", [])
        if isinstance(prefill_messages, list):
            callback.prime(
                [message for message in prefill_messages if isinstance(message, str)]
            )
        return callback.run(golden.input, prompt_text=prompt.interpolate())

    algorithm = CompatibleMIPROV2(
        num_candidates=num_candidates,
        num_trials=num_trials,
        minibatch_size=3,
        minibatch_full_eval_steps=3,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
        num_demo_sets=3,
        random_seed=42,
    )

    optimizer_model = FixedOpenRouterModel(
        model=openrouter_model,
        api_key=openrouter_api_key or None,
    )
    algorithm.optimizer_model = optimizer_model
    optimizer = PromptOptimizer(
        model_callback=model_callback,
        metrics=[ExactMatchMetric()],
        optimizer_model=optimizer_model,
        algorithm=algorithm,
        async_config=AsyncConfig(run_async=False, max_concurrent=1),
    )

    try:
        best_prompt = optimizer.optimize(
            prompt=Prompt(text_template=prompt_text),
            goldens=goldens,
        )
    finally:
        if hasattr(callback, "close"):
            callback.close()  # type: ignore[attr-defined]

    return best_prompt.text_template or prompt_text


def _parse_runtime_options(items: Sequence[str]) -> dict[str, str]:
    runtime_options: dict[str, str] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key.strip():
            raise ValueError(
                f"Runtime option '{item}' must use the KEY=VALUE format."
            )
        runtime_options[key.strip()] = value
    return runtime_options


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run generic MIPROv2 prompt optimization using an evaluation definition."
    )
    parser.add_argument(
        "--definition",
        required=True,
        help=(
            "Evaluation definition in module:attribute format, for example "
            "`agentic_runtime.manage_notes.evaluation:NOTES_EVALUATION`."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("packages/evaluation/src/evaluation/optimized_prompt.txt"),
        help="Target file for the optimized prompt.",
    )
    parser.add_argument(
        "--openrouter-model",
        type=str,
        default="openai/gpt-4o-mini",
        help="OpenRouter model used by MIPROv2.",
    )
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default="",
        help="Optional OpenRouter API key override.",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=6,
        help="Number of candidates in MIPROv2.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=12,
        help="Number of Bayesian optimization trials in MIPROv2.",
    )
    parser.add_argument(
        "--runtime-option",
        action="append",
        default=[],
        help="Runtime option passed to the domain callback in KEY=VALUE format.",
    )
    args = parser.parse_args()

    try:
        definition = load_evaluation_definition(args.definition)
        output_path = AgentPromptOptimization(
            definition=definition,
            output_path=args.output_path,
            openrouter_model=args.openrouter_model,
            openrouter_api_key=args.openrouter_api_key or None,
            num_candidates=args.num_candidates,
            num_trials=args.num_trials,
            runtime_options=_parse_runtime_options(args.runtime_option),
        ).run()
    except DeepEvalError as error:
        message = (
            "Failed to run MIPROv2.\n"
            "- Make sure `optuna` is installed.\n"
            "- Make sure `OPENROUTER_API_KEY` or `--openrouter-api-key` is set.\n"
            f"Details: {error}"
        )
        raise SystemExit(message) from error
    except Exception as error:
        raise SystemExit(str(error)) from error

    print(f"Saved optimized prompt: {output_path}")


if __name__ == "__main__":
    main()
