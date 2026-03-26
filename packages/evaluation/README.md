## Evaluation module

DeepEval coverage for the notes flow (`add_note`, `edit_note`, `get_note`, `list_notes`) lives in:

- `packages/evaluation/tests/test_notes_deepeval_metrics.py`

The tests evaluate:

- `ToolCorrectnessMetric`
- `StepEfficiencyMetric`

### OpenRouter benchmark for NotesToolScenario

There is also an opt-in benchmark test for comparing local candidate models
(MLX / Hugging Face IDs) on the same notes scenarios, with OpenRouter used as
the LLM judge for Deepeval metrics:

- `ToolCorrectnessMetric`
- `StepEfficiencyMetric`
- `TaskCompletionMetric`

File:

- `packages/evaluation/tests/test_notes_openrouter_benchmark.py`

Why opt-in:

- It makes paid OpenRouter API calls for judging.
- Candidate models are loaded locally (MLX / HF via `mlx_lm`).
- It can take a while (multiple models x multiple scenarios x judge calls).

Run (full benchmark):

```bash
export OPENROUTER_API_KEY=...
export RUN_OPENROUTER_BENCHMARKS=1
uv run --project packages/evaluation pytest -s packages/evaluation/tests/test_notes_openrouter_benchmark.py
```

Useful cost-control env vars:

```bash
# Judge model used by Deepeval TaskCompletionMetric (default: openai/gpt-4o-mini)
export NOTES_OPENROUTER_JUDGE_MODEL=openai/gpt-4o-mini

# Limit scenarios (for a quick smoke run)
export NOTES_OPENROUTER_MAX_SCENARIOS=5

# Or run only selected scenarios by dataset names
export NOTES_OPENROUTER_SCENARIOS=add_note,read_note,list_notes

# Override candidate models (comma-separated)
export NOTES_OPENROUTER_MODELS=microsoft/phi-4-mini-flash-reasoning,google/gemma-3-4b-it

# Optional: run extra prompt-format / instruction-following benchmark cases
export RUN_OPENROUTER_PROMPT_CASES=1

# Optional: run synthetic multi-turn conversation benchmark (46 generated conversations by default)
export RUN_OPENROUTER_CONVERSATION_BENCHMARKS=1
```

Reports:

- `packages/evaluation/src/evaluation/notes_openrouter_benchmark_report.json`
- `packages/evaluation/src/evaluation/notes_openrouter_prompt_cases_report.json` (when `RUN_OPENROUTER_PROMPT_CASES=1`)
- `packages/evaluation/src/evaluation/notes_openrouter_conversation_benchmark_report.json` (when `RUN_OPENROUTER_CONVERSATION_BENCHMARKS=1`)

Each report now contains:

- `top_models` (top 1-2 quick shortlist)
- `full_rank_list` (all candidate models with score breakdown)
- `ranked_summaries` (full metric details)

### Print full rank list as a table

Small CLI helper:

```bash
uv run --project packages/evaluation notes-benchmark-rank /path/to/report.json
```

Examples:

```bash
# Notes tool benchmark
uv run --project packages/evaluation notes-benchmark-rank \
  packages/evaluation/src/evaluation/notes_openrouter_benchmark_report.json

# Prompt-style benchmark (show only top 3)
uv run --project packages/evaluation notes-benchmark-rank \
  packages/evaluation/src/evaluation/notes_openrouter_prompt_cases_report.json \
  --top 3

# Conversation benchmark
uv run --project packages/evaluation notes-benchmark-rank \
  packages/evaluation/src/evaluation/notes_openrouter_conversation_benchmark_report.json
```

Note:

- `OpenRouter` is used only as the judge model in this benchmark.
- Candidate model IDs can be local MLX repos like `mlx-community/*` or other HF model IDs,
  but they still need to be loadable by your local `mlx_lm` setup.

Both use Polish user examples and run through `agentic_runtime.agents_manager.notes_agent`.

### Synthetic conversation dataset (multi-turn)

To generate many multi-turn conversations (including context/prefill flows) based on
`NOTES_TOOL_SCENARIOS`, use:

- `evaluation.notes_eval_dataset.build_notes_conversation_scenarios()`
- `evaluation.notes_eval_dataset.build_notes_conversation_examples()`

These produce conversations with:

- assistant greeting
- user messages
- expected `<tool_call>` outputs
- simulated tool results (`add_note`, `edit_note`, `get_note`, `list_notes`)

### Prompt optimization (MIPROv2)

The optimization entry point is a shared NOTE prompt optimizer for all four note intents:

- `packages/evaluation/src/evaluation/notes_prompt_optimization_miprov2.py`

Run:

```bash
uv run --project packages/evaluation note-prompt-optimize
```

Skrypt używa `FixedOpenRouterModel`, więc nie wymaga `OPENAI_API_KEY`.
Wymagane jest ustawienie:

```bash
export OPENROUTER_API_KEY=...
```

Opcjonalnie wybierz model OpenRouter:

```bash
uv run --project packages/evaluation note-prompt-optimize --openrouter-model openai/gpt-4o-mini
```

If needed, install optimizer extras first:

```bash
uv add --project packages/evaluation 'optuna>=3.6.0'
```
