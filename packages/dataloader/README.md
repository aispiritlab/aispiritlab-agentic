# Dataloader

Utilities for generating Flow PHP repository-grounded DeepFabric datasets and
normalizing them into SFT-ready JSONL exports for code agents such as Claude
Code or OpenCode.

Default model split:

- training target in config root: `ollama / qwen3.5:4b`
- topic generation: `openrouter / minimax/minimax-m2.5`
- default topic model base URL: `https://openrouter.ai/api/v1`
- agent trace generation: `openrouter / google/gemini-2.5-flash`

## Flow PHP Repo Dataset

The workspace exposes a `deepfabric` proxy, so this works from the repository
root without adding DeepFabric to the locked project dependencies:

```bash
uv run deepfabric generate packages/dataloader/data/flow_php_repo/flow_php_repo_qwen35_config.yaml
```

Under the hood the proxy delegates to:

```bash
uv tool run --from deepfabric deepfabric ...
```

By default the generated configs need:

```bash
export OPENROUTER_API_KEY=your-openrouter-key
```

By default `write-configs` resolves the Flow checkout from `FLOW_REPO_ROOT`.
If that variable is unset, it falls back to a sibling `../flow` checkout when
present, then `~/Projects/flow`.

If you still have older configs that used MiniMax for `generation.llm`, rerun
`write-configs`. That older setup can produce repeated `DF-P01 JSON parse error`
failures in agent mode.

If you want to use one provider for both topics and generation, regenerate the
configs explicitly. For native Gemini generation:

```bash
uv run --project packages/dataloader flow-php-repo-dataset write-configs \
  --topics-provider gemini \
  --topics-model gemini-2.5-flash \
  --generation-provider gemini \
  --generation-model gemini-2.5-flash
```

If you want to use MiniMax directly instead of OpenRouter for topics and generation,
regenerate the configs with the MiniMax OpenAI-compatible endpoint and use your MiniMax API key:

```bash
export MINIMAX_API_KEY=your-minimax-key
uv run --project packages/dataloader flow-php-repo-dataset write-configs \
  --topics-provider openai \
  --topics-model MiniMax-M2.5 \
  --topics-base-url https://api.minimax.io/v1 \
  --generation-provider openai \
  --generation-model MiniMax-M2.5 \
  --generation-base-url https://api.minimax.io/v1
```

For China-region MiniMax accounts, use:

```bash
uv run --project packages/dataloader flow-php-repo-dataset write-configs \
  --topics-provider openai \
  --topics-model MiniMax-M2.5 \
  --topics-base-url https://api.minimaxi.com/v1 \
  --generation-provider openai \
  --generation-model MiniMax-M2.5 \
  --generation-base-url https://api.minimaxi.com/v1
```

Generate the two mode-specific configs from the local Flow repository snapshot:

```bash
uv run --project packages/dataloader flow-php-repo-dataset write-configs
```

`write-configs` now defaults to the `finetune` profile, which increases both
topic coverage and sample count for better SFT volume:

- `topics.depth=4`
- `topics.degree=4`
- `output.num_samples="200%"` (two cycles over the discovered topic set)
- `output.batch_size=4`

If your local Flow checkout lives elsewhere:

```bash
uv run --project packages/dataloader flow-php-repo-dataset write-configs \
  --repo-root /absolute/path/to/flow
```

The generated configs are written under `packages/dataloader/data/flow_php_repo/`
and use current DeepFabric output controls:

- `topics.max_concurrent=8`
- `generation.sample_retries=3`
- `output.num_samples="200%"`
- `output.checkpoint.interval=100`

The VFS seed now pulls a broader example-oriented slice from
`/Users/mkubaszek/Projects/flow` via `--repo-root`, including:

- repository and package `README.md` files
- selected DSL `functions.php` files
- representative integration tests such as landing-page example coverage
- package-level contribution and usage material
- maintainer-facing ETL and type-system files such as `Analyze.php`, `StatisticsCollector.php`, `Cast.php`, and timezone/memory-related tests

The generated prompts are also biased toward maintainer-style task formats via a
synthetic `MAINTAINER_TASKS.md` seed. That lets DeepFabric produce tasks shaped
more like accepted Flow change requests, for example:

- proposal-style prompts with `Describe the Proposal` and `API Adjustments`
- implementation briefs with `# Context`, `# Your Task`, and `# Requirements`
- approved-edit tasks that ask for concrete code changes plus tests in exact files

Available profiles:

- `baseline` for cheap smoke-test generations
- `finetune` for the default higher-volume training pass
- `xl` for larger corpus generation when you want even more topic cycling

Examples:

```bash
uv run --project packages/dataloader flow-php-repo-dataset write-configs --profile baseline
uv run --project packages/dataloader flow-php-repo-dataset write-configs --profile xl
```

Generate raw DeepFabric samples for both modes:

```bash
uv run deepfabric generate packages/dataloader/data/flow_php_repo/flow_php_repo_qwen35_config.yaml
uv run deepfabric generate packages/dataloader/data/flow_php_repo/flow_php_repo_qwen35_approved_edit_config.yaml
```

Normalize, validate, and export the final datasets:

```bash
uv run --project packages/dataloader flow-php-repo-dataset normalize \
  --read-only-input packages/dataloader/data/flow_php_repo/flow-php-repo-read-only.raw.jsonl \
  --approved-edit-input packages/dataloader/data/flow_php_repo/flow-php-repo-approved-edit.raw.jsonl \
  --chat-output packages/dataloader/data/flow_php_repo/flow-php-sft-chat.jsonl \
  --agent-output packages/dataloader/data/flow_php_repo/flow-php-sft-agent.jsonl
```

The end-to-end helper target runs the same steps:

```bash
make flow-php-repo-dataset
```

Outputs:

- `packages/dataloader/data/flow_php_repo/flow-php-sft-chat.jsonl`
- `packages/dataloader/data/flow_php_repo/flow-php-sft-agent.jsonl`
