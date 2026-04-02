## providers

Subprocess-based inference provider management for local model servers.

### What it provides

- Download provider binaries from upstream releases
- Start and stop local inference servers
- Talk to OpenAI-compatible servers through `openai` or `httpx`

### Local development

```bash
uv run --project packages/providers pytest -q
```

### CLI

```bash
uv run --project packages/providers providers list
uv run --project packages/providers providers llama-cpp download
uv run --project packages/providers providers llama-cpp start --model /path/to/model.gguf
uv run --project packages/providers providers llama-cpp status
uv run --project packages/providers providers llama-cpp stop
```
