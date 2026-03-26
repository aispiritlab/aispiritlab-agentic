# AI Spirit Agent

Multi-agent system built as a modular Python workspace. The project separates reusable agent infrastructure from application-specific agents, making it easy to build new agent applications from shared building blocks.

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager
- macOS recommended (MLX acceleration for local models)

## Quickstart

```bash
uv sync
```

Run the docs site:

```bash
cd apps/docs
npm install
npm run dev
```

Run the chat interface:

1. Terminal 1: `make mlflow-ui`
2. Terminal 2: `make registry`
3. Terminal 3: `make chat`

## Architecture

```
packages/
  agentic/             Core agent SDK (Agent, CoreAgentic, tools, prompts, models)
  agentic_runtime/     Generic agent orchestration framework (runtime, messaging, distributed)
  chat/                Reusable Gradio UI building blocks
  personal_assistant/  Concrete PA application (5 agents, Gradio UI, domain events)
  cli/                 CLI interface
  registry/            Prompt registry (MLflow-backed)
  evaluation/          Evaluation and benchmarking
  knowledge_base/      Vector search and note storage
  core/                Shared utilities
  dataloader/          Dataset loading
  workshops/           Educational labs
```

### Package dependency graph

```
personal_assistant --> agentic_runtime (framework)
                   --> agentic (SDK)
                   --> chat (UI library)
                   --> registry, evaluation, knowledge_base

chat (UI library)  --> gradio

agentic_runtime    --> agentic (SDK)

cli                --> personal_assistant
```

### Key packages

**agentic** -- Core agent SDK. Provides `Agent`, `CoreAgentic`, model providers, prompt builders, toolsets, and workflow primitives. This is the low-level foundation.

**agentic_runtime** -- Generic orchestration framework. `AgenticRuntime` accepts workflows, a router, and output handlers as constructor parameters. Also provides messaging infrastructure, distributed runtime (Redis Streams), storage, tracing, and turn execution. No application-specific code.

**chat** -- Reusable Gradio UI building blocks: `add_message()`, `append_voice_response()`, `ChatAppConfig`, `install_shutdown_handlers()`, `launch()`. Any agent application can compose its UI from these components.

**personal_assistant** -- The concrete application. Contains 5 specialized agents (manage_notes, organizer, discovery_notes, sage, personalize), a router, domain events, output handlers, and the full Gradio UI. Entry point: `personal-assistant`.

## Personal Assistant Agents

| Agent | Purpose |
|-------|---------|
| **manage_notes** | CRUD operations on Obsidian notes |
| **organizer** | PARA-based note classification (event-driven) |
| **discovery_notes** | Semantic search via RAG knowledge base |
| **sage** | Decision support using 7-step process (thinking model) |
| **personalize** | User onboarding and vault configuration |

## Commands

Run `make help` to see all available commands.

### Development

```bash
make install          # Install dependencies
make lint             # Check code style
make format           # Auto-format code
make mypy             # Type checking
make clean            # Remove compiled files and caches
```

### Testing

```bash
make test             # All unit tests
make test-pa          # Personal assistant tests
make test-runtime     # Framework tests
make test-agentic     # SDK tests
make test-e2e         # Workflow smoke tests (requires live model)
make test-e2e-live    # Full end-to-end tests (requires live model)
```

### Running

```bash
make chat             # Gradio chat interface
make cli              # Interactive CLI
make registry         # Prompt registry
make generate-kb      # Generate knowledge base
```

## Distributed Mode (Lab 6)

1. Copy `.env.example` to `.env` and set `API_BASE_URL`, `LANGSEARCH_API_KEY`
2. Start the stack: `make lab6-up`
3. Open chat at `http://localhost:7860`
4. Showcase client: `uv run --package workshops workshops lab6`
5. Stop: `make lab6-down`

## Configuration

Settings are loaded from environment variables or `.env` file. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen3.5-4B` | Main LLM model |
| `ORCHESTRATION_MODEL_NAME` | `Qwen/Qwen3.5-2B` | Router model |
| `THINKINK_MODEL` | `Qwen/Qwen3.5-9B` | Thinking model (Sage) |
| `API_BASE_URL` | `http://localhost:1234` | LLM API endpoint |
| `AGENTIC_TRANSPORT` | `in_memory` | `in_memory` or `redis_streams` |
| `CHAT_SERVER_PORT` | `7860` | Gradio server port |
