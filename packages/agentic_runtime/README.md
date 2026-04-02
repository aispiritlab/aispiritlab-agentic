# Agentic Runtime

Generic agent orchestration framework with message bus, event sourcing, and distributed support via Redis Streams.

## Features

- **AgenticRuntime** — workflow registration, routing, and turn execution
- **Message bus** — publish/subscribe for events (`UserMessage`, `TurnStarted`, `TurnCompleted`)
- **SQLiteMessageStore** — persistent conversation log with event sourcing
- **WorkflowOutputHandler** — subscriber pattern for automatic output handling
- **Distributed mode** — Redis Streams transport for multi-process agent systems
- **Fine-tuning export** — JSONL dataset generation from conversation history

## Usage

```python
from agentic_runtime import AgenticRuntime, WorkflowOutputHandler

runtime = AgenticRuntime()
runtime.register_workflow("notes", notes_workflow)
runtime.register_workflow("search", search_workflow)

response = await runtime.process("Find my notes about Python")
```

## Distributed mode

Set `AGENTIC_TRANSPORT=redis_streams` and `REDIS_URL` to run agents across multiple processes:

```bash
AGENTIC_TRANSPORT=redis_streams REDIS_URL=redis://localhost:6379 uv run personal-assistant
```

## Testing

```bash
make test-runtime            # unit tests
make test-resilience         # distributed resilience tests
```
