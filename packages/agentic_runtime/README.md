# Agentic Runtime

Generic agent orchestration framework with message bus, event sourcing, and distributed support over Apache Iggy.

## Features

- **AgenticRuntime** — workflow registration, routing, and turn execution
- **Message bus** — publish/subscribe for events (`UserMessage`, `TurnStarted`, `TurnCompleted`)
- **SQLiteMessageStore** — persistent conversation log with event sourcing
- **WorkflowOutputHandler** — subscriber pattern for automatic output handling
- **Distributed mode** — Apache Iggy transport, through the Laser SDK, for multi-process agent systems
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

Set `AGENTIC_TRANSPORT=laser` and `LASER_CONNECTION_STRING` to run agents across
multiple processes. `make iggy` starts a local broker.

```bash
AGENTIC_TRANSPORT=laser LASER_CONNECTION_STRING=iggy:iggy@127.0.0.1:8090 uv run personal-assistant
```

One Iggy stream holds every topic: `control` and `health` carry the registry,
`messages.<agent>` one agent's inbox, `dead-letter.<agent>` what it could not
handle. See `distributed/transport.py` for what an offset does that an `XACK`
did not.

## Testing

```bash
make test-runtime            # unit tests
make iggy                    # a local broker, in another shell
make test-resilience         # distributed resilience tests
```

`tests/test_laser_transport.py` covers the offset ledger with no broker at all —
it is the part an offset-based log gets wrong, so it must always run.
`tests/test_laser_transport_integration.py` needs a real one and skips without
it; `AGENTIC_TEST_LASER_CONNECTION` points it somewhere other than the default.
