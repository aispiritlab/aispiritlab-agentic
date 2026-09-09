# Personal Assistant

Multi-agent system for note management, knowledge discovery, and decision support. Provides a Gradio chat interface with voice input and image generation.

## Agents

| Agent | Purpose |
|-------|---------|
| **Router** | Routes user messages to the correct workflow |
| **Sage** | Provides advice and insight based on context |
| **ManageNotes** | CRUD operations on notes (create, edit, delete) |
| **DiscoveryNotes** | Discovers relevant notes from the vault |
| **Organizer** | Auto-organizes newly created notes |
| **Personalize** | Learns user preferences over time |

## Architecture

`PARuntime` orchestrates all workflows through a message bus:

```
User message → Router → [Sage | ManageNotes | DiscoveryNotes] → Organizer (post-hook)
```

Each workflow is an event-driven pipeline built on `agentic_runtime.AgenticRuntime`.

## Usage

### Gradio chat UI

```bash
make chat
# or
uv run personal-assistant
```

The UI provides three modes:
- **Agenci** — full agent routing (notes, search, advice)
- **Chat** — direct LLM conversation
- **Generate image** — text-to-image via MFLUX

### As a library

```python
from personal_assistant import get_runtime, ai_spirit_agent

runtime = get_runtime(user="alice", workspace="default")
response = ai_spirit_agent("Summarize my recent notes", runtime=runtime)
```

### Direct agent access

```python
from personal_assistant import sage_agent, chat_agent

result = sage_agent("What should I focus on today?")
result = chat_agent("Explain event sourcing in Python")
```

## Configuration

Inherits from `agentic_runtime.Settings`:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `http://localhost:1234` | LLM server endpoint |
| `AGENTIC_MODEL` | — | Model name for inference |
| `AGENTIC_TRANSPORT` | `local` | `local` or `laser` |

## Testing

```bash
make test-pa                 # unit tests
make test-e2e                # workflow smoke tests (requires live model)
make test-e2e-live           # full end-to-end tests
```
