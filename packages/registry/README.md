# Registry

Centralized prompt template management backed by MLflow.

## Features

- **Prompts enum** — typed access to all registered prompts (`MANAGE_NOTES`, `SAGE`, `GREETING`, etc.)
- **get_prompt()** — retrieve prompt template text by name
- **RegisterPrompt** — decorator for registering custom prompts
- **MLflow integration** — versioned prompt storage and experiment tracking

## Usage

```python
from registry import Prompts, get_prompt

template = get_prompt(Prompts.SAGE)
```

### Start the registry server

```bash
make registry
# or
uv run registry
```
