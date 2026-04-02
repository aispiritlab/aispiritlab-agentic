# Agentic Graph

Visual agent composition tool. Build multi-agent workflows by connecting nodes on a canvas, then validate, compile, and execute them.

## Features

- **Graph builder** — drag-and-drop agent node editor
- **Validation** — structural checks before compilation
- **Code generation** — export graph as executable Python code
- **Graph runtime** — run compiled graphs and stream events
- **Embeddable UI** — Gradio tab component for integration into larger apps

## Usage

```bash
uv run agentic-graph
```

### As a library

```python
from agentic_graph import AgenticGraphBuilder, GraphRuntime, validate_graph

builder = AgenticGraphBuilder()
graph = builder.compile(graph_data)
issues = validate_graph(graph)

runtime = GraphRuntime(graph)
result = await runtime.run("Summarize the document")
```

### Embed in Gradio app

```python
from agentic_graph import build_agent_builder_tab

with gr.Blocks() as app:
    build_agent_builder_tab()
```
