# Workshops

Educational labs progressing from simple agents to distributed multi-agent systems.

## Labs

| Lab | Topic | Key Concepts |
|-----|-------|--------------|
| **Lab 0** | Simple agent | Tools, prompts, basic LLM call |
| **Lab 1** | Multi-agent workflows | Critic/Writer workflow, deciders, events |
| **Lab 2** | Planner agent | PlannerAgent, for-loop orchestration |
| **Lab 3** | Event-driven agents | Output handlers, event routing (no loops) |
| **Lab 4** | Full runtime | WorkshopRuntime, message bus, lifecycle events |
| **Lab 5** | Multi-provider | ChatPrompt vs QwenPrompt, dual-model support |
| **Lab 6** | Distributed agents | DistributedAgenticRuntime, Redis Streams |

## Usage

```bash
uv run workshops           # interactive lab selector
```

### Run individual labs

```bash
uv run workshops lab0
uv run workshops lab3
uv run workshops lab6
```

### Distributed lab (Lab 6)

```bash
make lab6-up               # start Redis + distributed agents
make lab6-down             # stop the stack
```
