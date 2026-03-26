# Architecture Diagrams

## 1. General Architecture Overview

All packages with their internal components shown as nested boxes.

```mermaid
flowchart TB
  subgraph Interfaces
    cli[CLI]
    tui[TUI / Textual]
    chat[Chat / Gradio]
    voice[Voice / MLX Parakeet]
  end

  subgraph Workshops
    lab0["Lab 0: Simple Agent + Tools"]
    lab1["Lab 1: Writer-Critic Events"]
    lab2["Lab 2: Planner For-Loop"]
    lab3["Lab 3: Event-Driven Planner"]
    lab4["Lab 4: Full Runtime + Bus"]
    lab5["Lab 5: Vision Image → Art"]
  end

  subgraph agentic ["agentic"]
    subgraph Core
      agent["Agent / CoreAgentic"]
      llm_call["LLM Call"]
      prompts["PromptBuilder"]
    end

    subgraph ToolsModels ["Tools & Models"]
      tools["Tools / Toolsets"]
      models["Models / ModelConfig"]
    end

    subgraph Providers
      mlx["MLX LM"]
      mlx_vlm["MLX VLM"]
      mlx_audio["MLX Audio"]
      api["API / OpenAI"]
      onnx["ONNX ASR"]
    end

    subgraph WorkflowSys ["Workflow Subsystem"]
      stream["MessageStream"]
      consumer["MessageConsumer"]
      decider["Decider fn"]
      reactor["Reactor\n(LLM / MultiTurn)"]
      routing["TechnicalRoutingFn"]
      bus["InMemoryMessageBus"]
      output_h["OutputHandlerDispatcher"]
      turn_exec["TurnExecutor"]
      builder["WorkflowBuilder"]
      wf_runtime["WorkflowRuntime"]
    end
  end

  subgraph agentic_runtime ["agentic_runtime"]
    ag_runtime["AgenticRuntime"]
    router["RouterAgent"]
    msg_store["SQLiteMessageStore"]
    subgraph DomainWorkflows ["Domain Workflows"]
      personalize["Personalize"]
      manage_notes["ManageNotes"]
      discovery["DiscoveryNotes"]
      sage["Sage"]
      organizer["Organizer"]
    end
    subgraph OutputHandlers ["Output Handlers"]
      oh_organizer["Organizer Handler\n(CreatedNote)"]
      oh_rag["RAG Handler\n(NoteUpdated/Deleted)"]
    end
  end

  subgraph KB ["knowledge_base"]
    knowledge["KnowledgeBase"]
    qdrant["Qdrant Vector Store"]
  end

  Interfaces --> agentic_runtime
  Interfaces --> agentic
  Workshops --> agentic
  agentic_runtime --> agentic
  oh_organizer --> organizer
  oh_rag --> KB
  discovery --> KB
```

---

## 2. Chat & CLI Interface Flow

How user input flows from any interface through the runtime modes.

```mermaid
sequenceDiagram
  box rgb(21, 67, 130) Interface
    actor User
    participant UI as Chat (Gradio) / CLI / TUI
  end

  box rgb(153, 84, 10) Runtime
    participant RT as AgenticRuntime
    participant Router as RouterAgent
  end

  box rgb(27, 94, 32) Workflow
    participant WF as Selected Workflow
  end

  box rgb(81, 16, 100) Infrastructure
    participant Bus as MessageBus
    participant Store as SQLiteMessageStore
  end

  User->>UI: text / voice / file upload

  alt Voice input
    UI->>UI: convert_audio(MLX Parakeet) → text
  end

  alt Mode: Agenci (default)
    UI->>RT: ai_spirit_agent(text)
    RT->>RT: run(text) → UserMessage
    RT->>Router: route(text, available_workflows)
    Router-->>RT: workflow_name
    RT->>WF: workflow.handle(UserMessage)
    WF-->>RT: WorkflowExecution
    RT->>Bus: publish messages
    Bus->>Store: persist
    RT-->>UI: response text
  else Mode: Chat (direct LLM)
    UI->>RT: chat_agent(text)
    RT->>RT: run_chat(text) → LLMCall
    RT-->>UI: response text
  else Mode: Generate Image
    UI->>RT: generate_image_agent(text)
    RT->>RT: run_generate_image(text) → MFLUX
    RT-->>UI: ImageGenerationResult
  end

  UI-->>User: display response / image
```

---

## 3. AgenticRuntime — Routing & Workflow Dispatch

Full lifecycle of a user message through the runtime: routing, turn execution, event dispatch.

```mermaid
sequenceDiagram
  box rgb(21, 67, 130) Entry
    actor User
  end

  box rgb(153, 84, 10) Runtime
    participant RT as AgenticRuntime
    participant Router as RouterAgent
  end

  box rgb(27, 94, 32) Execution
    participant TE as TurnExecutor
  end

  box rgb(81, 16, 100) Infrastructure
    participant Bus as InMemoryMessageBus
    participant OHD as OutputHandlerDispatcher
    participant Store as SQLiteMessageStore
  end

  User->>RT: run(text)
  RT->>RT: UserMessage(domain="general")

  RT->>RT: _routable_workflows()
  Note over RT: Excludes organizer (event-only).<br/>Excludes personalize if finished.

  RT->>Router: route(text, workflows_summary)
  Router-->>RT: workflow_name (e.g. "manage_notes")

  RT->>RT: _plan_general_turn(message, workflow)
  Note over RT: TurnPlan with handler, pre_messages<br/>(PromptSnapshot + router decision)

  RT->>TE: execute(TurnPlan)
  TE->>Bus: publish(TurnStarted)
  TE->>Bus: publish(UserMessage)
  Bus->>Store: persist

  TE->>TE: handler(UserMessage) → WorkflowExecution

  TE->>Bus: publish(AssistantMessage)
  TE->>Bus: publish_many(emitted_events)

  Bus->>OHD: dispatch each event
  Note over OHD: CreatedNote → Organizer Handler<br/>NoteUpdated → RAG Handler

  TE->>Bus: publish(TurnCompleted)
  Bus->>Store: persist all

  RT-->>User: response text
```

---

## 4. ManageNotes Workflow (Decider + Events)

Workflow with custom decider that parses tool calls into domain events (CreatedNote, NoteUpdated).

```mermaid
sequenceDiagram
  box rgb(27, 94, 32) Execution
    participant TE as TurnExecutor
  end

  box rgb(13, 71, 161) Workflow Engine
    participant Stream as MessageStream
    participant Consumer as MessageConsumer
    participant Decider as manage_notes_decider
    participant Reactor as LLMReactor
  end

  box rgb(153, 84, 10) Agent
    participant Agent as ManageNotesAgent
    participant Tools as Toolset
  end

  box rgb(81, 16, 100) Infrastructure
    participant Bus as MessageBus
  end

  TE->>Stream: append(UserMessage)
  TE->>Consumer: consume(stream, decider, routing)

  Consumer->>Stream: read_next() → UserMessage
  Consumer->>Decider: decider(UserMessage)
  Decider-->>Consumer: [UserMessage]
  Consumer->>Reactor: invoke(UserMessage)
  Reactor->>Agent: respond(text)
  Agent->>Tools: add_note(name, content)
  Tools-->>Agent: ToolRunResult
  Agent-->>Reactor: LLMResponse (tool_calls in metadata)
  Reactor-->>Consumer: LLMResponse
  Consumer->>Stream: append(LLMResponse)

  Consumer->>Stream: read_next() → LLMResponse
  Consumer->>Decider: decider(LLMResponse)
  Note over Decider: Parse tool_calls:<br/>AddNote → [CreatedNote, NoteUpdated]
  Decider-->>Consumer: [CreatedNote, NoteUpdated]

  Consumer->>Stream: append(CreatedNote)
  Consumer->>Stream: append(NoteUpdated)
  Note over Consumer: No reactor for events → stay in history

  Consumer-->>TE: done
  TE->>TE: build WorkflowExecution(text, emitted_events)

  TE->>Bus: publish(AssistantMessage)
  TE->>Bus: publish(CreatedNote)
  TE->>Bus: publish(NoteUpdated)

  Note over Bus: CreatedNote → Organizer Handler<br/>NoteUpdated → RAG Handler
```

---

## 5. Sage Workflow (MultiTurn + Thinking)

Sage uses a thinking model. The MultiTurnLLMReactor strips `<think>` tags in post-processing.

```mermaid
sequenceDiagram
  box rgb(27, 94, 32) Execution
    participant TE as TurnExecutor
  end

  box rgb(13, 71, 161) Workflow Engine
    participant WF as SageWorkflow
    participant Reactor as MultiTurnLLMReactor
  end

  box rgb(153, 84, 10) Agent
    participant Agent as SageAgent
    participant LLM as Thinking Model
  end

  TE->>WF: handle(UserMessage)
  WF->>Reactor: invoke(UserMessage)
  Reactor->>Agent: respond(text)
  Agent->>LLM: prompt (Sage system prompt)
  LLM-->>Agent: response with <think>...</think>
  Agent-->>Reactor: AgentResult

  Note over Reactor: No tools → single turn
  Reactor->>Reactor: post_process: strip <think> tags
  Reactor-->>WF: LLMResponse (clean text)

  WF-->>TE: WorkflowExecution(text)
  Note over TE: No events emitted
```

---

## 6. DiscoveryNotes Workflow (RAG Search)

Uses semantic search tool to query the Qdrant knowledge base.

```mermaid
sequenceDiagram
  box rgb(27, 94, 32) Execution
    participant TE as TurnExecutor
  end

  box rgb(13, 71, 161) Workflow
    participant WF as DiscoveryNotesWorkflow
  end

  box rgb(153, 84, 10) Agent
    participant Agent as DiscoveryNotesAgent
    participant LLM as LLM
    participant Tools as Toolset
  end

  box rgb(81, 16, 100) Knowledge Base
    participant KB as Qdrant
  end

  TE->>WF: handle(UserMessage)
  WF->>Agent: respond(text)
  Agent->>LLM: prompt + tool schemas
  LLM-->>Agent: tool_call: semantic_search(query)
  Agent->>Tools: semantic_search(query)
  Tools->>KB: vector search
  KB-->>Tools: matching notes
  Tools-->>Agent: ToolRunResult
  Agent-->>WF: AgentResult (response text)
  WF-->>TE: WorkflowExecution(text)
  Note over TE: No events emitted
```

---

## 7. Personalize Workflow

Collects user preferences (name, vault). After completion, the workflow is excluded from routing.

```mermaid
sequenceDiagram
  box rgb(27, 94, 32) Execution
    participant TE as TurnExecutor
  end

  box rgb(13, 71, 161) Workflow
    participant WF as PersonalizeWorkflow
  end

  box rgb(153, 84, 10) Agent
    participant Agent as PersonalizeAgent
    participant LLM as LLM
    participant Tools as Toolset
  end

  box rgb(62, 39, 35) Storage
    participant Prefs as personalization.json
  end

  TE->>WF: handle(UserMessage)
  WF->>Agent: respond(text)
  Agent->>LLM: prompt (greeting) + tool schemas
  LLM-->>Agent: tool_call: set_preferences(name, vault)
  Agent->>Tools: set_preferences(name, vault)
  Tools->>Prefs: save to disk
  Prefs-->>Tools: ok
  Tools-->>Agent: ToolRunResult
  Agent-->>WF: AgentResult
  WF-->>TE: WorkflowExecution(text)

  Note over TE: After personalization complete,<br/>workflow excluded from routing
```

---

## 8. Organizer Workflow (Event-Driven, Not Routable)

Triggered only by CreatedNote events from the message bus. Classifies notes with PARA tags.

```mermaid
sequenceDiagram
  box rgb(81, 16, 100) Infrastructure
    participant Bus as MessageBus
    participant OH as Organizer OutputHandler
  end

  box rgb(13, 71, 161) Workflow Engine
    participant Stream as MessageStream
    participant Consumer as MessageConsumer
    participant Decider as organizer_decider
    participant Reactor as LLMReactor
  end

  box rgb(153, 84, 10) Agent
    participant Agent as OrganizerAgent
    participant Tools as Toolset
  end

  Bus->>OH: CreatedNote(note_name, note_content)
  OH->>Stream: append(CreatedNote)

  Stream->>Consumer: consume(stream, decider, routing)
  Consumer->>Stream: read_next() → CreatedNote
  Consumer->>Decider: decider(CreatedNote)
  Note over Decider: Convert to UserMessage:<br/>"Nazwa: X\nTreść: Y"
  Decider-->>Consumer: [UserMessage]

  Consumer->>Reactor: invoke(UserMessage)
  Reactor->>Agent: respond(formatted text)
  Agent->>Tools: tag_note(name, tags)
  Tools-->>Agent: ToolRunResult
  Agent-->>Reactor: LLMResponse (PARA classification)
  Reactor-->>Consumer: LLMResponse
  Consumer->>Stream: append(LLMResponse)

  Consumer->>Stream: read_next() → LLMResponse
  Consumer->>Decider: decider(LLMResponse) → []
  Note over Consumer: Stream empty → done

  OH-->>Bus: WorkflowExecution (classification result)
```

---

## 9. Cross-Workflow Event Flow (ManageNotes → Organizer → RAG)

Shows how creating a note triggers both the Organizer (PARA tagging) and RAG (knowledge base indexing) in parallel via output handlers.

```mermaid
sequenceDiagram
  box rgb(21, 67, 130) Entry
    actor User
  end

  box rgb(27, 94, 32) ManageNotes
    participant MN as ManageNotes Workflow
  end

  box rgb(81, 16, 100) Infrastructure
    participant Bus as MessageBus
    participant Store as SQLiteMessageStore
  end

  box rgb(153, 84, 10) Organizer Pipeline
    participant OH_Org as Organizer Handler
    participant Org as Organizer Workflow
  end

  box rgb(0, 77, 64) RAG Pipeline
    participant OH_RAG as RAG Handler
    participant KB as KnowledgeBase
  end

  User->>MN: "Create a note about AI"
  MN->>MN: Agent creates note via add_note tool
  MN-->>Bus: publish(AssistantMessage)
  MN-->>Bus: publish(CreatedNote)
  MN-->>Bus: publish(NoteUpdated)
  Bus->>Store: persist all

  par Organizer Handler
    Bus->>OH_Org: CreatedNote
    OH_Org->>Org: handle(CreatedNote)
    Org->>Org: Decider → UserMessage → LLMReactor
    Org->>Org: Agent classifies with PARA tags
    Org-->>OH_Org: WorkflowExecution (tags applied)
  and RAG Handler
    Bus->>OH_RAG: NoteUpdated
    OH_RAG->>KB: submit_update(note_path)
    KB->>KB: re-index note in Qdrant
    KB-->>OH_RAG: done
  end

  Bus-->>User: original response from ManageNotes
```

---

## 10. Lab 0 — Simple Agent + Tools

Direct agent call with tool execution. No workflow, no events.

```mermaid
sequenceDiagram
  box rgb(21, 67, 130) Interface
    actor User
    participant App as Lab0App
  end

  box rgb(153, 84, 10) Agent
    participant Agent as Agent
    participant LLM as Qwen LLM (MLX)
  end

  box rgb(27, 94, 32) Tools
    participant Tools as Toolset
  end

  User->>App: "What time is it?"
  App->>Agent: agent.run(message)
  Agent->>LLM: prompt + tool schemas
  LLM-->>Agent: AgentResult (tool_calls)
  Note over Agent: tool_calls = [get_current_time]
  Agent-->>App: AgentResult
  App->>Tools: toolsets.run_tool(tool_call)
  Tools-->>App: ToolRunResult
  App-->>User: Display tool call + result
```

---

## 11. Lab 1 — Writer-Critic with Events

Two agents coordinated via a domain event (WriterCompleted). Introduces the Decider pattern and `dispatch_output_handlers`.

```mermaid
sequenceDiagram
  box rgb(21, 67, 130) Interface
    actor User
    participant App as Lab1App
  end

  box rgb(27, 94, 32) Writer Workflow
    participant WF1 as run_workflow (Writer)
    participant WDecider as writer_decider
    participant WReactor as LLMReactor (Writer)
  end

  box rgb(81, 16, 100) Event Dispatch
    participant Dispatch as dispatch_output_handlers
  end

  box rgb(153, 84, 10) Critic Workflow
    participant WF2 as run_workflow (Critic)
    participant CDecider as critic_decider
    participant CReactor as LLMReactor (Critic)
  end

  User->>App: "Write about AI"
  App->>WF1: run_workflow(UserMessage)

  WF1->>WDecider: UserMessage
  WDecider-->>WF1: [UserMessage]
  WF1->>WReactor: invoke(UserMessage)
  WReactor-->>WF1: LLMResponse (writer text)
  WF1->>WDecider: LLMResponse
  WDecider-->>WF1: [WriterCompleted event]

  Note over WF1: No reactor for WriterCompleted → stream ends

  WF1-->>App: WorkflowExecution (text + WriterCompleted)
  App->>App: display Writer output

  App->>Dispatch: dispatch(handlers, emitted_events)
  Dispatch->>WF2: run_workflow(UserMessage with writer output)

  WF2->>CDecider: UserMessage
  CDecider-->>WF2: [UserMessage]
  WF2->>CReactor: invoke(UserMessage)
  CReactor-->>WF2: LLMResponse (critique)
  WF2->>CDecider: LLMResponse
  CDecider-->>WF2: [] (terminate)

  WF2-->>Dispatch: WorkflowExecution (critique)
  Dispatch-->>App: critic response
  App-->>User: Writer output + Critic feedback
```

---

## 12. Lab 2 — Planner For-Loop Delegation

Multi-agent orchestration with imperative for-loop. The planner creates tasks, then each is dispatched sequentially.

```mermaid
sequenceDiagram
  box rgb(21, 67, 130) Interface
    actor User
    participant App as Lab2App
  end

  box rgb(153, 84, 10) Planner
    participant Planner as PlannerAgent
  end

  box rgb(136, 14, 11) Imperative Orchestration
    participant ForLoop as for-loop
  end

  box rgb(27, 94, 32) Worker Workflows
    participant R_WF as run_workflow (Researcher)
    participant W_WF as run_workflow (Writer)
  end

  User->>App: "Research quantum computing"
  App->>Planner: planner.plan(message)
  Planner-->>App: [TaskDelegated(researcher), TaskDelegated(writer)]
  App->>App: display Plan

  Note over App,ForLoop: Imperative for-loop over tasks

  ForLoop->>R_WF: run_workflow(task_1)
  R_WF-->>ForLoop: WorkflowExecution
  ForLoop->>App: TaskCompleted(researcher)

  ForLoop->>W_WF: run_workflow(task_2)
  W_WF-->>ForLoop: WorkflowExecution
  ForLoop->>App: TaskCompleted(writer)

  App->>Planner: planner.summarize(all results)
  Planner-->>App: summary
  App-->>User: Plan + Results + Summary
```

---

## 13. Lab 3 — Event-Driven Planner (No For-Loops)

Same goal as Lab 2, but `dispatch_output_handlers` replaces the for-loop with declarative event routing.

```mermaid
sequenceDiagram
  box rgb(21, 67, 130) Interface
    actor User
    participant App as Lab3App
  end

  box rgb(153, 84, 10) Planner
    participant Planner as PlannerAgent
  end

  box rgb(81, 16, 100) Event Dispatch
    participant Dispatch as dispatch_output_handlers
    participant Handler as WorkerDispatchHandler
  end

  box rgb(27, 94, 32) Worker Workflows
    participant R_WF as run_workflow (Researcher)
    participant W_WF as run_workflow (Writer)
  end

  User->>App: "Research quantum computing"
  App->>Planner: planner.plan(message)
  Planner-->>App: [TaskDelegated(researcher), TaskDelegated(writer)]
  App->>App: display Plan

  App->>Dispatch: dispatch(handlers, task_events)

  Note over Dispatch,Handler: Event-driven dispatch replaces for-loop

  Dispatch->>Handler: TaskDelegated(researcher)
  Handler->>R_WF: run_workflow(UserMessage)
  R_WF-->>Handler: WorkflowExecution
  Handler->>App: on_completed(TaskCompleted)

  Dispatch->>Handler: TaskDelegated(writer)
  Handler->>W_WF: run_workflow(UserMessage)
  W_WF-->>Handler: WorkflowExecution
  Handler->>App: on_completed(TaskCompleted)

  Dispatch-->>App: all results

  App->>Planner: planner.summarize(results)
  Planner-->>App: summary
  App-->>User: Plan + Results + Summary
```

---

## 14. Lab 4 — Full Runtime with Message Bus

Complete infrastructure: central MessageBus, TurnExecutor lifecycle, OutputHandlerDispatcher, and message log for observability.

```mermaid
sequenceDiagram
  box rgb(21, 67, 130) Interface
    actor User
    participant App as Lab4App
  end

  box rgb(153, 84, 10) Runtime
    participant RT as WorkshopRuntime
    participant TE as TurnExecutor
  end

  box rgb(81, 16, 100) Infrastructure
    participant Bus as InMemoryMessageBus
    participant OHD as OutputHandlerDispatcher
    participant Log as Message Log
  end

  box rgb(27, 94, 32) Workflows
    participant PlanWF as Planner Workflow
    participant WorkerWF as Worker Workflow
  end

  User->>App: "Research quantum computing"
  App->>RT: runtime.run(message, "planner")
  RT->>TE: execute(TurnPlan)

  TE->>Bus: publish(TurnStarted)
  Bus->>Log: store
  TE->>Bus: publish(UserMessage)
  Bus->>Log: store

  TE->>PlanWF: handler(UserMessage)
  PlanWF-->>TE: WorkflowExecution + TaskDelegated events

  TE->>Bus: publish(AssistantMessage)
  Bus->>Log: store
  TE->>Bus: publish_many(TaskDelegated events)

  Note over Bus,OHD: Bus auto-dispatches to registered handlers

  Bus->>OHD: dispatch(TaskDelegated:researcher)
  OHD->>WorkerWF: handle(UserMessage)
  WorkerWF-->>OHD: WorkflowExecution
  OHD->>App: on_completed(TaskCompleted)

  Bus->>OHD: dispatch(TaskDelegated:writer)
  OHD->>WorkerWF: handle(UserMessage)
  WorkerWF-->>OHD: WorkflowExecution
  OHD->>App: on_completed(TaskCompleted)

  TE->>Bus: publish(TurnCompleted)
  Bus->>Log: store

  RT-->>App: plan text
  App-->>User: Plan + Results + Summary
```

---

## 15. Consumer Loop Detail

The core message processing loop used by all workflows. Decider decides WHAT happens, Reactor handles HOW.

```mermaid
sequenceDiagram
  box rgb(153, 84, 10) Caller
    participant Caller as run_workflow()
  end

  box rgb(13, 71, 161) Stream
    participant Stream as InMemoryMessageStream
  end

  box rgb(27, 94, 32) Consumer Loop
    participant Consumer as MessageConsumer
    participant Decider as Decider fn
    participant Routing as TechnicalRoutingFn
  end

  box rgb(136, 14, 11) Side Effects
    participant Reactor as Reactor
  end

  Caller->>Stream: append(UserMessage)
  Caller->>Consumer: consume(stream, decider, routing_fn)

  loop while stream.read_next() is not None
    Consumer->>Stream: read_next()
    Stream-->>Consumer: msg

    Consumer->>Decider: decider(msg)
    Decider-->>Consumer: commands: Sequence[Message]

    loop for each command
      Consumer->>Routing: routing_fn(command)

      alt Reactor found
        Routing-->>Consumer: reactor
        Consumer->>Reactor: reactor.invoke(command)
        Reactor-->>Consumer: output Message
        Consumer->>Stream: append(output)
      else No reactor
        Routing-->>Consumer: None
        Consumer->>Stream: append(command as-is)
        Note over Stream: Event stays in history
      end
    end
  end

  Note over Consumer: Stream empty → done
  Caller->>Stream: all_messages()
  Stream-->>Caller: build WorkflowExecution
```

---

## 16. Full End-to-End Flow — Chat to All Agents

Complete swimlane showing the entire message journey from Chat/CLI through routing to every domain workflow, with event-driven side effects.

```mermaid
sequenceDiagram
  box rgb(21, 67, 130) Interface
    actor User
    participant UI as Chat / CLI / TUI
  end

  box rgb(153, 84, 10) Runtime & Routing
    participant RT as AgenticRuntime
    participant Router as RouterAgent
    participant TE as TurnExecutor
  end

  box rgb(81, 16, 100) Infrastructure
    participant Bus as MessageBus
    participant OHD as OutputHandlerDispatcher
    participant Store as SQLiteMessageStore
  end

  box rgb(27, 94, 32) Domain Workflows
    participant PW as PersonalizeWorkflow
    participant MN as ManageNotesWorkflow
    participant DN as DiscoveryNotesWorkflow
    participant SG as SageWorkflow
    participant OR as OrganizerWorkflow
  end

  box rgb(0, 77, 64) Knowledge Base
    participant KB as Qdrant
  end

  User->>UI: text / voice
  UI->>RT: ai_spirit_agent(text)
  RT->>RT: UserMessage(domain="general")

  RT->>Router: route(text, workflows_summary)
  Note over Router: Available: personalize, manage_notes,<br/>discovery_notes, sage<br/>(organizer excluded — event-only)

  alt Router selects "personalize"
    Router-->>RT: "personalize"
    RT->>TE: execute(TurnPlan)
    TE->>Bus: publish(TurnStarted + UserMessage)
    Bus->>Store: persist
    TE->>PW: handle(UserMessage)
    PW->>PW: Agent → set_preferences tool
    PW-->>TE: WorkflowExecution
    TE->>Bus: publish(AssistantMessage + TurnCompleted)
    Bus->>Store: persist
    RT-->>UI: greeting + preferences saved

  else Router selects "manage_notes"
    Router-->>RT: "manage_notes"
    RT->>TE: execute(TurnPlan)
    TE->>Bus: publish(TurnStarted + UserMessage)
    Bus->>Store: persist
    TE->>MN: handle(UserMessage)
    MN->>MN: Agent → add_note tool
    MN->>MN: Decider emits CreatedNote + NoteUpdated
    MN-->>TE: WorkflowExecution(text, events)
    TE->>Bus: publish(AssistantMessage)
    TE->>Bus: publish(CreatedNote)
    TE->>Bus: publish(NoteUpdated)

    par Organizer (auto-triggered)
      Bus->>OHD: CreatedNote
      OHD->>OR: handle(CreatedNote)
      OR->>OR: Decider → UserMessage → Agent → tag_note
      OR-->>OHD: PARA classification
    and RAG (auto-triggered)
      Bus->>OHD: NoteUpdated
      OHD->>KB: submit_update(note_path)
      KB->>KB: re-index in Qdrant
    end

    TE->>Bus: publish(TurnCompleted)
    Bus->>Store: persist
    RT-->>UI: note created + confirmed

  else Router selects "discovery_notes"
    Router-->>RT: "discovery_notes"
    RT->>TE: execute(TurnPlan)
    TE->>Bus: publish(TurnStarted + UserMessage)
    Bus->>Store: persist
    TE->>DN: handle(UserMessage)
    DN->>DN: Agent → semantic_search tool
    DN->>KB: vector search
    KB-->>DN: matching notes
    DN-->>TE: WorkflowExecution
    TE->>Bus: publish(AssistantMessage + TurnCompleted)
    Bus->>Store: persist
    RT-->>UI: search results

  else Router selects "sage"
    Router-->>RT: "sage"
    RT->>TE: execute(TurnPlan)
    TE->>Bus: publish(TurnStarted + UserMessage)
    Bus->>Store: persist
    TE->>SG: handle(UserMessage)
    SG->>SG: Agent → Thinking Model
    SG->>SG: strip <think> tags
    SG-->>TE: WorkflowExecution
    TE->>Bus: publish(AssistantMessage + TurnCompleted)
    Bus->>Store: persist
    RT-->>UI: sage response
  end

  UI-->>User: display response
```

---

## 17. Lab 5 — Vision: Image Summary → Art Writer

User sends image path + text. VLM agent describes the image. Art Writer creates a creative piece from the description. Same pattern as Lab 1 (two agents bridged by domain event) with a new modality.

```mermaid
sequenceDiagram
  box rgb(21, 67, 130) Interface
    actor User
    participant App as Lab5App
  end

  box rgb(27, 94, 32) Image Summary Workflow
    participant WF1 as run_workflow (Summarizer)
    participant IDecider as image_decider
    participant VReactor as VLMReactor
  end

  box rgb(81, 16, 100) Event Dispatch
    participant Dispatch as dispatch_output_handlers
  end

  box rgb(153, 84, 10) Art Writer Workflow
    participant WF2 as run_workflow (Art Writer)
    participant ADecider as art_writer_decider
    participant LReactor as LLMReactor
  end

  User->>App: "/path/to/photo.jpg | What emotions does this evoke?"
  App->>App: parse → ImageMessage(text, image_path)

  App->>WF1: run_workflow(ImageMessage)

  WF1->>IDecider: ImageMessage
  IDecider-->>WF1: [ImageMessage]
  WF1->>VReactor: invoke(ImageMessage)
  Note over VReactor: agent.respond(text, images=image_path)<br/>VLM sees both text and image
  VReactor-->>WF1: LLMResponse (image description)
  WF1->>IDecider: LLMResponse
  IDecider-->>WF1: [ImageDescribed event]

  Note over WF1: No reactor for ImageDescribed → stream ends

  WF1-->>App: WorkflowExecution (description + ImageDescribed)
  App->>App: display image description

  App->>Dispatch: dispatch(handlers, emitted_events)
  Dispatch->>WF2: run_workflow(UserMessage with description)

  WF2->>ADecider: UserMessage
  ADecider-->>WF2: [UserMessage]
  WF2->>LReactor: invoke(UserMessage)
  LReactor-->>WF2: LLMResponse (poem/prose)
  WF2->>ADecider: LLMResponse
  ADecider-->>WF2: [] (terminate)

  WF2-->>Dispatch: WorkflowExecution (creative text)
  Dispatch-->>App: art response
  App-->>User: Image description + Creative art piece
```
