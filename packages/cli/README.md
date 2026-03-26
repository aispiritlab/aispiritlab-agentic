# CLI - AI Spirit Agent Command-Line Interface

Command-line interface for interacting with the AI Spirit Agent.

## Features

- **Interactive TUI**: Textual + Rich chat client with persistent transcript, activity feed, and keyboard shortcuts
- **Single Query Mode**: Ask a single question and get a response
- **History Reset**: Clear active conversation state from CLI
- Built with Click, Textual, and Rich

## Installation

```bash
uv sync
```

## Usage

### Interactive Mode

Launch an interactive conversation session:

```bash
uv run ai-spirit-cli
```

This opens the Textual terminal client. Useful bindings:

- `Ctrl+Enter`: send message
- `Ctrl+L`: clear conversation state
- `1`: switch to routed agents mode
- `2`: switch to direct chat mode
- `3`: switch to image mode
- `Esc`: focus the composer

Or using the full module path:

```bash
uv run python -m cli
```

### Single Query Mode

Ask a single question:

```bash
uv run ai-spirit-cli chat "What is Flow PHP?"
```

Run a single personalization step:

```bash
uv run ai-spirit-cli personalizacja "Mam na imię Mateusz, vault to /Users/me/vault"
```

Print prompt template by enum name:

```bash
uv run ai-spirit-cli get-prompt Note
uv run ai-spirit-cli get-prompt Greetings
```

Start training command:

```bash
uv run ai-spirit-cli trenowanie start
```

## Commands

- **No command**: Launches the interactive Textual chat client
- **chat**: Ask a single question in chat mode and exit
- **personalizacja**: Send one message to the personalization flow and exit
- **get-prompt**: Print a prompt template (`Note` or `Greetings`)
- **trenowanie start**: Show training placeholder info (`Wkrotce`)
- **clear-history**: Clear active history and print fresh greeting

## Dependencies

- `agentic-system`: The agent logic layer
- `click>=8.3,<8.4`: Command-line interface framework
