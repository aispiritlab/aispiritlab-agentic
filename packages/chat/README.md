# Chat

Reusable Gradio UI building blocks for chat-based agent applications.

## Features

- **ChatHistory / ChatMessage** — type aliases for Gradio chatbot state
- **MultimodalMessage** — messages with text and file attachments
- **add_message()** — append messages to chat history
- **append_voice_response()** — transcribe audio and add to chat
- **ChatAppConfig** — server configuration (name, host, port)
- **launch()** — start Gradio app with graceful shutdown handlers

## Usage

```python
from chat import ChatHistory, add_message, launch, ChatAppConfig

history: ChatHistory = []
history = add_message(history, role="assistant", content="Hello!")

config = ChatAppConfig(server_name="0.0.0.0", server_port=7860)
launch(demo, config=config)
```
