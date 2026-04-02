## agentic ASR

Minimal SDK ASR API is available under `agentic.asr`:

```python
from agentic.asr import load_model, transcribe

load_model("Qwen/Qwen3-ASR-1.7B")

result = transcribe(
    byteobject,
    model="Qwen/Qwen3-ASR-1.7B",
    language="Polish",
    on_progress=lambda e: print(e["event"], e.get("progress", 0.0)),
    verbose=True,
)
print(result.text)
```

### Runtime Requirements

- `ffmpeg` is required for non-WAV encoded bytes (mp3, m4a, webm, etc.).
- WAV bytes are decoded through the vendored fast-path parser and do not require `ffmpeg`
  unless resampling is needed.

## Acknowledgements

The event sourcing primitives in `agentic.workflow` (event store, decider, processor, saga)
are inspired by [Emmett](https://github.com/event-driven-io/emmett) by
[@oskardudycz](https://github.com/oskardudycz) — a perfect example of event-driven
messaging done right. Thank you for the excellent reference that shaped the foundation
of this agentic SDK.
