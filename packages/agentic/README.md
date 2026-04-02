## agentic ASR

## Migration Note

Provider, model, and image-generation modules that used to live under `agentic` now live in
`providers`.

```python
# old
from agentic.models import ModelConfig, ModelProvider
from agentic.image_generation_call import MfluxImageCall

# new
from providers import ModelConfig, ModelProvider
from providers.image.mflux import MfluxImageCall
```

The old import paths were removed intentionally and are no longer part of the supported `agentic`
API.

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
