# Providers

Model providers, orchestrator, and subprocess-based inference management. Central package for all ML inference backends.

## Migration

The provider/model extraction from `agentic` is now complete. Import moved APIs from `providers` directly.

```python
# old
from agentic.models import ModelConfig, ModelProvider
from agentic.image_generation_call import MfluxImageCall

# new
from providers import ModelConfig, ModelProvider
from providers.image.mflux import MfluxImageCall
```

Compatibility shims for the removed `agentic.models`, `agentic.providers.*`, and
`agentic.image_generation_call` modules are not provided.

## Model Providers

| Provider | Backend | Type key |
|----------|---------|----------|
| **MlxProvider** | Apple MLX (text) | `mlx` |
| **MlxVlmProvider** | Apple MLX (vision) | `mlx-vlm` |
| **MlxAudioProvider** | Apple MLX (speech-to-text) | `mlx-audio` |
| **OpenAIProvider** | OpenAI-compatible HTTP API | `openai` |
| **VLLMProvider** | vLLM (native, OpenAI, Ray) | `vllm` |
| **TransformersProvider** | HuggingFace Transformers | `transformers` |
| **OnnxAsrProvider** | ONNX Runtime (ASR) | `onnx` |

All providers implement `ProviderProto` — a `Protocol` with four classmethods: `load_backend`, `build_model`, `close_backend`, `load`.

## Backend Dependencies

- Apple-local backends (`mlx`, `mlx-lm`, `mlx-vlm`, `mlx-audio`, `mflux`) are macOS-only.
- `TransformersProvider` requires a compatible `transformers` and `torch` install.
- `VLLMProvider` requires `vllm`; the Ray strategy also requires `ray`.
- `OnnxAsrProvider` requires `onnx-asr`.

## Orchestrator

`ModelProvider` manages model lifecycle (lazy loading, shared backends, reference counting, thread-safe inference locks):

```python
from providers import ModelProvider, ModelConfig

provider = ModelProvider("model-name", model_provider_type="openai", config=ModelConfig())
model = provider.get("model")
response = model.response("Hello")
provider.close()
```

## Image Generation

```python
from providers.image.mflux import MfluxImageCall

gen = MfluxImageCall(model_name="z-image-turbo")
result = gen.generate_image("A cat sitting on a cloud")
print(result.image_path)
```

## Subprocess Management

Download and manage local inference server binaries:

```bash
uv run providers list
uv run providers llama-cpp download
uv run providers llama-cpp start --model /path/to/model.gguf
uv run providers llama-cpp status
uv run providers llama-cpp stop
```

## Testing

```bash
uv run pytest packages/providers/tests -q
```
