# Tunning Model

Colab-first training assets for fine-tuning `Qwen/Qwen3.5-4B` on the
repository-grounded datasets produced by `packages/dataloader`.

## Main Deliverable

- Notebook:
  - `packages/tunning_model/notebooks/train_qwen35_4b_unsloth_colab.ipynb` (existing quality-focused flow)
  - `packages/tunning_model/notebooks/train_qwen3_8b_unsloth_colab.md` (single-page markdown runbook)
  - `packages/tunning_model/notebooks/train_qwen3_8b_unsloth_colab.ipynb` (single-page Colab notebook)
  - `packages/tunning_model/notebooks/train_qwen3_4b_simple_sft_colab.md` (simple `SFTTrainer` flow)
  - `packages/tunning_model/notebooks/train_qwen3_4b_simple_sft_colab.ipynb` (simple `SFTTrainer` flow)

The notebook is designed for Google Colab and focuses on quality over speed:

- dataset QA before training
- longer context (`8192`)
- higher-capacity LoRA (`r=32`)
- staged curriculum:
  1. chat foundation
  2. agent specialization
  3. mixed refresh
- held-out evaluation before export

## Required Inputs

Generate the normalized datasets outside Colab with the existing dataloader
pipeline:

```bash
make flow-php-repo-dataset
```

Upload or mount these files in Colab:

- `packages/dataloader/data/flow_php_repo/flow-php-sft-chat.jsonl`
- `packages/dataloader/data/flow_php_repo/flow-php-sft-agent.jsonl`

The notebook expects the current dataloader schema:

- chat rows: `messages`, `metadata`
- agent rows: `messages`, `tools`, `metadata`, optional `reasoning`

## Training Defaults

The notebook uses the upstream Hugging Face model:

- `Qwen/Qwen3.5-4B`

The notebook is configured for 16-bit LoRA training. It prefers `bf16` on
supported GPUs and falls back to `fp16` only when BF16 is unavailable.

Default training profile:

- `max_seq_length=8192`
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=16`
- `lora_r=32`
- `lora_alpha=64`
- `lora_dropout=0.05`
- `packing=False`
- assistant-only loss masking

Stage learning rates:

- stage 1 chat: `1e-4`
- stage 2 agent: `8e-5`
- stage 3 mixed refresh: `5e-5`

## Exports

The notebook produces:

- local LoRA adapters for each stage
- a merged 16-bit `safetensors` checkpoint
- a GGUF Hugging Face repo with:
  - `q4_k_m`
  - `q8_0`

The merged 16-bit Hugging Face repo is the canonical artifact for any later
manual MLX conversion.

## Colab Flow

1. Open the notebook in Colab.
2. Run the dependency bootstrap cell once and let the runtime restart.
3. Mount Drive or upload the two normalized JSONL exports.
4. Set `HF_TOKEN`, `HF_NAMESPACE`, and dataset paths in the config cell.
5. Run the notebook top to bottom.
6. Review held-out evaluation and smoke-test output before pushing artifacts.

## References

- [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Unsloth notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks)
- [Unsloth Qwen guide](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune)
- [Unsloth chat templates](https://unsloth.ai/docs/basics/chat-templates)
- [Unsloth saving to GGUF](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)
- [Unsloth saving to vLLM / merged 16-bit](https://unsloth.ai/docs/basics/saving-and-using-models/saving-to-vllm)
