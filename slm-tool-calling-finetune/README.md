# SLM Tool-Calling Fine-Tune

Fine-tunes **Qwen3.5-4B** for tool/function calling using **MLX** on Apple Silicon. No CUDA, no PyTorch — Metal GPU only.

## Requirements

- macOS 15+ on Apple Silicon (M-series)
- Python 3.11 or 3.12
- ~20GB free disk space (model weights + data)

## Setup

```bash
# From repo root
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r slm-tool-calling-finetune/requirements.txt
```

## Project Structure

```
slm-tool-calling-finetune/
├── config.py              # All hyperparameters and paths (single source of truth)
├── lora_config.yaml       # LoRA settings (rank 16, alpha 32)
├── requirements.txt       # MLX-only dependencies
├── download_model.py      # Download and verify base model from HuggingFace
├── data/
│   ├── preprocess.py      # Convert xLAM dataset → JSONL training format
│   └── augment_negatives.py  # Add negative examples (no-tool-needed cases)
├── inference.py           # Run inference with trained adapter
├── evaluate.py            # Evaluate tool-calling accuracy
├── train.sh               # Launch MLX LoRA training
├── fuse.sh                # Fuse adapter into base model
├── export_gguf.sh         # Export to GGUF format
├── adapters/              # LoRA adapter weights (output of training)
├── fused_model/           # Fused model weights (output of fuse.sh)
└── tests/
    └── test_tool_calling.py
```

## Commands

### Download base model

```bash
cd slm-tool-calling-finetune
python download_model.py
```

Downloads `mlx-community/Qwen3.5-4B-MLX-bf16` (~8GB) and verifies:
- Tokenizer vocab and ChatML tokens (`<|im_start|>`, `<|im_end|>`)
- Chat template functionality
- Basic generation

> **Note:** This model does not include native `<|function_calls|>` tokens. The preprocessing pipeline uses XML-style tool-call tags instead (`TOOL_CALL_FORMAT = "xml"` in `config.py`).

### Explore dataset

```bash
cd slm-tool-calling-finetune
python data/explore_dataset.py
```

Requires HuggingFace authentication (`huggingface_hub.login()`). Downloads and inspects
`Salesforce/xlam-function-calling-60k` (60,000 examples).

**Dataset schema (confirmed by exploration):**

| Column | Type | Notes |
|--------|------|-------|
| `id` | int | Row index |
| `query` | str | User query |
| `tools` | str (JSON) | List of `{name, description, parameters}` |
| `answers` | str (JSON) | List of `{name, arguments}` — `arguments` is a **dict**, not a string |

**Key findings:**
- `arguments` is a native dict — only one `json.loads()` needed (spec said two; that is incorrect for this dataset version)
- 52.8% of examples contain parallel tool calls (>1 call per query)
- 0 parse failures across first 1,000 rows

## Configuration

All hyperparameters are in `config.py`. Key values:

| Setting | Value |
|---------|-------|
| Model | `mlx-community/Qwen3.5-4B-MLX-bf16` |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Batch size | 4 |
| Learning rate | 1e-5 |
| Max seq length | 2048 |
| Tool call format | xml |
