# Engineering Specification: Fine-Tuning Qwen 3.5-4B for Tool Calling on Apple Silicon

**Date:** March 20, 2026
**Target hardware:** MacBook Pro, Apple Silicon (M4-series), 128GB unified memory
**Target runtime:** Claude Code

---

## 1. Objective

Fine-tune **Qwen 3.5-4B** to excel at tool calling using the **Salesforce xLAM 60K** dataset via **Apple MLX** with LoRA on a Mac with 128GB unified memory. No NVIDIA GPU. No CUDA.

---

## 2. Component Selection

### 2.1 Model: Qwen 3.5-4B (MLX format)

- **HF ID:** `mlx-community/Qwen3.5-4B-MLX-bf16` (full precision — we have the memory)
- **Alternative:** `mlx-community/Qwen3.5-4B-MLX-4bit` (for faster iteration during development)
- **Why bf16 over 4-bit:** 128GB unified memory easily fits a 4B model at full precision (~8GB weights). Training on bf16 gives better gradient signal and higher quality output. No quantization error.
- **Why Qwen 3.5-4B:** Native tool-calling tokens (`<|function_calls|>` / `<|/function_calls|>`), Terminal-Bench 2.0 score of 52.5, 262K context window, pre-converted MLX weights available on HuggingFace

### 2.2 Dataset: Salesforce/xlam-function-calling-60k

- **HF ID:** `Salesforce/xlam-function-calling-60k`
- **Size:** 60,000 examples, 21 API domains, 3,673 unique APIs
- **Quality:** 3-stage verification (format → execution → semantic). Every example validated by running the function call.
- **Schema:** 4 columns — `id`, `query` (string), `tools` (JSON string), `answers` (JSON string)
- **Gotcha:** `answers[].arguments` is a JSON string INSIDE a JSON string. Must call `json.loads()` twice.

### 2.3 Framework: MLX / mlx-lm

- **Package:** `mlx-lm` (Apple's official LLM fine-tuning toolkit built on MLX)
- **Why MLX:** Purpose-built for Apple Silicon unified memory. Zero-copy between CPU and GPU. 2-3x faster than PyTorch MPS for LLM training. Native bf16 support. CLI-first workflow.
- **Why NOT Unsloth:** Unsloth is NVIDIA/CUDA only. Apple Silicon support is listed as "in the works" but not available.
- **Why NOT PyTorch MPS:** Missing operations, no bf16 support, requires CPU fallback for many ops, 2-3x slower than MLX.

---

## 3. Key Differences from NVIDIA Workflow

| Aspect | NVIDIA/CUDA | Apple Silicon/MLX |
|--------|------------|-------------------|
| Framework | Unsloth + HF TRL | mlx-lm |
| Training API | Python SFTTrainer | `mlx_lm.lora` CLI command |
| Data format | HF Datasets with `text` column | JSONL files (train.jsonl, valid.jsonl) |
| LoRA library | PEFT | Built into mlx-lm |
| Quantization | bitsandbytes | MLX native quantize |
| Precision | bf16 via CUDA | bf16 native on Apple Silicon |
| Optimizer | adamw_8bit | Adam (MLX native) |
| Merge/export | Unsloth save_pretrained_merged | `mlx_lm.fuse` |
| GGUF export | Unsloth save_pretrained_gguf | `mlx_lm.fuse` → llama.cpp convert |

**Packages NOT available on macOS:**
- `bitsandbytes` — not needed (128GB memory, use bf16)
- `flash-attn` — not needed (MLX has optimized attention)
- `apex` — not needed (MLX handles mixed precision)
- CUDA toolkit — not applicable

---

## 4. Environment & Dependencies

### 4.1 System Requirements

```
macOS:      15.0+ (Sequoia or later)
Python:     3.11 or 3.12
Chip:       Apple Silicon (M1/M2/M3/M4 series)
Memory:     128GB unified (8GB minimum for 4-bit, 16GB for bf16)
Disk:       ~20GB free (model weights + dataset + adapters)
Xcode CLT:  Required (for Metal compiler)
```

### 4.2 Dependencies

```
mlx>=0.22.0
mlx-lm>=0.22.0
numpy
huggingface-hub
datasets
```

That's it. No PyTorch, no transformers, no PEFT, no TRL, no bitsandbytes.

---

## 5. Data Format

MLX-LM expects **JSONL files** in a specific directory structure:

```
data/
├── train.jsonl
├── valid.jsonl
└── test.jsonl    (optional)
```

### 5.1 Chat Format (messages-based)

Each line in the JSONL is a conversation:

```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant with access to the following functions. Use them if required.\n\n[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\"}},\"required\":[\"location\"]}}}]"},
  {"role": "user", "content": "What's the weather in NYC?"},
  {"role": "assistant", "content": "<|function_calls|>\n[{\"name\":\"get_weather\",\"arguments\":{\"location\":\"New York City\"}}]\n<|/function_calls|>"}
]}
```

MLX-LM automatically applies the model's chat template from the tokenizer during training. The `messages` array maps directly to the template.

### 5.2 xLAM → JSONL Conversion

Each xLAM row transforms as:

1. Parse `tools` (JSON string → list of tool defs)
2. Parse `answers` (JSON string → list of calls, then parse each `arguments` string)
3. Normalize tool defs to OpenAI schema
4. Build system message: instruction text + JSON tool definitions
5. Build user message: the `query` field
6. Build assistant message: wrap calls in `<|function_calls|>` tokens
7. Write as one JSONL line with `{"messages": [...]}`

### 5.3 Negative Examples

Augment with ~12,000 examples where tools are present but the assistant answers naturally (no `<|function_calls|>`). Same JSONL format, same system message with tools, but the assistant content is plain text.

---

## 6. LoRA Configuration

### 6.1 Config File

MLX-LM accepts a LoRA config as a YAML or JSON file via `--lora-config`:

```yaml
# lora_config.yaml
rank: 16
alpha: 32
dropout: 0.0
scale: 2.0   # alpha / rank
```

### 6.2 Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--model` | `mlx-community/Qwen3.5-4B-MLX-bf16` | Full precision, pre-converted for MLX |
| `--data` | `./data` | Directory containing train.jsonl, valid.jsonl |
| `--train` | (flag) | Enable training mode |
| `--batch-size` | 4 | Conservative start; increase to 8 or 16 with 128GB |
| `--num-layers` | 16 | Apply LoRA to last 16 transformer layers |
| `--iters` | 1000 | Training iterations (NOT epochs) — see calculation below |
| `--learning-rate` | 1e-5 | Standard for MLX LoRA fine-tuning |
| `--steps-per-eval` | 100 | Evaluate every 100 steps |
| `--val-batches` | 25 | Number of validation batches per eval |
| `--steps-per-report` | 10 | Log training loss every 10 steps |
| `--adapter-path` | `./adapters` | Output directory for trained adapters |
| `--lora-config` | `./lora_config.yaml` | LoRA hyperparameters |
| `--max-seq-length` | 2048 | Max tokens per training sequence |
| `--grad-checkpoint` | (flag) | Enable gradient checkpointing (saves memory) |

### 6.3 Iterations Calculation

MLX uses iterations, not epochs. To train for 3 epochs on 66,000 examples (54K positive + 12K negative):

```
iterations = (dataset_size / batch_size) * num_epochs
           = (66000 / 4) * 3
           = 49,500 iterations
```

For a first run, start with fewer iterations (1,000–5,000) to validate the pipeline works, then scale up.

### 6.4 Estimated Training Time

On Apple Silicon M-series Ultra with 128GB:
- ~100-150 tokens/second for 4B model LoRA training
- 60K examples × 3 epochs ≈ **3-5 hours**
- Quick validation run (1000 iters) ≈ **15-20 minutes**

---

## 7. Training Command

The complete training command:

```bash
mlx_lm.lora \
  --model mlx-community/Qwen3.5-4B-MLX-bf16 \
  --data ./data \
  --train \
  --batch-size 4 \
  --num-layers 16 \
  --learning-rate 1e-5 \
  --iters 49500 \
  --steps-per-eval 200 \
  --val-batches 25 \
  --steps-per-report 10 \
  --adapter-path ./adapters \
  --lora-config ./lora_config.yaml \
  --max-seq-length 2048 \
  --grad-checkpoint
```

---

## 8. Model Export & Deployment

### 8.1 Fuse LoRA Adapters

After training, merge adapters back into the base model:

```bash
mlx_lm.fuse \
  --model mlx-community/Qwen3.5-4B-MLX-bf16 \
  --adapter-path ./adapters \
  --save-path ./fused_model
```

This produces a standalone MLX model directory that can be used for inference without needing the base model + adapters separately.

### 8.2 GGUF Export

MLX cannot directly export Qwen models to GGUF (known limitation). The workflow is:

1. Fuse to MLX format (Step 8.1)
2. Use `mlx_lm.fuse --de-quantize` to produce HF-compatible fp16 weights
3. Convert to GGUF using llama.cpp's `convert_hf_to_gguf.py`
4. Quantize with llama.cpp's `llama-quantize`

```bash
# Step 1: Fuse with de-quantize to get HF-format weights
mlx_lm.fuse \
  --model mlx-community/Qwen3.5-4B-MLX-bf16 \
  --adapter-path ./adapters \
  --save-path ./hf_model \
  --de-quantize

# Step 2: Convert to GGUF (requires llama.cpp repo)
python llama.cpp/convert_hf_to_gguf.py ./hf_model --outfile ./model.gguf

# Step 3: Quantize
./llama.cpp/llama-quantize ./model.gguf ./model-q4_k_m.gguf q4_k_m
```

### 8.3 Deployment Options

| Target | Format | How |
|--------|--------|-----|
| **MLX native** (Mac) | Fused MLX model | `mlx_lm.generate --model ./fused_model` |
| **Ollama** | GGUF | Convert per 8.2, create Modelfile |
| **llama.cpp** | GGUF | Convert per 8.2, load directly |
| **HuggingFace** | De-quantized HF format | Push `./hf_model` to Hub |
| **vLLM** | HF format | Use de-quantized model with `--tool-call-parser qwen3_coder` |

---

## 9. Inference

### 9.1 MLX Native Inference

```bash
mlx_lm.generate \
  --model ./fused_model \
  --prompt "What's the weather in San Francisco?" \
  --max-tokens 256 \
  --temp 0.1
```

### 9.2 Inference with Adapters (without fusing)

```bash
mlx_lm.generate \
  --model mlx-community/Qwen3.5-4B-MLX-bf16 \
  --adapter-path ./adapters \
  --prompt "What's the weather in San Francisco?" \
  --max-tokens 256
```

### 9.3 Python Inference

```python
from mlx_lm import load, generate

model, tokenizer = load("./fused_model")

messages = [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "What's the weather in NYC?"},
]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

response = generate(model, tokenizer, prompt=prompt, max_tokens=256, temp=0.1)
print(response)
```

---

## 10. Evaluation Metrics

| Metric | Target |
|--------|--------|
| Tool Selection Accuracy | > 85% |
| Argument Accuracy | > 80% |
| JSON Validity Rate | > 98% |
| False Positive Rate | < 5% |
| False Negative Rate | < 10% |

---

## 11. Project File Structure

```
slm-tool-calling-finetune/
├── requirements.txt
├── lora_config.yaml
├── data/
│   ├── preprocess.py          # xLAM → JSONL conversion
│   ├── augment_negatives.py   # Add no-tool-call examples
│   ├── train.jsonl            # Generated by preprocess.py
│   └── valid.jsonl            # Generated by preprocess.py
├── train.sh                   # mlx_lm.lora command wrapper
├── fuse.sh                    # mlx_lm.fuse command wrapper
├── export_gguf.sh             # GGUF conversion pipeline
├── inference.py               # Python inference + tool call parser
├── evaluate.py                # Run eval metrics on validation set
├── adapters/                  # LoRA adapter output (created by training)
├── fused_model/               # Fused model output (created by fuse.sh)
└── tests/
    └── test_tool_calling.py   # Integration tests
```

---

## 12. Execution Order

1. Install dependencies (`pip install mlx-lm datasets huggingface-hub numpy`)
2. Run `data/preprocess.py` — downloads xLAM, converts to JSONL
3. Run `data/augment_negatives.py` — adds negative examples, creates final train.jsonl + valid.jsonl
4. Run `train.sh` — executes `mlx_lm.lora` training
5. Run `evaluate.py` — tests model against validation set
6. Run `fuse.sh` — merges adapters into standalone model
7. Run `export_gguf.sh` — converts to GGUF for Ollama/llama.cpp (optional)
8. Run `tests/test_tool_calling.py` — integration tests

---

## 13. Known Risks & Contingencies

| Risk | Likelihood | Contingency |
|------|-----------|-------------|
| MLX silent data format errors | High | Validate JSONL thoroughly before training. MLX has minimal error messages for bad data. |
| Training too slow on first run | Medium | Start with `--iters 1000` to validate pipeline, then scale up. Use 4-bit model for development. |
| Qwen chat template mismatch | Medium | Verify `tokenizer.apply_chat_template()` produces expected output. Print samples before training. |
| GGUF export fails for Qwen | Medium | Known limitation. Use MLX native format for Mac deployment, or de-quantize → llama.cpp convert. |
| Memory pressure on 128GB | Low | 4B bf16 LoRA uses ~13GB. Not a concern. Monitor with `sudo powermetrics --samplers gpu_power`. |
| Loss doesn't decrease | Low | Try `--learning-rate 5e-6` or `--learning-rate 2e-5`. Check data for format issues. |
