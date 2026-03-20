# Ticket 11: Fuse Model & Export

**Dependencies:** Ticket 08
**Estimated time:** 15 minutes
**Spec reference:** Section 8

---

## Objective

Create `fuse.sh` and `export_gguf.sh` — scripts to merge LoRA adapters into a standalone model and optionally export to GGUF for Ollama/llama.cpp.

---

## Step 1: Create `fuse.sh`

Create `slm-tool-calling-finetune/fuse.sh`:

```bash
#!/bin/bash
# Fuse LoRA adapters into a standalone MLX model
set -euo pipefail
cd "$(dirname "$0")"

MODEL="mlx-community/Qwen3.5-4B-MLX-bf16"
ADAPTER_PATH="./adapters"
FUSED_PATH="./fused_model"

echo "🔧 Fusing adapters into standalone model..."
echo "   Base model: ${MODEL}"
echo "   Adapters: ${ADAPTER_PATH}"
echo "   Output: ${FUSED_PATH}"

if [[ ! -d "${ADAPTER_PATH}" ]]; then
    echo "❌ Adapters not found at ${ADAPTER_PATH}. Run training first."
    exit 1
fi

mlx_lm.fuse \
  --model "${MODEL}" \
  --adapter-path "${ADAPTER_PATH}" \
  --save-path "${FUSED_PATH}"

echo ""
echo "✅ Fused model saved to ${FUSED_PATH}"
echo ""
echo "Test with:"
echo "  mlx_lm.generate --model ${FUSED_PATH} --prompt 'Hello' --max-tokens 50"
echo ""
echo "Or run inference.py:"
echo "  python inference.py"
```

Make executable:

```bash
chmod +x slm-tool-calling-finetune/fuse.sh
```

---

## Step 2: Create `export_gguf.sh`

Create `slm-tool-calling-finetune/export_gguf.sh`:

```bash
#!/bin/bash
# Export fused model to GGUF format for Ollama/llama.cpp
#
# Prerequisites:
#   - fuse.sh has been run
#   - llama.cpp repo is cloned and built
#
# NOTE: Qwen GGUF export has known limitations. If conversion fails,
# the fused MLX model can be used directly on Mac via mlx_lm.generate.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="mlx-community/Qwen3.5-4B-MLX-bf16"
ADAPTER_PATH="./adapters"
HF_PATH="./hf_model"
GGUF_OUT="./model.gguf"
GGUF_QUANT="./model-q4_k_m.gguf"

# Check for llama.cpp
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-../llama.cpp}"
if [[ ! -d "${LLAMA_CPP_DIR}" ]]; then
    echo "⚠️  llama.cpp not found at ${LLAMA_CPP_DIR}"
    echo "   Clone it first:"
    echo "     git clone https://github.com/ggerganov/llama.cpp.git ${LLAMA_CPP_DIR}"
    echo "     cd ${LLAMA_CPP_DIR} && make"
    echo ""
    echo "   Or set LLAMA_CPP_DIR to point to your installation."
    exit 1
fi

echo "Step 1: Fusing with de-quantize to HF format..."
mlx_lm.fuse \
  --model "${MODEL}" \
  --adapter-path "${ADAPTER_PATH}" \
  --save-path "${HF_PATH}" \
  --de-quantize

echo ""
echo "Step 2: Converting to GGUF..."
python "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" \
  "${HF_PATH}" \
  --outfile "${GGUF_OUT}"

echo ""
echo "Step 3: Quantizing to q4_k_m..."
"${LLAMA_CPP_DIR}/llama-quantize" \
  "${GGUF_OUT}" \
  "${GGUF_QUANT}" \
  q4_k_m

echo ""
echo "✅ GGUF export complete:"
echo "   Full:      ${GGUF_OUT}"
echo "   Quantized: ${GGUF_QUANT}"
echo ""
echo "To use with Ollama:"
echo "   echo 'FROM ${GGUF_QUANT}' > Modelfile"
echo "   ollama create qwen35-tool-calling -f Modelfile"
echo "   ollama run qwen35-tool-calling"
```

Make executable:

```bash
chmod +x slm-tool-calling-finetune/export_gguf.sh
```

---

## Step 3: Run fuse

```bash
cd slm-tool-calling-finetune
./fuse.sh
```

---

## Step 4: Verify fused model

```bash
# Quick generation test
mlx_lm.generate \
  --model ./fused_model \
  --prompt "Hello, how are you?" \
  --max-tokens 50

# Check directory contents
ls -lh fused_model/
```

---

## Step 5: Export GGUF (optional)

Only needed if you want to deploy via Ollama or llama.cpp:

```bash
# First, clone and build llama.cpp if you don't have it
git clone https://github.com/ggerganov/llama.cpp.git ../llama.cpp
cd ../llama.cpp && make && cd -

# Then export
./export_gguf.sh
```

**Note:** Qwen GGUF export has known limitations. If `convert_hf_to_gguf.py` fails, the fused MLX model works directly on Mac via `mlx_lm.generate` or `inference.py`.

---

## Acceptance Criteria

- [ ] `fuse.sh` runs and produces `./fused_model/` directory
- [ ] Fused model generates output via `mlx_lm.generate`
- [ ] `inference.py` loads the fused model (without `--use-adapters` flag)
- [ ] `export_gguf.sh` exists (even if GGUF export is deferred)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Fuse fails | Check adapter path exists and has valid files. |
| GGUF conversion fails | Known issue with Qwen models. Use MLX format directly. The fused model works on Mac without GGUF. |
| Fused model generates garbage | The adapters may have overtrained. Try adapters from an earlier checkpoint. |
