# Ticket 07: LoRA Config & Training Script

**Dependencies:** Ticket 02
**Estimated time:** 10 minutes
**Spec reference:** Sections 6, 7

---

## Objective

Create the `train.sh` wrapper script that invokes `mlx_lm.lora` with all hyperparameters from the spec. MLX training is CLI-driven — there's no Python Trainer class to write.

---

## Step 1: Verify lora_config.yaml exists

Confirm this was created in Ticket 02:

```bash
cat slm-tool-calling-finetune/lora_config.yaml
```

Should show:
```yaml
rank: 16
alpha: 32
dropout: 0.0
scale: 2.0
```

---

## Step 2: Create `train.sh`

Create `slm-tool-calling-finetune/train.sh`:

```bash
#!/bin/bash
# Train Qwen 3.5-4B for tool calling using MLX LoRA
#
# Prerequisites:
#   - data/train.jsonl and data/valid.jsonl exist (Tickets 05-06)
#   - lora_config.yaml exists (Ticket 02)
#
# Usage:
#   ./train.sh              # Full training run (~3-5 hours)
#   ./train.sh --quick      # Quick validation run (1000 iters, ~15 min)

set -euo pipefail
cd "$(dirname "$0")"

MODEL="mlx-community/Qwen3.5-4B-MLX-bf16"
DATA_DIR="./data"
ADAPTER_PATH="./adapters"
LORA_CONFIG="./lora_config.yaml"

# Default: full training
ITERS=49500  # (66000 / 4) * 3 epochs

# Quick mode override
if [[ "${1:-}" == "--quick" ]]; then
    ITERS=1000
    echo "🔧 Quick mode: ${ITERS} iterations (~15 minutes)"
else
    echo "🚀 Full training: ${ITERS} iterations (~3-5 hours)"
fi

# Pre-flight checks
if [[ ! -f "${DATA_DIR}/train.jsonl" ]]; then
    echo "❌ ${DATA_DIR}/train.jsonl not found. Run data/preprocess.py and data/augment_negatives.py first."
    exit 1
fi

if [[ ! -f "${DATA_DIR}/valid.jsonl" ]]; then
    echo "❌ ${DATA_DIR}/valid.jsonl not found. Run data/preprocess.py first."
    exit 1
fi

TRAIN_COUNT=$(wc -l < "${DATA_DIR}/train.jsonl")
VALID_COUNT=$(wc -l < "${DATA_DIR}/valid.jsonl")
echo "📊 Training examples: ${TRAIN_COUNT}"
echo "📊 Validation examples: ${VALID_COUNT}"
echo ""

# Run training
mlx_lm.lora \
  --model "${MODEL}" \
  --data "${DATA_DIR}" \
  --train \
  --batch-size 4 \
  --num-layers 16 \
  --learning-rate 1e-5 \
  --iters "${ITERS}" \
  --steps-per-eval 200 \
  --val-batches 25 \
  --steps-per-report 10 \
  --adapter-path "${ADAPTER_PATH}" \
  --lora-config "${LORA_CONFIG}" \
  --max-seq-length 2048 \
  --grad-checkpoint

echo ""
echo "✅ Training complete."
echo "   Adapters saved to: ${ADAPTER_PATH}"
echo "   Next: run evaluate.py (Ticket 10) or fuse.sh (Ticket 11)"
```

Make it executable:

```bash
chmod +x slm-tool-calling-finetune/train.sh
```

---

## Step 3: Validate the command works (dry run)

Before running full training, verify mlx_lm.lora accepts all arguments:

```bash
cd slm-tool-calling-finetune
mlx_lm.lora --help
```

Check that all flags used in `train.sh` appear in the help output. If any flag is unrecognized, the version of mlx-lm may differ — adjust accordingly.

---

## Step 4: Quick smoke test

```bash
cd slm-tool-calling-finetune
./train.sh --quick 2>&1 | head -50
```

Let it run for ~30 seconds, then kill with Ctrl+C. Verify:
- No import errors
- Model loading message appears
- Training starts (loss values printing)
- No data format errors

---

## Acceptance Criteria

- [ ] `train.sh` exists and is executable
- [ ] `lora_config.yaml` exists with rank=16, alpha=32
- [ ] `--quick` mode runs ~1000 iterations
- [ ] Full mode calculates correct iteration count for 3 epochs
- [ ] All mlx_lm.lora flags are valid for the installed version
- [ ] Pre-flight checks verify data files exist
- [ ] Smoke test shows training starts successfully

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `mlx_lm.lora: command not found` | Activate venv: `source .venv/bin/activate`. Or use `python -m mlx_lm.lora`. |
| Unrecognized flag | Run `mlx_lm.lora --help` and adjust. Flag names may differ between mlx-lm versions. |
| Data format error | MLX gives minimal error messages. Validate JSONL: `head -1 data/train.jsonl \| python -m json.tool` |
| OOM | Shouldn't happen with 128GB. If it does: reduce `--batch-size` to 2, or use 4-bit model. |
