# Ticket 13: Model Export (Merge, GGUF, Hub)

**Dependencies:** Ticket 09
**Estimated time:** 15 minutes
**Spec reference:** Section 10 (Model Export & Deployment)

---

## Objective

Create `export.py` — a script that takes the trained LoRA adapters and exports the model in multiple formats: merged full-precision weights, GGUF for Ollama/llama.cpp, and optionally pushes to HuggingFace Hub.

---

## Step 1: Create `export.py`

Create `slm-tool-calling-finetune/export.py`:

```python
"""
Model export script.

Takes trained LoRA adapters and exports in multiple formats:
  1. LoRA adapters only (~100-200MB)
  2. Merged full-precision model (~8GB)
  3. GGUF quantized model (~2-3GB) for Ollama/llama.cpp
  4. (Optional) Push to HuggingFace Hub

Usage:
    python export.py                    # Export all formats
    python export.py --format merged    # Export only merged model
    python export.py --format gguf      # Export only GGUF
    python export.py --push-hub         # Also push to HuggingFace Hub

Prerequisites:
    Training must be complete (Ticket 09)
"""

import argparse
import os
from unsloth import FastLanguageModel
from config import (
    LORA_OUTPUT_DIR, MERGED_OUTPUT_DIR, GGUF_OUTPUT_DIR,
    GGUF_QUANTIZATION, HUB_MODEL_NAME, MAX_SEQ_LENGTH,
    DTYPE, LOAD_IN_4BIT,
)


def load_trained_model():
    """Load the model with trained LoRA adapters."""
    print(f"Loading model from {LORA_OUTPUT_DIR}...")

    if not os.path.exists(LORA_OUTPUT_DIR):
        raise FileNotFoundError(
            f"LoRA adapters not found at {LORA_OUTPUT_DIR}. "
            "Run training (Ticket 09) first."
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LORA_OUTPUT_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    print(f"✅ Model loaded from LoRA adapters")
    return model, tokenizer


def export_merged(model, tokenizer):
    """
    Merge LoRA adapters into the base model and save at full precision.
    Output: ~8GB model that can be loaded directly without Unsloth/PEFT.
    """
    print()
    print("=" * 60)
    print("EXPORTING MERGED MODEL (16-bit)")
    print("=" * 60)
    print(f"Output: {MERGED_OUTPUT_DIR}")
    print()

    os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)

    model.save_pretrained_merged(
        MERGED_OUTPUT_DIR,
        tokenizer,
        save_method="merged_16bit",
    )

    # Verify output
    expected_files = ["config.json", "tokenizer.json"]
    for f in expected_files:
        path = os.path.join(MERGED_OUTPUT_DIR, f)
        if os.path.exists(path):
            print(f"  ✅ {f}")
        else:
            print(f"  ⚠️  Missing: {f}")

    # Check total size
    total_size = sum(
        os.path.getsize(os.path.join(MERGED_OUTPUT_DIR, f))
        for f in os.listdir(MERGED_OUTPUT_DIR)
        if os.path.isfile(os.path.join(MERGED_OUTPUT_DIR, f))
    )
    print(f"\n  Total size: {total_size / 1e9:.1f} GB")
    print(f"  ✅ Merged model saved")


def export_gguf(model, tokenizer):
    """
    Export model as GGUF quantized format.
    Output: ~2-3GB model for llama.cpp and Ollama.
    """
    print()
    print("=" * 60)
    print(f"EXPORTING GGUF ({GGUF_QUANTIZATION})")
    print("=" * 60)
    print(f"Output: {GGUF_OUTPUT_DIR}")
    print()

    os.makedirs(GGUF_OUTPUT_DIR, exist_ok=True)

    model.save_pretrained_gguf(
        GGUF_OUTPUT_DIR,
        tokenizer,
        quantization_method=GGUF_QUANTIZATION,
    )

    # Check output
    gguf_files = [f for f in os.listdir(GGUF_OUTPUT_DIR) if f.endswith('.gguf')]
    if gguf_files:
        for f in gguf_files:
            size = os.path.getsize(os.path.join(GGUF_OUTPUT_DIR, f)) / 1e9
            print(f"  ✅ {f} ({size:.1f} GB)")
    else:
        print("  ⚠️  No .gguf files found — export may have failed")

    print(f"\n  ✅ GGUF export complete")
    print(f"\n  To use with Ollama:")
    print(f"    1. Create a Modelfile:")
    print(f'       echo "FROM ./{GGUF_OUTPUT_DIR}/{gguf_files[0] if gguf_files else "model.gguf"}" > Modelfile')
    print(f"    2. Create the model:")
    print(f"       ollama create qwen35-tool-calling -f Modelfile")
    print(f"    3. Run it:")
    print(f"       ollama run qwen35-tool-calling")


def push_to_hub(model, tokenizer, token=None):
    """
    Push merged model to HuggingFace Hub.
    Requires authentication.
    """
    print()
    print("=" * 60)
    print("PUSHING TO HUGGINGFACE HUB")
    print("=" * 60)
    print(f"Repository: {HUB_MODEL_NAME}")
    print()

    if HUB_MODEL_NAME == "your-username/qwen3.5-4b-tool-calling":
        print("⚠️  HUB_MODEL_NAME is still set to the default placeholder.")
        print("   Update it in config.py before pushing to the Hub.")
        print("   Skipping push.")
        return

    kwargs = {}
    if token:
        kwargs["token"] = token

    model.push_to_hub_merged(
        HUB_MODEL_NAME,
        tokenizer,
        save_method="merged_16bit",
        **kwargs,
    )

    print(f"  ✅ Pushed to https://huggingface.co/{HUB_MODEL_NAME}")


def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned model")
    parser.add_argument(
        "--format",
        type=str,
        choices=["all", "merged", "gguf", "lora"],
        default="all",
        help="Export format (default: all)",
    )
    parser.add_argument("--push-hub", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace API token")
    args = parser.parse_args()

    # Load model
    model, tokenizer = load_trained_model()

    # Export based on selected format
    if args.format in ("all", "merged"):
        export_merged(model, tokenizer)

    if args.format in ("all", "gguf"):
        export_gguf(model, tokenizer)

    if args.format == "lora":
        print("LoRA adapters are already saved at:", LORA_OUTPUT_DIR)

    if args.push_hub:
        push_to_hub(model, tokenizer, args.hf_token)

    print()
    print("=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print()
    print("Exported formats:")
    if args.format in ("all", "lora"):
        print(f"  LoRA adapters: {LORA_OUTPUT_DIR}")
    if args.format in ("all", "merged"):
        print(f"  Merged model:  {MERGED_OUTPUT_DIR}")
    if args.format in ("all", "gguf"):
        print(f"  GGUF model:    {GGUF_OUTPUT_DIR}")
    print()
    print("Deployment options:")
    print("  vLLM:      Load from merged model with --tool-call-parser qwen3_coder")
    print("  Ollama:    Create Modelfile from GGUF (see instructions above)")
    print("  llama.cpp: Load GGUF directly")
    print("  HF:        AutoModelForCausalLM.from_pretrained(merged_path)")


if __name__ == "__main__":
    main()
```

---

## Step 2: Run export

```bash
cd slm-tool-calling-finetune

# Export all formats
python export.py

# Or export individually
python export.py --format merged
python export.py --format gguf
```

---

## Step 3: Verify exports

```bash
# Check merged model
ls -lh outputs/merged_model/

# Check GGUF
ls -lh outputs/gguf_model/*.gguf

# Quick smoke test: load merged model with vanilla HF
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('./outputs/merged_model', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('./outputs/merged_model')
print(f'✅ Merged model loads with vanilla HuggingFace')
print(f'   Parameters: {sum(p.numel() for p in model.parameters()):,}')
"
```

---

## Acceptance Criteria

- [ ] `export.py` runs without errors
- [ ] Merged model saved to `outputs/merged_model/` (~8GB)
- [ ] GGUF model saved to `outputs/gguf_model/` (~2-3GB)
- [ ] Merged model loads with standard `AutoModelForCausalLM.from_pretrained()`
- [ ] GGUF file has `.gguf` extension
- [ ] Ollama instructions are printed
- [ ] Hub push is skipped if `HUB_MODEL_NAME` is still the placeholder

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM during merge | Merging requires loading both base model and adapters. Try on a machine with more RAM, or use `save_method="merged_4bit"`. |
| GGUF export fails | Ensure `llama.cpp` conversion tools are available. Unsloth handles this internally but may need additional dependencies. |
| Merged model is very large | Expected: ~8GB for 4B params at 16-bit. This is normal. |
| Hub push authentication error | Run `huggingface-cli login` first, or pass `--hf-token YOUR_TOKEN`. |
| vLLM can't load the model | Pin vLLM to 0.7.3. Version 0.8.0+ has known incompatibilities with Unsloth exports. |
