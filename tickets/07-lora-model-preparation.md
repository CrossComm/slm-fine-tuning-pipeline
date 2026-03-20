# Ticket 07: LoRA Configuration & Model Preparation

**Dependencies:** Ticket 03
**Estimated time:** 15 minutes
**Spec reference:** Section 6 (LoRA Configuration)

---

## Objective

Apply LoRA adapters to the loaded Qwen 3.5-4B model using Unsloth's `get_peft_model()`. Verify the adapter configuration, count trainable parameters, and confirm VRAM usage is within budget.

---

## Step 1: Create the model preparation module

Create `slm-tool-calling-finetune/model_setup.py`:

```python
"""
Model loading and LoRA adapter setup.

Loads Qwen 3.5-4B via Unsloth and applies LoRA adapters.
This module is imported by train.py and inference.py.

Usage (standalone test):
    python model_setup.py
"""

import torch
from unsloth import FastLanguageModel
from config import (
    MODEL_NAME, MAX_SEQ_LENGTH, DTYPE, LOAD_IN_4BIT,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    LORA_BIAS, GRADIENT_CHECKPOINTING, RANDOM_STATE,
    USE_RSLORA, LOFTQ_CONFIG,
)


def load_base_model():
    """
    Load the base Qwen 3.5-4B model and tokenizer via Unsloth.

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading base model: {MODEL_NAME}")
    print(f"  max_seq_length: {MAX_SEQ_LENGTH}")
    print(f"  load_in_4bit: {LOAD_IN_4BIT} (should be False)")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    # Fix pad_token if needed
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        if "<|finetune_right_pad_id|>" in tokenizer.get_vocab():
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            model.resize_token_embeddings(len(tokenizer))
        print(f"  ⚠️  Fixed pad_token to: '{tokenizer.pad_token}'")

    print(f"  ✅ Base model loaded")
    return model, tokenizer


def apply_lora(model):
    """
    Apply LoRA adapters to the model using Unsloth's optimized implementation.

    CRITICAL NOTES:
    - lora_dropout MUST be 0 for Unsloth kernel fusion to work
    - target_modules includes ALL linear layers for best structured output quality
    - gradient_checkpointing="unsloth" uses 30% less VRAM than HF default

    Returns:
        model with LoRA adapters applied
    """
    print(f"\nApplying LoRA adapters:")
    print(f"  r (rank): {LORA_R}")
    print(f"  alpha: {LORA_ALPHA}")
    print(f"  dropout: {LORA_DROPOUT}")
    print(f"  target_modules: {LORA_TARGET_MODULES}")
    print(f"  gradient_checkpointing: {GRADIENT_CHECKPOINTING}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias=LORA_BIAS,
        use_gradient_checkpointing=GRADIENT_CHECKPOINTING,
        random_state=RANDOM_STATE,
        use_rslora=USE_RSLORA,
        loftq_config=LOFTQ_CONFIG,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = trainable_params / total_params * 100

    print(f"\n  Parameter summary:")
    print(f"    Total:     {total_params:,}")
    print(f"    Trainable: {trainable_params:,} ({trainable_pct:.2f}%)")
    print(f"    Frozen:    {total_params - trainable_params:,}")
    print(f"  ✅ LoRA adapters applied")

    return model


def get_model_and_tokenizer():
    """
    Full setup: load base model and apply LoRA.
    This is the main entry point for other scripts.

    Returns:
        tuple: (model_with_lora, tokenizer)
    """
    model, tokenizer = load_base_model()
    model = apply_lora(model)
    return model, tokenizer


def main():
    """Standalone test: load model, apply LoRA, print stats."""
    print("=" * 60)
    print("MODEL SETUP VERIFICATION")
    print("=" * 60)
    print()

    model, tokenizer = get_model_and_tokenizer()

    print()

    # VRAM check
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_mem / 1e9
        free = total - reserved

        print(f"GPU Memory after LoRA setup:")
        print(f"  Allocated: {allocated:.1f} GB")
        print(f"  Reserved:  {reserved:.1f} GB")
        print(f"  Total:     {total:.1f} GB")
        print(f"  Free:      {free:.1f} GB")
        print()

        if free < 4:
            print("⚠️  WARNING: Less than 4GB free. Training may OOM.")
            print("   Consider reducing per_device_train_batch_size to 2.")
        elif free < 6:
            print("⚠️  NOTE: Tight on memory. Use batch_size=2 if OOM during training.")
        else:
            print("✅ Sufficient VRAM headroom for training.")

    # Quick forward pass test
    print()
    print("Testing forward pass...")
    test_input = tokenizer("Hello, world!", return_tensors="pt")
    if torch.cuda.is_available():
        test_input = {k: v.cuda() for k, v in test_input.items()}
    with torch.no_grad():
        output = model(**test_input)
    print(f"  Output shape: {output.logits.shape}")
    print(f"  ✅ Forward pass successful")

    print()
    print("=" * 60)
    print("MODEL SETUP VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Step 2: Run verification

```bash
cd slm-tool-calling-finetune
python model_setup.py
```

**Expected output:**
- Base model loads successfully
- LoRA adapters applied with ~65M trainable params (~1.6% of total)
- pad_token != eos_token
- Forward pass succeeds
- At least 4GB VRAM free after setup

---

## Acceptance Criteria

- [ ] `model_setup.py` runs without errors
- [ ] Trainable parameters are ~1.5-2% of total (LoRA is working, not full fine-tune)
- [ ] `lora_dropout` is 0 (confirmed in output)
- [ ] All 7 target modules are listed (q/k/v/o_proj, gate/up/down_proj)
- [ ] Gradient checkpointing is set to "unsloth"
- [ ] Forward pass completes without errors
- [ ] VRAM usage leaves at least 4GB free on 16GB GPU
- [ ] `get_model_and_tokenizer()` function is importable by other scripts

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM during model load | Close other GPU processes. Check `nvidia-smi` for other users. |
| "Module not found" for target modules | The module names may differ in Qwen 3.5. Print `model.named_modules()` to find the correct names. |
| LoRA dropout warning | Unsloth may warn about dropout=0 being required. This is expected — just confirm it's set to 0. |
| Forward pass fails | Check dtype compatibility. Try explicitly setting `dtype=torch.bfloat16` in config. |
| Trainable params seem too high (>5%) | Verify `load_in_4bit=False` and that LoRA (not full fine-tune) is being applied. |
