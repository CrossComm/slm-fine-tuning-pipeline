# Ticket 03: Model Download & Tokenizer Verification

**Dependencies:** Ticket 01, Ticket 02
**Estimated time:** 10 minutes (plus download time)
**Spec reference:** Sections 2.1 (Model), 3.3 (Pad Token), 5 (Model Loading)

---

## Objective

Download the Qwen 3.5-4B model via Unsloth and verify that the tokenizer is correctly configured — especially the pad_token fix and the presence of native tool-calling tokens.

---

## Step 1: Create the model download & verification script

Create `slm-tool-calling-finetune/download_model.py`:

```python
"""
Download Qwen 3.5-4B via Unsloth and verify tokenizer configuration.
This script should be run once to cache the model weights locally.
"""

import sys
import torch
from unsloth import FastLanguageModel
from config import MODEL_NAME, MAX_SEQ_LENGTH, DTYPE, LOAD_IN_4BIT


def main():
    print(f"Downloading model: {MODEL_NAME}")
    print(f"Max sequence length: {MAX_SEQ_LENGTH}")
    print(f"Load in 4-bit: {LOAD_IN_4BIT} (should be False)")
    print()

    # =========================================================================
    # Step 1: Load model and tokenizer
    # =========================================================================
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    print(f"✅ Model loaded successfully")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Dtype: {next(model.parameters()).dtype}")
    print()

    # =========================================================================
    # Step 2: Verify pad_token != eos_token
    # =========================================================================
    print("Tokenizer checks:")
    print(f"   pad_token: '{tokenizer.pad_token}' (id: {tokenizer.pad_token_id})")
    print(f"   eos_token: '{tokenizer.eos_token}' (id: {tokenizer.eos_token_id})")
    print(f"   bos_token: '{tokenizer.bos_token}' (id: {tokenizer.bos_token_id})")

    if tokenizer.pad_token == tokenizer.eos_token:
        print()
        print("⚠️  WARNING: pad_token equals eos_token!")
        print("   This will cause infinite generation during inference.")
        print("   Attempting fix: setting pad_token to '<|endoftext|>' alternative...")

        # Try to find a suitable pad token
        if "<|finetune_right_pad_id|>" in tokenizer.get_vocab():
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
            print(f"   Fixed: pad_token set to '{tokenizer.pad_token}'")
        else:
            # Add a new pad token
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            model.resize_token_embeddings(len(tokenizer))
            print(f"   Fixed: added new pad_token '<|pad|>'")
    else:
        print("   ✅ pad_token != eos_token (correct)")

    print()

    # =========================================================================
    # Step 3: Verify native tool-calling tokens exist
    # =========================================================================
    vocab = tokenizer.get_vocab()
    required_tokens = ["<|function_calls|>", "<|/function_calls|>"]
    tool_tokens_found = []
    tool_tokens_missing = []

    for token in required_tokens:
        if token in vocab:
            tool_tokens_found.append(token)
            print(f"   ✅ Found tool token: '{token}' (id: {vocab[token]})")
        else:
            tool_tokens_missing.append(token)
            print(f"   ❌ MISSING tool token: '{token}'")

    print()

    if tool_tokens_missing:
        print("⚠️  WARNING: Some native tool-calling tokens are missing.")
        print("   The model may not have native tool-calling support.")
        print("   Check that you're using the correct model version.")
        print(f"   Missing: {tool_tokens_missing}")
        print()
        print("   FALLBACK: If tokens are truly missing, the preprocessing")
        print("   pipeline will use <tool_call>/<tool_call> XML tags instead.")
        print("   Update config.py TOOL_CALL_FORMAT accordingly.")
    else:
        print("✅ All native tool-calling tokens present")

    print()

    # =========================================================================
    # Step 4: Check ChatML tokens
    # =========================================================================
    chatml_tokens = ["<|im_start|>", "<|im_end|>"]
    for token in chatml_tokens:
        if token in vocab:
            print(f"   ✅ ChatML token: '{token}' (id: {vocab[token]})")
        else:
            print(f"   ❌ MISSING ChatML token: '{token}'")

    print()

    # =========================================================================
    # Step 5: Test chat template
    # =========================================================================
    print("Testing chat template application...")
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]

    try:
        formatted = tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        print(f"   ✅ Chat template works")
        print(f"   Sample output (first 200 chars):")
        print(f"   {formatted[:200]}")
    except Exception as e:
        print(f"   ❌ Chat template failed: {e}")

    print()

    # =========================================================================
    # Step 6: Memory check
    # =========================================================================
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU Memory:")
        print(f"   Allocated: {allocated:.1f} GB")
        print(f"   Reserved:  {reserved:.1f} GB")
        print(f"   Total:     {total:.1f} GB")
        print(f"   Free:      {total - reserved:.1f} GB")
        print()

        if total - reserved < 5:
            print("⚠️  WARNING: Less than 5GB free after model load.")
            print("   Training may OOM. Consider reducing batch size.")

    print()
    print("=" * 60)
    print("MODEL DOWNLOAD & VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Step 2: Run the script

```bash
cd slm-tool-calling-finetune
python download_model.py
```

**Expected output:** All checkmarks (✅). If any ❌ or ⚠️ appear, follow the guidance printed by the script.

**Note:** First run will download ~8GB of model weights. Subsequent runs use the HuggingFace cache.

---

## Step 3: Record results

After running, note these values from the output for reference in later tickets:

- Actual dtype used (should be bfloat16)
- Parameter count (should be ~4B)
- GPU memory after load (should be ~8-9GB)
- Whether tool-calling tokens were found
- Whether pad_token fix was needed

---

## Acceptance Criteria

- [ ] Model downloads and loads without errors
- [ ] `pad_token != eos_token` (either natively or after fix)
- [ ] Native tool-calling tokens (`<|function_calls|>`, `<|/function_calls|>`) are present in vocabulary OR a fallback plan is documented
- [ ] ChatML tokens (`<|im_start|>`, `<|im_end|>`) are present
- [ ] `apply_chat_template()` works on test messages
- [ ] GPU memory after load leaves at least 5GB free on a 16GB card
- [ ] `download_model.py` script exists and is reusable

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Download fails / timeout | Set `HF_HUB_DOWNLOAD_TIMEOUT=300` env var. Check internet connection. |
| Model not found on HuggingFace | Verify the model ID `unsloth/Qwen3.5-4B` exists. If not, try `Qwen/Qwen3.5-4B` and update `config.py`. |
| OOM during model load | You may not have enough VRAM. Try `unsloth/Qwen3.5-2B` as fallback and update `config.py`. |
| Tool-calling tokens missing | The model may use different token names. Search the vocab for "function" or "tool": `[k for k in vocab if 'function' in k.lower() or 'tool' in k.lower()]` |
| pad_token fix needed | The script handles this automatically. Just note it happened for debugging. |
| `gated repo` error | You need to accept the model's license on HuggingFace. Run `huggingface-cli login` and accept terms at the model page. |
