# Ticket 03: Download & Verify Model

**Dependencies:** Ticket 01
**Estimated time:** 10 minutes (plus ~8GB download)
**Spec reference:** Section 2.1

---

## Objective

Download the Qwen 3.5-4B MLX model from HuggingFace and verify it loads correctly, has the expected tool-calling tokens, and the chat template works.

---

## Step 1: Create verification script

Create `slm-tool-calling-finetune/download_model.py`:

```python
"""Download Qwen 3.5-4B MLX and verify tokenizer/tool-calling tokens."""

from mlx_lm import load
from config import MODEL_NAME


def main():
    print(f"Downloading and loading: {MODEL_NAME}")
    print("(First run downloads ~8GB of weights)\n")

    # Load model and tokenizer
    model, tokenizer = load(MODEL_NAME)
    print(f"✅ Model loaded")
    print(f"   Vocab size: {len(tokenizer.get_vocab())}")

    # Check tool-calling tokens
    vocab = tokenizer.get_vocab()
    tool_tokens = ["<|function_calls|>", "<|/function_calls|>"]
    for token in tool_tokens:
        if token in vocab:
            print(f"   ✅ Found: '{token}' (id: {vocab[token]})")
        else:
            print(f"   ⚠️  Missing: '{token}'")

    # Check ChatML tokens
    chatml_tokens = ["<|im_start|>", "<|im_end|>"]
    for token in chatml_tokens:
        if token in vocab:
            print(f"   ✅ Found: '{token}' (id: {vocab[token]})")
        else:
            print(f"   ⚠️  Missing: '{token}'")

    # Test chat template
    print("\nTesting chat template...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print(f"   ✅ Chat template works")
        print(f"   Preview (first 200 chars):\n   {formatted[:200]}")
    except Exception as e:
        print(f"   ❌ Chat template failed: {e}")

    # Quick generation test
    print("\nTesting generation...")
    from mlx_lm import generate
    response = generate(model, tokenizer, prompt="Hello", max_tokens=20, verbose=False)
    print(f"   ✅ Generation works: '{response[:100]}'")

    print("\n✅ Model download and verification complete.")


if __name__ == "__main__":
    main()
```

---

## Step 2: Run it

```bash
cd slm-tool-calling-finetune
python download_model.py
```

---

## Step 3: Record findings

Note these from the output:
- Whether tool-calling tokens (`<|function_calls|>`, `<|/function_calls|>`) exist in vocab
- Whether ChatML tokens exist
- What the chat template output looks like
- Whether generation produces coherent output

If tool-calling tokens are MISSING, the preprocessing pipeline (Ticket 05) must use a fallback format (e.g., `<tool_call>` XML tags). Update config.py accordingly.

---

## Acceptance Criteria

- [ ] Model downloads and loads via `mlx_lm.load()`
- [ ] Tool-calling token presence is documented
- [ ] ChatML tokens are present
- [ ] `apply_chat_template()` works
- [ ] Generation produces output without errors

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Model not found | Check exact HF ID. Try `huggingface-hub` CLI: `huggingface-cli scan-cache`. The model may be at a different path. Search `huggingface.co/mlx-community` for Qwen3.5. |
| Download fails | Set `HF_HUB_DOWNLOAD_TIMEOUT=300`. Check internet. |
| `mlx_lm.load` fails | Model format may be incompatible. Try the 4-bit variant: `mlx-community/Qwen3.5-4B-MLX-4bit`. |
| Gated repo error | Accept model license at the HuggingFace model page, then `huggingface-cli login`. |
| Out of memory | Shouldn't happen with 128GB. If it does, try the 4-bit variant. |
