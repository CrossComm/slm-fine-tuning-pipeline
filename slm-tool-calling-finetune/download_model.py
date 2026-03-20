"""Download Qwen 3.5-4B MLX and verify tokenizer/tool-calling tokens."""

from mlx_lm import load
from config import MODEL_NAME


def main():
    """Download the model from HuggingFace and verify it loads correctly.

    Checks for tool-calling tokens, ChatML tokens, chat template functionality,
    and basic generation. Prints a summary of findings.
    """
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
