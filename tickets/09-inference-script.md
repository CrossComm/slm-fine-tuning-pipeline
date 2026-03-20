# Ticket 09: Inference Script & Tool Call Parser

**Dependencies:** Ticket 02
**Estimated time:** 20 minutes
**Spec reference:** Section 9

---

## Objective

Create `inference.py` — loads the fine-tuned model (fused or base+adapters), generates responses, and parses tool calls from output. Used for interactive testing and by the evaluation script.

---

## Step 1: Create `inference.py`

Create `slm-tool-calling-finetune/inference.py`:

```python
"""
Inference and tool-call parsing for the fine-tuned Qwen 3.5-4B model.

Supports:
  - Loading fused model or base model + adapters
  - Tool call extraction from native <|function_calls|> format
  - Single query and interactive modes

Usage:
    python inference.py                         # Interactive mode
    python inference.py --query "..."           # Single query
    python inference.py --use-adapters          # Use base + adapters (before fusing)
"""

import argparse
import json
import re
from mlx_lm import load, generate
from config import MODEL_NAME, ADAPTER_PATH, FUSED_MODEL_PATH, SYSTEM_PROMPT


# =========================================================================
# TOOL CALL PARSER
# =========================================================================

def extract_tool_calls(response: str) -> list[dict]:
    """
    Extract tool calls from Qwen 3.5 native format.
    Returns list of {"name": ..., "arguments": {...}} dicts.
    """
    pattern = r'<\|function_calls\|>\s*(.*?)\s*<\|/function_calls\|>'
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return []
    try:
        calls = json.loads(match.group(1))
        if isinstance(calls, dict):
            calls = [calls]
        return calls if isinstance(calls, list) else []
    except json.JSONDecodeError:
        return []


def has_tool_call(response: str) -> bool:
    """Check if response contains tool-calling tokens."""
    return "<|function_calls|>" in response


# =========================================================================
# MODEL LOADING
# =========================================================================

def load_model(use_adapters=False):
    """
    Load model for inference.
    - use_adapters=True: load base model + LoRA adapters (before fusing)
    - use_adapters=False: load fused model (after fusing)
    """
    if use_adapters:
        print(f"Loading base model + adapters...")
        model, tokenizer = load(MODEL_NAME, adapter_path=ADAPTER_PATH)
    else:
        print(f"Loading fused model from {FUSED_MODEL_PATH}...")
        model, tokenizer = load(FUSED_MODEL_PATH)

    print("✅ Model loaded")
    return model, tokenizer


# =========================================================================
# INFERENCE
# =========================================================================

def run_inference(model, tokenizer, query: str, tools: list[dict],
                  max_tokens=512, temp=0.1) -> str:
    """Generate a response for a query with available tools."""
    tools_json = json.dumps(tools, ensure_ascii=False)
    system_content = f"{SYSTEM_PROMPT}\n\n{tools_json}"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    response = generate(
        model, tokenizer, prompt=prompt,
        max_tokens=max_tokens, temp=temp, verbose=False,
    )
    return response


# =========================================================================
# DEFAULT TOOLS FOR TESTING
# =========================================================================

DEFAULT_TOOLS = [
    {"type":"function","function":{"name":"get_weather","description":"Get current weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string","description":"City and state"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}}},
    {"type":"function","function":{"name":"search_web","description":"Search the web for information","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Search query"}},"required":["query"]}}},
    {"type":"function","function":{"name":"send_email","description":"Send an email","parameters":{"type":"object","properties":{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}},"required":["to","subject","body"]}}},
]


# =========================================================================
# INTERACTIVE MODE
# =========================================================================

def interactive_mode(model, tokenizer):
    print("\n" + "=" * 60)
    print("INTERACTIVE TOOL-CALLING TEST")
    print("=" * 60)
    print("Tools: get_weather, search_web, send_email")
    print("Type 'quit' to exit\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() == "quit":
            break

        response = run_inference(model, tokenizer, query, DEFAULT_TOOLS)
        calls = extract_tool_calls(response)

        if calls:
            print("Assistant: [TOOL CALL]")
            for c in calls:
                print(f"  → {c['name']}({json.dumps(c.get('arguments',{}), indent=2)})")
        else:
            # Strip any trailing special tokens for display
            clean = response.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
            print(f"Assistant: {clean}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Single query")
    parser.add_argument("--use-adapters", action="store_true",
                        help="Use base model + adapters instead of fused model")
    args = parser.parse_args()

    model, tokenizer = load_model(use_adapters=args.use_adapters)

    if args.query:
        response = run_inference(model, tokenizer, args.query, DEFAULT_TOOLS)
        calls = extract_tool_calls(response)
        if calls:
            for c in calls:
                print(f"{c['name']}({json.dumps(c.get('arguments',{}))})")
        else:
            print(response)
    else:
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
```

---

## Step 2: Test the parser (no model needed)

```bash
cd slm-tool-calling-finetune
python -c "
from inference import extract_tool_calls, has_tool_call

# Single call
r1 = '<|function_calls|>\n[{\"name\":\"get_weather\",\"arguments\":{\"location\":\"NYC\"}}]\n<|/function_calls|>'
assert len(extract_tool_calls(r1)) == 1 and extract_tool_calls(r1)[0]['name'] == 'get_weather'
print('✅ Single call')

# Parallel calls
r2 = '<|function_calls|>\n[{\"name\":\"a\",\"arguments\":{}},{\"name\":\"b\",\"arguments\":{}}]\n<|/function_calls|>'
assert len(extract_tool_calls(r2)) == 2
print('✅ Parallel calls')

# No call
assert extract_tool_calls('The capital is Paris.') == []
assert not has_tool_call('The capital is Paris.')
print('✅ No call')

# Bad JSON
assert extract_tool_calls('<|function_calls|>bad json<|/function_calls|>') == []
print('✅ Bad JSON handled')

print('\nAll parser tests passed!')
"
```

---

## Acceptance Criteria

- [ ] `inference.py` exists with `extract_tool_calls()`, `has_tool_call()`, `run_inference()`, `load_model()`
- [ ] All 4 parser tests pass
- [ ] Supports both `--use-adapters` (pre-fuse) and default (post-fuse) loading
- [ ] Interactive mode works
- [ ] Single query mode works
