# Ticket 06: Negative Example Augmentation

**Dependencies:** Ticket 05
**Estimated time:** 15 minutes
**Spec reference:** Section 5.3

---

## Objective

Add ~12,000 negative examples to `train.jsonl` — conversations where tools ARE in the system message but the assistant answers naturally without calling any tools. This prevents the model from learning "always call a tool."

---

## Step 1: Create `data/augment_negatives.py`

```python
"""
Add negative examples (no tool call) to train.jsonl.
Tools are present in system message but assistant responds with plain text.
"""

import json
import os
import sys
import random
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    NEGATIVE_DATASET_NAME, NEGATIVE_EXAMPLE_COUNT,
    SPLIT_SEED, SYSTEM_PROMPT, DATA_DIR,
)

# Realistic tool sets to randomly include in negative examples
TOOL_SETS = [
    [{"type":"function","function":{"name":"get_weather","description":"Get current weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}},
     {"type":"function","function":{"name":"search_web","description":"Search the web","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}],
    [{"type":"function","function":{"name":"send_email","description":"Send an email","parameters":{"type":"object","properties":{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}},"required":["to","subject","body"]}}},
     {"type":"function","function":{"name":"calculate","description":"Evaluate math expression","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}}],
    [{"type":"function","function":{"name":"get_stock_price","description":"Get stock price","parameters":{"type":"object","properties":{"symbol":{"type":"string"}},"required":["symbol"]}}},
     {"type":"function","function":{"name":"translate_text","description":"Translate text","parameters":{"type":"object","properties":{"text":{"type":"string"},"target":{"type":"string"}},"required":["text","target"]}}}],
]


def main():
    print("=" * 60)
    print("NEGATIVE EXAMPLE AUGMENTATION")
    print("=" * 60)

    random.seed(SPLIT_SEED)
    train_path = os.path.join(DATA_DIR, "train.jsonl")

    # Count existing positive examples
    with open(train_path) as f:
        original_count = sum(1 for _ in f)
    print(f"\nExisting training examples: {original_count:,}")

    # Load Alpaca for negative source
    print(f"Loading {NEGATIVE_DATASET_NAME}...")
    alpaca = load_dataset(NEGATIVE_DATASET_NAME, split="train")
    print(f"✅ Loaded {len(alpaca):,} examples")

    # Filter: keep general Q&A, skip anything that sounds like it needs a tool
    tool_keywords = ["api", "http", "url", "endpoint", "fetch", "request", "database"]
    candidates = []
    for row in alpaca:
        instruction = row.get("instruction", "")
        output = row.get("output", "")
        input_text = row.get("input", "")
        if input_text:
            instruction = f"{instruction}\n{input_text}"
        if not instruction or not output or len(output) < 20:
            continue
        if any(kw in instruction.lower() for kw in tool_keywords):
            continue
        candidates.append((instruction, output))

    print(f"Filtered candidates: {len(candidates):,}")

    # Sample
    if len(candidates) > NEGATIVE_EXAMPLE_COUNT:
        candidates = random.sample(candidates, NEGATIVE_EXAMPLE_COUNT)
    print(f"Selected: {len(candidates):,}")

    # Create negative examples and append to train.jsonl
    negatives = []
    for instruction, output in candidates:
        tools = random.choice(TOOL_SETS)
        tools_json = json.dumps(tools, ensure_ascii=False)
        system_content = f"{SYSTEM_PROMPT}\n\n{tools_json}"

        negatives.append({
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output},  # NO tool call
            ]
        })

    # Read existing, merge, shuffle, rewrite
    print("\nMerging and shuffling...")
    with open(train_path) as f:
        existing = [json.loads(line) for line in f]

    merged = existing + negatives
    random.shuffle(merged)

    with open(train_path, "w") as f:
        for ex in merged:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Count positive vs negative
    pos = sum(1 for ex in merged if "<|function_calls|>" in ex["messages"][-1]["content"])
    neg = len(merged) - pos

    print(f"\n✅ Augmented train.jsonl:")
    print(f"   Total: {len(merged):,}")
    print(f"   Positive (tool call): {pos:,} ({pos/len(merged)*100:.1f}%)")
    print(f"   Negative (no call):   {neg:,} ({neg/len(merged)*100:.1f}%)")
    print(f"   File size: {os.path.getsize(train_path)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
```

---

## Step 2: Run augmentation

```bash
cd slm-tool-calling-finetune
python data/augment_negatives.py
```

---

## Step 3: Verify

```bash
wc -l data/train.jsonl
# Should be ~66,000 (54K + 12K)

# Spot-check a negative example
grep -c "function_calls" data/train.jsonl
# Should be ~54,000 (only positives have function_calls)
```

---

## Acceptance Criteria

- [ ] train.jsonl now has ~66,000 lines
- [ ] ~75% positive, ~25% negative ratio
- [ ] Negative examples have tools in system message but NO `<|function_calls|>` in assistant
- [ ] Dataset is shuffled
- [ ] valid.jsonl is NOT modified
