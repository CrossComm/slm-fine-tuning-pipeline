# Ticket 05: Preprocess xLAM → JSONL

**Dependencies:** Ticket 02, 03, 04
**Estimated time:** 25 minutes
**Spec reference:** Sections 5, 6

---

## Objective

Build `data/preprocess.py` — converts the raw xLAM dataset into JSONL chat-format files that MLX-LM expects. This is the most critical script. Bad data = bad model.

**Output:** `data/train.jsonl` and `data/valid.jsonl`

---

## Key Context

MLX-LM expects JSONL files where each line is:
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

MLX-LM automatically applies the model's chat template during training. We provide the raw messages; the tokenizer handles formatting.

---

## Step 1: Create `data/preprocess.py`

```python
"""
Convert xLAM dataset to JSONL chat format for MLX-LM training.

Output: data/train.jsonl, data/valid.jsonl

Each JSONL line is:
{"messages": [
    {"role": "system", "content": "<instruction + tool definitions>"},
    {"role": "user", "content": "<query>"},
    {"role": "assistant", "content": "<tool calls in native format>"}
]}
"""

import json
import os
import sys
import random
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATASET_NAME, SYSTEM_PROMPT, VAL_SPLIT_RATIO, SPLIT_SEED, DATA_DIR,
)


def parse_json_field(value):
    """Parse a field that may be a JSON string or already a Python object."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def normalize_tool(tool):
    """Normalize tool definition to OpenAI schema format."""
    # Already in OpenAI wrapper format
    if "type" in tool and tool.get("type") == "function" and "function" in tool:
        return tool

    # Flat format (name/description/parameters at top level)
    if "name" in tool:
        return {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", tool.get("parameter", {})),
            }
        }
    return None


def normalize_call(answer):
    """Normalize an answer to {"name": ..., "arguments": {...}}."""
    if not isinstance(answer, dict):
        return None

    name = answer.get("name", answer.get("function", ""))
    args = answer.get("arguments", answer.get("parameters", {}))

    # arguments is often a JSON string (double-encoded)
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}

    if not name:
        return None
    return {"name": name, "arguments": args}


def convert_row(row):
    """
    Convert one xLAM row to a messages dict for JSONL.
    Returns None if the row is invalid.
    """
    query = row.get("query", "")
    tools = parse_json_field(row.get("tools"))
    answers = parse_json_field(row.get("answers"))

    if not query or not tools or not answers:
        return None
    if not isinstance(tools, list) or not isinstance(answers, list):
        return None

    # Normalize tools
    norm_tools = [t for t in (normalize_tool(tool) for tool in tools) if t]
    if not norm_tools:
        return None

    # Normalize calls
    norm_calls = [c for c in (normalize_call(a) for a in answers) if c]
    if not norm_calls:
        return None

    # Build system message: instruction + tool definitions
    tools_json = json.dumps(norm_tools, ensure_ascii=False)
    system_content = f"{SYSTEM_PROMPT}\n\n{tools_json}"

    # Build assistant response with native tool-calling format
    calls_json = json.dumps(norm_calls, ensure_ascii=False)
    assistant_content = f"<|function_calls|>\n{calls_json}\n<|/function_calls|>"

    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def main():
    print("=" * 60)
    print("PREPROCESSING xLAM → JSONL FOR MLX")
    print("=" * 60)

    random.seed(SPLIT_SEED)

    # Load dataset
    print(f"\nLoading {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"✅ Loaded {len(dataset):,} rows")

    # Convert
    print("\nConverting to JSONL messages format...")
    examples = []
    skipped = 0
    for i, row in enumerate(dataset):
        result = convert_row(row)
        if result:
            examples.append(result)
        else:
            skipped += 1
        if (i + 1) % 10000 == 0:
            print(f"   {i+1:,} / {len(dataset):,} (kept: {len(examples):,}, skipped: {skipped:,})")

    print(f"\n   Total: {len(dataset):,}")
    print(f"   Kept:  {len(examples):,}")
    print(f"   Skipped: {skipped:,} ({skipped/len(dataset)*100:.1f}%)")

    # Shuffle and split
    random.shuffle(examples)
    val_count = int(len(examples) * VAL_SPLIT_RATIO)
    val_examples = examples[:val_count]
    train_examples = examples[val_count:]

    print(f"\n   Train: {len(train_examples):,}")
    print(f"   Valid: {len(val_examples):,}")

    # Write JSONL files
    os.makedirs(DATA_DIR, exist_ok=True)

    train_path = os.path.join(DATA_DIR, "train.jsonl")
    valid_path = os.path.join(DATA_DIR, "valid.jsonl")

    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(valid_path, "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n✅ Written:")
    print(f"   {train_path} ({os.path.getsize(train_path)/1e6:.1f} MB)")
    print(f"   {valid_path} ({os.path.getsize(valid_path)/1e6:.1f} MB)")

    # Show sample
    print("\n" + "=" * 60)
    print("SAMPLE (first training example)")
    print("=" * 60)
    sample = train_examples[0]
    for msg in sample["messages"]:
        role = msg["role"]
        content = msg["content"][:200]
        print(f"\n  [{role}]: {content}{'...' if len(msg['content']) > 200 else ''}")

    print("\n✅ Preprocessing complete.")


if __name__ == "__main__":
    main()
```

---

## Step 2: Run preprocessing

```bash
cd slm-tool-calling-finetune
python data/preprocess.py
```

---

## Step 3: Validate output

```bash
# Check files exist and have content
wc -l data/train.jsonl data/valid.jsonl

# Validate first line is valid JSON
head -1 data/train.jsonl | python -m json.tool > /dev/null && echo "✅ Valid JSON"

# Check a sample
head -1 data/train.jsonl | python -c "
import json, sys
line = json.loads(sys.stdin.read())
print(f'Keys: {list(line.keys())}')
print(f'Message count: {len(line[\"messages\"])}')
for m in line['messages']:
    print(f'  {m[\"role\"]}: {m[\"content\"][:80]}...')
"
```

---

## Acceptance Criteria

- [ ] `data/preprocess.py` runs without errors
- [ ] `data/train.jsonl` exists with ~54,000 lines
- [ ] `data/valid.jsonl` exists with ~6,000 lines
- [ ] Each JSONL line has `{"messages": [...]}` structure
- [ ] Each messages array has system (with tools), user (with query), assistant (with `<|function_calls|>`)
- [ ] Skip rate < 5%
- [ ] All lines are valid JSON (no corruption)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| High skip rate (>10%) | xLAM schema differs from expected. Check Ticket 04 findings and adjust `normalize_tool()` / `normalize_call()`. |
| Double-encoded arguments | The script handles this in `normalize_call()`. Verify with Ticket 04 output. |
| File too large (>500MB) | Normal for 60K examples with tool definitions. MLX handles this fine. |
| Unicode errors | Use `ensure_ascii=False` in `json.dumps()` (already set). |
