# Ticket 05: Data Preprocessing Pipeline

**Dependencies:** Ticket 02, Ticket 03, Ticket 04
**Estimated time:** 30 minutes
**Spec reference:** Sections 7 (Chat Template), 8 (Data Preprocessing)

---

## Objective

Build `data/preprocess.py` — the pipeline that transforms raw xLAM dataset rows into fully formatted Qwen 3.5 ChatML conversations ready for SFTTrainer. This is the most critical script in the project. Bad preprocessing = bad model.

---

## Step 1: Understand the transformation

Each xLAM row must be converted from:

```
Raw: {query, tools (JSON string), answers (JSON string)}
```

To:

```
Formatted ChatML conversation string with:
  - System message containing tool definitions
  - User message with the query
  - Assistant message with <|function_calls|> tokens wrapping the tool call array
  - EOS token at the end
```

---

## Step 2: Create `data/preprocess.py`

Create `slm-tool-calling-finetune/data/preprocess.py` with the following logic. Read the inline comments carefully — they explain every decision.

```python
"""
Preprocessing pipeline for xLAM → Qwen 3.5 tool-calling format.

Transforms raw xLAM dataset rows into formatted ChatML conversations
using Qwen 3.5's native <|function_calls|> tokens.

Usage:
    python data/preprocess.py

Output:
    Saves processed dataset to config.PROCESSED_DATASET_DIR
"""

import json
import os
import sys
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_NAME, DATASET_NAME, MAX_SEQ_LENGTH,
    SYSTEM_PROMPT, TEST_SPLIT_RATIO, SPLIT_SEED,
    PROCESSED_DATASET_DIR,
)


def load_tokenizer():
    """Load the tokenizer for chat template application."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Ensure pad_token is set and different from eos_token
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        if "<|finetune_right_pad_id|>" in tokenizer.get_vocab():
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
        else:
            tokenizer.pad_token = "<|endoftext|>"

    return tokenizer


def parse_json_field(value):
    """
    Parse a JSON field that may be a string or already a Python object.
    The xLAM dataset stores tools and answers as JSON strings in some versions
    and as native objects in others. Handle both cases.
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def normalize_tool_definition(tool):
    """
    Normalize a tool definition to OpenAI function-calling schema.

    The xLAM dataset may use slightly different key names or nesting.
    This function ensures every tool follows the format from Spec Section 7.2:

    {
        "type": "function",
        "function": {
            "name": "...",
            "description": "...",
            "parameters": { ... }
        }
    }
    """
    # If tool already has the OpenAI wrapper format
    if "type" in tool and tool.get("type") == "function" and "function" in tool:
        return tool

    # If tool is a flat dict with name/description/parameters at top level
    if "name" in tool:
        normalized = {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
            }
        }
        # Handle parameters — might be "parameters" or "parameter"
        params = tool.get("parameters", tool.get("parameter", {}))
        if isinstance(params, dict):
            normalized["function"]["parameters"] = params
        else:
            normalized["function"]["parameters"] = {
                "type": "object",
                "properties": {},
                "required": [],
            }
        return normalized

    # Unknown format — return as-is and let filtering catch it
    return tool


def normalize_answer(answer):
    """
    Normalize an answer/tool-call to the format:
    {"name": "function_name", "arguments": {"key": "value"}}
    """
    if isinstance(answer, dict):
        name = answer.get("name", answer.get("function", ""))
        args = answer.get("arguments", answer.get("parameters", {}))

        # Arguments might be a JSON string
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}

        return {"name": name, "arguments": args}

    return None


def format_example(row, tokenizer):
    """
    Transform a single xLAM row into a formatted ChatML conversation string.

    Returns:
        dict with "text" key containing the formatted conversation,
        or None if the row is invalid.
    """
    # Parse JSON fields
    tools = parse_json_field(row.get("tools"))
    answers = parse_json_field(row.get("answers"))
    query = row.get("query", "")

    # Validate
    if not tools or not answers or not query:
        return None
    if not isinstance(tools, list) or not isinstance(answers, list):
        return None
    if len(tools) == 0 or len(answers) == 0:
        return None

    # Normalize tool definitions
    normalized_tools = []
    for tool in tools:
        norm = normalize_tool_definition(tool)
        if norm:
            normalized_tools.append(norm)

    if not normalized_tools:
        return None

    # Normalize answers into tool call format
    normalized_calls = []
    for answer in answers:
        norm = normalize_answer(answer)
        if norm and norm["name"]:
            normalized_calls.append(norm)

    if not normalized_calls:
        return None

    # Build the tool call string using Qwen 3.5 native format
    tool_calls_json = json.dumps(normalized_calls, ensure_ascii=False)
    assistant_content = f"<|function_calls|>\n{tool_calls_json}\n<|/function_calls|>"

    # Build the system message with tool definitions
    tools_json = json.dumps(normalized_tools, indent=2, ensure_ascii=False)
    system_content = f"{SYSTEM_PROMPT}\n\n{tools_json}"

    # Build messages array for chat template
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query},
        {"role": "assistant", "content": assistant_content},
    ]

    # Apply chat template
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        # Ensure EOS token is at the end
        if not text.endswith(tokenizer.eos_token):
            text += tokenizer.eos_token
    except Exception as e:
        print(f"Chat template error: {e}")
        return None

    # Check sequence length
    token_count = len(tokenizer.encode(text))
    if token_count > MAX_SEQ_LENGTH:
        return None  # Skip examples that are too long

    return {"text": text}


def main():
    print("=" * 60)
    print("PREPROCESSING xLAM DATASET FOR QWEN 3.5 TOOL CALLING")
    print("=" * 60)
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer()
    print(f"✅ Tokenizer loaded (vocab size: {len(tokenizer)})")
    print()

    # Load dataset
    print(f"Loading dataset: {DATASET_NAME}...")
    raw_dataset = load_dataset(DATASET_NAME, split="train")
    print(f"✅ Dataset loaded ({len(raw_dataset):,} examples)")
    print()

    # Process examples
    print("Processing examples...")
    processed_examples = []
    skipped = 0

    for i, row in enumerate(raw_dataset):
        result = format_example(row, tokenizer)
        if result is not None:
            processed_examples.append(result)
        else:
            skipped += 1

        if (i + 1) % 10000 == 0:
            print(f"   Processed {i+1:,} / {len(raw_dataset):,} "
                  f"(kept: {len(processed_examples):,}, skipped: {skipped:,})")

    print()
    print(f"Processing complete:")
    print(f"   Total raw:     {len(raw_dataset):,}")
    print(f"   Kept:          {len(processed_examples):,}")
    print(f"   Skipped:       {skipped:,} ({skipped/len(raw_dataset)*100:.1f}%)")
    print()

    # Create HuggingFace dataset
    processed_dataset = Dataset.from_list(processed_examples)

    # Split into train/validation
    split = processed_dataset.train_test_split(
        test_size=TEST_SPLIT_RATIO,
        seed=SPLIT_SEED,
    )
    train_dataset = split["train"]
    val_dataset = split["test"]

    print(f"Split:")
    print(f"   Training:   {len(train_dataset):,}")
    print(f"   Validation: {len(val_dataset):,}")
    print()

    # Save to disk
    os.makedirs(PROCESSED_DATASET_DIR, exist_ok=True)
    train_path = os.path.join(PROCESSED_DATASET_DIR, "train")
    val_path = os.path.join(PROCESSED_DATASET_DIR, "val")

    train_dataset.save_to_disk(train_path)
    val_dataset.save_to_disk(val_path)

    print(f"✅ Saved to disk:")
    print(f"   Train: {train_path}")
    print(f"   Val:   {val_path}")
    print()

    # Show sample output
    print("=" * 60)
    print("SAMPLE PROCESSED EXAMPLE")
    print("=" * 60)
    print(processed_examples[0]["text"][:1000])
    print("..." if len(processed_examples[0]["text"]) > 1000 else "")
    print()

    # Token length statistics
    print("Token length statistics (first 1000 examples):")
    lengths = []
    for ex in processed_examples[:1000]:
        lengths.append(len(tokenizer.encode(ex["text"])))
    print(f"   Min: {min(lengths)}")
    print(f"   Max: {max(lengths)}")
    print(f"   Avg: {sum(lengths)/len(lengths):.0f}")
    print(f"   Median: {sorted(lengths)[len(lengths)//2]}")
    print()

    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Step 3: Run preprocessing

```bash
cd slm-tool-calling-finetune
python data/preprocess.py
```

**Expected output:**
- ~58,000–60,000 examples kept (< 5% skipped)
- Train/val split at 90/10
- Sample output shows proper ChatML formatting with `<|function_calls|>` tokens
- Token lengths mostly under 2048

---

## Step 4: Verify output

After running, verify the saved dataset loads correctly:

```python
from datasets import load_from_disk
train = load_from_disk("./data/processed/train")
val = load_from_disk("./data/processed/val")
print(f"Train: {len(train)}, Val: {len(val)}")
print(f"Columns: {train.column_names}")
print(f"Sample text (first 300 chars): {train[0]['text'][:300]}")
```

---

## Acceptance Criteria

- [ ] `data/preprocess.py` runs without errors
- [ ] Skip rate is < 5% of total examples
- [ ] Output dataset has a `text` column with formatted ChatML strings
- [ ] Each formatted example contains `<|im_start|>system`, `<|im_start|>user`, `<|im_start|>assistant`
- [ ] Tool calls are wrapped in `<|function_calls|>` / `<|/function_calls|>` tokens
- [ ] Tool definitions in system message follow OpenAI schema
- [ ] Each example ends with the EOS token
- [ ] No examples exceed MAX_SEQ_LENGTH (2048) tokens
- [ ] Train/val split is saved to disk and reloadable
- [ ] Token length statistics are reasonable (avg < 1500)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| High skip rate (>10%) | Check `normalize_tool_definition()` — the xLAM schema may differ from what's expected. Run Ticket 04's explorer to compare. |
| `apply_chat_template` fails | The tokenizer may not have a chat template. Build the ChatML string manually using `<\|im_start\|>` / `<\|im_end\|>` tokens. |
| Memory error during processing | Process in batches: load the dataset with streaming, process 10K at a time, and concatenate. |
| JSON encoding errors | Some tool descriptions may contain special characters. Use `ensure_ascii=False` and handle encoding. |
| Token lengths > 2048 for many examples | Consider increasing `MAX_SEQ_LENGTH` in config to 4096 if GPU memory allows. |
