# Ticket 06: Negative Example Augmentation

**Dependencies:** Ticket 05
**Estimated time:** 20 minutes
**Spec reference:** Section 8.5 (Including Negative Examples)

---

## Objective

Augment the preprocessed training dataset with ~12,000 negative examples — conversations where tools ARE defined in the system message but the correct response is a natural language answer WITHOUT calling any tools. This prevents the model from learning "always call a tool when tools are present."

---

## Why This Matters

The xLAM dataset is 100% positive examples (every example calls a tool). Without negative examples, the model learns:

> "If tools are in the system message, I MUST call one."

This causes false positives in production — the model calls tools when the user just wants a conversational answer. Target ratio after augmentation: ~75% tool-calling, ~25% no-tool-call.

---

## Step 1: Create the augmentation script

Create `slm-tool-calling-finetune/data/augment_negatives.py`:

```python
"""
Augment the training dataset with negative examples (no tool calls).

Downloads yahma/alpaca-cleaned, selects general Q&A examples,
wraps them in Qwen 3.5 chat template WITH tool definitions in the
system message but WITHOUT any <|function_calls|> in the response.

Usage:
    python data/augment_negatives.py

Prerequisite:
    Ticket 05 (preprocess.py) must have been run first.
"""

import json
import os
import sys
import random
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    NEGATIVE_DATASET_NAME, NEGATIVE_EXAMPLE_COUNT, SPLIT_SEED,
    SYSTEM_PROMPT, PROCESSED_DATASET_DIR, MODEL_NAME,
)
from transformers import AutoTokenizer


# A set of realistic tool definitions to randomly include in negative examples.
# The model needs to see tools in the system message and learn NOT to call them.
DUMMY_TOOL_SETS = [
    [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City and state"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        }
    ],
    [
        {
            "type": "function",
            "function": {
                "name": "send_email",
                "description": "Send an email to a recipient",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "description": "Recipient email"},
                        "subject": {"type": "string", "description": "Email subject"},
                        "body": {"type": "string", "description": "Email body"}
                    },
                    "required": ["to", "subject", "body"]
                }
            }
        }
    ],
    [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform a mathematical calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression to evaluate"}
                    },
                    "required": ["expression"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "translate_text",
                "description": "Translate text between languages",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to translate"},
                        "target_language": {"type": "string", "description": "Target language code"}
                    },
                    "required": ["text", "target_language"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Get the current stock price",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock ticker symbol"}
                    },
                    "required": ["symbol"]
                }
            }
        }
    ],
]


def load_tokenizer():
    """Load tokenizer with pad token fix."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        if "<|finetune_right_pad_id|>" in tokenizer.get_vocab():
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
        else:
            tokenizer.pad_token = "<|endoftext|>"
    return tokenizer


def create_negative_example(instruction, output, tokenizer):
    """
    Create a negative example: tools in system message, but the assistant
    responds with natural language (no tool calls).
    """
    # Randomly select a tool set
    tools = random.choice(DUMMY_TOOL_SETS)
    tools_json = json.dumps(tools, indent=2, ensure_ascii=False)
    system_content = f"{SYSTEM_PROMPT}\n\n{tools_json}"

    # Build messages — assistant responds WITHOUT tool calls
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output},
    ]

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if not text.endswith(tokenizer.eos_token):
            text += tokenizer.eos_token
        return {"text": text}
    except Exception:
        return None


def main():
    print("=" * 60)
    print("NEGATIVE EXAMPLE AUGMENTATION")
    print("=" * 60)
    print()

    random.seed(SPLIT_SEED)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer()
    print(f"✅ Tokenizer loaded")
    print()

    # Load existing processed training data
    train_path = os.path.join(PROCESSED_DATASET_DIR, "train")
    print(f"Loading existing training data from {train_path}...")
    train_dataset = load_from_disk(train_path)
    original_count = len(train_dataset)
    print(f"✅ Loaded {original_count:,} positive examples")
    print()

    # Load Alpaca dataset for negative examples
    print(f"Loading negative example source: {NEGATIVE_DATASET_NAME}...")
    alpaca = load_dataset(NEGATIVE_DATASET_NAME, split="train")
    print(f"✅ Loaded {len(alpaca):,} Alpaca examples")
    print()

    # Filter Alpaca examples:
    # - Must have both instruction and output
    # - Skip examples that look like they need a tool (contain URLs, API references, etc.)
    # - Skip very short outputs (< 20 chars) — not useful for training
    print("Filtering suitable negative examples...")
    candidates = []
    tool_keywords = ["api", "http", "url", "endpoint", "fetch", "request", "database", "query"]

    for row in alpaca:
        instruction = row.get("instruction", "")
        output = row.get("output", "")
        input_text = row.get("input", "")

        # Combine instruction with input if present
        if input_text:
            instruction = f"{instruction}\n{input_text}"

        # Filter criteria
        if not instruction or not output:
            continue
        if len(output) < 20:
            continue
        # Skip examples that sound like they genuinely need a tool
        lower_inst = instruction.lower()
        if any(kw in lower_inst for kw in tool_keywords):
            continue

        candidates.append((instruction, output))

    print(f"   Candidates after filtering: {len(candidates):,}")

    # Randomly sample
    if len(candidates) > NEGATIVE_EXAMPLE_COUNT:
        candidates = random.sample(candidates, NEGATIVE_EXAMPLE_COUNT)
    print(f"   Selected: {len(candidates):,}")
    print()

    # Create negative examples
    print("Creating negative examples...")
    negative_examples = []
    for i, (instruction, output) in enumerate(candidates):
        result = create_negative_example(instruction, output, tokenizer)
        if result:
            negative_examples.append(result)

        if (i + 1) % 5000 == 0:
            print(f"   Processed {i+1:,} / {len(candidates):,}")

    print(f"   Created: {len(negative_examples):,} negative examples")
    print()

    # Create negative dataset and merge with positive
    negative_dataset = Dataset.from_list(negative_examples)

    # Merge
    augmented_train = concatenate_datasets([train_dataset, negative_dataset])

    # Shuffle the merged dataset
    augmented_train = augmented_train.shuffle(seed=SPLIT_SEED)

    print(f"Augmented dataset:")
    print(f"   Positive examples: {original_count:,}")
    print(f"   Negative examples: {len(negative_examples):,}")
    print(f"   Total:             {len(augmented_train):,}")
    print(f"   Negative ratio:    {len(negative_examples)/len(augmented_train)*100:.1f}%")
    print()

    # Save augmented dataset (overwrite the train split)
    augmented_train.save_to_disk(train_path)
    print(f"✅ Saved augmented training set to {train_path}")
    print()

    # Show a sample negative example
    print("=" * 60)
    print("SAMPLE NEGATIVE EXAMPLE")
    print("=" * 60)
    # Find a negative example (it won't have <|function_calls|>)
    for ex in negative_examples[:5]:
        if "<|function_calls|>" not in ex["text"]:
            print(ex["text"][:800])
            break
    print()

    print("=" * 60)
    print("AUGMENTATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Step 2: Run the augmentation

```bash
cd slm-tool-calling-finetune
python data/augment_negatives.py
```

---

## Step 3: Verify the augmented dataset

```python
from datasets import load_from_disk

train = load_from_disk("./data/processed/train")
print(f"Total training examples: {len(train):,}")

# Count positive vs negative
pos = sum(1 for ex in train if "<|function_calls|>" in ex["text"])
neg = len(train) - pos
print(f"Positive (tool call): {pos:,} ({pos/len(train)*100:.1f}%)")
print(f"Negative (no call):   {neg:,} ({neg/len(train)*100:.1f}%)")
```

**Expected:** ~75% positive, ~25% negative.

---

## Acceptance Criteria

- [ ] `data/augment_negatives.py` runs without errors
- [ ] ~12,000 negative examples are created
- [ ] Negative examples have tool definitions in system message but NO `<|function_calls|>` in assistant response
- [ ] Augmented training set is shuffled and saved to disk
- [ ] Final ratio is approximately 75% positive / 25% negative
- [ ] Validation set is NOT modified (only training set is augmented)
- [ ] Sample negative example looks natural and well-formatted

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Alpaca dataset not found | Check the HF ID `yahma/alpaca-cleaned`. Alternative: use `tatsu-lab/alpaca`. |
| Too few candidates after filtering | Relax the keyword filter or reduce `NEGATIVE_EXAMPLE_COUNT` in config. |
| Memory error during concatenation | Process in chunks: save negative examples separately, then merge using `datasets.concatenate_datasets()`. |
| Negative ratio too low/high | Adjust `NEGATIVE_EXAMPLE_COUNT` in config.py. Target is 10,000–15,000. |
