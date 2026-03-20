# Ticket 04: Download & Explore xLAM Dataset

**Dependencies:** Ticket 01, Ticket 02
**Estimated time:** 10 minutes
**Spec reference:** Sections 2.2 (Dataset), 8.1 (xLAM Schema)

---

## Objective

Download the Salesforce xLAM 60K dataset from HuggingFace, explore its structure, understand the column formats, and verify it matches the spec's expectations. This exploration directly informs the preprocessing pipeline in Ticket 05.

---

## Step 1: Create the dataset exploration script

Create `slm-tool-calling-finetune/data/explore_dataset.py`:

```python
"""
Download and explore the xLAM function-calling dataset.
Outputs schema info, sample rows, and statistics needed for preprocessing.
"""

import json
from datasets import load_dataset
from config import DATASET_NAME


def main():
    print(f"Loading dataset: {DATASET_NAME}")
    print()

    # =========================================================================
    # Step 1: Download dataset
    # =========================================================================
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"✅ Dataset loaded")
    print(f"   Total examples: {len(dataset):,}")
    print(f"   Columns: {dataset.column_names}")
    print(f"   Features: {dataset.features}")
    print()

    # =========================================================================
    # Step 2: Examine column types and content
    # =========================================================================
    sample = dataset[0]

    print("=" * 60)
    print("SAMPLE ROW (index 0)")
    print("=" * 60)

    for col in dataset.column_names:
        value = sample[col]
        print(f"\n--- Column: '{col}' ---")
        print(f"    Type: {type(value).__name__}")
        print(f"    Length: {len(str(value))}")

        # Show truncated value
        val_str = str(value)
        if len(val_str) > 500:
            print(f"    Value (first 500 chars): {val_str[:500]}...")
        else:
            print(f"    Value: {val_str}")

    print()

    # =========================================================================
    # Step 3: Parse JSON columns and examine structure
    # =========================================================================
    print("=" * 60)
    print("PARSED JSON STRUCTURE")
    print("=" * 60)

    # Parse 'tools' column
    print("\n--- 'tools' column (parsed) ---")
    try:
        tools = json.loads(sample["tools"]) if isinstance(sample["tools"], str) else sample["tools"]
        print(f"    Type after parse: {type(tools).__name__}")
        if isinstance(tools, list):
            print(f"    Number of tools: {len(tools)}")
            if len(tools) > 0:
                print(f"    First tool keys: {list(tools[0].keys()) if isinstance(tools[0], dict) else 'N/A'}")
                print(f"    First tool (pretty):")
                print(json.dumps(tools[0], indent=6))
    except (json.JSONDecodeError, TypeError) as e:
        print(f"    ❌ Failed to parse: {e}")
        print(f"    Raw value type: {type(sample['tools'])}")

    # Parse 'answers' column
    print("\n--- 'answers' column (parsed) ---")
    try:
        answers = json.loads(sample["answers"]) if isinstance(sample["answers"], str) else sample["answers"]
        print(f"    Type after parse: {type(answers).__name__}")
        if isinstance(answers, list):
            print(f"    Number of answers: {len(answers)}")
            if len(answers) > 0:
                print(f"    First answer keys: {list(answers[0].keys()) if isinstance(answers[0], dict) else 'N/A'}")
                print(f"    First answer (pretty):")
                print(json.dumps(answers[0], indent=6))
    except (json.JSONDecodeError, TypeError) as e:
        print(f"    ❌ Failed to parse: {e}")
        print(f"    Raw value type: {type(sample['answers'])}")

    # Show query
    print(f"\n--- 'query' column ---")
    print(f"    Value: {sample['query']}")

    print()

    # =========================================================================
    # Step 4: Statistics across the full dataset
    # =========================================================================
    print("=" * 60)
    print("DATASET STATISTICS (sampling first 1000 rows)")
    print("=" * 60)

    num_tools_list = []
    num_calls_list = []
    query_lengths = []
    parse_failures = 0

    for i in range(min(1000, len(dataset))):
        row = dataset[i]
        query_lengths.append(len(row["query"]))

        try:
            tools = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
            answers = json.loads(row["answers"]) if isinstance(row["answers"], str) else row["answers"]
            num_tools_list.append(len(tools) if isinstance(tools, list) else 0)
            num_calls_list.append(len(answers) if isinstance(answers, list) else 0)
        except (json.JSONDecodeError, TypeError):
            parse_failures += 1

    print(f"    Parse failures: {parse_failures} / 1000")
    print(f"    Query length - min: {min(query_lengths)}, max: {max(query_lengths)}, avg: {sum(query_lengths)/len(query_lengths):.0f}")
    if num_tools_list:
        print(f"    Tools per example - min: {min(num_tools_list)}, max: {max(num_tools_list)}, avg: {sum(num_tools_list)/len(num_tools_list):.1f}")
    if num_calls_list:
        print(f"    Calls per example - min: {min(num_calls_list)}, max: {max(num_calls_list)}, avg: {sum(num_calls_list)/len(num_calls_list):.1f}")

    # Count parallel calls
    parallel_calls = sum(1 for n in num_calls_list if n > 1)
    print(f"    Examples with parallel calls (>1): {parallel_calls} / 1000 ({parallel_calls/10:.1f}%)")

    print()

    # =========================================================================
    # Step 5: Examine 3 diverse examples
    # =========================================================================
    print("=" * 60)
    print("3 DIVERSE EXAMPLES")
    print("=" * 60)

    for idx in [0, 500, 999]:
        if idx >= len(dataset):
            continue
        row = dataset[idx]
        print(f"\n--- Example {idx} ---")
        print(f"    Query: {row['query'][:200]}...")
        try:
            answers = json.loads(row["answers"]) if isinstance(row["answers"], str) else row["answers"]
            print(f"    Calls: {json.dumps(answers, indent=6)[:300]}...")
        except:
            print(f"    Calls: [parse error]")

    print()
    print("=" * 60)
    print("EXPLORATION COMPLETE")
    print("=" * 60)
    print()
    print("KEY FINDINGS TO CARRY FORWARD:")
    print("  1. Note the exact keys in the 'tools' dict (for schema normalization)")
    print("  2. Note the exact keys in the 'answers' dict (for tool call formatting)")
    print("  3. Note whether columns are strings (need json.loads) or native objects")
    print(f"  4. Parse failure rate: {parse_failures/10:.1f}% (should be < 1%)")


if __name__ == "__main__":
    main()
```

---

## Step 2: Run the exploration

```bash
cd slm-tool-calling-finetune
python data/explore_dataset.py
```

---

## Step 3: Document findings

After running, update a comment at the top of `data/explore_dataset.py` with your findings:

- Exact column names and types
- Whether `tools` and `answers` are strings (need `json.loads`) or native lists
- Exact key names in tool definitions (e.g., `name`, `description`, `parameters` vs. `parameter`)
- Exact key names in answer objects (e.g., `name`, `arguments` vs. `parameters`)
- Parse failure rate
- Distribution of tool counts and call counts

These findings are CRITICAL for Ticket 05 (preprocessing). If the schema differs from what the spec assumes, the preprocessing pipeline must be adjusted.

---

## Acceptance Criteria

- [ ] Dataset downloads from HuggingFace without errors
- [ ] Total example count is ~60,000
- [ ] Column names are identified and documented
- [ ] JSON parsing of `tools` and `answers` columns works
- [ ] Tool definition structure is understood (keys, nesting)
- [ ] Answer/call structure is understood (keys, format)
- [ ] Parse failure rate is documented (should be < 1%)
- [ ] Statistics (tools per example, calls per example) are recorded
- [ ] At least 3 diverse examples have been examined

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Dataset not found | Check the exact HF ID: `Salesforce/xlam-function-calling-60k`. Try browsing it at `huggingface.co/datasets/Salesforce/xlam-function-calling-60k`. |
| Download fails | Set `HF_DATASETS_CACHE` to a directory with enough space. Check internet connection. |
| Memory error loading full dataset | Use `load_dataset(..., streaming=True)` and iterate instead of loading all at once. |
| JSON parse errors on most rows | The column may already be a native Python object (list/dict), not a string. Check `type(sample["tools"])` — if it's already a list, skip `json.loads()`. |
