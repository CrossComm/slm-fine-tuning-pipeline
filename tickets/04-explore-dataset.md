# Ticket 04: Download & Explore xLAM Dataset

**Dependencies:** Ticket 01
**Estimated time:** 10 minutes
**Spec reference:** Section 2.2

---

## Objective

Download the xLAM 60K dataset, examine its structure, and document the exact schema. This directly informs the preprocessing pipeline in Ticket 05.

---

## Step 1: Create exploration script

Create `slm-tool-calling-finetune/data/explore_dataset.py`:

```python
"""Download and explore the xLAM dataset. Document schema for preprocessing."""

import json
from datasets import load_dataset
from config import DATASET_NAME


def main():
    print(f"Loading: {DATASET_NAME}\n")
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"✅ Loaded {len(dataset):,} examples")
    print(f"   Columns: {dataset.column_names}\n")

    # Examine first row
    row = dataset[0]
    print("=" * 60)
    print("ROW 0 — RAW VALUES")
    print("=" * 60)
    for col in dataset.column_names:
        val = str(row[col])
        print(f"\n  [{col}] type={type(row[col]).__name__}, len={len(val)}")
        print(f"  {val[:300]}{'...' if len(val) > 300 else ''}")

    # Parse JSON columns
    print("\n" + "=" * 60)
    print("PARSED STRUCTURES")
    print("=" * 60)

    tools = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
    answers = json.loads(row["answers"]) if isinstance(row["answers"], str) else row["answers"]

    print(f"\n  tools: type={type(tools).__name__}, count={len(tools)}")
    print(f"  First tool keys: {list(tools[0].keys()) if tools else 'N/A'}")
    print(f"  First tool:\n{json.dumps(tools[0], indent=4)[:500]}")

    print(f"\n  answers: type={type(answers).__name__}, count={len(answers)}")
    print(f"  First answer keys: {list(answers[0].keys()) if answers else 'N/A'}")
    print(f"  First answer:\n{json.dumps(answers[0], indent=4)[:500]}")

    # CRITICAL: Check if arguments is a string (double-encoded)
    if answers and "arguments" in answers[0]:
        args = answers[0]["arguments"]
        print(f"\n  ⚠️  arguments type: {type(args).__name__}")
        if isinstance(args, str):
            print("  ⚠️  arguments IS a JSON string — needs json.loads() twice!")
            parsed_args = json.loads(args)
            print(f"  Parsed arguments: {json.dumps(parsed_args, indent=4)[:300]}")
        else:
            print("  arguments is already a dict — single json.loads() is enough")

    # Statistics on first 1000 rows
    print("\n" + "=" * 60)
    print("STATISTICS (first 1000 rows)")
    print("=" * 60)
    tools_counts, calls_counts, query_lens = [], [], []
    parse_fails = 0

    for i in range(min(1000, len(dataset))):
        r = dataset[i]
        query_lens.append(len(r["query"]))
        try:
            t = json.loads(r["tools"]) if isinstance(r["tools"], str) else r["tools"]
            a = json.loads(r["answers"]) if isinstance(r["answers"], str) else r["answers"]
            tools_counts.append(len(t))
            calls_counts.append(len(a))
        except:
            parse_fails += 1

    print(f"  Parse failures: {parse_fails}/1000")
    print(f"  Query length — min:{min(query_lens)} max:{max(query_lens)} avg:{sum(query_lens)//len(query_lens)}")
    print(f"  Tools/example — min:{min(tools_counts)} max:{max(tools_counts)} avg:{sum(tools_counts)/len(tools_counts):.1f}")
    print(f"  Calls/example — min:{min(calls_counts)} max:{max(calls_counts)} avg:{sum(calls_counts)/len(calls_counts):.1f}")
    parallel = sum(1 for n in calls_counts if n > 1)
    print(f"  Parallel calls (>1): {parallel}/1000 ({parallel/10:.1f}%)")

    print("\n✅ Exploration complete.")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
```

---

## Step 2: Run exploration

```bash
cd slm-tool-calling-finetune
python data/explore_dataset.py
```

---

## Step 3: Document findings for Ticket 05

Record:
- Whether `tools` and `answers` are JSON strings (need `json.loads()`) or native objects
- Whether `answers[].arguments` is double-encoded (JSON string inside JSON string)
- Exact key names in tools (e.g., `name`, `description`, `parameters`)
- Exact key names in answers (e.g., `name`, `arguments`)
- Parse failure rate (should be < 1%)

---

## Acceptance Criteria

- [ ] Dataset loads from HuggingFace
- [ ] Column schema documented (types, keys)
- [ ] Double-encoding of `arguments` field confirmed/denied
- [ ] Statistics recorded (tools per example, calls per example)
- [ ] Parse failure rate < 1%
