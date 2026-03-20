"""Download and explore the xLAM dataset. Document schema for preprocessing."""

import json
import os
import sys

# Allow running directly from the data/ subdirectory by resolving the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from config import DATASET_NAME

SAMPLE_SIZE = 1000


def main():
    """Download the xLAM dataset and print schema, sample rows, and statistics.

    Loads the full training split, inspects raw column types, parses JSON
    fields, confirms whether arguments are double-encoded, and prints aggregate
    statistics over the first 1000 rows to inform the preprocessing pipeline.
    """
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

    # Statistics on first SAMPLE_SIZE rows
    print("\n" + "=" * 60)
    print(f"STATISTICS (first {SAMPLE_SIZE} rows)")
    print("=" * 60)
    tools_counts, calls_counts, query_lens = [], [], []
    parse_fails = 0

    for i in range(min(SAMPLE_SIZE, len(dataset))):
        r = dataset[i]
        query_lens.append(len(r["query"]))
        try:
            t = json.loads(r["tools"]) if isinstance(r["tools"], str) else r["tools"]
            a = json.loads(r["answers"]) if isinstance(r["answers"], str) else r["answers"]
            tools_counts.append(len(t))
            calls_counts.append(len(a))
        except (json.JSONDecodeError, ValueError, KeyError):
            parse_fails += 1

    sample_count = len(query_lens)
    parsed_count = len(tools_counts)
    print(f"  Parse failures: {parse_fails}/{sample_count}")
    print(f"  Query length — min:{min(query_lens)} max:{max(query_lens)} avg:{sum(query_lens)//sample_count}")

    if parsed_count == 0:
        print("  Tools/example — no data (all rows failed to parse)")
        print("  Calls/example — no data (all rows failed to parse)")
        print("  Parallel calls (>1) — no data (all rows failed to parse)")
    else:
        if parsed_count < sample_count:
            print(f"  Note: tools/calls stats cover {parsed_count} successfully parsed rows only")
        parallel = sum(1 for n in calls_counts if n > 1)
        print(f"  Tools/example — min:{min(tools_counts)} max:{max(tools_counts)} avg:{sum(tools_counts)/parsed_count:.1f}")
        print(f"  Calls/example — min:{min(calls_counts)} max:{max(calls_counts)} avg:{sum(calls_counts)/len(calls_counts):.1f}")
        print(f"  Parallel calls (>1): {parallel}/{len(calls_counts)} ({parallel/len(calls_counts)*100:.1f}%)")

    print("\n✅ Exploration complete.")


if __name__ == "__main__":
    main()
