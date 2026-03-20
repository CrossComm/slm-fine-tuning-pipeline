"""Download and explore the xLAM dataset. Document schema for preprocessing."""

import json
import os
import sys

# Allow running directly from the data/ subdirectory by resolving the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from config import DATASET_NAME

SAMPLE_SIZE = 1000


def check_arguments_encoding(answers):
    """Check whether the first answer's arguments field is double-encoded.

    Args:
        answers: List of answer dicts parsed from the xLAM dataset.

    Returns:
        A dict with keys 'double_encoded' (bool) and 'arguments' (the parsed
        value), or None if answers is empty or has no 'arguments' key.
    """
    if not answers or "arguments" not in answers[0]:
        return None
    args = answers[0]["arguments"]
    if isinstance(args, str):
        return {"double_encoded": True, "arguments": json.loads(args)}
    return {"double_encoded": False, "arguments": args}


def collect_stats(rows):
    """Collect aggregate statistics over an iterable of dataset rows.

    Each row must be a dict with 'query' (str), 'tools' (JSON str or list),
    and 'answers' (JSON str or list) keys. Rows that fail to parse are counted
    in parse_fails but do not contribute to tools_counts or calls_counts.

    Args:
        rows: Iterable of dataset row dicts.

    Returns:
        A dict with keys:
            query_lens (list[int]): Length in chars of each query.
            tools_counts (list[int]): Number of tools per successfully parsed row.
            calls_counts (list[int]): Number of answer calls per successfully parsed row.
            parse_fails (int): Number of rows that raised a parse error.
    """
    tools_counts, calls_counts, query_lens = [], [], []
    parse_fails = 0

    for r in rows:
        query_lens.append(len(r["query"]))
        try:
            t = json.loads(r["tools"]) if isinstance(r["tools"], str) else r["tools"]
            a = json.loads(r["answers"]) if isinstance(r["answers"], str) else r["answers"]
            tools_counts.append(len(t))
            calls_counts.append(len(a))
        except (json.JSONDecodeError, ValueError, KeyError):
            parse_fails += 1

    return {
        "query_lens": query_lens,
        "tools_counts": tools_counts,
        "calls_counts": calls_counts,
        "parse_fails": parse_fails,
    }


def print_raw_row(row, column_names):
    """Print the raw values of a single dataset row.

    Args:
        row: A dataset row dict.
        column_names: Ordered list of column names to print.
    """
    print("=" * 60)
    print("ROW 0 — RAW VALUES")
    print("=" * 60)
    for col in column_names:
        val = str(row[col])
        print(f"\n  [{col}] type={type(row[col]).__name__}, len={len(val)}")
        print(f"  {val[:300]}{'...' if len(val) > 300 else ''}")


def print_parsed_structures(row):
    """Parse and print the tools and answers JSON fields from a dataset row.

    Also calls check_arguments_encoding and prints the double-encoding result.

    Args:
        row: A dataset row dict with 'tools' and 'answers' keys.
    """
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

    encoding = check_arguments_encoding(answers)
    if encoding is not None:
        print(f"\n  ⚠️  arguments type: {'str' if encoding['double_encoded'] else 'dict'}")
        if encoding["double_encoded"]:
            print("  ⚠️  arguments IS a JSON string — needs json.loads() twice!")
            print(f"  Parsed arguments: {json.dumps(encoding['arguments'], indent=4)[:300]}")
        else:
            print("  arguments is already a dict — single json.loads() is enough")


def print_stats(stats, sample_size):
    """Print formatted aggregate statistics from collect_stats output.

    Args:
        stats: Dict returned by collect_stats().
        sample_size: The number of rows that were sampled (for display).
    """
    query_lens = stats["query_lens"]
    tools_counts = stats["tools_counts"]
    calls_counts = stats["calls_counts"]
    parse_fails = stats["parse_fails"]

    sample_count = len(query_lens)
    parsed_count = len(tools_counts)

    print("\n" + "=" * 60)
    print(f"STATISTICS (first {sample_size} rows)")
    print("=" * 60)
    print(f"  Parse failures: {parse_fails}/{sample_count}")

    if sample_count > 0:
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
        print(f"  Calls/example — min:{min(calls_counts)} max:{max(calls_counts)} avg:{sum(calls_counts)/parsed_count:.1f}")
        print(f"  Parallel calls (>1): {parallel}/{parsed_count} ({parallel/parsed_count*100:.1f}%)")


def main():
    """Download the xLAM dataset and print schema, sample rows, and statistics.

    Loads the full training split, inspects raw column types, parses JSON
    fields, confirms whether arguments are double-encoded, and prints aggregate
    statistics over the first SAMPLE_SIZE rows to inform the preprocessing pipeline.
    """
    print(f"Loading: {DATASET_NAME}\n")
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"✅ Loaded {len(dataset):,} examples")
    print(f"   Columns: {dataset.column_names}\n")

    row = dataset[0]
    print_raw_row(row, dataset.column_names)
    print_parsed_structures(row)

    stats = collect_stats(dataset[i] for i in range(min(SAMPLE_SIZE, len(dataset))))
    print_stats(stats, SAMPLE_SIZE)

    print("\n✅ Exploration complete.")


if __name__ == "__main__":
    main()
