# Ticket 10: Evaluation Script

**Dependencies:** Ticket 05, 09
**Estimated time:** 20 minutes
**Spec reference:** Section 10

---

## Objective

Create `evaluate.py` — runs the model against the validation set and computes all 5 metrics from the spec.

---

## Step 1: Create `evaluate.py`

Create `slm-tool-calling-finetune/evaluate.py`:

```python
"""
Evaluate the fine-tuned model against the validation set.

Metrics:
  - Tool Selection Accuracy (> 85%)
  - Argument Accuracy (> 80%)
  - JSON Validity Rate (> 98%)
  - False Positive Rate (< 5%)
  - False Negative Rate (< 10%)

Usage:
    python evaluate.py                         # Full eval on valid.jsonl
    python evaluate.py --max-samples 50        # Quick eval
    python evaluate.py --use-adapters          # Before fusing
"""

import argparse
import json
import os
import sys
import time
from inference import load_model, run_inference, extract_tool_calls, has_tool_call
from config import (
    DATA_DIR, TOOL_SELECTION_ACCURACY_TARGET, ARGUMENT_ACCURACY_TARGET,
    JSON_VALIDITY_TARGET, FALSE_POSITIVE_RATE_TARGET, FALSE_NEGATIVE_RATE_TARGET,
)


def load_validation_set(max_samples=None):
    """Load valid.jsonl and parse expected behavior."""
    valid_path = os.path.join(DATA_DIR, "valid.jsonl")
    examples = []
    with open(valid_path) as f:
        for line in f:
            examples.append(json.loads(line))
    if max_samples:
        examples = examples[:max_samples]
    return examples


def parse_example(example):
    """Extract tools, query, and expected calls from a JSONL example."""
    messages = example["messages"]
    system_msg = next((m for m in messages if m["role"] == "system"), None)
    user_msg = next((m for m in messages if m["role"] == "user"), None)
    assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)

    if not system_msg or not user_msg or not assistant_msg:
        return None

    # Extract tools from system message
    content = system_msg["content"]
    bracket_start = content.find("[")
    tools = []
    if bracket_start >= 0:
        try:
            tools = json.loads(content[bracket_start:])
        except json.JSONDecodeError:
            pass

    # Check if assistant response has tool calls
    expected_calls = extract_tool_calls(assistant_msg["content"])
    has_expected = has_tool_call(assistant_msg["content"])

    return {
        "tools": tools,
        "query": user_msg["content"],
        "expected_calls": expected_calls,
        "has_expected_calls": has_expected,
    }


def run_evaluation(model, tokenizer, examples, max_samples=None):
    """Run full evaluation pipeline."""
    total = len(examples)
    print(f"Evaluating {total} examples...\n")

    metrics = {
        "total": total, "errors": 0,
        "tool_correct": 0, "arg_correct": 0,
        "json_valid": 0, "json_total": 0,
        "tp": 0, "fp": 0, "tn": 0, "fn": 0,
        "pos_expected": 0, "neg_expected": 0,
    }

    start = time.time()

    for i, example in enumerate(examples):
        parsed = parse_example(example)
        if not parsed or not parsed["query"]:
            metrics["errors"] += 1
            continue

        try:
            response = run_inference(
                model, tokenizer,
                parsed["query"], parsed["tools"],
                max_tokens=256, temp=0.0,
            )
            predicted = extract_tool_calls(response)
            model_called = len(predicted) > 0

            if parsed["has_expected_calls"]:
                metrics["pos_expected"] += 1
                if model_called:
                    metrics["tp"] += 1
                    metrics["json_valid"] += 1
                    metrics["json_total"] += 1

                    # Tool selection
                    exp_names = sorted([c["name"] for c in parsed["expected_calls"]])
                    pred_names = sorted([c["name"] for c in predicted])
                    if exp_names == pred_names:
                        metrics["tool_correct"] += 1

                        # Argument check (only if tools match)
                        exp_by_name = {c["name"]: c for c in parsed["expected_calls"]}
                        pred_by_name = {c["name"]: c for c in predicted}
                        if all(exp_by_name[n].get("arguments") == pred_by_name[n].get("arguments")
                               for n in exp_names):
                            metrics["arg_correct"] += 1
                else:
                    metrics["fn"] += 1
            else:
                metrics["neg_expected"] += 1
                if model_called:
                    metrics["fp"] += 1
                    metrics["json_valid"] += 1
                    metrics["json_total"] += 1
                else:
                    metrics["tn"] += 1

        except Exception as e:
            metrics["errors"] += 1
            if i < 3:
                print(f"  Error on {i}: {e}")

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate / 60
            print(f"  {i+1}/{total} ({rate:.1f}/sec, ETA: {eta:.1f}m)")

    return metrics


def print_results(m):
    """Print metrics with pass/fail."""
    tp = m["tp"]
    tool_acc = m["tool_correct"] / tp if tp else 0
    arg_acc = m["arg_correct"] / tp if tp else 0
    json_rate = m["json_valid"] / m["json_total"] if m["json_total"] else 1.0
    fp_rate = m["fp"] / m["neg_expected"] if m["neg_expected"] else 0
    fn_rate = m["fn"] / m["pos_expected"] if m["pos_expected"] else 0

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n{'Metric':<30} {'Score':>8} {'Target':>10} {'Status':>8}")
    print("-" * 60)

    results = [
        ("Tool Selection Accuracy", tool_acc, TOOL_SELECTION_ACCURACY_TARGET, True),
        ("Argument Accuracy", arg_acc, ARGUMENT_ACCURACY_TARGET, True),
        ("JSON Validity Rate", json_rate, JSON_VALIDITY_TARGET, True),
        ("False Positive Rate", fp_rate, FALSE_POSITIVE_RATE_TARGET, False),
        ("False Negative Rate", fn_rate, FALSE_NEGATIVE_RATE_TARGET, False),
    ]

    all_pass = True
    for name, score, target, higher_better in results:
        passed = score >= target if higher_better else score <= target
        if not passed:
            all_pass = False
        d = "≥" if higher_better else "≤"
        s = "✅" if passed else "❌"
        print(f"{name:<30} {score:>7.1%} {d}{target:>7.1%} {s}")

    print(f"\nConfusion: TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")
    print(f"Errors: {m['errors']}")
    print(f"\n{'🎉 ALL PASS' if all_pass else '⚠️  SOME METRICS BELOW TARGET'}")
    return all_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--use-adapters", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_model(use_adapters=args.use_adapters)
    examples = load_validation_set(args.max_samples)
    metrics = run_evaluation(model, tokenizer, examples)
    passed = print_results(metrics)

    # Save results
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/eval_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved to outputs/eval_results.json")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
```

---

## Step 2: Test parsing (before training)

```bash
cd slm-tool-calling-finetune
python -c "
from evaluate import parse_example
ex = {'messages': [
    {'role':'system','content':'You are helpful.\n\n[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{}}}]'},
    {'role':'user','content':'test query'},
    {'role':'assistant','content':'<|function_calls|>\n[{\"name\":\"f\",\"arguments\":{}}]\n<|/function_calls|>'}
]}
p = parse_example(ex)
print(f'Query: {p[\"query\"]}')
print(f'Tools: {len(p[\"tools\"])}')
print(f'Calls: {p[\"expected_calls\"]}')
print('✅ Parser works')
"
```

---

## Acceptance Criteria

- [ ] `evaluate.py` computes all 5 metrics from the spec
- [ ] Quick mode works with `--max-samples 10`
- [ ] Supports `--use-adapters` for pre-fuse evaluation
- [ ] Prints pass/fail against targets
- [ ] Saves results to `outputs/eval_results.json`
- [ ] Exit code 0 on pass, 1 on fail
