# Ticket 11: Evaluation Script & Metrics

**Dependencies:** Ticket 05, Ticket 10
**Estimated time:** 25 minutes
**Spec reference:** Section 12 (Evaluation)

---

## Objective

Create `evaluate.py` — a script that runs the fine-tuned model against the held-out validation set and computes all metrics from the spec: Tool Selection Accuracy, Argument Accuracy, JSON Validity Rate, False Positive Rate, False Negative Rate, and AST Match Score.

---

## Step 1: Create `evaluate.py`

Create `slm-tool-calling-finetune/evaluate.py`:

```python
"""
Evaluation script for the fine-tuned tool-calling model.

Runs the model against the validation set and computes:
  - Tool Selection Accuracy (target: > 85%)
  - Argument Accuracy (target: > 80%)
  - JSON Validity Rate (target: > 98%)
  - False Positive Rate (target: < 5%)
  - False Negative Rate (target: < 10%)
  - AST Match Score (target: > 80%)

Usage:
    python evaluate.py
    python evaluate.py --max-samples 100   # Quick eval on subset
    python evaluate.py --adapter-path ./outputs/lora_adapters

Prerequisites:
    - Ticket 09: Training complete
    - Ticket 10: inference.py available
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch
from datasets import load_from_disk

from config import (
    PROCESSED_DATASET_DIR, LORA_OUTPUT_DIR,
    TOOL_SELECTION_ACCURACY_TARGET, ARGUMENT_ACCURACY_TARGET,
    JSON_VALIDITY_TARGET, FALSE_POSITIVE_RATE_TARGET,
    FALSE_NEGATIVE_RATE_TARGET, AST_MATCH_SCORE_TARGET,
    SYSTEM_PROMPT, MAX_SEQ_LENGTH,
)
from inference import (
    load_finetuned_model, generate_response,
    extract_tool_calls, has_tool_call,
)


def parse_expected_from_text(text: str) -> dict:
    """
    Parse a preprocessed training example to extract the expected
    tool calls and tool definitions.

    Returns:
        dict with keys: tools, query, expected_calls, has_expected_calls
    """
    result = {
        "tools": [],
        "query": "",
        "expected_calls": [],
        "has_expected_calls": False,
    }

    # Extract the query (user message content)
    import re

    # Find user content between <|im_start|>user and <|im_end|>
    user_match = re.search(
        r'<\|im_start\|>user\s*\n(.*?)<\|im_end\|>',
        text, re.DOTALL
    )
    if user_match:
        result["query"] = user_match.group(1).strip()

    # Find system content to extract tools
    system_match = re.search(
        r'<\|im_start\|>system\s*\n(.*?)<\|im_end\|>',
        text, re.DOTALL
    )
    if system_match:
        system_content = system_match.group(1).strip()
        # Try to extract JSON array of tools from system message
        # Find the first [ that starts the tools array
        bracket_start = system_content.find('[')
        if bracket_start >= 0:
            try:
                tools_json = system_content[bracket_start:]
                result["tools"] = json.loads(tools_json)
            except json.JSONDecodeError:
                pass

    # Find expected tool calls in assistant response
    calls_match = re.search(
        r'<\|function_calls\|>\s*(.*?)\s*<\|/function_calls\|>',
        text, re.DOTALL
    )
    if calls_match:
        try:
            expected = json.loads(calls_match.group(1))
            if isinstance(expected, dict):
                expected = [expected]
            result["expected_calls"] = expected
            result["has_expected_calls"] = True
        except json.JSONDecodeError:
            pass

    # If no function_calls found, this is a negative example
    if not result["has_expected_calls"]:
        result["has_expected_calls"] = False

    return result


def compare_tool_calls(expected: list[dict], predicted: list[dict]) -> dict:
    """
    Compare expected vs predicted tool calls.

    Returns:
        dict with detailed comparison metrics
    """
    result = {
        "tool_selection_correct": False,
        "all_args_correct": False,
        "json_valid": True,  # If we got here, JSON was valid
        "expected_names": [],
        "predicted_names": [],
    }

    expected_names = sorted([c.get("name", "") for c in expected])
    predicted_names = sorted([c.get("name", "") for c in predicted])

    result["expected_names"] = expected_names
    result["predicted_names"] = predicted_names

    # Tool Selection: correct if same set of function names called
    result["tool_selection_correct"] = expected_names == predicted_names

    # Argument Accuracy: for each matching function, check if args match
    if result["tool_selection_correct"] and len(expected) == len(predicted):
        # Match by name
        expected_by_name = {c["name"]: c for c in expected}
        predicted_by_name = {c["name"]: c for c in predicted}

        all_correct = True
        for name in expected_names:
            exp_args = expected_by_name.get(name, {}).get("arguments", {})
            pred_args = predicted_by_name.get(name, {}).get("arguments", {})

            if exp_args != pred_args:
                all_correct = False
                break

        result["all_args_correct"] = all_correct

    return result


def run_evaluation(model, tokenizer, dataset, max_samples=None):
    """
    Run the full evaluation pipeline.

    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        dataset: Validation dataset
        max_samples: Limit number of samples (for quick testing)

    Returns:
        dict with all metrics
    """
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    total = len(dataset)
    print(f"Evaluating on {total} examples...")
    print()

    # Counters
    metrics = {
        "total": total,
        "tool_selection_correct": 0,
        "argument_correct": 0,
        "json_valid": 0,
        "json_invalid": 0,
        "true_positive": 0,   # Expected call, correctly called
        "false_positive": 0,  # No call expected, but model called
        "true_negative": 0,   # No call expected, model didn't call
        "false_negative": 0,  # Call expected, model didn't call
        "total_positive_expected": 0,
        "total_negative_expected": 0,
        "ast_matches": 0,
        "errors": 0,
    }

    start_time = time.time()

    for i, example in enumerate(dataset):
        text = example["text"]

        # Parse expected behavior from the preprocessed example
        expected = parse_expected_from_text(text)

        if not expected["query"] or not expected["tools"]:
            metrics["errors"] += 1
            continue

        try:
            # Generate model response
            response = generate_response(
                model, tokenizer,
                expected["query"],
                expected["tools"],
                max_new_tokens=256,
                temperature=0.0,  # Greedy for evaluation
            )

            # Parse predicted tool calls
            predicted_calls = extract_tool_calls(response)
            model_called_tool = len(predicted_calls) > 0

            if expected["has_expected_calls"]:
                metrics["total_positive_expected"] += 1

                if model_called_tool:
                    metrics["true_positive"] += 1
                    metrics["json_valid"] += 1

                    # Compare tool calls
                    comparison = compare_tool_calls(
                        expected["expected_calls"],
                        predicted_calls,
                    )

                    if comparison["tool_selection_correct"]:
                        metrics["tool_selection_correct"] += 1

                    if comparison["all_args_correct"]:
                        metrics["argument_correct"] += 1

                    # AST match = both name and args correct
                    if comparison["tool_selection_correct"] and comparison["all_args_correct"]:
                        metrics["ast_matches"] += 1

                else:
                    metrics["false_negative"] += 1

            else:
                metrics["total_negative_expected"] += 1

                if model_called_tool:
                    metrics["false_positive"] += 1
                    metrics["json_valid"] += 1  # JSON was valid even if shouldn't have called
                else:
                    metrics["true_negative"] += 1

        except Exception as e:
            metrics["errors"] += 1
            if i < 5:  # Print first few errors
                print(f"  Error on example {i}: {e}")

        # Progress
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            print(f"  {i+1}/{total} ({rate:.1f} ex/sec, ETA: {eta/60:.1f} min)")

    return metrics


def print_results(metrics: dict):
    """Print formatted evaluation results with pass/fail status."""
    total = metrics["total"]
    errors = metrics["errors"]
    evaluated = total - errors

    # Calculate rates
    pos_expected = metrics["total_positive_expected"]
    neg_expected = metrics["total_negative_expected"]
    tp = metrics["true_positive"]

    tool_selection_acc = metrics["tool_selection_correct"] / tp if tp > 0 else 0
    argument_acc = metrics["argument_correct"] / tp if tp > 0 else 0
    json_valid_rate = metrics["json_valid"] / (tp + metrics["false_positive"]) if (tp + metrics["false_positive"]) > 0 else 1.0
    false_pos_rate = metrics["false_positive"] / neg_expected if neg_expected > 0 else 0
    false_neg_rate = metrics["false_negative"] / pos_expected if pos_expected > 0 else 0
    ast_match = metrics["ast_matches"] / tp if tp > 0 else 0

    print()
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print()
    print(f"Total examples:     {total}")
    print(f"Evaluated:          {evaluated}")
    print(f"Errors/skipped:     {errors}")
    print(f"Positive expected:  {pos_expected}")
    print(f"Negative expected:  {neg_expected}")
    print()

    # Results table
    results = [
        ("Tool Selection Accuracy", tool_selection_acc, TOOL_SELECTION_ACCURACY_TARGET, True),
        ("Argument Accuracy", argument_acc, ARGUMENT_ACCURACY_TARGET, True),
        ("JSON Validity Rate", json_valid_rate, JSON_VALIDITY_TARGET, True),
        ("False Positive Rate", false_pos_rate, FALSE_POSITIVE_RATE_TARGET, False),
        ("False Negative Rate", false_neg_rate, FALSE_NEGATIVE_RATE_TARGET, False),
        ("AST Match Score", ast_match, AST_MATCH_SCORE_TARGET, True),
    ]

    print(f"{'Metric':<30} {'Score':>8} {'Target':>8} {'Status':>8}")
    print("-" * 60)

    all_passed = True
    for name, score, target, higher_is_better in results:
        if higher_is_better:
            passed = score >= target
        else:
            passed = score <= target

        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_passed = False

        direction = "≥" if higher_is_better else "≤"
        print(f"{name:<30} {score:>7.1%} {direction}{target:>6.1%} {status}")

    print()
    print(f"Confusion matrix:")
    print(f"  True Positive:  {metrics['true_positive']}")
    print(f"  True Negative:  {metrics['true_negative']}")
    print(f"  False Positive: {metrics['false_positive']}")
    print(f"  False Negative: {metrics['false_negative']}")
    print()

    if all_passed:
        print("🎉 ALL METRICS PASS — Model meets targets!")
    else:
        print("⚠️  Some metrics below target — see troubleshooting in ticket.")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Evaluate tool-calling model")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to evaluate (for quick testing)")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to LoRA adapters")
    args = parser.parse_args()

    # Load model
    model, tokenizer = load_finetuned_model(args.adapter_path)

    # Load validation set
    val_path = os.path.join(PROCESSED_DATASET_DIR, "val")
    val_dataset = load_from_disk(val_path)
    print(f"Loaded {len(val_dataset)} validation examples")

    # Run evaluation
    metrics = run_evaluation(model, tokenizer, val_dataset, args.max_samples)

    # Print results
    passed = print_results(metrics)

    # Save results to file
    results_path = os.path.join("outputs", "eval_results.json")
    os.makedirs("outputs", exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nDetailed results saved to {results_path}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
```

---

## Step 2: Test the parsing function standalone

```bash
cd slm-tool-calling-finetune
python -c "
from evaluate import parse_expected_from_text

sample = '''<|im_start|>system
You are a helpful assistant with access to the following functions. Use them if required.

[{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\"}}, \"required\": [\"location\"]}}}]
<|im_end|>
<|im_start|>user
What is the weather in NYC?
<|im_end|>
<|im_start|>assistant
<|function_calls|>
[{\"name\": \"get_weather\", \"arguments\": {\"location\": \"NYC\"}}]
<|/function_calls|>
<|im_end|>'''

parsed = parse_expected_from_text(sample)
print(f'Query: {parsed[\"query\"]}')
print(f'Tools found: {len(parsed[\"tools\"])}')
print(f'Expected calls: {parsed[\"expected_calls\"]}')
print(f'Has expected calls: {parsed[\"has_expected_calls\"]}')
print('✅ Parser works')
"
```

---

## Acceptance Criteria

- [ ] `evaluate.py` runs without errors (even if model isn't trained yet — it should fail gracefully on model load)
- [ ] `parse_expected_from_text()` correctly extracts query, tools, and expected calls from preprocessed examples
- [ ] All 6 metrics from the spec are computed and printed
- [ ] Results show pass/fail against spec targets
- [ ] Confusion matrix is printed (TP, TN, FP, FN)
- [ ] Results are saved to `outputs/eval_results.json`
- [ ] Quick eval mode works with `--max-samples 10`
- [ ] Exit code is 0 if all metrics pass, 1 if any fail

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| All scores are 0 | Check `parse_expected_from_text()` — the regex may not match the actual format. Print a raw example and compare. |
| Very slow evaluation | Use `--max-samples 100` for quick testing. Full eval on 6K examples takes 1-2 hours. |
| False positive rate is very high | The negative examples may not be in the val set (they were only added to train). This is expected — evaluate FP separately. |
| JSON validity is low | The model may need more training or constrained decoding. |
| Tool selection is good but args are wrong | This is common — args require more precise learning. Consider increasing LoRA rank to 32. |
