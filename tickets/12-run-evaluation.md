# Ticket 12: Run Evaluation on Hold-Out Set

**Dependencies:** Ticket 09, Ticket 11
**Estimated time:** 30 minutes – 2 hours (depending on sample count)
**Spec reference:** Section 12 (Evaluation)

---

## Objective

Run the evaluation script against the held-out validation set. Analyze results and determine if the model meets the spec's quality targets.

---

## Step 1: Quick sanity check (10 samples)

Before running the full evaluation, do a quick check to make sure everything works:

```bash
cd slm-tool-calling-finetune
python evaluate.py --max-samples 10
```

This should complete in 1-2 minutes. Verify:
- No crashes
- The output shows metrics (even if scores are low on just 10 samples)
- The confusion matrix makes sense

---

## Step 2: Medium evaluation (200 samples)

```bash
python evaluate.py --max-samples 200
```

This gives statistically more meaningful results. Takes ~10-20 minutes. Look for:
- Tool Selection Accuracy trending toward > 85%
- JSON Validity Rate should be close to 98%+
- False Positive Rate should be low

---

## Step 3: Full evaluation

```bash
python evaluate.py 2>&1 | tee eval_output.log
```

This evaluates all ~6,000 validation examples. Takes 1-2 hours depending on GPU.

---

## Step 4: Analyze results

After evaluation completes, review the results:

```bash
cat outputs/eval_results.json | python -m json.tool
```

### Target metrics from the spec:

| Metric | Target | Your Result |
|--------|--------|-------------|
| Tool Selection Accuracy | > 85% | ___ |
| Argument Accuracy | > 80% | ___ |
| JSON Validity Rate | > 98% | ___ |
| False Positive Rate | < 5% | ___ |
| False Negative Rate | < 10% | ___ |
| AST Match Score | > 80% | ___ |

---

## Step 5: If metrics fail

If the model doesn't meet targets, try these interventions in order:

### JSON Validity Rate < 98%
1. Increase training epochs to 4-5 in config.py
2. Add constrained decoding (install `outlines` library and modify inference.py)
3. Verify preprocessing is creating valid JSON in training examples

### Tool Selection Accuracy < 85%
1. Check if the model confuses similar tools — may need better tool descriptions
2. Increase LoRA rank from 16 to 32
3. Add more training data (supplement with Glaive v2)

### False Positive Rate > 5%
1. Increase negative examples from 12K to 18K
2. Re-run Ticket 06 with `NEGATIVE_EXAMPLE_COUNT = 18000`
3. Re-run training

### Argument Accuracy < 80%
1. This is the hardest metric — args require precise understanding
2. Increase LoRA rank to 32
3. Consider increasing max_seq_length to 4096 if truncation is causing arg loss

### False Negative Rate > 10%
1. Check if training examples have enough variety of tool types
2. The model may be under-trained — add 1-2 more epochs

---

## Acceptance Criteria

- [ ] Evaluation completes on the full validation set without crashes
- [ ] All 6 metrics are reported with pass/fail status
- [ ] Results are saved to `outputs/eval_results.json`
- [ ] Results are documented (fill in the table above)
- [ ] If any metric fails, an intervention plan is documented
- [ ] `eval_output.log` contains the full evaluation output

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Evaluation is extremely slow | Check GPU utilization with `nvidia-smi`. If low, increase `max_new_tokens` setting or check data loading. |
| All predictions are empty (no tool calls) | Model may not have learned the tool-calling format. Check a few raw outputs: `python inference.py --query "What's the weather in NYC?"` |
| OOM during evaluation | Generation uses more memory than training. Restart with `--max-samples 500` to verify, then try again after clearing GPU cache. |
| Metrics all at 0% | The `parse_expected_from_text()` regex may not match your preprocessed format. Print a raw example and the parsed output to debug. |
