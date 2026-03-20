# Ticket 08: Run Training

**Dependencies:** Ticket 05, 06, 07
**Estimated time:** 3-5 hours (Apple Silicon M-series)
**Spec reference:** Section 7

---

## Objective

Execute the training run to completion. Monitor loss, verify convergence, and confirm adapters are saved.

---

## Step 1: Quick validation run first

Always run a quick test before the multi-hour full run:

```bash
cd slm-tool-calling-finetune
./train.sh --quick 2>&1 | tee training_quick.log
```

This runs 1,000 iterations (~15 minutes). Verify:
- Loss starts at ~2-3 and decreases over 100+ steps
- No errors or NaN values
- Eval loss is reported at step 200
- Adapters are saved to `./adapters/`

If the quick run looks good, proceed to full training.

---

## Step 2: Full training run

```bash
cd slm-tool-calling-finetune
./train.sh 2>&1 | tee training.log
```

On a Mac with a detachable session, use `tmux` or `screen`:

```bash
tmux new -s training
cd slm-tool-calling-finetune
./train.sh 2>&1 | tee training.log
# Detach: Ctrl+B then D
# Reattach: tmux attach -t training
```

---

## Step 3: Monitor training

### Healthy training:
- Training loss decreases from ~2-3 to < 1.0 over first few thousand steps
- Eval loss tracks training loss (slightly higher is normal)
- Loss stabilizes after ~1 epoch (16,500 steps at batch_size=4)
- No NaN values

### Monitor memory:
```bash
# In another terminal
sudo powermetrics --samplers gpu_power -i 5000
# Or simply:
top -l1 | head -10
```

### Warning signs:

| Symptom | Action |
|---------|--------|
| Loss is NaN | Kill. Check data format. Reduce learning rate to 5e-6. |
| Loss doesn't decrease after 500 steps | Kill. Try learning rate 2e-5 or 5e-6. |
| Loss oscillates wildly | Kill. Increase batch size to 8. |
| Very slow (<10 tokens/sec) | Check that Metal is being used. Verify with `Activity Monitor → GPU`. |
| Loss goes UP after step 30,000+ | Overfitting. Best adapters were saved at earlier eval. Use those. |

---

## Step 4: Verify completion

```bash
# Check adapters were saved
ls -la slm-tool-calling-finetune/adapters/

# Check training log for final metrics
tail -30 training.log
```

**Expected files in `adapters/`:**
- `adapters.safetensors` (or `adapters.npz`)
- Possibly config files

---

## Step 5: Record metrics

- Final training loss: ___
- Final eval loss: ___
- Total iterations: ___
- Total time: ___

---

## Acceptance Criteria

- [ ] Training completed all iterations without crashing
- [ ] Final training loss < 1.5 (ideally < 1.0)
- [ ] No NaN loss values
- [ ] Adapter files exist in `./adapters/`
- [ ] `training.log` contains full output
