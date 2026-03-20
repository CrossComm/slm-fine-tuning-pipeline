# Ticket 09: Run Training

**Dependencies:** Ticket 08
**Estimated time:** 4–7 hours (GPU-bound)
**Spec reference:** Section 9.3 (Estimated Training Time)

---

## Objective

Execute the training script and monitor it to completion. This is the long-running step. The model will train for 3 epochs over ~66,000 examples (54K positive + 12K negative).

---

## Step 1: Pre-flight checks

Before starting the multi-hour training run, verify everything:

```bash
cd slm-tool-calling-finetune

# Check GPU is available and has enough VRAM
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB')"

# Check dataset exists and has correct size
python -c "from datasets import load_from_disk; t=load_from_disk('./data/processed/train'); v=load_from_disk('./data/processed/val'); print(f'Train: {len(t):,}, Val: {len(v):,}')"

# Check disk space (need ~10GB for checkpoints)
df -h .

# Check no other GPU processes are running
nvidia-smi
```

---

## Step 2: Run training

```bash
cd slm-tool-calling-finetune
python train.py 2>&1 | tee training.log
```

The `tee` command saves output to `training.log` while also printing to terminal, so you have a record even if the terminal disconnects.

**For remote/SSH sessions**, use `tmux` or `screen` to prevent training from stopping on disconnect:

```bash
tmux new -s training
cd slm-tool-calling-finetune
python train.py 2>&1 | tee training.log
# Detach with Ctrl+B, then D
# Reattach with: tmux attach -t training
```

---

## Step 3: Monitor training

While training runs, monitor these signals:

### Healthy training looks like:

- **Loss decreases** over the first 100-200 steps (from ~2-3 down to ~1.0-1.5)
- **Loss stabilizes** around 0.5-1.0 after first epoch
- **Eval loss** tracks training loss (slightly higher is normal)
- **No OOM errors**
- **GPU utilization** at 90-100% (check with `nvidia-smi`)

### Warning signs:

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| Loss is NaN | Precision issue | Kill training. Set `bf16=False, fp16=True` in config or reduce LR to 5e-5. Restart from scratch. |
| Loss doesn't decrease after 200 steps | Learning rate too low/high | Kill training. Try `LEARNING_RATE = 5e-5` or `1e-4`. Restart. |
| Loss oscillates wildly | Batch size too small | Kill training. Increase `GRADIENT_ACCUMULATION_STEPS` to 8. Restart. |
| OOM error | Not enough VRAM | Reduce `PER_DEVICE_TRAIN_BATCH_SIZE` to 2 in config. Restart. |
| Loss goes UP after epoch 2 | Overfitting | This is fine — the best checkpoint was saved earlier. Use that one. |
| Training is extremely slow (<1 step/sec) | Packing or data loading issue | Check `packing=True` in config. Check `dataloader_num_workers`. |

### Monitor GPU in another terminal:

```bash
watch -n 5 nvidia-smi
```

---

## Step 4: Verify training completed

After training finishes, check:

```bash
# Check the training log for final metrics
tail -50 slm-tool-calling-finetune/training.log

# Check that LoRA adapters were saved
ls -la slm-tool-calling-finetune/outputs/lora_adapters/

# Check checkpoint files
ls -la slm-tool-calling-finetune/outputs/
```

**Expected files in `outputs/lora_adapters/`:**
- `adapter_config.json`
- `adapter_model.safetensors` (or `.bin`)
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`

---

## Step 5: Record training metrics

Document these values from the training output:

- Final training loss: ___
- Final eval loss: ___
- Total training steps: ___
- Total training time: ___
- Best checkpoint step: ___

---

## Acceptance Criteria

- [ ] Training completed all 3 epochs without crashing
- [ ] Final training loss is < 1.5 (ideally < 1.0)
- [ ] Final eval loss is < 2.0 (ideally < 1.5)
- [ ] LoRA adapter files exist in `outputs/lora_adapters/`
- [ ] At least one checkpoint exists in `outputs/`
- [ ] No NaN loss values appeared during training
- [ ] `training.log` file contains the full training output

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Training killed by OOM mid-run | Reduce batch size in config.py, restart from scratch (not from checkpoint — LoRA checkpoints can be fragile). |
| SSH disconnected during training | If you used tmux/screen, reattach. If not, training is lost — restart with tmux next time. |
| Training seems stuck (no output) | Check `nvidia-smi` — if GPU is at 100%, it's working. Packing can make initial data loading slow. |
| Loss is good but eval loss is much higher | Normal if gap is < 0.5. If gap > 1.0, you may be overfitting — reduce epochs to 2. |
| Disk full error | Delete old checkpoints: `ls -lt outputs/checkpoint-*` and remove the oldest. |
| Want to resume from checkpoint | `trainer.train(resume_from_checkpoint="outputs/checkpoint-XXXX")` — but be cautious with LoRA resume stability. |
