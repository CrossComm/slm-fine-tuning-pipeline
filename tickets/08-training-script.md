# Ticket 08: Training Script (SFTTrainer)

**Dependencies:** Ticket 05, Ticket 06, Ticket 07
**Estimated time:** 25 minutes
**Spec reference:** Section 9 (Training Configuration)

---

## Objective

Create `train.py` — the main training script that loads the LoRA-configured model, loads the preprocessed dataset, configures SFTTrainer with all hyperparameters from the spec, and runs training. This script must be runnable as a single command with no interactive input.

---

## Step 1: Create `train.py`

Create `slm-tool-calling-finetune/train.py`:

```python
"""
Main training script for Qwen 3.5-4B tool-calling fine-tuning.

Loads the preprocessed dataset, configures SFTTrainer with all
hyperparameters from config.py, and runs training.

Usage:
    python train.py

Prerequisites:
    - Ticket 05: data/preprocess.py has been run
    - Ticket 06: data/augment_negatives.py has been run
    - Ticket 07: model_setup.py has been verified

Output:
    - Checkpoints saved to config.OUTPUT_DIR
    - LoRA adapters saved to config.LORA_OUTPUT_DIR
"""

import os
import torch
from datasets import load_from_disk
from trl import SFTTrainer
from transformers import TrainingArguments

from config import (
    # Training
    OUTPUT_DIR, NUM_TRAIN_EPOCHS, PER_DEVICE_TRAIN_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS, LEARNING_RATE, LR_SCHEDULER_TYPE,
    WARMUP_RATIO, BF16, FP16, OPTIM, MAX_GRAD_NORM,
    LOGGING_STEPS, EVAL_STRATEGY, EVAL_STEPS, SAVE_STRATEGY,
    SAVE_STEPS, SAVE_TOTAL_LIMIT, SEED, DATALOADER_NUM_WORKERS,
    REPORT_TO, PACKING, DATASET_TEXT_FIELD, MAX_SEQ_LENGTH,
    # Paths
    PROCESSED_DATASET_DIR, LORA_OUTPUT_DIR,
)
from model_setup import get_model_and_tokenizer


def main():
    print("=" * 60)
    print("TRAINING: Qwen 3.5-4B Tool-Calling Fine-Tune")
    print("=" * 60)
    print()

    # =========================================================================
    # Step 1: Load model and tokenizer with LoRA
    # =========================================================================
    print("Step 1: Loading model with LoRA adapters...")
    model, tokenizer = get_model_and_tokenizer()
    print()

    # =========================================================================
    # Step 2: Load preprocessed datasets
    # =========================================================================
    print("Step 2: Loading preprocessed datasets...")
    train_path = os.path.join(PROCESSED_DATASET_DIR, "train")
    val_path = os.path.join(PROCESSED_DATASET_DIR, "val")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            "Run data/preprocess.py and data/augment_negatives.py first."
        )
    if not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Validation data not found at {val_path}. "
            "Run data/preprocess.py first."
        )

    train_dataset = load_from_disk(train_path)
    val_dataset = load_from_disk(val_path)

    print(f"  Training examples:   {len(train_dataset):,}")
    print(f"  Validation examples: {len(val_dataset):,}")
    print(f"  Dataset columns:     {train_dataset.column_names}")
    print()

    # Verify the text column exists
    if DATASET_TEXT_FIELD not in train_dataset.column_names:
        raise ValueError(
            f"Column '{DATASET_TEXT_FIELD}' not found in dataset. "
            f"Available columns: {train_dataset.column_names}"
        )

    # =========================================================================
    # Step 3: Configure training arguments
    # =========================================================================
    print("Step 3: Configuring training arguments...")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        bf16=BF16,
        fp16=FP16,
        gradient_checkpointing=True,
        optim=OPTIM,
        max_grad_norm=MAX_GRAD_NORM,
        logging_steps=LOGGING_STEPS,
        eval_strategy=EVAL_STRATEGY,
        eval_steps=EVAL_STEPS,
        save_strategy=SAVE_STRATEGY,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        seed=SEED,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        report_to=REPORT_TO,
    )

    effective_batch = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    total_steps = (len(train_dataset) // effective_batch) * NUM_TRAIN_EPOCHS

    print(f"  Effective batch size: {effective_batch}")
    print(f"  Estimated total steps: {total_steps:,}")
    print(f"  Warmup steps: {int(total_steps * WARMUP_RATIO):,}")
    print(f"  Eval every: {EVAL_STEPS} steps")
    print(f"  Save every: {SAVE_STEPS} steps")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Optimizer: {OPTIM}")
    print(f"  Precision: {'bf16' if BF16 else 'fp16' if FP16 else 'fp32'}")
    print(f"  Packing: {PACKING}")
    print()

    # =========================================================================
    # Step 4: Create SFTTrainer
    # =========================================================================
    print("Step 4: Creating SFTTrainer...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=PACKING,
        dataset_text_field=DATASET_TEXT_FIELD,
    )

    print(f"  ✅ SFTTrainer created")
    print()

    # =========================================================================
    # Step 5: Pre-training memory check
    # =========================================================================
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"Pre-training GPU memory:")
        print(f"  Allocated: {allocated:.1f} GB / {total:.1f} GB")
        print()

    # =========================================================================
    # Step 6: Run training
    # =========================================================================
    print("Step 5: Starting training...")
    print(f"  Epochs: {NUM_TRAIN_EPOCHS}")
    print(f"  This will take approximately 4-7 hours depending on GPU.")
    print("  Monitor loss in the logs below.")
    print()
    print("-" * 60)

    train_result = trainer.train()

    print("-" * 60)
    print()
    print("Training complete!")
    print(f"  Final training loss: {train_result.training_loss:.4f}")
    print(f"  Total steps: {train_result.global_step}")
    print()

    # =========================================================================
    # Step 7: Save final model
    # =========================================================================
    print("Step 6: Saving LoRA adapters...")
    os.makedirs(LORA_OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(LORA_OUTPUT_DIR)
    tokenizer.save_pretrained(LORA_OUTPUT_DIR)
    print(f"  ✅ LoRA adapters saved to {LORA_OUTPUT_DIR}")
    print()

    # =========================================================================
    # Step 8: Log training metrics
    # =========================================================================
    print("Training metrics:")
    metrics = train_result.metrics
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value}")

    # Run final evaluation
    print()
    print("Running final evaluation...")
    eval_results = trainer.evaluate()
    print("Evaluation metrics:")
    for key, value in sorted(eval_results.items()):
        print(f"  {key}: {value}")

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  LoRA adapters: {LORA_OUTPUT_DIR}")
    print(f"  Checkpoints:   {OUTPUT_DIR}")
    print()
    print("Next steps:")
    print("  1. Run evaluate.py to test on hold-out set (Ticket 12)")
    print("  2. Run export.py to merge and export model (Ticket 13)")


if __name__ == "__main__":
    main()
```

---

## Step 2: Verify the script loads without errors (dry run)

Before actually training, verify everything initializes correctly:

```bash
cd slm-tool-calling-finetune
python -c "
from train import *
print('All imports successful')
print(f'Config values:')
print(f'  Batch size: {PER_DEVICE_TRAIN_BATCH_SIZE}')
print(f'  Gradient accum: {GRADIENT_ACCUMULATION_STEPS}')
print(f'  Learning rate: {LEARNING_RATE}')
print(f'  Epochs: {NUM_TRAIN_EPOCHS}')
"
```

---

## OOM Fallback Plan

If training OOMs, modify `config.py` with these changes (in order of preference):

1. **First try:** `PER_DEVICE_TRAIN_BATCH_SIZE = 2`, `GRADIENT_ACCUMULATION_STEPS = 8`
2. **Second try:** Also set `MAX_SEQ_LENGTH = 1024`
3. **Last resort:** Switch to `MODEL_NAME = "unsloth/Qwen3.5-2B"` and re-run Tickets 03 and 07

---

## Acceptance Criteria

- [ ] `train.py` exists and all imports resolve
- [ ] Script loads model, dataset, and creates SFTTrainer without errors
- [ ] All hyperparameters match config.py (which matches the engineering spec)
- [ ] SFTTrainer uses `packing=True` and `dataset_text_field="text"`
- [ ] Training arguments use `optim="adamw_8bit"` and `bf16=True`
- [ ] LoRA adapters are saved to `./outputs/lora_adapters` after training completes
- [ ] Final evaluation is run and metrics are printed
- [ ] Script is runnable as `python train.py` with no interactive input

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM during first training step | Reduce batch size (see fallback plan above). |
| OOM during evaluation | Set `per_device_eval_batch_size=1` in TrainingArguments. |
| Loss is NaN from the start | Check that bf16=True and fp16=False. Mixed precision issues can cause NaN. |
| Loss doesn't decrease after 100 steps | Learning rate may be too high or too low. Try 5e-5 or 1e-4. |
| "No module named model_setup" | Run from the `slm-tool-calling-finetune/` directory. |
| Dataset column error | Verify the preprocessed dataset has a `text` column: `load_from_disk("./data/processed/train").column_names` |
| Checkpoint disk space error | Reduce `SAVE_TOTAL_LIMIT` or increase disk space. Each checkpoint is ~200MB (LoRA only). |
