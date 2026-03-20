# Ticket 02: Project Scaffolding & Config Module

**Dependencies:** Ticket 01 (environment setup)
**Estimated time:** 10 minutes
**Spec reference:** Sections 13 (File Structure), 4.3 (Versions), 6 (LoRA Config), 9 (Training Config)

---

## Objective

Create the project directory structure and a central `config.py` module that holds ALL hyperparameters, paths, and model identifiers. Every other script imports from `config.py` — no magic numbers scattered across files.

---

## Step 1: Create directory structure

```bash
mkdir -p slm-tool-calling-finetune/data
mkdir -p slm-tool-calling-finetune/tests
mkdir -p slm-tool-calling-finetune/outputs
```

Verify:

```bash
find slm-tool-calling-finetune -type d
```

**Expected:**

```
slm-tool-calling-finetune
slm-tool-calling-finetune/data
slm-tool-calling-finetune/tests
slm-tool-calling-finetune/outputs
```

---

## Step 2: Create `config.py`

Create `slm-tool-calling-finetune/config.py` with the following content. Every value comes directly from the engineering spec — do NOT change any values.

```python
"""
Central configuration for Qwen 3.5-4B tool-calling fine-tuning.
All hyperparameters, paths, and model identifiers live here.
Every other script imports from this module.
"""

# =============================================================================
# MODEL
# =============================================================================
MODEL_NAME = "unsloth/Qwen3.5-4B"
MAX_SEQ_LENGTH = 2048
DTYPE = None  # Auto-detect (bf16 on supported GPUs)
LOAD_IN_4BIT = False  # IMPORTANT: Do NOT use 4-bit. Use bf16 LoRA.

# =============================================================================
# LORA
# =============================================================================
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0  # Must be 0 — non-zero disables Unsloth kernel fusion
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
LORA_BIAS = "none"
GRADIENT_CHECKPOINTING = "unsloth"  # Unsloth's optimized version
RANDOM_STATE = 42
USE_RSLORA = False
LOFTQ_CONFIG = None

# =============================================================================
# DATASET
# =============================================================================
DATASET_NAME = "Salesforce/xlam-function-calling-60k"
NEGATIVE_DATASET_NAME = "yahma/alpaca-cleaned"
NEGATIVE_EXAMPLE_COUNT = 12000  # ~25% of total when combined with 60K positives
TEST_SPLIT_RATIO = 0.1
SPLIT_SEED = 42

# =============================================================================
# CHAT TEMPLATE
# =============================================================================
SYSTEM_PROMPT = (
    "You are a helpful assistant with access to the following functions. "
    "Use them if required."
)

# =============================================================================
# TRAINING
# =============================================================================
OUTPUT_DIR = "./outputs"
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 4  # Reduce to 2 if OOM
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 4 * 4 = 16
LEARNING_RATE = 2e-4
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.1
BF16 = True
FP16 = False
OPTIM = "adamw_8bit"
MAX_GRAD_NORM = 1.0
LOGGING_STEPS = 10
EVAL_STRATEGY = "steps"
EVAL_STEPS = 200
SAVE_STRATEGY = "steps"
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 3
SEED = 42
DATALOADER_NUM_WORKERS = 2
REPORT_TO = "none"  # Change to "wandb" if W&B is configured
PACKING = True
DATASET_TEXT_FIELD = "text"

# =============================================================================
# EXPORT
# =============================================================================
LORA_OUTPUT_DIR = "./outputs/lora_adapters"
MERGED_OUTPUT_DIR = "./outputs/merged_model"
GGUF_OUTPUT_DIR = "./outputs/gguf_model"
GGUF_QUANTIZATION = "q4_k_m"
HUB_MODEL_NAME = "your-username/qwen3.5-4b-tool-calling"  # UPDATE THIS

# =============================================================================
# EVALUATION
# =============================================================================
TOOL_SELECTION_ACCURACY_TARGET = 0.85
ARGUMENT_ACCURACY_TARGET = 0.80
JSON_VALIDITY_TARGET = 0.98
FALSE_POSITIVE_RATE_TARGET = 0.05
FALSE_NEGATIVE_RATE_TARGET = 0.10
AST_MATCH_SCORE_TARGET = 0.80

# =============================================================================
# PATHS (relative to project root)
# =============================================================================
PROCESSED_DATASET_DIR = "./data/processed"
RAW_DATASET_DIR = "./data/raw"
```

---

## Step 3: Create `__init__.py` files

```bash
touch slm-tool-calling-finetune/__init__.py
touch slm-tool-calling-finetune/data/__init__.py
touch slm-tool-calling-finetune/tests/__init__.py
```

---

## Step 4: Create placeholder files

Create empty placeholder files for all scripts that will be implemented in later tickets:

```bash
touch slm-tool-calling-finetune/data/preprocess.py
touch slm-tool-calling-finetune/train.py
touch slm-tool-calling-finetune/inference.py
touch slm-tool-calling-finetune/evaluate.py
touch slm-tool-calling-finetune/export.py
touch slm-tool-calling-finetune/tests/test_tool_calling.py
```

---

## Step 5: Verify config imports

```bash
cd slm-tool-calling-finetune
python -c "from config import *; print(f'Model: {MODEL_NAME}'); print(f'Dataset: {DATASET_NAME}'); print(f'LoRA rank: {LORA_R}'); print('✅ Config imports OK')"
cd ..
```

**Expected:**

```
Model: unsloth/Qwen3.5-4B
Dataset: Salesforce/xlam-function-calling-60k
LoRA rank: 16
✅ Config imports OK
```

---

## Acceptance Criteria

- [ ] Directory structure matches spec Section 13 exactly
- [ ] `config.py` exists with ALL hyperparameters from the spec
- [ ] All values in `config.py` match the engineering spec exactly
- [ ] `from config import *` works without errors
- [ ] All placeholder files exist (even if empty)
- [ ] No hardcoded values exist outside `config.py`

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Import error on config | Make sure you're running from the `slm-tool-calling-finetune/` directory or add it to `PYTHONPATH` |
| Directory already exists | That's fine — `mkdir -p` is idempotent |
