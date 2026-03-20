"""
Central configuration. All hyperparameters, paths, and model identifiers.
Every other script imports from here.
"""

# =============================================================================
# MODEL
# =============================================================================
MODEL_NAME = "mlx-community/Qwen3.5-4B-MLX-bf16"
MODEL_NAME_4BIT = "mlx-community/Qwen3.5-4B-MLX-4bit"  # For quick dev iterations
MAX_SEQ_LENGTH = 2048

# Tool-calling token format.
# NOTE: Qwen3.5-4B-MLX-bf16 does NOT have <|function_calls|> tokens in its
# vocabulary (confirmed by download_model.py). Preprocessing must use XML-style
# tags instead of native function-call tokens.
TOOL_CALL_FORMAT = "xml"  # "native" | "xml"

# =============================================================================
# LORA (matches lora_config.yaml)
# =============================================================================
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
LORA_SCALE = 2.0  # alpha / rank
NUM_LORA_LAYERS = 16

# =============================================================================
# TRAINING
# =============================================================================
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
STEPS_PER_EVAL = 200
VAL_BATCHES = 25
STEPS_PER_REPORT = 10
GRAD_CHECKPOINT = True

# =============================================================================
# DATASET
# =============================================================================
DATASET_NAME = "Salesforce/xlam-function-calling-60k"
NEGATIVE_DATASET_NAME = "yahma/alpaca-cleaned"
NEGATIVE_EXAMPLE_COUNT = 12000
VAL_SPLIT_RATIO = 0.1
SPLIT_SEED = 42

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to the following functions. "
    "Use them if required."
)

# =============================================================================
# PATHS (relative to project root)
# =============================================================================
DATA_DIR = "./data"
ADAPTER_PATH = "./adapters"
FUSED_MODEL_PATH = "./fused_model"
HF_MODEL_PATH = "./hf_model"
LORA_CONFIG_PATH = "./lora_config.yaml"

# =============================================================================
# EVALUATION TARGETS
# =============================================================================
TOOL_SELECTION_ACCURACY_TARGET = 0.85
ARGUMENT_ACCURACY_TARGET = 0.80
JSON_VALIDITY_TARGET = 0.98
FALSE_POSITIVE_RATE_TARGET = 0.05
FALSE_NEGATIVE_RATE_TARGET = 0.10
