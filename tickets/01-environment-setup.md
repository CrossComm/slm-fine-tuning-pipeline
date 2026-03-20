# Ticket 01: Environment Setup & Dependency Installation

**Dependencies:** None
**Estimated time:** 15 minutes
**Spec reference:** Section 4 (Environment & Dependencies)

---

## Objective

Set up a Python environment with all required packages for fine-tuning Qwen 3.5-4B with Unsloth. This is a blank machine — everything must be installed from scratch.

---

## Pre-Flight Checks

Before installing anything, verify the system meets requirements:

### Step 1: Check Python version

```bash
python3 --version
```

**Expected:** Python 3.11.x or 3.12.x. If not installed or wrong version:

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# macOS (for development only — training requires NVIDIA GPU)
brew install python@3.12
```

### Step 2: Check CUDA version

```bash
nvidia-smi
nvcc --version
```

**Expected:** CUDA 11.8, 12.1, 12.4, or 12.6. Note the exact version — you'll need it for the Unsloth install variant.

If `nvidia-smi` fails, NVIDIA drivers are not installed. Install them:

```bash
# Ubuntu
sudo apt install -y nvidia-driver-535
sudo reboot
```

### Step 3: Check available GPU VRAM

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

**Expected:** At least 16GB (e.g., RTX 4080 16GB, RTX 4090 24GB, A4000 16GB).

### Step 4: Check available disk space

```bash
df -h .
```

**Expected:** At least 35GB free. Model weights (~8GB) + dataset (~500MB) + checkpoints (~24GB) + GGUF export (~3GB).

### Step 5: Check available RAM

```bash
free -h
```

**Expected:** At least 16GB, 32GB recommended. Dataset preprocessing loads the full dataset into memory.

---

## Create Virtual Environment

### Step 6: Create and activate venv

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Verify you're in the venv:

```bash
which python
# Should show: /path/to/.venv/bin/python
```

### Step 7: Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

---

## Install Dependencies

### Step 8: Install PyTorch (CUDA-specific)

Check your CUDA version from Step 2 and use the matching command:

```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 9: Install Unsloth

```bash
pip install unsloth
```

If this fails (common on some systems), use the from-source method:

```bash
# Replace cu124-torch250 with your CUDA/PyTorch version combo
pip install --upgrade --no-cache-dir --no-deps \
  "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git"
```

### Step 10: Install remaining ML dependencies

```bash
pip install --upgrade \
  trl>=0.15.0 \
  transformers>=4.48.0 \
  datasets>=3.0.0 \
  accelerate>=1.3.0 \
  bitsandbytes>=0.45.0 \
  peft>=0.14.0
```

### Step 11: Install utility packages

```bash
pip install wandb   # Optional: for experiment tracking
```

### Step 12: Create requirements.txt

Create this file at `./slm-tool-calling-finetune/requirements.txt` by freezing the current environment:

```bash
mkdir -p slm-tool-calling-finetune
pip freeze > slm-tool-calling-finetune/requirements.txt
```

---

## Verification

### Step 13: Verify all imports work

Create and run a verification script:

```python
# verify_setup.py
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

import transformers
print(f"Transformers version: {transformers.__version__}")

import trl
print(f"TRL version: {trl.__version__}")

import datasets
print(f"Datasets version: {datasets.__version__}")

import peft
print(f"PEFT version: {peft.__version__}")

from unsloth import FastLanguageModel
print("Unsloth imported successfully")

print("\n✅ All dependencies verified.")
```

Run it:

```bash
python verify_setup.py
```

**Expected output:** All versions print, CUDA is available, Unsloth imports without error.

---

## Acceptance Criteria

- [ ] Python 3.11 or 3.12 is active in the venv
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] GPU has >= 16GB VRAM
- [ ] All packages import without errors
- [ ] `requirements.txt` exists at `slm-tool-calling-finetune/requirements.txt`
- [ ] `verify_setup.py` prints "All dependencies verified"

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `nvidia-smi` not found | Install NVIDIA drivers: `sudo apt install nvidia-driver-535` then reboot |
| `torch.cuda.is_available()` returns False | PyTorch was installed without CUDA support. Reinstall with the correct `--index-url` for your CUDA version |
| Unsloth pip install fails | Use the from-source method in Step 9. Check that your CUDA/PyTorch combo is supported at `docs.unsloth.ai` |
| `bitsandbytes` import error | Run `pip install bitsandbytes --no-cache-dir` or build from source |
| Permission denied errors | Make sure you activated the venv (`source .venv/bin/activate`) |
| Disk space error during install | Clear pip cache: `pip cache purge` |
