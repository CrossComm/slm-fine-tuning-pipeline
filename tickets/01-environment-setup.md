# Ticket 01: Environment Setup (macOS + MLX)

**Dependencies:** None
**Estimated time:** 10 minutes
**Spec reference:** Section 4

---

## Objective

Set up a Python environment on macOS with MLX and mlx-lm. This is Apple Silicon — NO CUDA, NO PyTorch needed for training.

---

## Step 1: Verify system requirements

```bash
# Check macOS version (need 15.0+)
sw_vers

# Check chip (must be Apple Silicon)
sysctl -n machdep.cpu.brand_string
# OR
system_profiler SPHardwareDataType | grep "Chip"

# Check total memory
sysctl -n hw.memsize | awk '{print $0/1073741824 " GB"}'

# Check Xcode Command Line Tools (required for Metal compiler)
xcode-select -p
# If not installed:
# xcode-select --install

# Check disk space
df -h .
```

**Expected:** macOS 15+, Apple Silicon chip, 128GB memory, Xcode CLT installed, 20GB+ free disk.

---

## Step 2: Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

---

## Step 3: Install MLX and mlx-lm

```bash
pip install mlx-lm
```

This single install pulls in `mlx`, `mlx-lm`, and their dependencies. That's the core framework.

---

## Step 4: Install utility packages

```bash
pip install datasets huggingface-hub numpy
```

These are used only for downloading/preprocessing the xLAM dataset. They are NOT used during training.

---

## Step 5: Verify installation

```python
# verify_setup.py
import mlx.core as mx
print(f"MLX version: {mx.__version__}")
print(f"Default device: {mx.default_device()}")
print(f"Metal available: {mx.metal.is_available()}")
if mx.metal.is_available():
    print(f"Metal device: {mx.metal.device_info()['device_name'] if hasattr(mx.metal, 'device_info') else 'detected'}")

import mlx_lm
print(f"mlx-lm version: {mlx_lm.__version__}")

# Quick test: create a tensor on GPU
a = mx.ones((1000, 1000))
b = mx.ones((1000, 1000))
c = a @ b
mx.eval(c)
print(f"Matrix multiply test passed (result shape: {c.shape})")

print("\n✅ All dependencies verified.")
```

Run it:

```bash
python verify_setup.py
```

**Expected:** MLX imports, Metal is available, matrix multiply works.

---

## Step 6: Freeze requirements

```bash
pip freeze > requirements.txt
```

---

## Acceptance Criteria

- [ ] macOS 15+ on Apple Silicon confirmed
- [ ] Python 3.11 or 3.12 in venv
- [ ] `mlx-lm` imports without errors
- [ ] `mx.metal.is_available()` returns True
- [ ] Matrix multiply test passes on Metal
- [ ] `requirements.txt` generated

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `mlx` fails to import | Ensure macOS 15+ and Apple Silicon. MLX does not work on Intel Macs. |
| Metal not available | Install Xcode Command Line Tools: `xcode-select --install` |
| pip install fails | Try `pip install --no-cache-dir mlx-lm` |
| Wrong Python version | Use `python3.12 -m venv .venv` explicitly |
