"""Verify MLX environment setup for SLM fine-tuning project."""
import mlx.core as mx

print(f"MLX version: {mx.__version__}")
print(f"Default device: {mx.default_device()}")
print(f"Metal available: {mx.metal.is_available()}")
if mx.metal.is_available():
    info = mx.metal.device_info()
    print(f"Metal device: {info.get('device_name', 'detected')}")

import mlx_lm
print(f"mlx-lm version: {mlx_lm.__version__}")

# Quick test: create a tensor on GPU
a = mx.ones((1000, 1000))
b = mx.ones((1000, 1000))
c = a @ b
mx.eval(c)
print(f"Matrix multiply test passed (result shape: {c.shape})")

print("\n✅ All dependencies verified.")
