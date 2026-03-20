# SLM Tool-Calling Fine-Tuning — Ticket Index

**Project:** Fine-tune Qwen 3.5-4B for tool calling using MLX on Apple Silicon
**Hardware:** MacBook Pro, M4-series, 128GB unified memory
**Spec:** See `../engineering-spec.md`

## Execution Order

| # | Ticket | Dependencies | Est. Time |
|---|--------|-------------|-----------|
| 01 | Environment Setup (macOS + MLX) | None | 10 min |
| 02 | Project Scaffolding & Config | 01 | 10 min |
| 03 | Download & Verify Model | 01 | 10 min |
| 04 | Download & Explore xLAM Dataset | 01 | 10 min |
| 05 | Preprocess xLAM → JSONL | 02, 03, 04 | 25 min |
| 06 | Negative Example Augmentation | 05 | 15 min |
| 07 | LoRA Config & Training Script | 02 | 10 min |
| 08 | Run Training | 05, 06, 07 | 3-5 hrs |
| 09 | Inference Script & Tool Call Parser | 02 | 20 min |
| 10 | Evaluation Script | 05, 09 | 20 min |
| 11 | Fuse Model & Export | 08 | 15 min |
| 12 | Integration Tests | 08, 09, 10 | 25 min |

## Notes for Claude Code

- This runs on **macOS with Apple Silicon** — there is NO CUDA, NO NVIDIA GPU
- The framework is **MLX** (`mlx-lm`), NOT Unsloth, NOT PyTorch
- Training is done via the `mlx_lm.lora` CLI command, NOT a Python Trainer class
- Data must be **JSONL files** (train.jsonl, valid.jsonl), NOT HuggingFace Datasets
- Do NOT install bitsandbytes, flash-attn, apex, or any CUDA package
- The engineering spec (`../engineering-spec.md`) is the source of truth
