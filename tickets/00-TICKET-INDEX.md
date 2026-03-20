# SLM Tool-Calling Fine-Tuning — Ticket Index

**Project:** Fine-tune Qwen 3.5-4B for tool calling using Unsloth + xLAM 60K
**Spec:** See `SLM-Tool-Calling-Engineering-Spec.md` in outputs folder

## Execution Order

Tickets MUST be completed in this order. Each ticket lists its dependencies.

| # | Ticket | Dependencies | Est. Time |
|---|--------|-------------|-----------|
| 01 | Environment Setup & Dependency Installation | None | 15 min |
| 02 | Project Scaffolding & Config Module | 01 | 10 min |
| 03 | Model Download & Tokenizer Verification | 01, 02 | 10 min |
| 04 | Download & Explore xLAM Dataset | 01, 02 | 10 min |
| 05 | Data Preprocessing Pipeline | 02, 03, 04 | 30 min |
| 06 | Negative Example Augmentation | 05 | 20 min |
| 07 | LoRA Configuration & Model Preparation | 03 | 15 min |
| 08 | Training Script (SFTTrainer) | 05, 06, 07 | 25 min |
| 09 | Run Training | 08 | 4-7 hrs (GPU) |
| 10 | Inference Script & Tool Call Parser | 02, 03 | 20 min |
| 11 | Evaluation Script & Metrics | 05, 10 | 25 min |
| 12 | Run Evaluation on Hold-Out Set | 09, 11 | 30 min |
| 13 | Model Export (Merge, GGUF, Hub) | 09 | 15 min |
| 14 | Integration Test Suite | 10, 12, 13 | 30 min |

## Notes for Claude Code

- Start at Ticket 01 and work sequentially
- Each ticket is self-contained with exact commands, code, and acceptance criteria
- If a ticket fails, check the "Troubleshooting" section at the bottom of that ticket before moving on
- The engineering spec (`SLM-Tool-Calling-Engineering-Spec.md`) is the source of truth for all design decisions — do NOT deviate from it without explicit user approval
- All code files go in `./slm-tool-calling-finetune/` within the project directory
