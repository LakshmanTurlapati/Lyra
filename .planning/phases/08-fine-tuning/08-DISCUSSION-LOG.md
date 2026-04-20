# Phase 8: Fine-Tuning - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.

**Date:** 2026-04-20
**Phase:** 08-fine-tuning
**Areas discussed:** Training hardware target, Hyperparameters, Output format and checkpointing

---

## Training Hardware Target

| Option | Description | Selected |
|--------|-------------|----------|
| MPS (Apple Silicon Mac) | Train locally, no cloud costs. Aligns with Phase 3. | Yes |
| Google Colab | T4 GPU, free but session limits. | |
| Cloud GPU | Overkill for 1.7B. | |

**User's choice:** MPS (Apple Silicon Mac)

---

## Hyperparameters

| Option | Description | Selected |
|--------|-------------|----------|
| SmolLM2 community defaults | LoRA r=16, alpha=32, lr=2e-4, 3 epochs, batch=4, grad_accum=4. | Yes |
| Aggressive LoRA | r=64, alpha=128, more parameters. | |
| Conservative minimal | r=8, alpha=16, minimal viable. | |

**User's choice:** SmolLM2 community defaults

---

## Output Format and Checkpointing

| Option | Description | Selected |
|--------|-------------|----------|
| Adapter + merged safetensors | Both LoRA adapter (~50MB) and merged model (~3.4GB). | Yes |
| Adapter only | Just LoRA adapter, merge later. | |
| Merged only | Full model, no adapter preservation. | |

**User's choice:** Adapter + merged safetensors

---

## Deferred Ideas

None
