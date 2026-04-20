# Phase 8: Fine-Tuning - Context

**Gathered:** 2026-04-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Fine-tune SmolLM2-1.7B on the assembled dataset (3,630 samples) using QLoRA with TRL SFTTrainer. Produce documented training scripts, hyperparameter configs, and model weights (adapter + merged). Covers TRNG-01, TRNG-02, TRNG-03.

</domain>

<decisions>
## Implementation Decisions

### Training Hardware
- **D-01:** MPS (Apple Silicon Mac) as primary training device. Consistent with Phase 3's MPS-first decision.
- **D-02:** Script must also support CPU fallback for environments without MPS/CUDA.
- **D-03:** No CUDA/cloud dependency required for v1. Training runs locally.

### Hyperparameters
- **D-04:** SmolLM2 community defaults as starting point:
  - LoRA rank: 16
  - LoRA alpha: 32
  - Learning rate: 2e-4
  - Epochs: 3
  - Batch size: 4
  - Gradient accumulation steps: 4 (effective batch size: 16)
- **D-05:** Target modules for LoRA: all attention layers (q_proj, k_proj, v_proj, o_proj) + gate/up/down MLP projections.
- **D-06:** 4-bit quantization (QLoRA with NF4) via bitsandbytes for memory efficiency on consumer hardware.

### Output Format
- **D-07:** Save BOTH LoRA adapter (~50MB) AND merged safetensors model (~3.4GB).
- **D-08:** Adapter saved to `models/lyra-adapter/` for iteration. Merged model saved to `models/lyra-merged/` for eval/deployment.
- **D-09:** Checkpoints saved every epoch. Final adapter is from best-performing checkpoint (lowest validation loss).

### Training Script Design
- **D-10:** Single documented training script `scripts/train.py` with argparse CLI. All hyperparameters configurable via flags with the defaults from D-04.
- **D-11:** Wandb integration optional (enabled via --wandb flag). Not required for v1 but available for experiment tracking.
- **D-12:** Training loads dataset from `datasets/assembled/` via HuggingFace DatasetDict.load_from_disk().

### Claude's Discretion
- Exact target modules beyond the standard attention/MLP set
- Warmup ratio and scheduler type
- Whether to use Unsloth's FastLanguageModel wrapper or raw transformers + peft
- Logging frequency and eval strategy
- Chat template configuration for SFTTrainer

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Dataset
- `datasets/assembled/` -- HuggingFace DatasetDict with train/validation/test splits
- `scripts/assemble_dataset.py` -- Assembly script (for reference on dataset format)

### Tech Stack (from CLAUDE.md)
- CLAUDE.md "Layer 3: Fine-Tuning Framework" section -- Unsloth, TRL, transformers, peft, bitsandbytes versions
- CLAUDE.md "Key Architecture Decisions: Why QLoRA" -- rationale for 4-bit approach

### Model
- SmolLM2-1.7B-Instruct on HuggingFace: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- Unsloth pre-quantized: `unsloth/SmolLM2-1.7B-Instruct`

### Prior Phase Patterns
- `scripts/` directory -- flat script pattern
- `configs/` -- YAML configuration pattern
- `tests/` -- pytest test pattern

### Project
- `.planning/PROJECT.md` -- Core constraints
- `.planning/REQUIREMENTS.md` -- TRNG-01, TRNG-02, TRNG-03 requirements

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `datasets/assembled/` -- Ready-to-load training data (DatasetDict format)
- `scripts/assemble_dataset.py` -- Reference for dataset loading pattern
- `requirements.txt` -- Will need torch, transformers, peft, trl, bitsandbytes, unsloth additions

### Established Patterns
- Flat `scripts/` directory with argparse CLI
- `models/` directory for output (new)
- `configs/` for YAML configuration

### Integration Points
- Loads assembled dataset via DatasetDict.load_from_disk()
- Outputs model weights consumed by Phase 9 (benchmarking)
- Wandb logs consumed by Phase 9 (training report)

</code_context>

<specifics>
## Specific Ideas

- MPS training means no bitsandbytes 4-bit quantization (bitsandbytes doesn't support MPS). The script will need to use 16-bit LoRA on MPS instead of 4-bit QLoRA. QLoRA path available when CUDA is detected.
- With 3,630 samples and batch size 4 + grad_accum 4 (effective batch 16): ~227 steps per epoch, ~681 steps total for 3 epochs. Feasible in 30-60 minutes on Apple Silicon.
- LoRA r=16 with all attention+MLP targets updates ~10% of parameters. Good balance for a small model.
- Adapter-first saves enable quick iteration: train multiple adapters with different hyperparams, compare, then merge the best one.

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 08-fine-tuning*
*Context gathered: 2026-04-20*
