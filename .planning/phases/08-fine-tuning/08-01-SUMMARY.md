---
phase: 08-fine-tuning
plan: 01
subsystem: training
tags: [lora, peft, trl, sft, smollm2, mps, fine-tuning, pytorch]

# Dependency graph
requires:
  - phase: 07-dataset-assembly
    provides: assembled DatasetDict at datasets/assembled/ with train/validation/test splits
provides:
  - scripts/train.py -- end-to-end fine-tuning script with argparse CLI and hardware auto-detection
  - tests/test_train.py -- 13 unit tests covering all non-GPU logic
  - requirements.txt updated with training dependencies (torch, peft, trl, accelerate)
affects: [08-02-training-execution, evaluation, deployment]

# Tech tracking
tech-stack:
  added: [torch>=2.6.0, peft>=0.19.0, trl>=1.2.0, accelerate>=1.13.0]
  patterns: [lazy-imports-for-testability, hardware-auto-detection, mock-module-injection-for-tests]

key-files:
  created:
    - scripts/train.py
    - tests/test_train.py
  modified:
    - requirements.txt

key-decisions:
  - "Lazy imports for torch/peft/trl inside functions rather than module-level -- enables testing without 3.4GB model download or ML library installation"
  - "Mock module injection pattern (sys.modules) for test environment -- tests create lightweight mock torch/peft/trl modules when real packages unavailable"
  - "fp32 default on MPS rather than bf16/fp16 -- most stable option per PyTorch Apple Silicon docs, avoids NaN loss issues"

patterns-established:
  - "Lazy ML imports: heavy ML libraries imported inside function bodies, not at module level, enabling unit testing on any machine"
  - "Hardware auto-detection: detect_hardware() returns (device, quant_config) tuple, centralizing all hardware-specific config"
  - "Mock module injection: tests install lightweight mock modules into sys.modules for packages not available in test environment"

requirements-completed: [TRNG-01, TRNG-02, TRNG-03]

# Metrics
duration: 6min
completed: 2026-04-21
---

# Phase 8 Plan 1: Training Script Summary

**LoRA fine-tuning script for SmolLM2-1.7B with MPS/CUDA/CPU auto-detection, full argparse CLI, and 13 mock-based unit tests**

## Performance

- **Duration:** 6 min
- **Started:** 2026-04-21T00:08:13Z
- **Completed:** 2026-04-21T00:14:17Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Complete fine-tuning script (364 lines) with hardware auto-detection for MPS, CUDA, and CPU
- Full argparse CLI with all D-04 hyperparameters, D-08 output paths, and D-11 optional wandb flag
- LoRA adapter targeting all 7 projection layers with SFTTrainer assistant-only loss
- 13 unit tests passing without any ML library installation using mock module injection pattern
- requirements.txt updated with 4 training dependencies (torch, peft, trl, accelerate)

## Task Commits

Each task was committed atomically:

1. **Task 1: Training script with hardware auto-detection and full CLI** - `bc3dab7` (feat)
2. **Task 2: Update requirements.txt with training dependencies** - `af05108` (chore)

## Files Created/Modified
- `scripts/train.py` - Complete fine-tuning script: hardware detection, argparse CLI, LoRA config, SFTConfig, training loop, adapter save, model merge
- `tests/test_train.py` - 13 unit tests covering detect_hardware (MPS/CUDA/CPU), CLI defaults/overrides/all-flags, LoRA config (default/custom), training args (MPS/CUDA/common/wandb), MPS fallback env
- `requirements.txt` - Added torch>=2.6.0, peft>=0.19.0, trl>=1.2.0, accelerate>=1.13.0 under "# Fine-tuning (Phase 8)" section

## Decisions Made
- **Lazy imports over module-level imports:** torch, peft, and trl are imported inside function bodies rather than at module top. This enables testing without installing heavy ML packages (torch alone is ~2GB). The env var `PYTORCH_ENABLE_MPS_FALLBACK=1` is still set at module level via os.environ.setdefault() which works without torch.
- **Mock module injection pattern:** Tests install lightweight mock modules (torch, peft, trl) into sys.modules when real packages are unavailable. Mocks replicate just enough interface (LoraConfig stores kwargs, SFTConfig normalizes report_to to list, torch.cuda/backends.mps have is_available MagicMocks) for function-level unit testing.
- **Excluded bitsandbytes and wandb from requirements:** bitsandbytes is CUDA-only and the primary target is MPS (per D-01, D-03). wandb is optional (per D-11) and only activated via --wandb flag.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Restructured train.py to use lazy imports for testability**
- **Found during:** Task 1 (TDD GREEN phase)
- **Issue:** Plan's RESEARCH.md skeleton showed torch/peft/trl as module-level imports. The test environment has Python 3.14 with transformers installed but no torch/peft/trl, making module-level imports fail on import.
- **Fix:** Moved all torch/peft/trl imports inside function bodies (detect_hardware, get_lora_config, get_training_args, main). build_parser() uses only stdlib argparse. Tests inject mock modules into sys.modules before importing scripts.train.
- **Files modified:** scripts/train.py, tests/test_train.py
- **Verification:** All 13 tests pass on Python 3.14 without torch/peft/trl installed
- **Committed in:** bc3dab7 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary for testability in the current environment. The script functions identically at runtime -- lazy imports have no performance impact since they only execute once.

## Issues Encountered
- Mock torch module needed `__spec__` attribute (via `importlib.machinery.ModuleSpec`) because the real transformers library calls `importlib.util.find_spec("torch")` during import, which inspects `__spec__`. Without it, collection failed with `ValueError: torch.__spec__ is None`.
- CUDA test for detect_hardware needed to patch `transformers.BitsAndBytesConfig` because the real class tries to resolve `torch.bfloat16` as a string via `getattr(torch, ...)`, which fails with the mock torch.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- scripts/train.py is ready to execute actual training (Plan 08-02)
- Training dependencies listed in requirements.txt but need installation before running: `pip install torch peft trl accelerate`
- Dataset at datasets/assembled/ (3,630 samples) is ready for consumption via DatasetDict.load_from_disk()
- Model merge step produces deployment-ready merged model at models/lyra-merged/

---
*Phase: 08-fine-tuning*
*Completed: 2026-04-21*
