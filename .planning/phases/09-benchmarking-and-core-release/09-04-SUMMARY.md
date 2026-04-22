---
phase: 09-benchmarking-and-core-release
plan: 04
subsystem: evaluation
tags: [lm-eval, eval-harness, mps, smollm2, benchmarking]

requires:
  - phase: 09-benchmarking-and-core-release (plans 02, 03)
    provides: eval_runner.py, eval_inference.py, eval_merge.py, eval_compare.py scripts
  - phase: 08-fine-tuning
    provides: models/lyra-merged trained model weights
provides:
  - results/base_knowledge.json — lm-eval knowledge benchmarks for SmolLM2-1.7B-Instruct
  - results/lyra_knowledge.json — lm-eval knowledge benchmarks for Lyra merged model
  - results/base_custom.json — custom inference eval (tool-call-format, code-syntax) for base model
  - results/lyra_custom.json — custom inference eval for Lyra model
  - results/base_full.json — merged EvalResult for base model (knowledge + custom)
  - results/lyra_full.json — merged EvalResult for Lyra model (knowledge + custom)
affects: [09-05-release-artifacts, investigation-phase]

tech-stack:
  added: [lm-eval[hf]==0.4.11 (re-installed in venv)]
  patterns: [sequential eval execution for MPS memory safety, --limit 100 for fair comparison]

key-files:
  created:
    - results/base_knowledge.json
    - results/lyra_knowledge.json
    - results/base_custom.json
    - results/lyra_custom.json
    - results/base_full.json
    - results/lyra_full.json

key-decisions:
  - "Used --limit 100 for lm-eval benchmarks to match base model sampling and keep runtime feasible on MPS (~20 min vs ~14 hours)"
  - "Ran all evals strictly sequentially — parallel eval crashed 64GB machine previously"

patterns-established:
  - "Sequential model evaluation: never run two model-loading processes simultaneously on MPS"
  - "Fair comparison: always use same --limit setting for base and candidate models"

requirements-completed: [EVAL-01]

duration: 60min
completed: 2026-04-21
---

# Phase 9, Plan 04: Run Model Evaluations Summary

**Six eval JSON files produced: Lyra shows tool-call-format regression (-0.22), slight ARC improvement (+0.05), and stable knowledge retention**

## Performance

- **Duration:** ~60 min (eval runs on MPS)
- **Started:** 2026-04-21
- **Tasks:** 3 completed (Task 1a pre-existing)
- **Files created:** 6

## Accomplishments
- Established base SmolLM2-1.7B-Instruct baseline across 5 benchmarks
- Completed Lyra model evaluation across all benchmarks
- Merged knowledge + custom results into full EvalResult files for comparison
- Produced valid 5-benchmark comparison table

## Comparison Results

| Category | Benchmark | Metric | Baseline | Lyra | Delta |
|----------|-----------|--------|----------|------|-------|
| knowledge | mmlu | acc | 0.5012 | 0.4604 | -0.0409 |
| knowledge | arc_challenge | acc_norm | 0.4500 | 0.5000 | +0.0500 |
| knowledge | hellaswag | acc_norm | 0.6600 | 0.6500 | -0.0100 |
| custom | tool-call-format | pass@1 | 0.4065 | 0.1870 | -0.2195 |
| custom | code-syntax | pass@1 | 0.2000 | 0.2000 | 0.0000 |

**Critical finding:** tool-call-format REGRESSED from 0.41 to 0.19 after fine-tuning. This is the opposite of the expected outcome and requires investigation before release.

## Task Commits

1. **Task 1a: Base model evaluations** — `c75253b` (pre-existing)
2. **Task 1b: Lyra model evaluations** — `74444a4` (feat)
3. **Task 2: Merge results** — `92a260a` (feat)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Re-installed lm-eval 0.4.11**
- **Found during:** Task 1b (Lyra knowledge benchmarks)
- **Issue:** lm-eval was not available in venv despite being used in Task 1a
- **Fix:** `pip install 'lm-eval[hf]==0.4.11'`
- **Verification:** Import successful, correct version confirmed

**2. [Rule 3 - Blocking] Used --limit 100 for fair comparison**
- **Found during:** Task 1b (Lyra knowledge benchmarks)
- **Issue:** Full benchmark (101K requests) would take ~14 hours on MPS
- **Fix:** Used `--limit 100` matching base model's commit c75253b approach
- **Verification:** Both models evaluated with same sampling methodology

---

**Total deviations:** 2 auto-fixed (both blocking)
**Impact on plan:** Both necessary for execution feasibility. No scope creep.

## Issues Encountered
- Tool-call-format regression is a significant finding — Lyra performs worse than base on the primary fine-tuning target
- Possible causes: training data format mismatch with eval format, catastrophic forgetting, insufficient/imbalanced training data

## Next Phase Readiness
- Eval results are complete and valid for comparison
- **BLOCKER:** Tool-call-format regression must be investigated and fixed before proceeding to release artifacts (Plan 09-05)
- Investigation phase 9.1 to be inserted for root cause analysis and retraining

---
*Phase: 09-benchmarking-and-core-release*
*Completed: 2026-04-21*
