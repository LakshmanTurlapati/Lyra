---
phase: 03-evaluation-framework
plan: 02
subsystem: evaluation
tags: [lm-eval, evalplus, bfcl, pytest, cli, argparse, subprocess]

# Dependency graph
requires:
  - phase: 03-evaluation-framework
    provides: EvalResult, CategoryResult, BenchmarkResult Pydantic schemas, EvalConfig, load_eval_config, compare_results
affects: [training-evaluation, model-publishing, fine-tuning]

provides:
  - eval_runner.py unified CLI invoking lm-eval, evalplus, and BFCL benchmark suites
  - Comprehensive test suite covering schemas, comparison logic, and runner orchestration
  - Device detection (MPS-primary, CPU-fallback) per D-02
  - Per-category JSON output matching EvalResult schema per D-07
  - Summary table stdout output per D-08

# Tech tracking
tech-stack:
  added: []
  patterns: [lazy imports for optional heavy deps, list-form subprocess for security, mocked benchmark orchestration testing]

key-files:
  created:
    - scripts/eval_runner.py
    - tests/test_eval_config.py
    - tests/test_eval_compare.py
    - tests/test_eval_runner.py
  modified: []

key-decisions:
  - "Lazy imports for torch, lm_eval, transformers inside functions -- avoids import errors when packages not installed, prints clear error message"
  - "Model path validation via regex pattern before passing to subprocess/library calls per T-03-07"
  - "sys.modules patching for mocking lazy imports in tests -- standard unittest.mock approach for import-time dependencies"

patterns-established:
  - "Lazy import pattern: try/except ImportError inside function body with logger.error and re-raise"
  - "Subprocess security: always list-form cmd, never shell=True, validate inputs before passing"
  - "Test mocking for lazy imports: patch.dict('sys.modules', {'module': mock}) pattern"

requirements-completed: [EVAL-03, EVAL-04]

# Metrics
duration: 4min
completed: 2026-04-20
---

# Phase 3 Plan 2: Eval Runner CLI and Test Suite Summary

**Unified eval runner CLI orchestrating lm-eval/evalplus/BFCL benchmark suites with MPS device detection and 30-test validation suite**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-20T14:29:58Z
- **Completed:** 2026-04-20T14:34:35Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Unified eval_runner.py CLI accepting --model, --benchmarks, --output, --config with three benchmark category functions
- Device detection implementing D-02 (MPS primary with PYTORCH_ENABLE_MPS_FALLBACK, CPU fallback, no CUDA)
- 30 unit tests passing across three test files covering schemas, comparison logic, and runner orchestration
- All benchmark calls mocked for fast test execution without model downloads

## Task Commits

Each task was committed atomically:

1. **Task 1: Create eval_runner.py unified evaluation CLI** - `ab99769` (feat)
2. **Task 2: Create test suite for eval framework** - `ab64bc1` (test)

## Files Created/Modified
- `scripts/eval_runner.py` - Main eval CLI with detect_device, run_knowledge/code/tool_calling_benchmarks, format_summary_table, main
- `tests/test_eval_config.py` - 10 unit tests for Pydantic schemas and config loading
- `tests/test_eval_compare.py` - 8 unit tests for comparison logic and table formatting
- `tests/test_eval_runner.py` - 14 tests (12 unit + 2 integration/slow) for runner orchestration with mocked benchmarks

## Decisions Made
- Lazy imports for torch/lm_eval/transformers inside function bodies to avoid import failures when packages not installed
- Model path validation via regex before passing to subprocess or library calls (T-03-07 mitigation)
- sys.modules patching approach for testing lazy imports (detect_device, run_knowledge_benchmarks)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed mock pattern for lazy torch import in tests**
- **Found during:** Task 2 (test suite creation)
- **Issue:** Initial @patch("scripts.eval_runner.torch") decorator failed because torch is imported lazily inside detect_device(), not at module level
- **Fix:** Changed to patch.dict("sys.modules", {"torch": mock_torch}) pattern which intercepts the import statement inside the function
- **Files modified:** tests/test_eval_runner.py
- **Verification:** All 30 tests pass
- **Committed in:** ab64bc1 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix in test mocking approach)
**Impact on plan:** Test mocking approach adjusted for lazy import pattern. No scope change.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required. Benchmark packages (lm-eval, evalplus, bfcl-eval) must be installed separately when running actual evaluations.

## Next Phase Readiness
- Eval runner ready to invoke standard benchmarks once packages are installed
- Compare CLI (from Plan 01) ready to diff results JSON files
- Full test suite validates orchestration logic without requiring model downloads
- Phase 03 evaluation framework complete -- ready for dataset generation phases (04, 05, 06)

## Self-Check: PASSED
