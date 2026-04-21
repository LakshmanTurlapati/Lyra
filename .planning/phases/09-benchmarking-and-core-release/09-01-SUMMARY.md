---
phase: 09-benchmarking-and-core-release
plan: 01
subsystem: testing
tags: [lm-eval, git-lfs, license, pytest, benchmarking]

# Dependency graph
requires:
  - phase: 08-fine-tuning-script
    provides: train.py and the complete eval framework from Phase 3

provides:
  - MIT LICENSE file with correct copyright holder
  - .gitattributes with git-lfs tracking for *.safetensors and *.bin
  - lm-eval 0.4.11 installed and importable in venv
  - Wave 0 test scaffolding: test_eval_inference.py, test_release_artifacts.py, extended test_eval_compare.py

affects: [09-02-eval-inference, 09-03-eval-compare, 09-05-release-artifacts]

# Tech tracking
tech-stack:
  added: [lm-eval[hf]==0.4.11, git-lfs 3.7.1]
  patterns: [RED test scaffolding before implementation (Wave 0 gap pattern), git-lfs track before model file staging]

key-files:
  created:
    - LICENSE
    - .gitattributes
    - tests/test_eval_inference.py
    - tests/test_release_artifacts.py
  modified:
    - requirements.txt
    - tests/test_eval_compare.py

key-decisions:
  - "git-lfs installed via brew (system tool) not pip -- .gitattributes committed before any model files staged per T-09-01"
  - "Wave 0 test scaffolding written in RED state intentionally -- Plans 02, 03, 05 will turn them GREEN"
  - "lm-eval pinned at 0.4.11 with [hf] extra for HuggingFace model backend support"

patterns-established:
  - "RED test scaffolding pattern: write tests before implementation so Plans 02/03 have clear done criteria"
  - "git-lfs tracking committed first, model files staged after -- enforces safe ordering"

requirements-completed: [REL-03, REL-04]

# Metrics
duration: 4min
completed: 2026-04-21
---

# Phase 9 Plan 01: Release Infrastructure and Wave 0 Test Scaffolding Summary

**MIT LICENSE, git-lfs .gitattributes for model files, lm-eval 0.4.11 installed, and 10+4+2 RED tests scaffolded across three Wave 0 gap test files**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-04-21T18:02:45Z
- **Completed:** 2026-04-21T18:06:15Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Created MIT LICENSE with "Lakshman Turlapati" as copyright holder (2026)
- Configured git-lfs via brew install and `git lfs track` for *.safetensors and *.bin patterns — .gitattributes committed before any model files staged (T-09-01 mitigation)
- Installed lm-eval[hf]==0.4.11 and pinned in requirements.txt; verified importable at correct version
- Created test_eval_inference.py with 10 unit tests for check_tool_call_format, check_code_syntax, and run_custom_eval (RED — awaiting Plan 02)
- Created test_release_artifacts.py with 4 artifact presence smoke tests; test_license_file and test_gitattributes_lfs already pass, test_model_card_frontmatter and test_dataset_card remain RED (awaiting Plan 05)
- Extended test_eval_compare.py with test_write_benchmark_md and test_mermaid_chart_present (RED — awaiting Plan 03); all 31 original eval tests still pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Create LICENSE, .gitattributes, install lm-eval and git-lfs** - `99dc987` (chore)
2. **Task 2: Write Wave 0 test scaffolding** - `1124ded` (test)

## Files Created/Modified

- `/Users/lakshman/Documents/Lyra/LICENSE` - MIT License with Lakshman Turlapati copyright (2026)
- `/Users/lakshman/Documents/Lyra/.gitattributes` - git-lfs tracking for *.safetensors and *.bin
- `/Users/lakshman/Documents/Lyra/requirements.txt` - Added lm-eval[hf]==0.4.11 under Phase 9 section
- `/Users/lakshman/Documents/Lyra/tests/test_eval_inference.py` - 10 unit tests for eval_inference helpers (RED)
- `/Users/lakshman/Documents/Lyra/tests/test_release_artifacts.py` - 4 release artifact smoke tests (2 RED, 2 GREEN)
- `/Users/lakshman/Documents/Lyra/tests/test_eval_compare.py` - Appended 2 new tests for write_benchmark_md and mermaid chart (RED)

## Decisions Made

- git-lfs installed via brew (system tool) rather than pip — not added to requirements.txt
- .gitattributes committed in this plan (before model files) per T-09-01 threat mitigation for tamper ordering
- Wave 0 gap tests written intentionally in RED state; Plans 02, 03, and 05 will implement the code to turn them GREEN
- lm-eval pinned with [hf] extra to ensure HuggingFace model backend is available for knowledge benchmarks

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. git-lfs was not previously installed and was installed fresh at 3.7.1 via Homebrew.
Pre-existing test_train.py failures (2 tests) were confirmed as pre-existing before this plan and are out of scope.

## Stub Scan

No stubs. This plan creates test scaffolding and infrastructure only — no data-rendering code.

## Threat Flags

No new threat surface introduced. .gitattributes, LICENSE, and requirements.txt contain no secrets and are consistent with the T-09-01/T-09-02/T-09-03 dispositions in the plan's threat register.

## Next Phase Readiness

- Plan 02 (eval_inference.py) can now implement check_tool_call_format, check_code_syntax, and run_custom_eval against the 10 RED tests in test_eval_inference.py
- Plan 03 (eval_compare.py extensions) can implement write_benchmark_md and format_mermaid_bar_chart against the 2 RED tests in test_eval_compare.py
- Plan 05 (release artifacts) can implement README.md frontmatter and datasets/README.md against test_model_card_frontmatter and test_dataset_card
- lm-eval 0.4.11 is ready for knowledge benchmarks in Plan 04

## Self-Check: PASSED

- FOUND: LICENSE
- FOUND: .gitattributes
- FOUND: tests/test_eval_inference.py
- FOUND: tests/test_release_artifacts.py
- FOUND: 09-01-SUMMARY.md
- FOUND commit: 99dc987 (chore: LICENSE, .gitattributes, lm-eval)
- FOUND commit: 1124ded (test: Wave 0 test scaffolding)

---
*Phase: 09-benchmarking-and-core-release*
*Completed: 2026-04-21*
