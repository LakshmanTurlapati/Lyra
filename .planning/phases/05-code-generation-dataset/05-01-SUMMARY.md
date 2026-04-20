---
phase: 05-code-generation-dataset
plan: 01
subsystem: dataset-generation
tags: [python, code-generation, sharegpt, tdd, pydantic, yaml]

# Dependency graph
requires:
  - phase: 01-data-format-and-pipeline-foundation
    provides: Conversation Pydantic model, system-prompts.yaml, code.yaml templates
  - phase: 04-tool-calling-dataset
    provides: generate_tool_data.py pattern (CLI, batch generators, validate_batch, write_batch)
provides:
  - scripts/generate_code_data.py with 3 category generators (utility, file-ops, debugging)
  - tests/test_generate_code_data.py with 29 tests covering all categories
  - CLI entry point for batch JSONL generation to datasets/code/
affects: [05-code-generation-dataset, 07-dataset-assembly]

# Tech tracking
tech-stack:
  added: []
  patterns: [language-weighted code pools per category, Bug/Fix debugging format, terse code-first response style]

key-files:
  created:
    - scripts/generate_code_data.py
    - tests/test_generate_code_data.py
  modified: []

key-decisions:
  - "Language-specific code pools with idiomatic patterns per language (Python type hints, Go error returns, Rust Result types)"
  - "Flat query template pools organized by topic area for maximum diversity"
  - "Debugging entries stored as (query, response) tuples per language/bug-type for exact Bug/Fix format control"

patterns-established:
  - "Code pool pattern: per-language dictionaries mapping query-suffix -> fenced code block for idiomatic responses"
  - "Weighted language selection via rng.choices() with UTILITY_LANG_WEIGHTS and FILEOPS_LANG_WEIGHTS dicts"
  - "Three-message code conversations: system prompt, user query, assistant code response (no tool calls)"

requirements-completed: [CODE-01, CODE-02, CODE-03]

# Metrics
duration: 8min
completed: 2026-04-20
---

# Phase 5 Plan 1: Code Generation Script Summary

**TDD-built code generation script with 3 category generators (utility/file-ops/debugging), language-weighted pools across 5 languages, and Bug/Fix debugging format**

## Performance

- **Duration:** 8 min
- **Started:** 2026-04-20T19:00:42Z
- **Completed:** 2026-04-20T19:08:57Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Built generate_code_data.py with 3 category generators following Phase 4's proven CLI pattern
- 29 passing tests covering all categories, format validation, language coverage, style compliance, and CLI
- Language-specific idiomatic code pools: Python (60+ entries), JavaScript (12), TypeScript (10), Go (10), Rust (10)
- Full test suite green (202 passed, 2 skipped, 0 failures)

## Task Commits

Each task was committed atomically:

1. **Task 1: Write failing tests for code generation script** - `39c600a` (test)
2. **Task 2: Implement generate_code_data.py to pass all tests** - `ff029dc` (feat)

## Files Created/Modified
- `scripts/generate_code_data.py` - Code generation script with 3 category generators, validate_batch, write_batch, CLI main
- `tests/test_generate_code_data.py` - 29 tests across 8 test classes covering all 3 categories

## Decisions Made
- Language-specific code pools with idiomatic patterns per language rather than template-only generation -- ensures Python uses type hints, Go uses error returns, Rust uses Result types per RESEARCH.md pitfall 5
- Debugging entries stored as explicit (query, response) tuples rather than template-generated -- gives exact control over Bug/Fix format per D-06
- Flat topic-organized query pools (7 topic areas for utility, 7 for file-ops, 7 bug types for debugging) matching code.yaml template structure

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed Python syntax errors in code pool string literals**
- **Found during:** Task 2
- **Issue:** Single-quoted strings containing `\'` sequences (in JavaScript/TypeScript .env loader regex) caused Python SyntaxError
- **Fix:** Converted affected entries to double-quoted strings with simplified regex
- **Files modified:** scripts/generate_code_data.py
- **Committed in:** ff029dc (part of Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Minor string quoting fix, no scope change.

## Issues Encountered
None beyond the auto-fixed string quoting issue.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Code generation script ready for batch production runs (Plan 2: generate raw batches)
- CLI tested: `python3 -m scripts.generate_code_data --category utility --count 50 --batch 1 --seed 42`
- All 3 categories validated: utility (5 languages), file-ops (3 languages), debugging (3 languages with Bug/Fix format)

---
*Phase: 05-code-generation-dataset*
*Completed: 2026-04-20*
