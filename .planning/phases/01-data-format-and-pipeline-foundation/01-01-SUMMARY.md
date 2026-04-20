---
phase: 01-data-format-and-pipeline-foundation
plan: 01
subsystem: data-format
tags: [pydantic, sharegpt, jsonl, validation, trl-native]

# Dependency graph
requires: []
provides:
  - Canonical TRL-native ShareGPT format specification with 5 tool call patterns
  - Pydantic validation schema (Conversation, Message, ToolCall, ToolSchema models)
  - validate_file() for JSONL line-by-line validation with error reporting
  - CLI entry point for standalone format validation
  - Shared pytest fixtures and test infrastructure
affects: [01-02, 01-03, phase-02, phase-04, phase-05, phase-06, phase-07]

# Tech tracking
tech-stack:
  added: [pydantic 2.12.5, pytest]
  patterns: [pydantic-model-validator, trl-native-format, jsonl-line-validation]

key-files:
  created:
    - specs/sharegpt-format.md
    - scripts/validate_format.py
    - tests/conftest.py
    - tests/test_format_validator.py
    - requirements.txt
    - pytest.ini
    - scripts/__init__.py
    - tests/__init__.py
    - .gitignore
  modified: []

key-decisions:
  - "TRL-native messages/role/content format over classic ShareGPT from/value format"
  - "Pydantic model_validator(mode='after') for structural rule enforcement"
  - "Strict first-message-must-be-system rule to prevent SmolLM2 default prompt injection"

patterns-established:
  - "Pydantic BaseModel hierarchy for conversation validation: FunctionDef -> ToolSchema, ToolCallFunction -> ToolCall -> Message -> Conversation"
  - "JSONL validation pattern: line-by-line parsing with per-line error collection"
  - "Shared fixtures in conftest.py return dicts (not Pydantic objects) to test the full dict-to-model validation path"

requirements-completed: [DATA-01]

# Metrics
duration: 4min
completed: 2026-04-20
---

# Phase 01 Plan 01: ShareGPT Format Specification and Validation Summary

**Pydantic-based TRL-native format validation with 9 role ordering rules, 5 tool call patterns, and 11 passing tests**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-20T09:23:30Z
- **Completed:** 2026-04-20T09:28:15Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments
- Canonical format specification document covering all 5 tool call patterns (single, multi-turn, parallel, MCP-style, CLI/shell) and 9 role ordering rules
- Pydantic validation schema with Conversation, Message, ToolCall, ToolSchema models that enforce structural correctness with specific error messages
- validate_file() function for JSONL batch validation with per-line error reporting and CLI entry point
- Test infrastructure with 6 shared fixtures and 11 passing unit tests covering valid formats and all major error cases

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for format validation** - `617272d` (test)
2. **Task 1 GREEN: Format spec and Pydantic validation** - `f869148` (feat)
3. **Task 2: Test infrastructure and canonical test suite** - `8ffe148` (test)

## Files Created/Modified
- `specs/sharegpt-format.md` - Canonical format specification with tool call patterns, role ordering rules, token budget, and storage conventions
- `scripts/validate_format.py` - Pydantic-based JSONL validation with CLI entry point (exports: Conversation, Message, ToolCall, ToolSchema, validate_file)
- `tests/conftest.py` - 6 shared pytest fixtures (valid_conversation, valid_tool_call_conversation, valid_parallel_tool_call_conversation, invalid_no_system, invalid_orphan_tool, invalid_undefined_tool)
- `tests/test_format_validator.py` - 11 unit tests covering DATA-01 requirement
- `requirements.txt` - Python dependencies (pydantic, transformers, pyyaml, pytest)
- `pytest.ini` - Pytest configuration pointing to tests/ directory
- `scripts/__init__.py` - Package init for script imports
- `tests/__init__.py` - Package init for test imports
- `.gitignore` - Python cache and build artifact exclusions

## Decisions Made
- Used TRL-native messages/role/content format instead of classic ShareGPT from/value -- TRL v1.2.0 SFTTrainer expects this natively, eliminating conversion at training time
- Chose Pydantic model_validator(mode='after') for structural validation -- allows all fields to be parsed first, then cross-field rules enforced in a single pass
- Enforced strict system-first rule to prevent SmolLM2's default system prompt injection that would waste tokens and create inconsistent training data

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added .gitignore for Python cache artifacts**
- **Found during:** Task 2 (test infrastructure commit)
- **Issue:** __pycache__/ and .pytest_cache/ directories were being tracked by git
- **Fix:** Created .gitignore with Python-standard exclusions
- **Files modified:** .gitignore
- **Verification:** git status no longer shows cache directories
- **Committed in:** 8ffe148 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** .gitignore is standard project hygiene. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Format specification and validation are complete and tested
- Plan 01-02 (tokenizer validation) can build on these Pydantic models
- Plan 01-03 (prompt templates) has the format spec as reference
- Phases 4-6 (data generation) have the data contract to target

## Self-Check: PASSED

All 9 created files verified present on disk. All 3 task commits verified in git history.

---
*Phase: 01-data-format-and-pipeline-foundation*
*Completed: 2026-04-20*
