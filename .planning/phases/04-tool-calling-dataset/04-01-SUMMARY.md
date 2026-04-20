---
phase: 04-tool-calling-dataset
plan: 01
subsystem: dataset-generation
tags: [yaml, pydantic, jsonl, tool-calling, sharegpt, batch-generation]

# Dependency graph
requires:
  - phase: 01-data-format-and-pipeline-foundation
    provides: "Pydantic Conversation model, validate_format.py, system-prompts.yaml, tool-calling.yaml templates"
provides:
  - "64-schema tool pool YAML (datasets/tool-calling/tool_schemas.yaml)"
  - "Batch generation script with 5 category generators (scripts/generate_tool_data.py)"
  - "Tests validating generated data passes Pydantic and format checks"
affects: [04-02, 04-03, 04-04, 05-code-dataset, 06-knowledge-dataset]

# Tech tracking
tech-stack:
  added: []
  patterns: ["schema pool YAML -> category batch generators -> Pydantic validation -> JSONL output", "seeded randomness per batch for reproducibility", "template-based query generation with placeholder filling"]

key-files:
  created:
    - "datasets/tool-calling/tool_schemas.yaml"
    - "scripts/generate_tool_data.py"
    - "tests/test_tool_schemas.py"
    - "tests/test_generate_tool_data.py"
  modified: []

key-decisions:
  - "64 schemas total: 41 developer, 18 everyday, 3 MCP meta, 2 CLI (64% developer weight per D-04)"
  - "Template-based query generation with placeholder filling for diversity instead of hardcoded queries"
  - "Seeded random.Random instances per batch function for reproducibility without global state"
  - "Edge cases allocated as first N samples in each batch for deterministic coverage"

patterns-established:
  - "Schema pool pattern: centralized YAML -> load_schemas() -> get_tools_for_category() -> per-sample tool selection"
  - "Batch generator pattern: category function(count, schemas, system_prompts, seed) -> list[dict] validated against Conversation model"
  - "CLI convention: python -m scripts.generate_tool_data --category X --count N --batch B producing {category}-batch-{batch:02d}.jsonl"

requirements-completed: [TOOL-01, TOOL-02, TOOL-03, TOOL-04, TOOL-05]

# Metrics
duration: 10min
completed: 2026-04-20
---

# Phase 4 Plan 1: Tool Schema Pool and Batch Generation Summary

**64-schema tool pool with 5-category batch generation script producing validated JSONL across single-call, CLI, multi-turn, parallel, and MCP patterns**

## Performance

- **Duration:** 10 min
- **Started:** 2026-04-20T16:05:24Z
- **Completed:** 2026-04-20T16:15:01Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Created 64 unique tool schemas across developer (41), everyday (18), MCP meta (3), and CLI (2) domains with max 3 parameters each
- Built batch generation script with per-category generators that produce valid JSONL with 25% edge cases
- All 173 existing tests pass plus 45 new tests (16 schema + 29 generation)
- CLI interface matching D-08 naming convention produces validated output files

## Task Commits

Each task was committed atomically:

1. **Task 1: Tool schema pool YAML** - `b06369f` (test: failing schema tests), `38f2a4d` (feat: 64 unique schemas)
2. **Task 2: Batch generation script** - `9148767` (test: failing generation tests), `8690e8d` (feat: generation script with 5 categories)

## Files Created/Modified
- `datasets/tool-calling/tool_schemas.yaml` - 64 unique tool schemas organized by domain (developer, everyday, mcp_meta, cli)
- `scripts/generate_tool_data.py` - Batch generation script with generate_single_call_batch, generate_cli_batch, generate_multi_turn_batch, generate_parallel_batch, generate_mcp_batch, validate_batch, write_batch, main
- `tests/test_tool_schemas.py` - 16 tests validating schema structure, counts, uniqueness, parameter limits
- `tests/test_generate_tool_data.py` - 29 tests validating all 5 categories, edge cases, CLI entry point, schema pool usage, query diversity

## Decisions Made
- Chose 64 schemas (within 50-100 range) with 64% developer weighting to match D-04 recommendation
- Used template-based query generation with value pool randomization rather than fully hardcoded queries -- provides diversity while maintaining control over content
- Used seeded random.Random instances (not global state) per batch function for deterministic reproducibility across runs
- Allocated edge cases as first N samples in each batch (count // 4) for predictable coverage in testing
- Generated safe-only CLI commands per T-04-03 threat mitigation -- no rm, DROP, chmod patterns

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Schema pool ready for all subsequent plans (04-02 through 04-04) to draw from
- Generation script ready to produce the ~3,300 raw samples across 5 categories
- Batch naming convention established: {category}-batch-{batch:02d}.jsonl
- All 5 category generators validated end-to-end through Pydantic Conversation model

---
*Phase: 04-tool-calling-dataset*
*Completed: 2026-04-20*
