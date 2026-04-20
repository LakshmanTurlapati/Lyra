---
phase: "04"
plan: "02"
subsystem: tool-calling-dataset
tags: [dataset-generation, single-call, cli, tool-calling, jsonl]
dependency_graph:
  requires: [04-01]
  provides: [single-call-batches, cli-batches]
  affects: [04-03, 04-04, phase-07]
tech_stack:
  added: []
  patterns: [template-based-generation, seeded-rng-per-batch, edge-case-distribution]
key_files:
  created:
    - datasets/tool-calling/single-call-batch-01.jsonl through single-call-batch-23.jsonl
    - datasets/tool-calling/cli-batch-01.jsonl through cli-batch-17.jsonl
  modified:
    - scripts/generate_tool_data.py
decisions:
  - Fixed edge case distribution to produce 22% no-tool samples (up from 12%) by separating no_tool_count from error_count explicitly
metrics:
  duration: 4min
  completed: 2026-04-20
  tasks: 2
  files: 41
---

# Phase 04 Plan 02: Single-Call and CLI Batch Generation Summary

Template-based generation of 1,980 tool-calling training samples across 40 JSONL batch files covering single-call function calling and CLI command patterns with 22% edge case coverage and zero destructive commands.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Generate 23 single-call batches (~1,155 samples) | b92775b | datasets/tool-calling/single-call-batch-{01..23}.jsonl, scripts/generate_tool_data.py |
| 2 | Generate 17 CLI batches (~825 samples) | 4abf709 | datasets/tool-calling/cli-batch-{01..17}.jsonl |

## Verification Results

| Check | Result |
|-------|--------|
| Single-call batch count | 23 files |
| CLI batch count | 17 files |
| Total batch files | 40 |
| Single-call total samples | 1,155 |
| CLI total samples | 825 |
| Combined total | 1,980 |
| Format validation (all files) | 100% pass |
| Single-call edge cases | 254/1,155 (22.0%) |
| CLI edge cases | 181/825 (21.9%) |
| CLI bash coverage | Confirmed |
| CLI git coverage | Confirmed |
| CLI file ops coverage | Confirmed |
| Destructive command check (T-04-03) | No matches |
| Existing test suite | 173 passed, 2 skipped |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed edge case distribution producing insufficient no-tool samples**
- **Found during:** Task 1 verification
- **Issue:** Original generate_single_call_batch and generate_cli_batch functions only produced ~12% no-tool-needed samples due to modulo-based edge case splitting. Acceptance criteria requires >= 20%.
- **Fix:** Replaced modulo-based distribution with explicit no_tool_count (22%) and error_count (4%) allocation. No-tool samples are now contiguous at start of batch for clean separation.
- **Files modified:** scripts/generate_tool_data.py
- **Commit:** b92775b (included with Task 1)

## Decisions Made

- Edge case distribution uses explicit count allocation (22% no-tool + 4% error) rather than modulo-based splitting for deterministic and predictable edge case coverage.

## Key Metrics

- Single-call: 1,155 samples across 23 batches (22 x 50 + 1 x 55)
- CLI: 825 samples across 17 batches (16 x 50 + 1 x 25)
- All samples use seeded Random for reproducibility (seed = batch_num * 42 for single-call, batch_num * 73 for CLI)
- CLI samples cover: bash commands (ls, find, grep, curl, ps, du), git operations (status, log, diff, branch), file operations (cat, head, tail, wc, tar)

## Self-Check: PASSED

- [x] datasets/tool-calling/single-call-batch-01.jsonl exists
- [x] datasets/tool-calling/single-call-batch-23.jsonl exists
- [x] datasets/tool-calling/cli-batch-01.jsonl exists
- [x] datasets/tool-calling/cli-batch-17.jsonl exists
- [x] Commit b92775b exists (Task 1)
- [x] Commit 4abf709 exists (Task 2)
