---
phase: 09-benchmarking-and-core-release
plan: "03"
subsystem: eval-pipeline
tags: [eval, markdown, mermaid, benchmarking, reporting]
dependency_graph:
  requires: [09-01]
  provides: [BENCHMARK.md generation via --markdown flag]
  affects: [scripts/eval_compare.py]
tech_stack:
  added: []
  patterns: [plain f-string formatting, argparse extension, fenced Mermaid xychart-beta]
key_files:
  created: []
  modified: [scripts/eval_compare.py]
decisions:
  - write_benchmark_md uses plain f-string table formatting consistent with format_compare_table (no external deps, project convention)
  - format_mermaid_bar_chart emits two bar series (baseline first, candidate second) per xychart-beta specification
  - Delta formatting uses "+{:.4f}" prefix for positive values, consistent with existing format_compare_table behavior
metrics:
  duration: "2min"
  completed: "2026-04-21"
  tasks: 1
  files: 1
---

# Phase 9 Plan 03: Markdown Report Generation Summary

Extended eval_compare.py with `write_benchmark_md` and `format_mermaid_bar_chart` functions plus a `--markdown` CLI flag that generates BENCHMARK.md with summary tables and Mermaid xychart-beta bar charts (D-04, D-05, D-06).

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Extend eval_compare.py with write_benchmark_md, format_mermaid_bar_chart, and --markdown flag | e8a3ae8 | scripts/eval_compare.py |

## What Was Built

`scripts/eval_compare.py` extended with:

1. `format_mermaid_bar_chart(results, base_name, candidate_name) -> str` — emits a fenced `xychart-beta` code block with x-axis benchmark labels and two bar series (baseline scores, candidate scores).

2. `write_benchmark_md(results, output_path, base_name, candidate_name) -> None` — writes BENCHMARK.md containing:
   - `# Benchmark Results` title with model names
   - Markdown summary table with Benchmark, Category, Metric, Base, Lyra, Delta columns
   - `## Score Comparison` section with the Mermaid bar chart

3. `--markdown OUTPUT_PATH` CLI argument in `main()` — when provided, calls `write_benchmark_md()` after the existing compare table is printed.

## Test Results

All 10 tests pass (8 pre-existing + 2 new):
- `test_write_benchmark_md` — verifies file contains `# Benchmark Results`, model names, benchmark names, and `+0.1000` delta format
- `test_mermaid_chart_present` — verifies file contains `` ```mermaid `` and `xychart-beta`

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None — `write_benchmark_md` writes fully-formed output from real `CompareResult` data. No placeholder values.

## Self-Check

- [x] `scripts/eval_compare.py` exists and contains `write_benchmark_md`, `format_mermaid_bar_chart`, and `--markdown` 
- [x] Commit e8a3ae8 exists in git log
- [x] 10/10 tests pass in `tests/test_eval_compare.py`
- [x] `python3 -m scripts.eval_compare --help` shows `--markdown` option
