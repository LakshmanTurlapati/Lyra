---
phase: "07"
plan: "01"
subsystem: dataset-assembly
tags: [dataset, assembly, splits, huggingface, tdd]
dependency_graph:
  requires: [curated JSONL from phases 04/05/06]
  provides: [assembled DatasetDict with train/validation/test splits]
  affects: [phase 08 fine-tuning consumes assembled dataset]
tech_stack:
  added: [datasets>=4.8.0]
  patterns: [manual stratified split, Dataset.from_list with on_mixed_types]
key_files:
  created:
    - scripts/assemble_dataset.py
    - tests/test_assemble_dataset.py
  modified:
    - requirements.txt
decisions:
  - Manual stratified split over datasets library stratify_by_column (ClassLabel type requirement incompatible with string domain values)
  - Fixture size 120 samples (60/30/30) to ensure minimum class counts per split for stratification testing
metrics:
  duration: 6min
  completed: 2026-04-20
  tasks: 1
  files: 3
---

# Phase 07 Plan 01: Dataset Assembly Script Summary

Manual stratified 90/5/5 split assembly script merging 3 curated JSONL domains into HuggingFace DatasetDict with domain metadata and _quality stripping.

## What Was Done

### Task 1: Test suite and assembly script implementation (TDD)

**RED phase:** Created comprehensive test suite with 13 tests covering:
- load_domain_jsonl (loads, strips _quality, adds domain, handles tools)
- Assembly output (all domains in each split, stratified proportions, total count, split ratios, domain column, no _quality, tools column, reproducibility)
- compute_stats (correct counts and percentages)
- CLI subcommands (assemble, stats)

**GREEN phase:** Implemented `scripts/assemble_dataset.py` with:
- `load_domain_jsonl(path, domain)`: Line-by-line JSON loading with error handling, _quality removal, domain tagging, tools defaulting
- `assemble(output_dir, seed, base_dir)`: Loads all 3 domains, manual stratified split preserving domain proportions, saves as DatasetDict
- `compute_stats(dataset_dict)`: Per-split domain distribution with counts and percentages
- `print_stats(dataset_dict)`: Formatted table output
- `validate_assembled(dataset_dict)`: Pydantic Conversation model validation on all samples
- CLI with argparse: `assemble` and `stats` subcommands

**Key implementation detail:** Used manual stratified split (group by domain, shuffle per group with seed, split each group 90/5/5) instead of `train_test_split(stratify_by_column=...)` because the library requires ClassLabel column type which loses string labels on cast-back.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] ClassLabel stratification incompatibility**
- **Found during:** GREEN phase implementation
- **Issue:** `datasets` library `train_test_split(stratify_by_column=...)` requires column to be ClassLabel type. Casting domain to ClassLabel and back to string produces integer indices ('0', '1', '2') instead of label strings.
- **Fix:** Implemented manual stratified split using Python's random.Random with seed for reproducibility. Groups samples by domain, shuffles each group, then allocates 90/5/5 per group.
- **Files modified:** scripts/assemble_dataset.py
- **Commit:** 37d0313

**2. [Rule 3 - Blocking] Fixture size too small for stratified split**
- **Found during:** GREEN phase testing
- **Issue:** With 40 total samples (20/10/10), the second-pass split had only 4 samples total with some classes having only 1 member, causing stratification to fail.
- **Fix:** Increased fixture to 120 samples (60/30/30) providing enough samples per class for meaningful stratified splits.
- **Files modified:** tests/test_assemble_dataset.py
- **Commit:** 37d0313

## Commits

| Commit | Type | Description |
|--------|------|-------------|
| f89472c | test | Add failing test suite for dataset assembly script (RED) |
| 37d0313 | feat | Implement dataset assembly script with stratified splits (GREEN) |

## Verification Results

- All 13 tests pass: `python -m pytest tests/test_assemble_dataset.py -x -q` exits 0
- All acceptance criteria grep checks pass (16/16)
- Module imports verified: `from scripts.assemble_dataset import assemble, compute_stats, print_stats, load_domain_jsonl`
- `datasets>=4.8.0` present in requirements.txt

## Self-Check: PASSED

- [x] scripts/assemble_dataset.py exists
- [x] tests/test_assemble_dataset.py exists
- [x] requirements.txt contains datasets>=4.8.0
- [x] Commit f89472c exists
- [x] Commit 37d0313 exists
- [x] All 13 tests pass
