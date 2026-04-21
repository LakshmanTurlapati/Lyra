---
phase: 09-benchmarking-and-core-release
plan: 02
subsystem: evaluation
tags: [eval, inference, merge, tool-calling, code-syntax, custom-eval]
dependency_graph:
  requires: [09-01]
  provides: [scripts/eval_inference.py, scripts/eval_merge.py]
  affects: [09-04]
tech_stack:
  added: []
  patterns: [module-level-monkeypatching, lazy-heavy-imports, pydantic-trust-boundary]
key_files:
  created:
    - scripts/eval_inference.py
    - scripts/eval_merge.py
  modified: []
decisions:
  - "Module-level None references (load_from_disk, run_inference_on_sample, _load_model_and_tokenizer) assigned lazily in run_custom_eval() body enable pytest monkeypatching without import-time side effects"
  - "check_code_syntax uses re.search with python|py alternation to match both ```python and ```py fenced code blocks"
  - "eval_merge.py uses model metadata from first file (knowledge results) and appends second file categories -- asymmetric merge is intentional per Pattern 3"
metrics:
  duration: 3min
  completed: 2026-04-21
  tasks_completed: 2
  files_created: 2
  files_modified: 0
---

# Phase 9 Plan 02: Custom Inference Eval and Result Merge Summary

**One-liner:** Custom inference eval (SmolLM2 XML tool-call format + ast.parse code syntax) and EvalResult JSON merge tool, with all 10 unit tests green.

## Completed Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Build eval_inference.py (TDD GREEN) | c699c3d | scripts/eval_inference.py |
| 2 | Build eval_merge.py (result merge tool) | 692fb2c | scripts/eval_merge.py |

## What Was Built

### eval_inference.py

Custom inference eval for tool-calling and code domains (D-02). Implements:

- `check_tool_call_format(output) -> bool`: regex-extracts `<tool_call>JSON</tool_call>` wrapper, validates JSON has "name" and "arguments" keys
- `check_code_syntax(output) -> bool`: regex-extracts ```python/```py blocks, validates via `ast.parse()` (no execution, T-09-05)
- `run_custom_eval(model_path, device, dataset_dir) -> CategoryResult`: iterates test split, skips knowledge samples, evaluates tool-calling and code outputs
- `_run_inference_on_sample`: builds prompt from messages excluding last assistant turn, applies chat template, generates greedily (do_sample=False for determinism)
- `_do_load_model_and_tokenizer`: lazy-imports torch/transformers, loads model to device
- CLI: --model (required), --dataset-dir (default: datasets/assembled), --output (required), --device (optional)

### eval_merge.py

Merges two EvalResult JSON files into one combined EvalResult. Implements:

- `merge_eval_results(first_path, second_path, output_path) -> EvalResult`: validates both inputs via `EvalResult.model_validate_json` at trust boundary (T-09-04), combines categories lists, writes merged JSON
- CLI: --first, --second, --output (all required)

## Verification Results

- `python3 -m pytest tests/test_eval_inference.py -x -q` -- 10/10 passed
- `python3 -m pytest tests/ -x -q --ignore=tests/test_eval_runner.py` -- 50 passed, 1 pre-existing Wave 0 stub failure (test_write_benchmark_md -- addressed in Plan 03)
- Manual sanity check: merge_eval_results correctly combines knowledge + custom categories into 2-category EvalResult

## Deviations from Plan

None - plan executed exactly as written.

The pre-existing `test_write_benchmark_md` failure (Wave 0 stub from Plan 01) is expected behavior: `write_benchmark_md` is not yet implemented in eval_compare.py. That test is a RED-state stub to be turned GREEN in Plan 03.

## Known Stubs

None. Both scripts are fully functional; no placeholder data flows to any output.

## Threat Flags

No new threat surface introduced beyond what was planned:
- T-09-04 mitigated: eval_merge.py uses model_validate_json at trust boundary
- T-09-05 mitigated: ast.parse() used in check_code_syntax (not exec/eval)
- T-09-06 accepted: regex on model output; no external trust boundary crossed

## Self-Check: PASSED

- [x] scripts/eval_inference.py exists
- [x] scripts/eval_merge.py exists
- [x] Task 1 commit c699c3d exists in git log
- [x] Task 2 commit 692fb2c exists in git log
- [x] All 10 unit tests pass
- [x] Import smoke test passes for both modules
