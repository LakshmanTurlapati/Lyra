---
phase: 9
slug: benchmarking-and-core-release
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-21
---

# Phase 9 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | tests/ directory |
| **Quick run command** | `python3 -m pytest tests/ -x -q` |
| **Full suite command** | `python3 -m pytest tests/ -v` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python3 -m pytest tests/ -x -q`
- **After each plan completes:** Run full suite
- **Regression check:** Existing 30 eval tests must still pass

---

## Validation Architecture

### Wave 0: Test Infrastructure
- Verify existing tests pass before any changes
- Install lm-eval and git-lfs dependencies

### Wave 1: Eval Execution
- Custom inference eval produces valid EvalResult JSON
- lm-eval-harness runs without errors on MPS
- JSON merge produces combined results

### Wave 2: Report & Release
- BENCHMARK.md renders valid Mermaid charts on GitHub
- Git LFS tracking works for .safetensors files
- LICENSE file has correct MIT text
