---
phase: 3
slug: evaluation-framework
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-20
---

# Phase 3 -- Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | tests/conftest.py |
| **Quick run command** | `python3 -m pytest tests/test_eval_runner.py -x -q` |
| **Full suite command** | `python3 -m pytest tests/ -q` |
| **Estimated runtime** | ~10 seconds (excluding model inference) |

---

## Sampling Rate

- **After every task commit:** Run `python3 -m pytest tests/test_eval_runner.py -x -q`
- **After every plan wave:** Run `python3 -m pytest tests/ -q`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | EVAL-03 | -- | N/A | unit | `python3 -m pytest tests/test_eval_runner.py -q` | W0 | pending |
| 03-01-02 | 01 | 1 | EVAL-04 | -- | N/A | unit | `python3 -m pytest tests/test_eval_reporting.py -q` | W0 | pending |
| 03-02-01 | 02 | 2 | EVAL-03 | -- | N/A | integration | `python3 -m pytest tests/test_eval_benchmarks.py -q` | W0 | pending |
| 03-02-02 | 02 | 2 | EVAL-04 | -- | N/A | unit | `python3 -m pytest tests/test_eval_compare.py -q` | W0 | pending |

---

## Wave 0 Requirements

- [ ] `tests/test_eval_runner.py` -- stubs for EVAL-03 eval runner CLI
- [ ] `tests/test_eval_reporting.py` -- stubs for EVAL-04 per-category reporting
- [ ] `tests/test_eval_benchmarks.py` -- stubs for benchmark integration
- [ ] `tests/test_eval_compare.py` -- stubs for comparison tool
- [ ] `tests/conftest.py` -- shared fixtures (mock model outputs, sample results)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Full benchmark run on SmolLM2-1.7B | EVAL-03 | Requires model download + MPS GPU time | Run eval_runner on HuggingFaceTB/SmolLM2-1.7B, verify results JSON |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
