---
phase: 2
slug: data-quality-and-curation-pipeline
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-20
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | tests/conftest.py (Wave 0 installs if needed) |
| **Quick run command** | `python -m pytest tests/test_curation.py -x -q` |
| **Full suite command** | `python -m pytest tests/ -q` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_curation.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -q`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | DATA-03 | — | N/A | unit | `python -m pytest tests/test_quality_scoring.py -q` | ❌ W0 | ⬜ pending |
| 02-01-02 | 01 | 1 | DATA-03 | — | N/A | unit | `python -m pytest tests/test_dedup.py -q` | ❌ W0 | ⬜ pending |
| 02-02-01 | 02 | 1 | DATA-04 | — | N/A | unit | `python -m pytest tests/test_pipeline_config.py -q` | ❌ W0 | ⬜ pending |
| 02-03-01 | 03 | 2 | DATA-05 | — | N/A | unit | `python -m pytest tests/test_style_validation.py -q` | ❌ W0 | ⬜ pending |
| 02-04-01 | 04 | 2 | DATA-03 | — | N/A | integration | `python -m pytest tests/test_pipeline_e2e.py -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending / ✅ green / ❌ red / ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_quality_scoring.py` — stubs for DATA-03 quality heuristics
- [ ] `tests/test_dedup.py` — stubs for DATA-03 deduplication
- [ ] `tests/test_pipeline_config.py` — stubs for DATA-04 config loading
- [ ] `tests/test_style_validation.py` — stubs for DATA-05 adaptive styles
- [ ] `tests/test_pipeline_e2e.py` — integration stubs for full pipeline run
- [ ] `tests/conftest.py` — shared fixtures (sample JSONL data, config fixtures)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| LLM-as-judge Tier 2 scoring | DATA-03 | Requires Claude Code session | Run judge prompts on 10 samples, verify scores align with Tier 1 |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
