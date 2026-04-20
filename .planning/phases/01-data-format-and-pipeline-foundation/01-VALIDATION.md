---
phase: 1
slug: data-format-and-pipeline-foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-20
---

# Phase 1 -- Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | none -- Wave 0 installs |
| **Quick run command** | `python -m pytest tests/ -x -q` |
| **Full suite command** | `python -m pytest tests/ -v` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/ -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -v`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD | TBD | TBD | DATA-01 | -- | N/A | unit | `python -m pytest tests/test_format.py` | -- W0 | pending |
| TBD | TBD | TBD | DATA-02 | -- | N/A | unit | `python -m pytest tests/test_validation.py` | -- W0 | pending |
| TBD | TBD | TBD | DATA-06 | -- | N/A | unit | `python -m pytest tests/test_templates.py` | -- W0 | pending |

*Status: pending*

---

## Wave 0 Requirements

- [ ] `tests/test_format.py` -- stubs for DATA-01 (ShareGPT format validation)
- [ ] `tests/test_validation.py` -- stubs for DATA-02 (tokenizer alignment)
- [ ] `tests/test_templates.py` -- stubs for DATA-06 (prompt template library)
- [ ] `tests/conftest.py` -- shared fixtures
- [ ] pytest installation -- if not installed

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Prompt templates are readable and organized | DATA-06 | Subjective quality check | Browse templates/ directory, verify categories and documentation |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
