---
phase: 5
slug: code-generation-dataset
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-20
---

# Phase 5 -- Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x (existing) + pipeline scripts |
| **Config file** | configs/pipeline.yaml |
| **Quick run command** | `python3 -m pytest tests/ -x -q` |
| **Full suite command** | `python3 -m pytest tests/ && python3 -m scripts.validate_format datasets/code/ && python3 -m scripts.validate_tokenizer datasets/code/` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python3 -m pytest tests/ -x -q`
- **After every plan wave:** Run full suite command
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Status |
|---------|------|------|-------------|-----------|-------------------|--------|
| 05-01-01 | 01 | 1 | CODE-01 | integration | `python3 -m scripts.validate_format datasets/code/utility-batch-*.jsonl` | Pending |
| 05-01-02 | 01 | 1 | CODE-02 | integration | `python3 -m scripts.validate_format datasets/code/file-ops-batch-*.jsonl` | Pending |
| 05-01-03 | 01 | 1 | CODE-03 | integration | `python3 -m scripts.validate_format datasets/code/debug-batch-*.jsonl` | Pending |

---

## Wave 0 Requirements

- Existing infrastructure covers all phase requirements.
- `scripts/validate_format.py` -- format validation exists
- `scripts/validate_tokenizer.py` -- token limit validation exists
- `scripts/curate_pipeline.py` -- quality pipeline exists
- `scripts/style_validator.py` -- code style enforcement (require_code_blocks, max_prose_ratio)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Code correctness | CODE-01-03 | Semantic correctness of generated code cannot be automated | Spot-check 10 samples per category for logical correctness |
| Language idiomacy | CODE-01 | Whether code follows language idioms requires human judgment | Review Go/Rust samples for idiomatic patterns |

---

## Validation Sign-Off

- [ ] All tasks have automated verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
