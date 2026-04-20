---
phase: 4
slug: tool-calling-dataset
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-20
---

# Phase 4 -- Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x (existing) + pipeline scripts |
| **Config file** | configs/pipeline.yaml |
| **Quick run command** | `python -m pytest tests/ -x -q` |
| **Full suite command** | `python -m pytest tests/ && python -m scripts.validate_format datasets/tool-calling/ && python -m scripts.validate_tokenizer datasets/tool-calling/` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/ -x -q`
- **After every plan wave:** Run full suite command
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | TOOL-01 | -- | N/A | integration | `python -m scripts.validate_format datasets/tool-calling/single-call-batch-*.jsonl` | Pending | Pending |
| 04-01-02 | 01 | 1 | TOOL-02 | -- | N/A | integration | `python -m scripts.validate_format datasets/tool-calling/multi-turn-batch-*.jsonl` | Pending | Pending |
| 04-01-03 | 01 | 1 | TOOL-03 | -- | N/A | integration | `python -m scripts.validate_format datasets/tool-calling/parallel-batch-*.jsonl` | Pending | Pending |
| 04-01-04 | 01 | 1 | TOOL-04 | -- | N/A | integration | `python -m scripts.validate_format datasets/tool-calling/mcp-batch-*.jsonl` | Pending | Pending |
| 04-01-05 | 01 | 1 | TOOL-05 | -- | N/A | integration | `python -m scripts.validate_format datasets/tool-calling/cli-batch-*.jsonl` | Pending | Pending |

*Status: Pending -- will be updated when plans are created*

---

## Wave 0 Requirements

- Existing infrastructure covers all phase requirements.
- `scripts/validate_format.py` -- format validation exists
- `scripts/validate_tokenizer.py` -- token limit validation exists
- `scripts/curate_pipeline.py` -- quality pipeline exists

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Content quality assessment | TOOL-01-05 | Semantic correctness of generated samples cannot be automated | Spot-check 10 random samples per category for logical coherence |
| Tool schema realism | TOOL-01-05 | Whether tool schemas represent realistic APIs requires human judgment | Review tool pool YAML for realism and variety |

---

## Validation Sign-Off

- [ ] All tasks have automated verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
