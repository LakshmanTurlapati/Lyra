# Phase 7: Dataset Assembly - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.

**Date:** 2026-04-20
**Phase:** 07-dataset-assembly
**Areas discussed:** Domain balance strategy, Split ratios, Assembly output format

---

## Domain Balance Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Downsample to smallest | Cap each at ~560. True 33/33/33. | |
| Use all, accept imbalance | Keep all 3,630. Distribution: 68/16.5/15.4. | Yes |
| Proportional cap at 2x smallest | Cap at ~1,120 each. | |

**User's choice:** Use all, accept imbalance
**Notes:** More data preferred over strict balance. Tool-calling dominance acceptable.

---

## Split Ratios

| Option | Description | Selected |
|--------|-------------|----------|
| 90/5/5 | Standard for small datasets. Maximizes training. | Yes |
| 80/10/10 | More eval data, less training. | |
| 85/10/5 | Larger validation for tuning. | |

**User's choice:** 90/5/5

---

## Assembly Output Format

| Option | Description | Selected |
|--------|-------------|----------|
| HuggingFace datasets format | Arrow-backed, push_to_hub ready, domain metadata. | Yes |
| Separate JSONL per split | Simple but no HF integration. | |
| Both | JSONL + HF format. | |

**User's choice:** HuggingFace datasets format

---

## Deferred Ideas

None
