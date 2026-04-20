# Phase 2: Data Quality and Curation Pipeline - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-04-20
**Phase:** 02-data-quality-and-curation-pipeline
**Areas discussed:** Quality scoring, Deduplication, Pipeline config, Adaptive styles

---

## Quality Scoring

| Option | Description | Selected |
|--------|-------------|----------|
| Heuristic rules (Recommended) | Python-computable signals: token count, role balance, no repetition, format compliance. Fast, deterministic, no API cost. | |
| LLM-as-judge | Send samples back to an LLM to rate quality 1-5. Higher accuracy but adds API cost. | Yes |
| Composite (heuristic + spot-check) | Heuristic scoring for bulk, manual spot-checks for calibration. | |

**User's choice:** LLM-as-judge
**Notes:** None

### Follow-up: Implementation method

| Option | Description | Selected |
|--------|-------------|----------|
| Introduce API SDK for scoring | Add Anthropic Python SDK specifically for quality scoring pipeline. | |
| Claude Code manual judging | Quality scoring happens in Claude Code sessions. No SDK. | Yes |
| Revise D-05 entirely | Introduce SDK for both generation and scoring. | |

**User's choice:** Claude Code manual judging
**Notes:** Maintains no-SDK constraint from Phase 1.

### Follow-up: Workflow integration

| Option | Description | Selected |
|--------|-------------|----------|
| Two-tier scoring | Python heuristics auto-score (pass/fail) + Claude Code adds quality score to calibration subset. | Yes |
| Manual-first, then threshold | Score batch manually, train heuristic classifier from ratings. | |
| All manual in Claude Code | Every sample scored by Claude in session. | |

**User's choice:** Two-tier scoring
**Notes:** None

### Follow-up: Heuristic signals (multi-select)

| Option | Description | Selected |
|--------|-------------|----------|
| Format compliance | Passes Pydantic validation, role ordering, within 2048 tokens. | Yes |
| Response completeness | No truncation, closed code blocks, conclusions reached. | Yes |
| Content diversity | No near-duplicate content within a batch. | Yes |
| Conversation naturalness | Balanced turn lengths, no artifacts, no meta-commentary. | Yes |

**User's choice:** All four signals selected
**Notes:** None

---

## Deduplication

| Option | Description | Selected |
|--------|-------------|----------|
| N-gram overlap (Recommended) | 3-4 gram Jaccard similarity. Fast, no dependencies, threshold-configurable. | Yes |
| Exact + MinHash | Exact match plus MinHash LSH for approximate duplicates at scale. | |
| Embedding similarity | Sentence-transformers embeddings with cosine threshold. | |

**User's choice:** N-gram overlap (Recommended)
**Notes:** None

### Follow-up: Dedup scope

| Option | Description | Selected |
|--------|-------------|----------|
| User prompts only | Near-identical queries are duplicates regardless of response. | |
| Full conversation | Compare entire conversation. Stricter. | |
| Both separately | Check prompts and responses independently. | |

**User's choice:** "You decide what's best" (Claude's discretion)
**Notes:** User deferred this to Claude's judgment.

---

## Pipeline Config

| Option | Description | Selected |
|--------|-------------|----------|
| Single YAML config (Recommended) | One pipeline.yaml with all settings. Consistent with Phase 1 YAML choice. | Yes |
| YAML + CLI overrides | Base YAML with CLI flag overrides for one-off changes. | |
| Per-domain configs | Separate config per domain. | |

**User's choice:** Single YAML config (Recommended)
**Notes:** None

### Follow-up: Per-domain overrides

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, per-domain sections | Global defaults with optional domain-specific overrides in same file. | Yes |
| No, uniform settings | Same thresholds for all domains. | |

**User's choice:** Yes, per-domain sections
**Notes:** None

---

## Adaptive Styles

| Option | Description | Selected |
|--------|-------------|----------|
| Template-driven (Recommended) | Style instructions in prompt templates. Pipeline validates style compliance via heuristics. | Yes |
| Post-generation scoring | Generate freely, score style appropriateness after. | |
| Both template + validation | Templates guide + pipeline validates. Belt-and-suspenders. | |

**User's choice:** Template-driven (Recommended)
**Notes:** None

### Follow-up: Style validation rules

| Option | Description | Selected |
|--------|-------------|----------|
| Token count ranges per domain | Code: 50-500, Knowledge: 200-1500, Tool calls: 20-200. | |
| Structural markers | Code must have code block, knowledge must have reasoning markers. | |
| You decide | Claude's discretion on specific rules. | Yes |

**User's choice:** You decide (Claude's discretion)
**Notes:** Must ensure domains produce distinguishably different output styles.

---

## Claude's Discretion

- Deduplication comparison scope (what to compare)
- Specific style validation heuristics
- Default threshold values
- N-gram size and similarity threshold defaults
- Quality score output format

## Deferred Ideas

None -- discussion stayed within phase scope
