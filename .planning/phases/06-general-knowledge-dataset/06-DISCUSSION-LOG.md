# Phase 6: General Knowledge Dataset - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.

**Date:** 2026-04-20
**Phase:** 06-general-knowledge-dataset
**Areas discussed:** Category distribution, Response depth and style, Topic diversity strategy, Dedup mitigation

---

## Category Distribution

| Option | Description | Selected |
|--------|-------------|----------|
| Q&A heavy | ~40% Q&A, ~35% explanations, ~25% reasoning. Most common use case. | Yes |
| Even thirds | ~33% each. | |
| Reasoning-heavy | ~40% reasoning. Hardest capability. | |

**User's choice:** Q&A heavy

---

## Response Depth and Style

| Option | Description | Selected |
|--------|-------------|----------|
| Category-adaptive | Reasoning: numbered steps. Q&A: concise. Explanations: structured. | Yes |
| Always detailed | Full reasoning regardless of question type. | |
| Always concise | Under 300 tokens always. Contradicts Phase 2. | |

**User's choice:** Category-adaptive

---

## Topic Diversity Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Tech and STEM weighted | ~40% tech, ~25% math, ~20% science, ~15% other. | Yes |
| Broad even coverage | Equal across all 6 domains. | |
| You decide | Claude's discretion. | |

**User's choice:** Tech and STEM weighted

---

## Dedup Mitigation

| Option | Description | Selected |
|--------|-------------|----------|
| Large topic pool + varied phrasing | 200+ unique topics, varied how/what/why phrasing. | Yes |
| Lower dedup threshold | Reduce from 0.7 to 0.5. | |
| Both combined | Large pool AND lower threshold. | |

**User's choice:** Large topic pool + varied phrasing

---

## Deferred Ideas

None -- discussion stayed within phase scope
