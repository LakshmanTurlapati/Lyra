# Phase 4: Tool Calling Dataset - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-04-20
**Phase:** 04-tool-calling-dataset
**Areas discussed:** Sample count & distribution, Tool schema diversity, Edge cases & boundaries, Generation workflow

---

## Sample Count & Distribution

| Option | Description | Selected |
|--------|-------------|----------|
| Weighted by complexity | Heavier on single-call (~35%) and CLI (~25%). Less on parallel (~15%) and MCP (~10%), multi-turn (~15%). | Yes |
| Even split (~20% each) | Equal representation across all 5 categories. | |
| Proportional to real usage | ~40% single, ~25% CLI, ~20% multi-turn, ~10% parallel, ~5% MCP. | |

**User's choice:** Weighted by complexity
**Notes:** None

### Follow-up: Volume strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Generate 2x, curate to ~1,667 | Generate ~3,300 raw samples, quality pipeline filters to ~1,667. | Yes |
| Target exactly ~1,667 | Generate close to target, only reject clear failures. | |
| Start with 500, iterate | Generate 500 first, evaluate, then scale. | |

**User's choice:** Generate 2x, curate to ~1,667
**Notes:** None

---

## Tool Schema Diversity

| Option | Description | Selected |
|--------|-------------|----------|
| Large pool (~50-100 unique tools) | Create diverse library of realistic tools. Teaches generalization. | Yes |
| Small curated set (~15-20 tools) | Focused set reused across many samples. | |
| Per-sample unique tools | Novel tool schemas for nearly every sample. | |

**User's choice:** Large pool (~50-100 unique tools)
**Notes:** None

### Follow-up: Tool types

| Option | Description | Selected |
|--------|-------------|----------|
| Developer-focused | APIs developers actually call: databases, file systems, HTTP, git, etc. | |
| Broad real-world mix | Developer tools + everyday tools (weather, calendar, email, search, maps). | |
| You decide | Claude's discretion on tool categories. | |

**User's choice:** Mix of developer-focused and broad real-world (custom answer)
**Notes:** User specified "mix of 1 and 2" -- developer tools weighted heavier but includes everyday tools.

---

## Edge Cases & Boundaries

| Option | Description | Selected |
|--------|-------------|----------|
| No-tool-needed responses | Assistant answers directly without calling a tool. | Yes |
| Tool error handling | Tool returns error/failure, assistant recovers gracefully. | Yes |
| Ambiguous requests | Could use a tool or answer directly, assistant makes judgment call. | Yes |
| Parameter edge cases | Optional params, empty strings, nested objects, arrays. | Yes |

**User's choice:** All four edge case types
**Notes:** None

### Follow-up: Edge case proportion

| Option | Description | Selected |
|--------|-------------|----------|
| ~25% edge cases | 75% clean tool use, 25% edge cases. Enough for robustness without dominating signal. | Yes |
| ~15% edge cases | Heavier on clean patterns. Edge cases as seasoning. | |
| ~35% edge cases | Aggressive robustness training. Risk of overly cautious model. | |

**User's choice:** ~25% edge cases
**Notes:** None

---

## Generation Workflow

| Option | Description | Selected |
|--------|-------------|----------|
| Category batches with validation | One category at a time. Generate batch, validate, fix, repeat. | Yes |
| All categories interleaved | Generate across all categories per session. | |
| Script-driven generation | Programmatic via API (contradicts D-05). | |

**User's choice:** Category batches with validation
**Notes:** None

### Follow-up: Batch size

| Option | Description | Selected |
|--------|-------------|----------|
| 50 samples per batch | One JSONL file per batch. Small enough to validate, large enough to be productive. | Yes |
| 100 samples per batch | Fewer sessions but harder to review. | |
| 25 samples per batch | Very granular, fast feedback, more sessions. | |

**User's choice:** 50 samples per batch
**Notes:** None

---

## Claude's Discretion

- Specific tool schemas in the 50-100 pool
- Exact edge case distribution within 25% allocation
- Topic variety within each category
- Batch file naming convention details
- Whether tools are shared across categories or kept separate

## Deferred Ideas

None -- discussion stayed within phase scope
