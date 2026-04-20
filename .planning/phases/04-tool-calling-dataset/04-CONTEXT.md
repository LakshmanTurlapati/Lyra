# Phase 4: Tool Calling Dataset - Context

**Gathered:** 2026-04-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Generate and curate all tool-call domain training data covering 5 patterns: JSON function calls, multi-turn conversations with tool results, parallel function execution, MCP-style tool use, and CLI/shell commands. Produces ~3,300 raw samples that get curated down to ~1,667 high-quality samples for training. Covers TOOL-01 through TOOL-05.

</domain>

<decisions>
## Implementation Decisions

### Sample Count and Distribution
- **D-01:** Generate ~3,300 raw tool-calling samples. Curation pipeline (Phase 2) filters down to ~1,667 final samples (the 33% tool-calling slice of 5K total).
- **D-02:** Weighted distribution by complexity and real-world frequency:
  - Single call: ~35% (~1,155 raw -> ~583 curated)
  - CLI commands: ~25% (~825 raw -> ~417 curated)
  - Multi-turn: ~15% (~495 raw -> ~250 curated)
  - Parallel calls: ~15% (~495 raw -> ~250 curated)
  - MCP patterns: ~10% (~330 raw -> ~167 curated)

### Tool Schema Diversity
- **D-03:** Large pool of 50-100 unique tool schemas across the dataset. Each sample draws from this pool to teach generalization across tool types.
- **D-04:** Mix of developer-focused tools (databases, file systems, HTTP, git, package managers, cloud services, monitoring) and everyday tools (weather, calendar, email, search, maps). Developer tools weighted heavier to align with Lyra's target use case.

### Edge Cases and Boundaries
- **D-05:** ~25% of samples are edge cases (split across four types), ~75% are clean happy-path tool use.
- **D-06:** Edge case types included:
  - No-tool-needed responses: assistant answers directly without calling a tool (teaches when NOT to use tools)
  - Tool error handling: tool returns failure, assistant recovers gracefully
  - Ambiguous requests: could use a tool or answer directly, assistant makes a judgment call
  - Parameter edge cases: optional params, empty strings, nested objects, arrays

### Generation Workflow
- **D-07:** Category batches with validation loops. Generate one category at a time, validate with pipeline, fix issues, repeat until category target met. Then move to next category.
- **D-08:** 50 samples per batch, one JSONL file per batch (e.g., `datasets/tool-calling/single-call-batch-01.jsonl`). Small enough to validate quickly, large enough to be productive.
- **D-09:** Order of generation: single-call first (simplest, establishes baseline), then CLI, multi-turn, parallel, MCP (increasing complexity).

### Claude's Discretion
- Specific tool schemas in the 50-100 pool (names, parameters, descriptions)
- Exact edge case distribution within the 25% allocation
- Topic variety within each category (which questions users ask, which domains tools cover)
- Batch file naming convention details
- Whether to reuse tools across categories or keep pools separate

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Data Format
- `specs/sharegpt-format.md` -- Canonical format spec: role ordering, tool_calls structure, tool response format. ALL generated data must conform.
- `scripts/validate_format.py` -- Pydantic validation models (Conversation, Message, ToolCall, ToolSchema). Run against every batch.
- `scripts/validate_tokenizer.py` -- Token count validation (2048 max). Run against every batch.

### Templates and System Prompts
- `templates/tool-calling.yaml` -- All 5 category definitions with example tools, topics, complexity levels, and generation_notes
- `templates/system-prompts.yaml` -- System prompts: tool_assistant, mcp_assistant, cli_assistant

### Curation Pipeline
- `scripts/curate_pipeline.py` -- Post-generation quality filtering, dedup, scoring
- `scripts/quality_scorer.py` -- Quality heuristics (format compliance, completeness, diversity, naturalness)
- `scripts/dedup.py` -- N-gram Jaccard deduplication
- `configs/pipeline.yaml` -- Pipeline configuration with quality thresholds

### Sample Reference
- `scripts/generate_sample.py` -- Hardcoded sample fixtures showing exact format for single-call, parallel, multi-turn, MCP, CLI patterns

### Project
- `.planning/PROJECT.md` -- Core constraints (5K samples, 33/33/33 split, MIT license, no API SDK)
- `.planning/REQUIREMENTS.md` -- TOOL-01 through TOOL-05 requirements

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/validate_format.py` -- Pydantic models for validation (Conversation, Message, ToolCall). Run on every generated batch.
- `scripts/validate_tokenizer.py` -- SmolLM2-1.7B token counting. Ensures 2048 limit per conversation.
- `scripts/curate_pipeline.py` -- Full pipeline: format check + quality score + dedup. Post-process all raw batches.
- `scripts/generate_sample.py` -- Reference implementations showing exact JSON structure for all 5 patterns.
- `templates/tool-calling.yaml` -- Category definitions, example tools, and generation notes.

### Established Patterns
- Flat `scripts/` directory with standalone Python files
- JSONL for dataset files (one conversation per line)
- `datasets/tool-calling/` directory for output
- Pydantic for structured validation
- YAML for configuration

### Integration Points
- Generated JSONL -> `scripts/curate_pipeline.py` for quality filtering
- Output stored in `datasets/tool-calling/` (will be consumed by Phase 7 assembly)
- Format validated against `specs/sharegpt-format.md` schema
- Token counts checked against SmolLM2-1.7B's 2048 limit

</code_context>

<specifics>
## Specific Ideas

- Generation happens in Claude Code sessions per D-05 (Phase 1). No API SDK, no batch processing infrastructure.
- The 2x overgeneration (3,300 raw -> 1,667 curated) gives the quality pipeline room to be selective without running short.
- Developer-heavy tool mix (with some everyday tools) aligns with Lyra's core value: "practically useful for day-to-day development tasks."
- 25% edge cases is enough to teach robustness without making the model overly cautious about calling tools.
- Weighted distribution ensures the model sees the most common real-world patterns (single calls, CLI) most often.

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 04-tool-calling-dataset*
*Context gathered: 2026-04-20*
