# Phase 2: Data Quality and Curation Pipeline - Context

**Gathered:** 2026-04-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Build a curation pipeline that filters, deduplicates, scores, and configures data generation so only high-quality samples proceed to training. Takes raw JSONL generated in Claude Code sessions and produces cleaned, scored output. Configurable via YAML for reuse across runs. Covers DATA-03, DATA-04, DATA-05.

</domain>

<decisions>
## Implementation Decisions

### Quality Scoring
- **D-01:** Two-tier scoring system. Tier 1: automated Python heuristics (pass/fail). Tier 2: manual LLM-as-judge via Claude Code sessions for a calibration subset.
- **D-02:** No Anthropic API SDK introduced for scoring -- judging happens in Claude Code sessions, consistent with D-05 from Phase 1.
- **D-03:** Automated heuristic signals (all four apply to every sample):
  - Format compliance (reuse validate_format.py Pydantic models)
  - Response completeness (no truncation, closed code blocks, conclusions reached)
  - Content diversity (near-duplicate detection within batch)
  - Conversation naturalness (balanced turn lengths, no copy-paste artifacts, no unintended meta-commentary)

### Deduplication
- **D-04:** N-gram overlap strategy (3-gram or 4-gram Jaccard similarity). No external dependencies (datasketch, sentence-transformers). Configurable similarity threshold.
- **D-05:** Deduplication scope is Claude's discretion -- determine the most effective comparison target (prompts, responses, or full conversation) based on empirical results.

### Pipeline Configuration
- **D-06:** Single YAML config file (pipeline.yaml) with sections for quality thresholds, dedup settings, topic distribution targets, and prompt template paths.
- **D-07:** Per-domain overrides within the single file -- global defaults with optional domain-specific sections (e.g., tool-calling can have stricter format thresholds than knowledge).
- **D-08:** Consistent with Phase 1's YAML choice for templates. Users copy and edit the config for different runs.

### Adaptive Output Styles
- **D-09:** Template-driven style enforcement. Style instructions baked into prompt templates (existing templates/ directory). Code templates guide terse responses, knowledge templates guide detailed chain-of-thought.
- **D-10:** Specific style validation rules (what measurably distinguishes terse from detailed) are Claude's discretion. Must ensure domains produce distinguishably different output styles.

### Claude's Discretion
- Deduplication comparison scope (prompt-only, response-only, full conversation, or combination)
- Specific style validation heuristics (token ranges, structural markers, or hybrid approach)
- Default threshold values in pipeline.yaml
- N-gram size (3 vs 4) and similarity threshold default
- Quality score output format (numeric, categorical, or both)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Data Format (from Phase 1)
- `specs/sharegpt-format.md` -- Format specification that the pipeline validates against
- `scripts/validate_format.py` -- Pydantic models to reuse for format compliance checking
- `scripts/validate_tokenizer.py` -- Tokenizer alignment validation (2048 token limit enforcement)

### Templates
- `templates/tool-calling.yaml` -- Tool calling domain templates with style instructions
- `templates/code.yaml` -- Code domain templates (terse style)
- `templates/knowledge.yaml` -- Knowledge domain templates (detailed style)
- `templates/system-prompts.yaml` -- Shared system prompts referenced by domain templates

### Research
- `.planning/research/PITFALLS.md` -- Tokenizer alignment silent failures; relevant to quality validation
- `.planning/research/ARCHITECTURE.md` -- Pipeline component boundaries

### Project
- `.planning/PROJECT.md` -- Core constraints (5K samples, 33/33/33 split, MIT license)
- `.planning/REQUIREMENTS.md` -- DATA-03, DATA-04, DATA-05 requirements

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/validate_format.py` -- Pydantic schema (Message, Conversation, ToolCall, ToolSchema models). Reuse directly for format compliance in Tier 1 scoring.
- `scripts/validate_tokenizer.py` -- Token count validation against SmolLM2-1.7B. Reuse for length-based quality checks.
- `scripts/generate_sample.py` -- Sample fixtures across all 3 domains. Useful as test input for the curation pipeline.

### Established Patterns
- Flat `scripts/` directory with standalone Python files (D-08 from Phase 1)
- Pydantic `model_validator(mode=after)` for structural rule enforcement
- YAML for configuration (templates use YAML already)
- `datasets/` directory with domain separation (tool-calling/, code/, knowledge/)

### Integration Points
- Pipeline reads raw JSONL from `datasets/{domain}/` directories
- Pipeline outputs filtered JSONL back to `datasets/{domain}/` (curated subdirectory or suffix)
- Config file lives at project root or `configs/` directory
- Phases 4-6 (data generation) produce the raw input this pipeline consumes

</code_context>

<specifics>
## Specific Ideas

- Two-tier scoring respects the no-API-SDK constraint while still getting LLM quality judgment via Claude Code sessions
- N-gram dedup chosen over embeddings to keep the project dependency-light (no sentence-transformers, no GPU for dedup)
- Per-domain config overrides enable different quality bars (tool-calling needs strict format, knowledge needs detail)
- Template-driven style means the pipeline validates what templates already instruct -- generation and validation are aligned

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 02-data-quality-and-curation-pipeline*
*Context gathered: 2026-04-20*
