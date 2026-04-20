# Phase 6: General Knowledge Dataset - Context

**Gathered:** 2026-04-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Generate and curate all general knowledge domain training data covering 3 categories: reasoning chains, factual Q&A, and explanations. Produces ~3,334 raw samples that get curated down to ~1,667 high-quality samples for training. Covers KNOW-01, KNOW-02, KNOW-03.

</domain>

<decisions>
## Implementation Decisions

### Sample Count and Distribution
- **D-01:** Generate ~3,334 raw knowledge samples. Curation pipeline filters down to ~1,667 final samples (the 33% knowledge slice of 5K total).
- **D-02:** Q&A-heavy distribution:
  - Factual Q&A: ~40% (~1,334 raw -> ~667 curated)
  - Explanations: ~35% (~1,167 raw -> ~584 curated)
  - Reasoning chains: ~25% (~834 raw -> ~417 curated)

### Response Depth and Style
- **D-03:** Category-adaptive response depth:
  - Reasoning chains: always show numbered steps + explicit conclusion. Target ~500-1200 tokens.
  - Factual Q&A: concise 1-5 sentences for simple facts. Target ~200-600 tokens.
  - Explanations: structured with examples/analogies, adaptive to topic complexity. Target ~400-1000 tokens.
- **D-04:** This is the "detailed" domain per Phase 2 style validation (D-09). Reasoning chains and explanations use chain-of-thought. Q&A stays concise but factually complete.

### Topic Diversity Strategy
- **D-05:** Tech and STEM weighted topic distribution:
  - Technology/computing: ~40%
  - Math/logic: ~25%
  - Science: ~20%
  - Other (history, geography, everyday): ~15%
- **D-06:** Aligns with Lyra's developer audience -- practical knowledge a developer would query a small model about.

### Dedup Mitigation (Lesson from Phase 5)
- **D-07:** Large topic pool of 200+ unique question topics (vs Phase 5's ~20 templates that caused 78% dedup rejection). Each batch uses different topic subsets.
- **D-08:** Varied question phrasing across how/what/why/explain/compare forms. Natural language diversity prevents n-gram overlap at the 0.7 Jaccard threshold.
- **D-09:** Generation script must produce samples with unique enough content that the curation pipeline retains 50%+ (vs Phase 5's 17.6% retention rate).

### Generation Infrastructure
- **D-10:** New script `scripts/generate_knowledge_data.py` following the same pattern as Phases 4-5 (CLI, category generators, batch of 50, inline validation).
- **D-11:** Category batches with validation loops. Order: factual Q&A first (simplest, largest volume), then explanations, then reasoning chains (most complex).

### Claude's Discretion
- Specific question topics within the 200+ pool
- Exact phrasing variations per topic
- Whether to include follow-up questions in multi-turn knowledge conversations
- Batch file naming convention (following pattern: `{category}-batch-{NN}.jsonl`)
- Test fixtures for the generation script

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Data Format
- `specs/sharegpt-format.md` -- Canonical format spec. ALL generated data must conform.
- `scripts/validate_format.py` -- Pydantic validation models. Run against every batch.
- `scripts/validate_tokenizer.py` -- Token count validation (2048 max).

### Templates and System Prompts
- `templates/knowledge.yaml` -- All 3 category definitions with topics, complexity, and generation_notes
- `templates/system-prompts.yaml` -- System prompts: knowledge_reasoning, knowledge_assistant

### Phase 4-5 Pattern (reference implementation)
- `scripts/generate_tool_data.py` -- Generation script pattern (CLI, category generators, batch loop)
- `scripts/generate_code_data.py` -- Second reference (same pattern, different domain)

### Curation Pipeline
- `scripts/curate_pipeline.py` -- Post-generation quality filtering, dedup, scoring
- `scripts/quality_scorer.py` -- Quality heuristics
- `scripts/dedup.py` -- N-gram Jaccard deduplication (0.7 threshold)
- `scripts/style_validator.py` -- Style enforcement (knowledge domain: detailed chain-of-thought)
- `configs/pipeline.yaml` -- Pipeline configuration with domain overrides

### Lessons Learned (Phase 5)
- `.planning/phases/05-code-generation-dataset/05-03-SUMMARY.md` -- Documents the dedup collapse and pipeline adjustments made

### Project
- `.planning/PROJECT.md` -- Core constraints (5K samples, 33/33/33 split, MIT license)
- `.planning/REQUIREMENTS.md` -- KNOW-01, KNOW-02, KNOW-03 requirements

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/generate_tool_data.py` -- Batch generation pattern reference
- `scripts/generate_code_data.py` -- Second reference (same pattern)
- `scripts/validate_format.py` -- Format validation (reuse directly)
- `scripts/validate_tokenizer.py` -- Token counting (reuse directly)
- `scripts/curate_pipeline.py` -- Full curation pipeline (reuse for post-processing)
- `scripts/style_validator.py` -- Already has knowledge domain config (detailed style enforcement)

### Established Patterns
- Flat `scripts/` directory with standalone Python files
- JSONL for dataset files
- `datasets/knowledge/` directory for output (exists but empty)
- 50 samples per batch, category batches with validation
- 2x overgeneration, curate to target

### Integration Points
- Generated JSONL -> `scripts/curate_pipeline.py` for quality filtering
- Output stored in `datasets/knowledge/` (consumed by Phase 7 assembly)
- Style validator enforces detailed responses for knowledge domain

</code_context>

<specifics>
## Specific Ideas

- 200+ topic pool is the key mitigation for Phase 5's dedup problem. Knowledge questions are naturally more diverse than code templates because they span many domains with varied phrasing.
- Tech/STEM weighting ensures the model learns knowledge a developer would actually query -- not random trivia.
- Category-adaptive depth means reasoning chains are always verbose (showing work) while Q&A is concise. This matches how humans expect different question types to be answered.
- Factual Q&A first in generation order because simple facts are easiest to generate uniquely (many facts, short responses, diverse domains).

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 06-general-knowledge-dataset*
*Context gathered: 2026-04-20*
