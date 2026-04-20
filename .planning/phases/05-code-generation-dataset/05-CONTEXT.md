# Phase 5: Code Generation Dataset - Context

**Gathered:** 2026-04-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Generate and curate all code generation domain training data covering 3 categories: utility functions, file operations, and debugging. Produces ~3,334 raw samples that get curated down to ~1,667 high-quality samples for training. Covers CODE-01, CODE-02, CODE-03.

</domain>

<decisions>
## Implementation Decisions

### Sample Count and Distribution
- **D-01:** Generate ~3,334 raw code generation samples. Curation pipeline filters down to ~1,667 final samples (the 33% code slice of 5K total).
- **D-02:** Utility-heavy distribution:
  - Utility functions: ~50% (~1,667 raw -> ~834 curated)
  - File operations: ~25% (~834 raw -> ~417 curated)
  - Debugging: ~25% (~834 raw -> ~417 curated)

### Language Distribution
- **D-03:** Python-heavy distribution for utility functions:
  - Python: ~40%
  - JavaScript/TypeScript combined: ~25%
  - Go: ~20%
  - Rust: ~15%
- **D-04:** For file operations and debugging (3 languages per template):
  - Python: ~50%
  - JavaScript: ~30%
  - TypeScript: ~20%

### Response Style
- **D-05:** Terse code-first style. Function/code block with 1-2 line comment explaining what it does. No preamble, no "Here's how to..." padding.
- **D-06:** Debugging samples use brief "Bug: X, Fix: Y" format before the corrected code. Concise identification, not lengthy explanations.
- **D-07:** Utility functions target ~200-500 tokens. File operations target ~300-800 tokens. Debugging targets ~400-800 tokens (per template guidance).

### Generation Infrastructure
- **D-08:** New script `scripts/generate_code_data.py` following the same pattern as Phase 4's `generate_tool_data.py` (CLI with argparse, category generators, batch loop of 50, inline validation via validate_format.py).
- **D-09:** Category batches with validation loops -- same workflow as Phase 4. One category at a time, 50 samples per batch.
- **D-10:** Order of generation: utility functions first (simplest, largest volume), then file operations, then debugging (most complex).

### Claude's Discretion
- Specific utility function topics beyond the template list
- Bug types and debugging scenarios beyond the template list
- Whether to include code comments in generated samples
- Test fixtures for the generation script
- Batch file naming convention (following Phase 4 pattern: `{category}-batch-{NN}.jsonl`)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Data Format
- `specs/sharegpt-format.md` -- Canonical format spec. ALL generated data must conform.
- `scripts/validate_format.py` -- Pydantic validation models. Run against every batch.
- `scripts/validate_tokenizer.py` -- Token count validation (2048 max). Run against every batch.

### Templates and System Prompts
- `templates/code.yaml` -- All 3 category definitions with languages, topics, complexity, and generation_notes
- `templates/system-prompts.yaml` -- System prompts: code_assistant, code_debugger

### Phase 4 Pattern (reference implementation)
- `scripts/generate_tool_data.py` -- Generation script pattern to follow (CLI, category generators, batch loop, inline validation)
- `datasets/tool-calling/tool_schemas.yaml` -- Schema pool pattern (not directly reusable but shows the approach)
- `tests/test_generate_tool_data.py` -- Test pattern to follow for the new generation script

### Curation Pipeline
- `scripts/curate_pipeline.py` -- Post-generation quality filtering, dedup, scoring
- `scripts/quality_scorer.py` -- Quality heuristics
- `scripts/dedup.py` -- N-gram Jaccard deduplication
- `configs/pipeline.yaml` -- Pipeline configuration

### Project
- `.planning/PROJECT.md` -- Core constraints (5K samples, 33/33/33 split, MIT license)
- `.planning/REQUIREMENTS.md` -- CODE-01, CODE-02, CODE-03 requirements

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/generate_tool_data.py` -- Reference implementation for batch generation pattern (CLI, validation loop, category generators)
- `scripts/validate_format.py` -- Pydantic models for format validation (reuse directly)
- `scripts/validate_tokenizer.py` -- SmolLM2-1.7B token counting (reuse directly)
- `scripts/curate_pipeline.py` -- Full curation pipeline (reuse for post-processing)
- `scripts/generate_sample.py` -- Contains code domain sample fixtures showing format

### Established Patterns
- Flat `scripts/` directory with standalone Python files
- JSONL for dataset files (one conversation per line)
- `datasets/code/` directory for output (exists but empty)
- 50 samples per batch, one JSONL file per batch
- Category batches with inline validation after each batch
- 2x overgeneration, curate to target

### Integration Points
- Generated JSONL -> `scripts/curate_pipeline.py` for quality filtering
- Output stored in `datasets/code/` (consumed by Phase 7 assembly)
- Format validated against `specs/sharegpt-format.md`
- Token counts checked against SmolLM2-1.7B's 2048 limit

</code_context>

<specifics>
## Specific Ideas

- Python-heavy distribution reflects real-world usage where small code helpers are most useful in Python
- Terse code-first style aligns with Phase 2's adaptive output style validation (D-09: code templates guide terse responses)
- Utility functions are 50% because that's where a 1.7B model is most practically useful -- quick, correct, self-contained helpers
- The separate generation script keeps concerns clean while following the proven Phase 4 pattern
- Debugging uses "Bug: X, Fix: Y" format to teach identification, not just correction

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 05-code-generation-dataset*
*Context gathered: 2026-04-20*
