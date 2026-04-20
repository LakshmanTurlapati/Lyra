# Phase 7: Dataset Assembly - Context

**Gathered:** 2026-04-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Merge all three domain datasets (tool-calling, code, knowledge) into a single final dataset with stratified train/validation/test splits. Produce a HuggingFace-compatible dataset ready for fine-tuning. Covers DATA-07.

</domain>

<decisions>
## Implementation Decisions

### Domain Balance
- **D-01:** Use ALL curated samples from all 3 domains without downsampling. Accept natural imbalance (tool-calling ~68%, code ~16.5%, knowledge ~15.4%). More training data preferred over strict 33/33/33 balance.
- **D-02:** Total dataset: ~3,630 samples (2,470 tool-calling + 600 code + 560 knowledge).
- **D-03:** Original 33/33/33 target revised -- imbalance accepted as a pragmatic decision given generation yield differences across domains.

### Split Ratios
- **D-04:** 90/5/5 train/validation/test split. Maximizes training data for a small dataset.
- **D-05:** Stratified splits -- each split contains proportional representation of all 3 domains. No domain should be absent from any split.
- **D-06:** Approximate split sizes: ~3,267 train / ~182 validation / ~182 test.

### Output Format
- **D-07:** HuggingFace `datasets` library format (Arrow-backed). Train/validation/test as named splits.
- **D-08:** Include a `domain` metadata column (values: "tool-calling", "code", "knowledge") so downstream analysis can filter by domain.
- **D-09:** Dataset ready for push_to_hub and direct consumption by Unsloth/TRL SFTTrainer.

### Assembly Process
- **D-10:** Assembly script reads curated JSONL from all 3 domain directories, adds domain metadata, performs stratified split, validates entire dataset, and saves as HF dataset format.

### Claude's Discretion
- Script naming and internal organization
- Whether to also output JSONL alongside HF format
- Exact stratification algorithm (sklearn train_test_split or manual)
- Stats command implementation details
- Directory structure for final assembled dataset

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Source Data
- `datasets/tool-calling/curated/tool-calling-curated.jsonl` -- 2,470 curated tool-calling samples
- `datasets/code/curated/code-curated.jsonl` -- 600 curated code samples
- `datasets/knowledge/curated/knowledge-curated.jsonl` -- 560 curated knowledge samples

### Data Format
- `specs/sharegpt-format.md` -- Format spec all samples conform to
- `scripts/validate_format.py` -- Validation (run on final assembled dataset)
- `scripts/validate_tokenizer.py` -- Token limit validation

### Pipeline
- `scripts/curate_pipeline.py` -- Quality pipeline (final validation pass)
- `configs/pipeline.yaml` -- Pipeline configuration

### Project
- `.planning/PROJECT.md` -- Core constraints
- `.planning/REQUIREMENTS.md` -- DATA-07 requirement

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/validate_format.py` -- Run final validation on assembled dataset
- `scripts/validate_tokenizer.py` -- Verify all samples within 2048 tokens
- HuggingFace `datasets` library -- already in project dependencies (CLAUDE.md tech stack)

### Established Patterns
- Flat `scripts/` directory
- JSONL as intermediate format
- Pydantic for validation
- argparse CLI entry points

### Integration Points
- Reads from `datasets/{domain}/curated/` directories
- Outputs to `datasets/assembled/` or similar
- Consumed by Phase 8 (fine-tuning) directly via HF datasets API

</code_context>

<specifics>
## Specific Ideas

- Using all samples means tool-calling is the dominant domain (~68%). This is acceptable because tool-calling is Lyra's primary differentiator and most practical use case for a 1.7B model.
- 90/5/5 split maximizes training data while still providing eval/test sets for Phase 9 benchmarking.
- HuggingFace format means the dataset can be loaded with one line: `dataset = load_dataset("path/to/assembled")` in the training script.
- Domain metadata column enables per-domain evaluation in Phase 9 without separate dataset files.

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 07-dataset-assembly*
*Context gathered: 2026-04-20*
