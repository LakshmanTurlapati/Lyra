# Phase 3: Evaluation Framework - Context

**Gathered:** 2026-04-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Build a unified evaluation runner that wraps standard, widely-approved benchmark suites (BFCL, HumanEval/MBPP, MMLU/ARC/HellaSwag) to evaluate SmolLM2-1.7B checkpoints. Provides per-category quality metrics, JSON output, and a comparison tool for base vs fine-tuned scoring. Covers EVAL-03, EVAL-04.

</domain>

<decisions>
## Implementation Decisions

### Eval Execution Interface
- **D-01:** Single script with CLI flags -- `python3 -m scripts.eval_runner --model path/to/model --benchmarks tool-calling,code,knowledge --output results.json`. Consistent with curate_pipeline.py argparse pattern.
- **D-02:** Device support: MPS (Apple Silicon) primary, CPU fallback. No CUDA support. Auto-detect MPS availability, fall back to CPU if unavailable.
- **D-03:** Model loading approach is Claude's discretion -- determine cleanest separation of inference logic from eval logic.

### Benchmark Strategy
- **D-04:** Use ONLY widely-approved, standard benchmark suites. No custom test cases or proprietary fixtures.
- **D-05:** Benchmark mapping by category:
  - Tool calling: BFCL (Berkeley Function Calling Leaderboard)
  - Code: HumanEval / MBPP via bigcode-evaluation-harness
  - General knowledge: MMLU, ARC, HellaSwag via lm-eval-harness
- **D-06:** Our value-add is the unified runner that invokes standard suites and aggregates results per-category. We are a thin orchestration layer, not a benchmark creator.

### Metric Reporting
- **D-07:** JSON output file: `results/{model_name}_{timestamp}.json` with structured scores per benchmark per category. Machine-readable, version-controllable.
- **D-08:** CLI prints a summary table to stdout after eval completes -- quick visual feedback showing category, benchmark, and scores.

### Baseline Comparison
- **D-09:** Run eval twice (once on base SmolLM2-1.7B, once on fine-tuned), producing separate JSON result files.
- **D-10:** Separate compare command reads two JSON files and prints delta table. No built-in history tracking -- simple file-based comparison.
- **D-11:** Compare output format is Claude's discretion.

### Claude's Discretion
- Model loading architecture (direct transformers vs inference wrapper module)
- Compare command output format (terminal table, JSON diff, or hybrid)
- Which specific lm-eval-harness tasks to include for MMLU/ARC/HellaSwag (task names, few-shot settings)
- Summary table formatting (tabulate, rich, or plain print)
- Results directory structure

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Evaluation Tools (from CLAUDE.md tech stack)
- CLAUDE.md "Layer 4: Evaluation" section -- lm-eval-harness 0.4.11, bigcode-evaluation-harness, BFCL v4

### Prior Phase Patterns
- `scripts/curate_pipeline.py` -- CLI argparse pattern to follow for eval_runner
- `scripts/pipeline_config.py` -- Pydantic config pattern (if eval needs config)
- `configs/pipeline.yaml` -- YAML config structure pattern

### Project
- `.planning/PROJECT.md` -- Core constraints (SmolLM2-1.7B, 5K samples, MIT license)
- `.planning/REQUIREMENTS.md` -- EVAL-03, EVAL-04 requirements
- `.planning/research/PITFALLS.md` -- Tokenizer alignment issues relevant to eval

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/curate_pipeline.py` -- CLI pattern (argparse, main function, --input/--output/--config flags)
- `scripts/validate_format.py` -- Pydantic models for ShareGPT format (useful if eval checks format compliance)
- `scripts/quality_scorer.py` -- Scoring pattern (returns dict with pass/fail + numeric score)

### Established Patterns
- Flat `scripts/` directory with standalone Python files
- argparse CLI entry points
- Pydantic for structured data validation
- YAML for configuration files
- `tests/` directory with pytest

### Integration Points
- Eval runner will need transformers + torch (new dependencies for model loading)
- Results stored in `results/` directory (new)
- Compare command reads JSON files from results/
- Standard eval harnesses installed as separate packages (lm-eval, bigcode-eval-harness)

</code_context>

<specifics>
## Specific Ideas

- Standard suites chosen to maximize credibility -- results are comparable to published leaderboards
- MPS-first reflects the development environment (Apple Silicon Mac)
- Thin orchestration layer keeps the project focused -- we don't reinvent benchmarking, we unify existing tools
- JSON output enables future automation (CI runs, tracking over multiple training iterations)

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 03-evaluation-framework*
*Context gathered: 2026-04-20*
