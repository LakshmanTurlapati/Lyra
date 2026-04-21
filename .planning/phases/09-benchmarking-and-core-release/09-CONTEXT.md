# Phase 9: Benchmarking and Core Release - Context

**Gathered:** 2026-04-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Run evaluations on the trained Lyra model (base SmolLM2-1.7B vs fine-tuned), produce a comparison report with Mermaid charts, and prepare a GitHub release with model weights, adapter, datasets, and scripts under MIT license. Covers EVAL-01, EVAL-02, REL-01, REL-02, REL-03, REL-04.

</domain>

<decisions>
## Implementation Decisions

### Benchmark Scope
- **D-01:** Run lm-eval-harness locally on MPS for knowledge benchmarks (MMLU, ARC-Challenge, HellaSwag). These work reliably on Apple Silicon.
- **D-02:** For tool-calling and code evals, use inference + pattern matching on our test split instead of BFCL/evalplus (which need CUDA). Run model inference on curated test prompts, check outputs for correct JSON tool call format, valid code syntax, and factual accuracy.
- **D-03:** No cloud/CUDA eval runs required. Everything runs locally on MPS with CPU fallback.

### Comparison Report
- **D-04:** Auto-generated `BENCHMARK.md` in repo root with tables showing base vs Lyra scores, deltas, and summary.
- **D-05:** Include Mermaid charts for visual comparison (bar charts for category scores, radar chart for overall profile).
- **D-06:** Report generated from JSON results files via the existing `eval_compare.py` tool, extended to output Markdown + Mermaid.

### Publishing Strategy
- **D-07:** No HuggingFace publishing for now. All artifacts stay on GitHub.
- **D-08:** Model card and dataset card still created as Markdown files in the repo (README.md serves as model card, datasets/README.md as dataset card).

### Release Packaging
- **D-09:** Ship everything: merged safetensors model, LoRA adapter weights, training/eval scripts, and assembled dataset files.
- **D-10:** Use Git LFS for large files (.safetensors, .bin, large .jsonl files). Keeps repo cloneable without bloating git history.
- **D-11:** MIT license file in repo root. License headers reference MIT in all scripts.

### Claude's Discretion
- Exact Mermaid chart types and styling
- Which lm-eval-harness few-shot settings to use (eval.yaml already has defaults)
- How to structure the custom inference eval (test harness script design)
- Report section ordering and narrative structure
- Git LFS track patterns and .gitattributes configuration

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Evaluation Framework (Phase 3)
- `scripts/eval_runner.py` -- Unified eval runner with CLI, benchmark dispatch, JSON output
- `scripts/eval_compare.py` -- Compare two JSON result files and print delta table
- `scripts/eval_config.py` -- Pydantic schemas for EvalResult, BenchmarkResult, CompareResult
- `configs/eval.yaml` -- Benchmark task lists, few-shot settings, batch size config

### Model Weights (Phase 8)
- `models/lyra-merged/` -- Full merged safetensors model for evaluation
- `models/lyra-adapter/` -- LoRA adapter weights
- `scripts/train.py` -- Training script (for documenting training params in model card)

### Dataset (Phase 7)
- `datasets/assembled/` -- HuggingFace DatasetDict with train/validation/test splits
- `scripts/assemble_dataset.py` -- Assembly script with stats command

### Project
- `.planning/PROJECT.md` -- Core constraints (SmolLM2-1.7B, MIT license)
- `.planning/REQUIREMENTS.md` -- EVAL-01, EVAL-02, REL-01 through REL-04

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/eval_runner.py` -- Full eval runner CLI, needs extension for custom inference evals
- `scripts/eval_compare.py` -- JSON comparison tool, needs Markdown + Mermaid output extension
- `scripts/eval_config.py` -- Pydantic models for structured eval results
- `configs/eval.yaml` -- Already configured with MMLU/ARC/HellaSwag tasks and few-shot settings

### Established Patterns
- Flat `scripts/` directory with argparse CLI entry points
- Pydantic for structured data validation
- JSON for machine-readable results, Markdown for human-readable output
- `results/` directory for eval output files

### Integration Points
- Eval runner loads model from `models/lyra-merged/` (or HF hub for base model)
- Compare tool reads two JSON result files and produces delta report
- BENCHMARK.md goes in repo root alongside README.md
- Git LFS configured via `.gitattributes` for models/ and large dataset files

</code_context>

<specifics>
## Specific Ideas

- Mermaid charts in BENCHMARK.md for visual comparison -- renders natively on GitHub
- Custom inference eval uses the test split (181 samples) as the evaluation set -- no data leakage since test split was held out during training
- Report should be scannable: summary table at top, detailed per-category breakdown below, Mermaid charts inline

</specifics>

<deferred>
## Deferred Ideas

- HuggingFace Hub publishing -- deferred, may revisit in Phase 10 or later
- Cloud CUDA eval runs for BFCL/evalplus standard scores -- not needed for v1

</deferred>

---

*Phase: 09-benchmarking-and-core-release*
*Context gathered: 2026-04-21*
