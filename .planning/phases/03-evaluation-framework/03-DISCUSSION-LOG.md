# Phase 3: Evaluation Framework - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-04-20
**Phase:** 03-evaluation-framework
**Areas discussed:** Eval execution interface, Custom benchmark format, Metric reporting, Baseline comparison

---

## Eval Execution Interface

| Option | Description | Selected |
|--------|-------------|----------|
| Single script with flags | python3 -m scripts.eval_runner --model path --benchmarks list. Consistent with curate_pipeline.py pattern. | yes |
| Config-driven | eval.yaml defines benchmarks, model path, output format | |
| Both (flags + optional config) | CLI flags for quick, config for reproducible | |

**User's choice:** Single script with flags
**Notes:** Consistent with existing curate_pipeline.py CLI pattern

---

### Model Loading

| Option | Description | Selected |
|--------|-------------|----------|
| HuggingFace transformers direct | Load via AutoModelForCausalLM | |
| Inference script wrapper | Thin wrapper separating model concerns | |
| You decide | Claude determines best approach | yes |

**User's choice:** You decide

---

### Device Support

| Option | Description | Selected |
|--------|-------------|----------|
| CPU default, GPU optional | Works anywhere, --device cuda for acceleration | |
| GPU required | Require CUDA, fail fast | |
| Auto-detect | Use GPU if available, fall back | |
| MPS only + CPU fallback | (user-provided) Target Mac dev, skip CUDA | yes |

**User's choice:** MPS only + CPU fallback
**Notes:** User explicitly requested MPS (Apple Silicon) as primary device

---

## Custom Benchmark Format

| Option | Description | Selected |
|--------|-------------|----------|
| JSON fixtures | evals/tool-calling.json with {input, expected, scoring_fn} | |
| Python test functions | Each benchmark is a Python function | |
| YAML task definitions | Consistent with templates/ YAML pattern | |
| Standard suites only | (user-provided) Use widely approved benchmarks only | yes |

**User's choice:** Use widely approved standard benchmark suites only -- no custom test cases
**Notes:** BFCL for tool calling, HumanEval/MBPP for code, MMLU/ARC/HellaSwag for knowledge. Value-add is unified runner + per-category reporting.

---

## Metric Reporting

| Option | Description | Selected |
|--------|-------------|----------|
| JSON output file | results/{model}_{timestamp}.json with structured scores | yes |
| Markdown report | Generate .md with tables and breakdowns | |
| Both JSON + terminal summary | JSON for machine, terminal for quick reading | |

**User's choice:** JSON output file + terminal summary table
**Notes:** Also confirmed: CLI prints summary table to stdout after eval completes

---

## Baseline Comparison

| Option | Description | Selected |
|--------|-------------|----------|
| Run twice, diff JSONs | Run base, save JSON. Run fine-tuned, save JSON. Compare command shows deltas. | yes |
| Built-in comparison mode | --compare base.json flag on runner | |
| Results history directory | Auto-save to results/, compare any two | |

**User's choice:** Run twice, diff JSONs
**Notes:** Compare output format deferred to Claude's discretion

---

## Claude's Discretion

- Model loading architecture (inference wrapper vs direct)
- Compare command output format
- lm-eval-harness task names and few-shot settings
- Summary table formatting library choice
- Results directory structure

## Deferred Ideas

None -- discussion stayed within phase scope
