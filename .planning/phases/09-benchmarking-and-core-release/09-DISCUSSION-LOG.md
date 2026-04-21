# Phase 9: Benchmarking and Core Release - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-04-21
**Phase:** 09-benchmarking-and-core-release
**Areas discussed:** Benchmark scope, HuggingFace publishing, Comparison report, Release packaging

---

## Benchmark Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Run what works locally | lm-eval-harness on MPS for knowledge; custom evals for code/tool-calling | ✓ |
| Try all standard suites | Attempt BFCL, evalplus, lm-eval-harness on MPS with fallbacks | |
| Cloud eval run | Use CUDA cloud instance for full standard benchmark suite | |

**User's choice:** Run what works locally
**Notes:** MPS-first approach consistent with all prior phases

### Follow-up: Custom evaluation approach

| Option | Description | Selected |
|--------|-------------|----------|
| Inference + pattern matching | Run model on test prompts, check JSON format, code syntax, accuracy | ✓ |
| Manual spot-check only | Generate 20-30 samples, manually review quality | |
| Both automated + manual | Automated pattern-matching + manual spot-check | |

**User's choice:** Inference + pattern matching

---

## HuggingFace Publishing

| Option | Description | Selected |
|--------|-------------|----------|
| Personal account | LakshmanTurlapati/lyra-smollm2-1.7b | |
| Create a Lyra org | lyra-project HF org | |
| You decide | Claude picks | |

**User's choice:** Not publishing for now -- using GitHub instead
**Notes:** User explicitly deferred HF publishing. All artifacts stay on GitHub.

---

## Comparison Report

| Option | Description | Selected |
|--------|-------------|----------|
| Markdown report | Auto-generated BENCHMARK.md with tables and summary | ✓ |
| JSON only | Machine-readable results in results/ | |
| Both Markdown + JSON | JSON data + Markdown report | |

**User's choice:** Markdown report with Mermaid charts
**Notes:** User specifically requested Mermaid charts for visual comparison

---

## Release Packaging

### What ships

| Option | Description | Selected |
|--------|-------------|----------|
| Model weights (merged safetensors) | Full merged model at models/lyra-merged/ | ✓ |
| LoRA adapter weights | Lightweight adapter at models/lyra-adapter/ | ✓ |
| Training & eval scripts | All scripts/ for reproducibility | ✓ |
| Dataset files | Assembled dataset and curated domain files | ✓ |

**User's choice:** All four -- ship everything

### Large file storage

| Option | Description | Selected |
|--------|-------------|----------|
| Git LFS | Track .safetensors, .bin, large .jsonl via LFS | ✓ |
| Gitignore + release artifacts | Large files as GitHub Release assets | |
| Just gitignore them | Keep local, document reproduction steps | |

**User's choice:** Git LFS

---

## Claude's Discretion

- Mermaid chart types and styling
- lm-eval-harness few-shot settings (defaults in eval.yaml)
- Custom inference eval script design
- Report narrative structure
- Git LFS .gitattributes patterns

## Deferred Ideas

- HuggingFace Hub publishing -- revisit later
- Cloud CUDA eval runs for standard BFCL/evalplus scores
