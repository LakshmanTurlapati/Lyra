---
phase: 09-benchmarking-and-core-release
status: passed
verified: 2026-04-22
requirements:
  - EVAL-01
  - EVAL-02
  - REL-01
  - REL-02
  - REL-03
  - REL-04
automated_checks:
  passed: 8
  failed: 0
human_verification: approved
gaps: 0
---

# Phase 09 Verification

## Goal

Users can see how Lyra compares to base SmolLM2-1.7B on standard/custom benchmarks and access local Git/GitHub-ready model, dataset, license, and documentation artifacts under MIT terms.

## Verdict

**PASSED** — Phase 09 deliverables are present, tested, and packaged.

The benchmark outcome itself remains a product concern: Lyra currently regresses on `tool-call-format` versus the base model. That concern is documented in `BENCHMARK.md`, `README.md`, and Phase 09.1. It does not block Phase 09 artifact packaging because the artifacts accurately report the measured result.

## Requirement Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| EVAL-01 | Passed | `results/base_full.json` and `results/lyra_full.json` contain knowledge and custom eval results across five benchmarks. |
| EVAL-02 | Passed | `BENCHMARK.md` generated from JSON results includes summary table and Mermaid chart. |
| REL-01 | Passed | `datasets/README.md` contains metadata, methodology, current split statistics, and limitations. |
| REL-02 | Passed | `README.md` starts with YAML model-card metadata and includes usage, training params, benchmark results, and limitations. |
| REL-03 | Passed | `models/lyra-merged/model.safetensors` and `models/lyra-adapter/adapter_model.safetensors` are tracked through Git LFS. |
| REL-04 | Passed | `LICENSE` is MIT and scripts have SPDX MIT headers. |

## Automated Checks

| Check | Result |
|-------|--------|
| `python -m pytest tests/ -x -q` | Passed: 290 passed, 2 skipped |
| `python -m pytest tests/test_release_artifacts.py -v` | Passed: 4 passed |
| `BENCHMARK.md` contains `xychart-beta` and `tool-call-format` | Passed |
| `README.md` starts with YAML frontmatter and contains `license: mit` and base model metadata | Passed |
| `datasets/README.md` contains dataset statistics and limitations | Passed |
| `git lfs ls-files` includes model safetensors and assembled dataset files | Passed |
| `git show HEAD:models/lyra-merged/model.safetensors` is an LFS pointer | Passed |
| Schema drift check | Passed: no drift detected |

## Human Verification

Checkpoint was approved by the user after artifact checks passed. The user accepted continuing despite the known tool-call-format regression because the release artifacts explicitly mark the model experimental and preserve the measured benchmark results.

## Gaps

None for Phase 09 artifact packaging.

## Residual Risks

- Tool-call-format regression remains unresolved and is tracked by Phase 09.1.
- HuggingFace Hub publishing is intentionally deferred by project decision D-07; Phase 09 produces repo-local model/dataset cards and Git LFS artifacts.
- LFS artifacts are committed locally; remote availability still depends on pushing the branch/repository with Git LFS enabled.
