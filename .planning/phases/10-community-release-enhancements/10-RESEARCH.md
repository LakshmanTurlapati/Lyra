# Phase 10: Community Release Enhancements - Research

**Researched:** 2026-04-24
**Domain:** GGUF quantization pipeline + dataset versioning + GitHub-native release engineering
**Confidence:** HIGH on toolchain and metadata behaviour; HIGH on GitHub size limits; HIGH on code-context (Lyra's existing scripts and repo state).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Publishing Scope**
- **D-01:** GitHub + Git LFS is the only distribution channel. No HuggingFace Model Hub weights, no HF dataset repo, no HF Space, no HF-hosted GGUF. Extends Phase 9 D-07 ("No HuggingFace publishing for now") through the community release. All artifacts live on GitHub and are pulled via clone or GitHub Releases.
- **D-02:** REL-06 (interactive Gradio demo Space on HuggingFace) is removed from v1. It moves to the `Out of Scope` table in REQUIREMENTS.md with reason "Community release is GitHub-native; interactive demo not pursued in v1." A corresponding ROADMAP.md update drops the Gradio Space from Phase 10's goal and success criteria.
- **D-03:** HuggingFace namespace decision is not made here. If REL-06 ever returns, namespace (personal account vs `lyra` org) is decided at publish time, not during planning.

**GGUF Pipeline (REL-05)**
- **D-04:** Conversion toolchain is canonical llama.cpp: `convert_hf_to_gguf.py` produces f16 GGUF from `models/lyra-merged/`, then `llama-quantize` produces the shipped variants. The full pipeline is wrapped in a reproducible shell script under `scripts/` so the conversion is one command from the repo.
- **D-05:** Quantization levels shipped: **Q4_K_M and Q8_0 only** — exactly what REL-05 specifies. No additional levels (Q5_K_M, Q6_K, etc.) in this phase; keeps the LFS footprint minimal and avoids a sprawling per-variant README matrix.
- **D-06:** Model variant quantized: the weights from Phase 09.1/09.2 that satisfy the tool-call-format success criterion (beats base SmolLM2-1.7B's 0.4065 on tool-call-format, or a documented revert-to-base release). Phase 10 does not begin until those weights exist. Current pre-09.1 `models/lyra-merged/` is not shipped to the community.
- **D-07:** SmolLM2 chat template is embedded into GGUF metadata at convert time via `convert_hf_to_gguf.py` (implicitly — see Standard Stack / Pitfall 1). llama.cpp / LM Studio then applies the exact template used during training and evaluation. This closes the same class of runtime template-drift bug that caused the Phase 09.1 D-03/D-04 regressions.

**Dataset Versioning (REL-07)**
- **D-08:** Versioning scheme is **SemVer** (`major.minor.patch`). Major = schema / format-breaking change. Minor = dataset re-curation or domain re-balance with the same schema. Patch = sample-level fixes or metadata corrections.
- **D-09:** Initial release is **v1.0.0 = the current 25K post-09.1 assembled dataset** — the dataset the released model trains on. No retroactive v0 for the original ~5K Phase 7 assembly.
- **D-10:** Version bundle contents: train/validation/test JSONL from `datasets/assembled/`, a dataset stats JSON (produced by `scripts/assemble_dataset.py`), a pinned snapshot of BENCHMARK.md for that version, and a CHANGELOG entry. All files are attached to a git tag + GitHub Release so the version is reproducible from the tag alone.
- **D-11:** Metrics per version are **pinned to the model evals of the model trained on that dataset version**. A new dataset version triggers a new training run + eval run; the resulting `results/eval_*.json` files and the generated BENCHMARK.md snapshot are attached to that version's release.

### Claude's Discretion
- Exact shell script name and structure for the GGUF conversion pipeline (e.g., `scripts/make_gguf.sh` vs `scripts/convert_gguf.py`).
- GGUF file-naming convention (e.g., `lyra-v1.0-q4_k_m.gguf`, `lyra-1.0.0-Q4_K_M.gguf`, or similar).
- Whether each GGUF variant ships with its own short README or a single combined README for the GGUF directory.
- CHANGELOG.md file location and format — project has no CHANGELOG yet.
- Dataset-version git tag naming (e.g., `dataset-v1.0.0`, `data-1.0.0`, `v1.0.0-dataset`). Must not collide with any future model-release tag scheme.
- Additions to `.gitattributes` to track `*.gguf` under Git LFS.
- Where inside `scripts/` the GGUF pipeline lives and how it's invoked from the repo README.
- llama.cpp version pinning strategy (specific commit SHA, release tag, or "latest supported").
- Optional post-quantization perplexity sanity check before publishing a GGUF.

### Deferred Ideas (OUT OF SCOPE)
- HuggingFace Model Hub publishing of Lyra weights — not pursued in v1.
- HuggingFace dataset repo for Lyra — not pursued in v1.
- Additional GGUF quantization levels beyond Q4_K_M / Q8_0 (Q5_K_M, Q6_K, etc.) — may be added in a later patch release.
- Continuous / automated release pipeline (GitHub Actions cutting releases on tag push) — manual release is sufficient for v1.
- Hosted inference / API / Ollama Modelfile publication — users deploy on their own infrastructure.
- Gradio demo Space UX direction is preserved in `.planning/phases/10-community-release-enhancements/10-CONTEXT.md` `<deferred>` for future reference but must not leak into Phase 10 plans.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REL-05 | GGUF quantized variants (Q4_K_M, Q8_0) published for LM Studio / llama.cpp | Standard Stack §GGUF Toolchain; Code Examples §Conversion & Quantization; Pitfall 1 (chat-template embed); §Environment Availability |
| REL-07 | Versioned dataset releases with documented changes and metrics per version | Standard Stack §Versioning; Architecture Patterns §Dataset Version Bundle; Code Examples §Stats JSON Augmentation; §Pitfall 3 (GitHub asset limits) |
| REL-06 | Interactive Gradio demo Space — **being removed from v1 per D-02** | Not researched; plan must apply REQUIREMENTS.md/ROADMAP.md edits per CONTEXT.md `<deferred>` |
</phase_requirements>

## Summary

Phase 10 is a **release-engineering phase** with two independent deliverables: (1) a reproducible GGUF conversion pipeline for the Lyra merged model, producing Q4_K_M and Q8_0 variants; and (2) a SemVer-based dataset versioning scheme where every dataset release is a git tag with an attached bundle (JSONL splits, stats JSON, pinned BENCHMARK.md snapshot, CHANGELOG entry) and matched model-evaluation JSONs.

The technical risk is **low** for the toolchain itself — llama.cpp's `convert_hf_to_gguf.py` + `llama-quantize` is a standard, well-documented pipeline — but **moderate for two project-specific cross-cuts**:
1. **Chat-template embed provenance.** `convert_hf_to_gguf.py` has **no `--chat-template` flag**. It auto-embeds the `chat_template` field from `tokenizer_config.json` into GGUF metadata as `tokenizer.chat_template` [VERIFIED: convert_hf_to_gguf.py source inspection, tail of argparse]. Lyra's `models/lyra-merged/tokenizer_config.json` already has this field populated. The risk is that a future retrain may strip the field; Phase 09.1 D-04 already instituted "persist chat template inline" so this is guarded, but the GGUF conversion script must **fail loudly** if the field is missing rather than silently embed a default.
2. **Phase 09.1/09.2 gating.** No community-facing GGUFs can ship until the Phase 09.1/09.2 tool-call-format regression resolves per D-06. The plan structure should split work into **non-gated prep** (infrastructure, scripts, CHANGELOG template, dataset stats augmentation) and **gated execution** (running conversion, cutting releases).

A third cross-cut surfaced in research: **the current `.gitattributes` does not track any file types under LFS** — it contains only a comment stating that large files live elsewhere [VERIFIED: `cat .gitattributes`]. `tests/test_release_artifacts.py::test_gitattributes_lfs` is consequently RED [VERIFIED: `pytest` run]. Phase 9 D-10 was a stated intent that never made it into the repo. Phase 10 must convert that intent into reality as a prerequisite for both deliverables (LFS tracks `*.gguf` for REL-05, tracks `*.jsonl` under `datasets/assembled/` for REL-07).

The current `models/lyra-merged/model.safetensors` is **3.2 GB** [VERIFIED: `du -sh`], exceeding GitHub's 2 GiB-per-file Release asset limit. This is not a Phase 10 issue (safetensors are not shipped per D-04/D-05 — only GGUFs are), but it confirms the overall strategy: **only GGUF variants are distributable via GitHub Releases**; the safetensors stay out of git (per existing `.gitignore`) and can be reconstructed from `models/lyra-adapter/` + base model if ever needed.

**Primary recommendation:** Split Phase 10 into four plans: **(1) Infrastructure prep** (non-gated: `.gitattributes` LFS rules, CHANGELOG template, `scripts/assemble_dataset.py` stats augmentation, `scripts/convert_gguf.sh` skeleton + unit tests, README scaffolding, REQUIREMENTS.md/ROADMAP.md edits per D-02), **(2) GGUF conversion execution** (gated on 09.1/09.2: run the pipeline, verify `tokenizer.chat_template` key present in outputs, smoke-test load in llama.cpp, optional perplexity sanity check), **(3) Dataset v1.0.0 release** (gated: assemble stats, pin metrics, tag `dataset-v1.0.0`, cut Release with bundle), **(4) Documentation & human verification** (README GGUF section, LM Studio / llama.cpp runbook, UAT checklist).

## Architectural Responsibility Map

Lyra is a Python/CLI project with no multi-tier architecture. "Tiers" here map to functional responsibility within the release pipeline.

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|--------------|----------------|-----------|
| GGUF conversion (safetensors → f16 GGUF) | `scripts/` (shell wrapper invoking llama.cpp `convert_hf_to_gguf.py`) | External toolchain (llama.cpp binary) | The repo does not vendor llama.cpp per CONTEXT.md §Integration Points; script must locate or fail |
| Quantization (f16 → Q4_K_M, Q8_0) | `scripts/` (same shell wrapper invoking `llama-quantize`) | External toolchain (llama.cpp binary) | Same as above; K-quants like Q4_K_M are not an `--outtype` of the convert script |
| GGUF smoke test (load + chat-template verify) | `scripts/` + `tests/` (pytest smoke test + manual UAT) | External toolchain (`gguf-dump` CLI or `gguf` Python package) | Verify `tokenizer.chat_template` key present; compare against `models/lyra-merged/chat_template.jinja` content |
| Dataset stats JSON emission | `scripts/assemble_dataset.py` (extend existing stats subcommand) | — | Existing code owns assembly; augmentation fits naturally |
| Pinned BENCHMARK.md snapshot per dataset version | `scripts/eval_compare.py --markdown` (existing, unchanged) | release automation (`scripts/release_dataset.sh` or docs-only manual steps) | Existing Phase 9 plan 09-03 already produces the snapshot; Phase 10 just captures it |
| Release bundle creation | Manual (GitHub Release UI or `gh release create` command documented in README) | — | Manual-release is sufficient per Deferred — no automation |
| Git LFS tracking | `.gitattributes` (declarative) | `git lfs` CLI (system tool, already installed via brew) | One-time repo-wide config, tested via `tests/test_release_artifacts.py` |
| CHANGELOG entry per version | `CHANGELOG.md` (repo root, new file) | Manual human edit per release | No automation; Keep-a-Changelog format per recommendation |
| Documentation (GGUF usage, version history) | `README.md` (repo root), `datasets/README.md`, `docs/` (if needed) | — | Lives with the artifact; no external doc site |

**Why this matters for planning:** The planner should structure tasks so that shell-script wrappers are implemented and unit-testable in Plan 1 (prep), external-tool invocations happen in Plan 2 (execution, gated), and docs/UAT live in Plan 4. The `scripts/` tier owns all automation; the planner should not introduce a new directory or package for this phase.

## Standard Stack

### Core

| Library / Tool | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| llama.cpp | b8913 (2026-04-24) [VERIFIED: web search] | Canonical GGUF conversion + quantization toolchain | D-04 mandates llama.cpp; it's the reference implementation for GGUF. Build number b8913 is current (llama.cpp releases almost daily). Recommend pinning to a specific build tag or commit in the conversion script. |
| `convert_hf_to_gguf.py` | ships with llama.cpp source | Converts HF safetensors → f16 GGUF with auto-embedded tokenizer/chat_template metadata | Part of llama.cpp source tree. **Not included in `pip install llama-cpp-python`** [VERIFIED: web search]; requires cloning llama.cpp or using a wrapper. |
| `llama-quantize` | ships with llama.cpp build artifacts | Quantizes f16 GGUF → Q4_K_M, Q8_0, and other types | Only way to produce K-quants (Q4_K_M, Q5_K_M, etc.) — `convert_hf_to_gguf.py --outtype` supports only f32/f16/bf16/q8_0/tq1_0/tq2_0/auto [VERIFIED: argparse source inspection]. Q4_K_M must go through this tool. |
| `gguf-dump` (or `GGUFReader` Python API) | gguf 0.18.0 [VERIFIED: PyPI] | Inspect GGUF metadata after conversion (verify `tokenizer.chat_template` present) | Ships with `gguf-py`/`gguf` PyPI package; standalone CLI in llama.cpp build. Both produce equivalent JSON dumps. |
| Git LFS | 3.7.1 [VERIFIED: `git-lfs --version`] | Tracks GGUF binaries and dataset JSONL under LFS so clones don't balloon | Already installed via brew per Phase 9 plan 09-01 STATE.md decision; current `.gitattributes` has no rules — this phase adds them. |
| GitHub Releases | web + `gh` CLI | Attaches release assets per D-10 / D-11 | Standard release channel; 2 GiB per-file asset limit [VERIFIED: docs.github.com]. |

**Quantization types produced in this phase (per D-05):**
| Type | Approx. bits/weight | Approx. output size for SmolLM2-1.7B | Use case |
|------|---------------------|--------------------------------------|----------|
| Q4_K_M | ~4.9 bpw | ~1.06 GB (reference: HuggingFace SmolLM2-1.7B-Instruct-GGUF) [CITED: huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF] | Balanced quality/size — recommended default for consumer hardware |
| Q8_0 | ~8.5 bpw | ~1.8 GB (estimate: SmolLM2-1.7B × 1.06 bytes/param + headers) | Near-original quality for users who prioritize accuracy |

Both comfortably fit under GitHub's 2 GiB Release asset limit.

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `datasets` | 4.8.4 (already pinned) | `DatasetDict.save_to_disk` + `load_from_disk` roundtrip for versioned bundles | Already used by `scripts/assemble_dataset.py`; no new dependency |
| `pydantic` | 2.12.5 (already pinned) | Schema-validate the stats JSON before it ships in the bundle | Already used; extend existing model or add a new `DatasetStats` model |
| `huggingface_hub` (indirect) | — | Not added in this phase | Deferred — D-01 prohibits HF publishing in v1 |
| `jq` (optional, system) | any | Pretty-print gguf-dump JSON output in the smoke test / runbook | System tool; not a code dependency. Runbook uses it only for human-readable inspection. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| llama.cpp `convert_hf_to_gguf.py` + `llama-quantize` | Unsloth's `model.save_pretrained_gguf("dir", tokenizer, quantization_method="q4_k_m")` [CITED: unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf] | Unsloth wraps llama.cpp internally but **does not guarantee chat_template embed** — docs warn "use the SAME chat template that was used when training the model." For a project whose last regression was caused by template drift, the more explicit llama.cpp path is safer. Plus, D-04 mandates llama.cpp directly. |
| llama.cpp CLI (system install) | `llama-cpp-python` | `llama-cpp-python` (v0.3.20) provides the `convert_hf_to_gguf` function but it is not the canonical script and adds a heavy C++ dependency. D-04 says "canonical llama.cpp", so stick with CLI. |
| Build llama.cpp from source | `brew install llama.cpp` (macOS) | Build-from-source pins a commit SHA (reproducibility ✓) but adds a CMake/compiler prerequisite. Brew ships stable 8890 [VERIFIED: `brew info llama.cpp`] which is one step behind HEAD and is sufficient for a release-engineering script. Recommend **brew as the documented default** with source-build as a fallback documented in README. |
| GitHub Releases for all artifacts | Keep GGUFs as LFS-tracked files in the repo tree | LFS-tracked file approach means every clone pulls GGUFs (bandwidth cost). GitHub Releases decouple artifact download from clone; only users who want the model hit the asset. **Recommend: release assets for GGUF binaries, LFS for long-lived tracked files (dataset JSONL in `datasets/assembled/`).** See Architecture Patterns §LFS vs Release Assets. |
| Keep-a-Changelog format | Common Changelog / freeform narrative | Keep-a-Changelog is the most widely recognized convention (Added/Changed/Deprecated/Removed/Fixed/Security), plays nicely with SemVer (D-08), and GitHub Release pages render it cleanly. Recommend Keep-a-Changelog 1.1.0. |

**Installation (developer prerequisite, documented in README):**
```bash
# Install llama.cpp CLI tools (macOS)
brew install llama.cpp
# Or build from source (any platform)
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build --config Release
# The convert_hf_to_gguf.py script lives at llama.cpp/convert_hf_to_gguf.py
# llama-quantize binary lives at llama.cpp/build/bin/llama-quantize
# gguf-dump CLI ships with the gguf-py install: pip install gguf==0.18.0
```

**Version verification:** I verified `llama-cpp-python: 0.3.20` and `gguf: 0.18.0` via `pip index` on PyPI [VERIFIED]. llama.cpp itself uses a rolling build-number tag (`b8913` current); the plan should pin a specific tag in the conversion script comment and re-verify before cutting the release.

## Architecture Patterns

### System Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        Phase 10 Release Pipeline                              │
└──────────────────────────────────────────────────────────────────────────────┘

                    Phase 09.1/09.2 weights (GATING D-06)
                              │
                              ▼
  ┌────────────────────────────────────────────────────┐
  │ models/lyra-merged/                                 │
  │  ├─ model.safetensors   (3.2 GB — not shipped)     │
  │  ├─ tokenizer_config.json  (has chat_template)     │
  │  ├─ tokenizer.json                                  │
  │  ├─ config.json                                     │
  │  └─ chat_template.jinja  (provenance artifact)     │
  └────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────┐
  │ scripts/convert_gguf.sh  (this phase creates)       │
  │  step 1: convert_hf_to_gguf.py --outtype f16        │
  │          → lyra-v1.0-f16.gguf  (intermediate)       │
  │          [auto-embeds tokenizer.chat_template       │
  │           from tokenizer_config.json]               │
  │  step 2: llama-quantize f16.gguf Q4_K_M.gguf Q4_K_M │
  │  step 3: llama-quantize f16.gguf Q8_0.gguf  Q8_0    │
  │  step 4: gguf-dump --no-tensors to verify            │
  │          tokenizer.chat_template key present         │
  │  step 5: (optional) llama-perplexity delta check    │
  └────────────────────────────────────────────────────┘
                              │
                              ├──────────────────► lyra-v1.0-q4_k_m.gguf  (~1 GB)
                              └──────────────────► lyra-v1.0-q8_0.gguf   (~1.8 GB)
                                                              │
                                                              ▼
                                          ┌────────────────────────────────┐
                                          │  GitHub Release: model-v1.0.0  │
                                          │  (attaches both .gguf files)    │
                                          └────────────────────────────────┘


  ┌────────────────────────────────────────────────────┐
  │ datasets/assembled/                                 │
  │  ├─ train/   (Arrow)                                │
  │  ├─ validation/                                     │
  │  ├─ test/                                           │
  │  └─ dataset_dict.json                               │
  └────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────┐
  │ scripts/assemble_dataset.py stats  (extend)         │
  │  + emit JSON to stdout/--output                     │
  │    { splits: {train,val,test}, domains: {...},      │
  │      token_histogram?: {...},                       │
  │      schema_version: "1.0.0" }                      │
  └────────────────────────────────────────────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ stats.json + JSONL dump │
                 │ + BENCHMARK.md snapshot │
                 │ + CHANGELOG.md excerpt  │
                 └─────────────────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ git tag dataset-v1.0.0  │
                 │ GitHub Release: bundle  │
                 └─────────────────────────┘
```

### Recommended Project Structure

```
lyra/
├── scripts/
│   ├── convert_gguf.sh           # NEW — top-level GGUF conversion wrapper (D-04)
│   ├── verify_gguf.py            # NEW — reads GGUF metadata, asserts tokenizer.chat_template present
│   ├── dataset_stats.py          # NEW (or extend assemble_dataset.py stats subcommand) — emits stats.json
│   └── release_bundle.sh         # NEW (optional) — packages tag bundle for a release asset
├── CHANGELOG.md                  # NEW — Keep-a-Changelog format, sections per version
├── .gitattributes                # EXTEND — add *.gguf, datasets/assembled/**/*.jsonl, *.safetensors LFS rules
├── README.md                     # EXTEND — add "GGUF variants" section + LM Studio / llama.cpp usage
├── datasets/
│   └── README.md                 # EXTEND — add "Dataset Versions" table with link to Releases
├── tests/
│   └── test_gguf_pipeline.py     # NEW — unit tests for convert_gguf.sh arg parsing + verify_gguf.py
│                                 #       (NOT an end-to-end conversion test — too slow/heavy)
└── gguf/                          # NOT recommended (do not commit GGUFs to repo; they live as Release assets)
```

### Pattern 1: GGUF conversion script — shell wrapper, not Python

**What:** A reproducible bash script that runs the two-stage pipeline (convert → quantize) with clear variable definitions and failure on any step.

**When to use:** Always, when the wrapped operation is primarily external CLI calls.

**Example:**
```bash
#!/usr/bin/env bash
# scripts/convert_gguf.sh — reproducible GGUF conversion pipeline (D-04)
# Usage: scripts/convert_gguf.sh <model_dir> <output_prefix>
# Requires: llama.cpp CLI tools (brew install llama.cpp) on $PATH; Python 3.10+
# Source: https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal/minicpmv4.5.md (pattern)
set -euo pipefail

MODEL_DIR="${1:-models/lyra-merged}"
OUTPUT_PREFIX="${2:-lyra-v1.0}"
OUTPUT_DIR="build/gguf"
mkdir -p "$OUTPUT_DIR"

# Step 1: HF -> f16 GGUF  (chat_template auto-embedded from tokenizer_config.json)
# convert_hf_to_gguf.py is the canonical name since ~2024; old convert-hf-to-gguf.py is a symlink in some builds
python "${LLAMA_CPP_DIR:-/opt/homebrew/share/llama.cpp}/convert_hf_to_gguf.py" \
    "$MODEL_DIR" \
    --outfile "$OUTPUT_DIR/${OUTPUT_PREFIX}-f16.gguf" \
    --outtype f16

# Step 2: f16 -> Q4_K_M
llama-quantize "$OUTPUT_DIR/${OUTPUT_PREFIX}-f16.gguf" \
    "$OUTPUT_DIR/${OUTPUT_PREFIX}-q4_k_m.gguf" Q4_K_M

# Step 3: f16 -> Q8_0
llama-quantize "$OUTPUT_DIR/${OUTPUT_PREFIX}-f16.gguf" \
    "$OUTPUT_DIR/${OUTPUT_PREFIX}-q8_0.gguf" Q8_0

# Step 4: Verify chat template embedded (D-07 non-negotiable)
python scripts/verify_gguf.py "$OUTPUT_DIR/${OUTPUT_PREFIX}-q4_k_m.gguf"
python scripts/verify_gguf.py "$OUTPUT_DIR/${OUTPUT_PREFIX}-q8_0.gguf"

echo "GGUF conversion complete:"
ls -lh "$OUTPUT_DIR/"*.gguf
```

### Pattern 2: GGUF metadata verification in Python

**What:** A small Python script that uses `gguf.GGUFReader` to assert metadata fields are present with expected content. Runs as part of the conversion pipeline and in CI.

**When to use:** Every time a GGUF is produced, before it leaves the build system.

**Example:**
```python
#!/usr/bin/env python3
# scripts/verify_gguf.py — verify GGUF metadata meets Lyra release requirements
# Source: https://context7.com llama.cpp /ggml-org/llama.cpp GGUFReader example
import sys
from pathlib import Path
from gguf import GGUFReader

REQUIRED_KEYS = ["tokenizer.chat_template", "general.architecture", "general.name"]

def main(gguf_path: str) -> int:
    reader = GGUFReader(gguf_path, mode="r")
    missing = []
    for key in REQUIRED_KEYS:
        if key not in reader.fields:
            missing.append(key)
    if missing:
        print(f"FAIL: {gguf_path} missing metadata keys: {missing}", file=sys.stderr)
        return 1
    chat_template = reader.fields["tokenizer.chat_template"].contents()
    if "<|im_start|>" not in chat_template:
        print(f"FAIL: {gguf_path} chat_template does not contain SmolLM2 markers <|im_start|>", file=sys.stderr)
        return 1
    print(f"OK: {gguf_path} — chat_template embedded ({len(chat_template)} chars)")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
```

### Pattern 3: Dataset version bundle — tag + release assets

**What:** Each dataset version is a git tag pointing at a commit that contains the assembled dataset artifacts (LFS-tracked), with a GitHub Release created from that tag attaching a zipped bundle and the pinned metrics artifacts.

**When to use:** Every dataset version bump (v1.0.0, v1.0.1, ...).

**Layout per version:**
- **In the repo at the tagged commit (LFS-tracked):**
  - `datasets/assembled/train/*.arrow` + `datasets/assembled/validation/*.arrow` + `datasets/assembled/test/*.arrow`
  - `datasets/assembled/dataset_dict.json`
  - `CHANGELOG.md` with the new version's entry at the top
  - `results/eval_*.json` (the evals used to compute the pinned metrics)
  - `BENCHMARK.md` (pinned snapshot)
- **Attached to the GitHub Release as discrete downloadable assets:**
  - `lyra-dataset-v1.0.0.tar.gz` — one-click bundle (entire `datasets/assembled/` + stats.json + BENCHMARK.md + CHANGELOG excerpt)
  - `lyra-dataset-v1.0.0-stats.json` — standalone stats for quick inspection without downloading the bundle
  - `BENCHMARK-v1.0.0.md` — standalone pinned snapshot
  - `lyra-v1.0-q4_k_m.gguf`, `lyra-v1.0-q8_0.gguf` — if this dataset version co-publishes a model release

### Pattern 4: LFS vs Release Assets — which file goes where

| File type | Location | Rationale |
|-----------|----------|-----------|
| `.gguf` quantized weights (~1–2 GB each) | **GitHub Release asset only** | Large, infrequent change. Users who don't need the model shouldn't pull them on clone. |
| `models/lyra-merged/*.safetensors` (3.2 GB) | **Neither in git nor released** | Exceeds 2 GiB per-file Release limit; reconstructable from `lyra-adapter` + base. Remains in `.gitignore`. |
| `datasets/assembled/**/*.arrow` + `dataset_dict.json` | **LFS-tracked in repo at the tagged commit** | Needs to be present in the tagged commit so `git checkout dataset-v1.0.0` gives a reproducible version. Total size ~67 MB — well under LFS limits. |
| `datasets/assembled/*.tar.gz` (bundle) | **GitHub Release asset only** | Convenience one-click download; do not commit archives to repo. |
| `BENCHMARK.md` (live) | **Plain repo file** | Small, human-readable. |
| `BENCHMARK-v1.0.0.md` (pinned snapshot) | **Both: in `docs/benchmarks/` LFS-tracked at tagged commit AND as Release asset** | Snapshot is immutable per CONTEXT.md `<specifics>`. |
| `results/eval_*.json` | **Plain repo file, tagged commit** | Small JSON, needs to be pinned to the version. |
| `CHANGELOG.md` | **Plain repo file, updated per release** | Human-edited. No LFS. |

### Anti-Patterns to Avoid

- **Committing GGUF files to the repo tree.** Even under LFS, every clone then pays the bandwidth cost. Release assets are the right home for user-facing binaries. Only exception would be if GitHub Releases becomes unusable, which it isn't.
- **Using Unsloth's `save_pretrained_gguf` without verifying chat-template embed.** Per Unsloth docs, chat template correctness is the caller's responsibility [CITED: unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf]. D-07 makes template embed non-negotiable, and llama.cpp's canonical path does it automatically by reading `tokenizer_config.json.chat_template`.
- **Hand-writing a "GGUF metadata patch" script.** `gguf_set_metadata.py` exists in llama.cpp's gguf-py; do not re-implement.
- **Cutting a release without first running the verify-gguf script.** The same class of bug that Phase 09.1 diagnosed (template drift between train/eval) can resurface in GGUF if the `tokenizer.chat_template` key is empty or wrong. Pattern 2 is the guard.
- **Treating dataset versioning as a "tag the repo" operation only.** The GitHub Release is what makes the version discoverable — tags alone don't show up on the Releases page.
- **Using CI to automate releases.** Explicitly deferred per `<deferred>`. Keep it manual for v1.
- **Re-generating BENCHMARK.md on every tag push.** Phase 10 pins the snapshot at release-cut time; later changes to the live BENCHMARK.md must not mutate pinned versions. Immutability per CONTEXT.md `<specifics>`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| HF model → GGUF conversion | Custom safetensors reader + GGUF writer | `llama.cpp/convert_hf_to_gguf.py` | 13,575+ LoC in canonical script; handles per-arch tensor mapping, vocab, metadata. Re-implementing is a months-long project. |
| Quantization (K-quants, etc.) | Custom int4/int8 rewriter | `llama-quantize` binary | Specialized per-tensor quantization with importance-matrix support. Don't touch. |
| GGUF metadata inspection | Custom binary parser | `gguf.GGUFReader` (Python) or `gguf-dump` (CLI) | Official `gguf-py` package [VERIFIED: PyPI gguf 0.18.0] handles endianness, versions, tensor catalogs. |
| Chat-template embed into GGUF | Custom metadata writer | **Leave it to `convert_hf_to_gguf.py`** which reads `tokenizer_config.json.chat_template` automatically | Zero code to write; just ensure the source tokenizer_config has the field. |
| SemVer parsing / validation | Regex in a shell script | `python -c "from packaging.version import Version; Version('1.0.0')"` | `packaging` is a transitive dependency of transformers; already installed. |
| CHANGELOG markdown format | Freeform | Keep-a-Changelog 1.1.0 [CITED: keepachangelog.com/en/1.1.0] | Well-known convention; plays nicely with SemVer; GitHub renders it. |
| Release asset upload | Manual web upload + `curl` | `gh release create TAG file1 file2 ...` | The `gh` CLI is a first-party GitHub tool; one-line release creation. |
| Perplexity measurement | Custom inference loop | `llama-perplexity -m model.gguf -f wiki.test.raw` | Official llama.cpp tool; KL-divergence option available for head-to-head quant comparison. |
| Dataset stats (token histogram, dedup rate) | `datasets.Dataset.map` with custom aggregators | Extend existing `scripts/assemble_dataset.py compute_stats()` | Function already exists; just add fields. No new module. |

**Key insight:** The release tooling is mostly **plumbing between existing tools**. The risk is in the plumbing (wrong filename conventions, missing verification steps, stale docs), not in the core conversion logic.

## Runtime State Inventory

> This phase creates new artifacts and introduces new infrastructure rather than renaming/refactoring existing state. Still, several categories deserve explicit documentation so the planner and future agents know what exists "in the wild" after Phase 10 execution.

| Category | Items Found | Action Required |
|----------|-------------|-----------------|
| Stored data | **None pre-existing.** After Phase 10: `datasets/assembled/**/*.arrow` (LFS-tracked under new `.gitattributes` rule); `CHANGELOG.md` at repo root; `BENCHMARK-v1.0.0.md` pinned snapshot (location TBD — recommend `docs/benchmarks/`). | Plan 1: author the files. Plan 3: populate the v1.0.0 pin. |
| Live service config | **None.** No HuggingFace namespace (D-03 defers that decision); no Gradio Space (D-02 removes); no hosted endpoints. | Nothing to migrate or reconfigure. |
| OS-registered state | **None.** No system services, no scheduled tasks, no daemons. Only user-invoked CLI scripts. | Nothing. |
| Secrets / env vars | **`GITHUB_TOKEN` (for `gh release create`)** — standard gh-CLI env var, read at release-cut time. **`HF_TOKEN`** — not needed per D-01. | README documents `GITHUB_TOKEN` setup in a runbook section. No new secrets introduced to .env. |
| Build artifacts | **New: `build/gguf/*.gguf` (gitignored; per-developer build output).** **Git LFS pointer files** replace the actual blobs for LFS-tracked content; `git lfs ls-files` currently returns nothing [VERIFIED]. After Phase 10, it will list `datasets/assembled/**` and possibly pinned BENCHMARK snapshots. | Plan 1: extend `.gitignore` with `build/` if not already present (spot-check: `build/` is not currently in `.gitignore`); extend `.gitattributes`; run `git lfs install` once per clone (already done per Phase 9 plan 09-01 STATE.md entry). |

**Nothing found in most categories:** Phase 10 is greenfield release engineering — it deposits new state rather than editing existing.

## Common Pitfalls

### Pitfall 1: Chat template not embedded or silently wrong

**What goes wrong:** A GGUF is shipped where `tokenizer.chat_template` metadata is missing, empty, or contains a generic template rather than the Lyra-trained one. LM Studio then applies its own default, and tool-call format regresses at runtime exactly like the Phase 09.1 D-03/D-04 eval regression.

**Why it happens:**
1. `convert_hf_to_gguf.py` reads `chat_template` from `tokenizer_config.json`. If that file has a null/missing field, the key is silently omitted from GGUF metadata.
2. `convert_hf_to_gguf.py` has **no `--chat-template` flag** to override — the template must come from the source model dir [VERIFIED: argparse inspection].
3. A future retrain might produce a merged model without the `chat_template` field in `tokenizer_config.json` (this was the exact Phase 09.1 D-04 bug until fixed).
4. LM Studio can auto-detect and override the chat template with a built-in one if its detection heuristic misfires [CITED: huggingface.co/Goekdeniz-Guelmez/.../discussions/1].

**How to avoid:**
1. **Precondition check in `convert_gguf.sh`:** before conversion, `jq -e '.chat_template' models/lyra-merged/tokenizer_config.json > /dev/null` or Python equivalent, fail loudly if missing.
2. **Postcondition check in `verify_gguf.py` (Pattern 2):** assert `tokenizer.chat_template` field exists in GGUF metadata and contains `<|im_start|>` SmolLM2 marker.
3. **Runbook instruction for users:** README includes the `gguf-dump --no-tensors` command so users can verify themselves; UAT checklist includes loading in LM Studio and running a tool-call prompt end-to-end.

**Warning signs:** LM Studio loads the model but tool-call outputs are malformed, regress vs base, or missing `<tool_call>` wrappers. `gguf-dump` shows no `tokenizer.chat_template` key.

### Pitfall 2: convert_hf_to_gguf.py doesn't produce Q4_K_M directly

**What goes wrong:** A plan task says "run convert_hf_to_gguf.py --outtype Q4_K_M" expecting a single command to produce Q4_K_M. It fails because Q4_K_M is not in the `--outtype` options.

**Why it happens:** The `--outtype` enum is `{f32,f16,bf16,q8_0,tq1_0,tq2_0,auto}` [VERIFIED: argparse source]. K-quants (Q4_K_M, Q5_K_M, Q6_K, etc.) are **quantization-only types** that are not valid conversion targets. They go through the separate `llama-quantize` tool that takes an f16 GGUF as input.

**How to avoid:** The `convert_gguf.sh` script does a two-stage pipeline: `f16` first via convert, then `llama-quantize` for each K-quant variant. Document the two-stage nature in the script comment.

**Warning signs:** Error message from convert_hf_to_gguf.py "argument --outtype: invalid choice: 'q4_k_m'".

### Pitfall 3: GitHub 2 GiB per-file release asset limit

**What goes wrong:** A task attempts to attach the full 3.2 GB `model.safetensors` or a ~2.5 GB combined bundle to a GitHub Release. The upload fails with HTTP 413 or a cryptic GitHub API error.

**Why it happens:** GitHub enforces **2 GiB per file** on Release assets [VERIFIED: docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github]. No total-size limit and no bandwidth limit, but per-file is hard.

**How to avoid:**
- **Do not attach `model.safetensors`** (per D-04/D-05 we only ship GGUFs — no safetensors on the release).
- Keep each GGUF variant as its own asset (Q4_K_M ~1 GB, Q8_0 ~1.8 GB — both fit).
- For the dataset bundle, check `du -sh datasets/assembled/` first. Current size is 67 MB [VERIFIED]; bundles will be well under the limit.
- If a future bundle ever approaches the limit, use `gh release upload` with multiple separate assets rather than one mega-archive.

**Warning signs:** `gh release create` fails; upload hangs past ~2 GB; GitHub returns HTTP 413.

### Pitfall 4: `.gitattributes` LFS rules don't apply retroactively

**What goes wrong:** A developer edits `.gitattributes` to add `*.gguf filter=lfs` but the files already in the repo history remain as regular blobs, bloating git objects.

**Why it happens:** Git LFS only intercepts files that are newly staged after the attribute matches. Existing tracked files stay as regular blobs unless `git lfs migrate import` is run.

**How to avoid:**
1. **Add LFS rules BEFORE the first GGUF is committed.** Plan 1 (prep, non-gated) creates the `.gitattributes` rules. Plan 2 (execution, gated) produces the GGUFs. This ordering is load-bearing.
2. For existing safetensors-like patterns the current repo doesn't contain (they're gitignored), we don't have a retroactive problem.
3. For `datasets/assembled/**/*.arrow` files that may currently be gitignored: verify `.gitignore` doesn't exclude them before adding an LFS rule. Current `.gitignore` has `datasets/assembled/` listed, so **both the `.gitignore` exclusion and the new LFS rule** must be reconciled — most likely remove `datasets/assembled/` from `.gitignore` and add an LFS rule for the Arrow files.

**Warning signs:** `git lfs ls-files` returns empty after a commit that should have been LFS-tracked; repo clone size balloons beyond expectation.

### Pitfall 5: Dataset version tag name collides with model-release tag

**What goes wrong:** A dataset version uses `v1.0.0` and a model release later also wants `v1.0.0`; the second `git tag` fails or overwrites.

**Why it happens:** Tags are a flat namespace in git. There's no built-in subsystem for "tag categories."

**How to avoid:** Namespace dataset tags with a prefix. Recommend **`dataset-v1.0.0`** (lowercase, hyphen-separated) because:
- Clear in `git tag -l` output
- Glob-friendly: `git tag -l 'dataset-v*'` lists only dataset versions
- Does not conflict with conventional `v1.0.0` tags that a model release might use
- Sorts lexically near `v*` but doesn't shadow

Model releases can then use `model-v1.0.0` or plain `v1.0.0`. CONTEXT.md `<decisions>` D-10/D-11 pin the dataset and model together at release time, so both tags will be cut together in practice (e.g., `dataset-v1.0.0` and `model-v1.0.0` on the same commit).

**Warning signs:** `git tag v1.0.0` error "already exists"; confusion in release page about which artifact a tag refers to.

### Pitfall 6: LM Studio overriding embedded chat template with a preset

**What goes wrong:** LM Studio loads the Lyra GGUF, detects a "chat template" from its catalog preset list (e.g., "ChatML"), and substitutes its own Jinja rather than the embedded `tokenizer.chat_template`. Outputs don't match eval-time formatting.

**Why it happens:** LM Studio attempts auto-detection from model name/GGUF metadata. For GGUFs whose chat template closely resembles a known preset, the override may engage [CITED: lmstudio-ai bug tracker issue 535; HuggingFace discussion with fix instructions].

**How to avoid:**
1. Include in README's UAT runbook: "After loading in LM Studio, navigate to My Models → model settings → Prompt Template and verify the Jinja content matches `models/lyra-merged/chat_template.jinja`."
2. Provide the exact chat template text in README as a reference the user can paste back in if LM Studio overrides it.
3. Test in UAT: run a tool-call prompt in LM Studio's chat UI; confirm output uses `<tool_call>...</tool_call>` wrappers.

**Warning signs:** Same prompt produces different outputs in `llama-cli` vs LM Studio; LM Studio doesn't emit `<tool_call>` wrappers.

## Code Examples

Verified patterns from official sources:

### Full conversion + quantization (documented example, SmolLM2-style)

```bash
# Source: llama.cpp docs/multimodal/minicpmv4.5.md (Context7 fetch) — generalized for SmolLM2
# Step 1: HF -> f16 GGUF (chat_template embedded automatically from tokenizer_config.json)
python llama.cpp/convert_hf_to_gguf.py models/lyra-merged \
    --outfile build/gguf/lyra-v1.0-f16.gguf \
    --outtype f16

# Step 2: f16 -> Q4_K_M (balanced quality/size)
./build/bin/llama-quantize build/gguf/lyra-v1.0-f16.gguf \
    build/gguf/lyra-v1.0-q4_k_m.gguf Q4_K_M

# Step 3: f16 -> Q8_0 (near-original quality)
./build/bin/llama-quantize build/gguf/lyra-v1.0-f16.gguf \
    build/gguf/lyra-v1.0-q8_0.gguf Q8_0
```

### Inspect GGUF metadata (CLI)

```bash
# Source: Context7 /ggml-org/llama.cpp llms.txt
gguf-dump build/gguf/lyra-v1.0-q4_k_m.gguf --no-tensors | head -40
# Or extract just the chat template:
gguf-dump build/gguf/lyra-v1.0-q4_k_m.gguf --no-tensors --json \
    | jq -r '.metadata["tokenizer.chat_template"].value'
```

### Inspect GGUF metadata (Python, for `tests/test_gguf_pipeline.py`)

```python
# Source: Context7 /ggml-org/llama.cpp llms.txt GGUFReader example
from gguf import GGUFReader

reader = GGUFReader("build/gguf/lyra-v1.0-q4_k_m.gguf", mode="r")
assert "tokenizer.chat_template" in reader.fields, "chat_template missing from GGUF metadata"
chat_template = reader.fields["tokenizer.chat_template"].contents()
assert "<|im_start|>" in chat_template, "SmolLM2 markers not found in embedded chat template"
```

### Perplexity sanity check (optional, Claude's discretion per CONTEXT.md)

```bash
# Source: knightli.com/en/2026/04/12/llama-quantize-gguf-guide (cross-verified with llama.cpp perplexity README)
# Run on a small held-out validation file (~100 samples extracted from datasets/assembled/validation/)
./build/bin/llama-perplexity -m build/gguf/lyra-v1.0-f16.gguf    -f val_samples.txt > perp_f16.txt
./build/bin/llama-perplexity -m build/gguf/lyra-v1.0-q4_k_m.gguf -f val_samples.txt > perp_q4km.txt

# Delta > 0.5 is a flag for catastrophic quantization loss; typical SmolLM2 Q4_K_M delta is ~0.1-0.3.
```

### Dataset stats JSON (extend existing `compute_stats`)

```python
# Source: scripts/assemble_dataset.py compute_stats() — extend with new fields
# Existing function already returns splits + domain breakdowns. Add:
def compute_stats(dataset_dict) -> dict:
    stats = {...}  # existing split/domain logic
    stats["schema_version"] = "1.0.0"
    stats["dataset_version"] = "1.0.0"  # matches the git tag dataset-v1.0.0
    stats["generated_at"] = datetime.now(timezone.utc).isoformat()
    # Token-length histogram per split (bucketed 0-500, 500-1000, 1000-2000, 2000+)
    # Dedup retention rate (sourced from curate_pipeline.py logs — document as "see CHANGELOG")
    # Format-validation pass rate (all samples pass Conversation.model_validate or assembly fails)
    return stats
```

### Create GitHub release with `gh` CLI

```bash
# Source: gh CLI docs (standard pattern)
# After git tag dataset-v1.0.0 -a -m "..." && git push --tags:
gh release create dataset-v1.0.0 \
    --title "Dataset v1.0.0" \
    --notes-file CHANGELOG-v1.0.0-excerpt.md \
    build/release/lyra-dataset-v1.0.0.tar.gz \
    build/release/lyra-dataset-v1.0.0-stats.json \
    build/release/BENCHMARK-v1.0.0.md

# For model release (typically same commit, paired tag):
gh release create model-v1.0.0 \
    --title "Model v1.0.0 (Lyra fine-tuned SmolLM2-1.7B)" \
    --notes "See CHANGELOG.md for dataset v1.0.0 entry. Uses dataset v1.0.0 under MIT." \
    build/gguf/lyra-v1.0-q4_k_m.gguf \
    build/gguf/lyra-v1.0-q8_0.gguf
```

### Keep-a-Changelog format (dataset entries)

```markdown
# Changelog

All notable changes to the Lyra dataset are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [Dataset v1.0.0] - 2026-04-XX

### Added
- Initial release of the 25K curated Lyra training dataset following the Phase 09.1
  retraining and data expansion (tool-calling: 23K, code: ~650, knowledge: ~650).
- Stratified 90/5/5 train/validation/test splits.
- Dataset stats JSON (`lyra-dataset-v1.0.0-stats.json`) with per-split domain
  breakdown and token-length histogram.
- Pinned model-evaluation metrics: see `BENCHMARK-v1.0.0.md` for the model trained
  on this dataset version.

### Changed
- Tool-calling dedup threshold increased from 0.7 to 0.9 (full-scope) to retain more
  template diversity at 25K scale.
- Domain balance shifted to ~90.2% tool-calling, 4.9% code, 4.9% knowledge (from the
  earlier Phase 7 ~33/33/33 assumption) to support Phase 09.1 regression investigation.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `convert-hf-to-gguf.py` (hyphenated) | `convert_hf_to_gguf.py` (underscored) | Renamed ~2024 in llama.cpp; hyphenated form now is a symlink in some builds | Scripts using old name may still work via symlink, but canonical is underscore |
| Phase 09.1 D-03 bug (tokenizer.chat_template stripped during merge) | Inline persistence of chat template inside tokenizer_config.json during `train.py` merge (D-04 fix) | Phase 09.1 | GGUF conversion relies on this field being present; D-07 is downstream of this fix |
| Unsloth's `save_pretrained_gguf` as one-liner | Two-stage llama.cpp pipeline with explicit verification | Phase 10 decision (D-04) | More transparency, explicit chat-template embed check; Unsloth is fine but opaque for this safety-critical pipeline |
| GitHub + HuggingFace dual publishing (implicit pre-Phase 9) | GitHub-only (D-01 / Phase 9 D-07) | Phase 9 and extended through Phase 10 | No HF Model Hub, no HF Dataset repo, no HF Space. Simplifies release surface. |

**Deprecated / outdated:**
- The older `convert-hf-to-gguf.py` filename is soft-deprecated; the canonical name is `convert_hf_to_gguf.py`.
- Any code or documentation that assumes `tokenizer_config.json` may not have `chat_template` populated — Lyra has institutionalized (Phase 09.1 D-04) that it must be inline.
- Early "ShareGPT `from`/`value`" format references in older READMEs — TRL-native `messages`/`role`/`content` is canonical per Phase 1 STATE.md decision.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Q8_0 file size for SmolLM2-1.7B is approximately 1.8 GB (not verified by producing the file; estimated from ~1.06 GB Q4_K_M and relative bpw ratio) [CITED: SmolLM2-1.7B-Instruct-GGUF] | Standard Stack §Core | Low — even if Q8_0 runs to ~1.9 GB, it's still under the 2 GiB GitHub Release limit. No flow breakage. |
| A2 | `/opt/homebrew/share/llama.cpp/convert_hf_to_gguf.py` is the brew-installed path of the script | Pattern 1 example | Medium — if brew ships only the compiled binaries (`llama-quantize`, `llama-cli`) and not the Python script, `convert_gguf.sh` needs to locate the script elsewhere (user-installed llama.cpp source clone). Verify at Plan 1 implementation time. |
| A3 | Token-length histogram is a reasonable default field for stats JSON (not explicitly requested by CONTEXT.md) | Code Examples §Dataset Stats JSON | Low — it's additive. If not wanted, drop the field; nothing else depends on it. |
| A4 | "datasets/assembled/**/*.arrow" LFS tracking is the right pattern for the versioned bundle | Pattern 4 | Medium — alternative is to skip LFS for the tagged-commit files and rely on Release-only attachments. LFS is recommended because it keeps `git checkout dataset-v1.0.0` reproducible without hitting the Release API. |
| A5 | `gh` CLI is installed on the developer's machine at release time | Code Examples §Create GitHub release | Low — `gh` is a standard install on most dev machines; if missing, the release can be cut manually via the web UI. Document both options in README runbook. |
| A6 | llama.cpp b8913 (latest at 2026-04-24) correctly handles SmolLM2 architecture | Standard Stack | Low — SmolLM2 uses `LlamaForCausalLM` architecture [VERIFIED: models/lyra-merged/config.json line 3]; this has been supported in llama.cpp since ~2023. Standard conversion path. |
| A7 | The dataset JSONL files produced for the bundle are TRL-native format (messages/role/content), matching what Phase 7 assembled | Pattern 3 | Low — `assemble_dataset.py` produces exactly this format [VERIFIED: source inspection]. The bundle is just `save_to_disk` output, so the format is whatever the Arrow files contain. |

## Open Questions (RESOLVED)

> All five open questions below have been answered via the "Recommendation:" paragraphs that follow each. The 4 PLAN.md files implement the recommended choices: (Q1) brew-default with `LLAMA_CPP_DIR` fallback, (Q2) `release/dataset-v1.0.0/BENCHMARK-v1.0.0.md` as the pinned snapshot path, (Q3) a single combined `CHANGELOG.md` with versioned sections, (Q4) extended `stats` subcommand with `--json --output` flags on existing `scripts/assemble_dataset.py`, (Q5) the 4-plan split (1 non-gated prep + 2 wave-2 releases + 1 wave-3 docs/UAT).

1. **Is `llama.cpp` brew formula sufficient, or should the project vendor a pinned source checkout?**
   - What we know: brew ships stable 8890; source-build gives commit-SHA reproducibility.
   - What's unclear: whether brew's bottled binary includes the Python `convert_hf_to_gguf.py` script in `share/llama.cpp/`, or only the compiled CLI tools.
   - Recommendation: **Plan 1 Task "verify llama.cpp install path"** does `brew install llama.cpp` on the dev machine, then greps for `convert_hf_to_gguf.py` in the install prefix. If missing, fall back to "clone llama.cpp at tag bXXXX" documented in README.

2. **Where does the pinned BENCHMARK.md snapshot live in the repo?**
   - What we know: BENCHMARK.md (live) is at repo root. Each dataset version pins a snapshot.
   - What's unclear: Whether pinned snapshots go in `docs/benchmarks/BENCHMARK-v1.0.0.md`, or directly in the Release asset only, or both.
   - Recommendation: **Both.** Keep the Release asset for discoverability; also keep a `docs/benchmarks/` directory in the repo for `git checkout dataset-v1.0.0 -- docs/benchmarks/BENCHMARK-v1.0.0.md` reproducibility. This mirrors the LFS rationale from Pattern 4.

3. **Should the `CHANGELOG.md` cover only dataset releases, or also model releases?**
   - What we know: D-10/D-11 couple dataset + model release at the same tag commit.
   - What's unclear: Whether one CHANGELOG.md handles both (with `[Dataset v1.0.0]` and `[Model v1.0.0]` as separate headers) or two files (`CHANGELOG.md` + `MODEL-CHANGELOG.md`).
   - Recommendation: **One combined CHANGELOG.md with versioned sections.** Simpler, one place to look, aligns with Keep-a-Changelog convention.

4. **Does `scripts/assemble_dataset.py stats` already emit JSON to a file, or only to stdout?**
   - What we know: Current `print_stats` formats a human-readable table to stdout. `compute_stats` returns a dict.
   - What's unclear: Whether Plan 1 extends the existing `stats` subcommand with a `--output` flag, or creates a new `dataset-stats-json` subcommand.
   - Recommendation: **Add `--json` and `--output PATH` flags to the existing `stats` subcommand.** Minimal change; preserves existing human-readable behavior as default.

5. **Phase 09.1/09.2 gating strategy — full-halt vs split plans.**
   - What we know: D-06 requires Phase 10 execution only after 09.1/09.2 weights satisfy the tool-call-format criterion.
   - Recommendation (Plan structure): **Split.** Plan 1 (non-gated prep: `.gitattributes`, CHANGELOG template, `scripts/convert_gguf.sh` + `verify_gguf.py` skeletons with unit tests, dataset stats augmentation, README scaffolding, REQUIREMENTS.md/ROADMAP.md edits per D-02). Plans 2–4 (gated on 09.1/09.2): conversion execution, dataset v1.0.0 release, documentation + UAT. The split is the release-engineering norm: ship infrastructure ahead of the artifact so the artifact cut is a one-command operation.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| `git-lfs` | LFS tracking of GGUF, Arrow files | ✓ | 3.7.1 [VERIFIED: `git lfs --version`] | — |
| `git` | Tag creation, release metadata | ✓ | (system) | — |
| Python 3.10+ | `convert_hf_to_gguf.py`, `verify_gguf.py`, `scripts/assemble_dataset.py` | ✓ | (venv) | — |
| `pytest` | Unit tests for scripts | ✓ | installed in venv | — |
| `gguf` (PyPI) | `GGUFReader` for verify script and tests | ✗ (not in requirements.txt) | 0.18.0 [VERIFIED: PyPI] | Install via `pip install gguf==0.18.0`; add to requirements.txt in Plan 1 |
| `llama.cpp` CLI tools (`llama-quantize`, `llama-perplexity`, `gguf-dump`) | GGUF conversion + quantization + inspection | ✗ | brew stable 8890 available [VERIFIED: `brew info llama.cpp`] | Install via `brew install llama.cpp` or build from source (documented in README) |
| `convert_hf_to_gguf.py` script | f16 GGUF production | ✗ | Part of llama.cpp source tree | Requires either brew install (provides script under `/opt/homebrew/share/llama.cpp/`) or `git clone https://github.com/ggml-org/llama.cpp` |
| `gh` (GitHub CLI) | Release creation | Likely ✓ | system (not checked) | Web UI release creation documented in README as fallback |
| `jq` | Pretty-print JSON in runbook / manual inspection | Likely ✓ (system) | — | Optional — `python -m json.tool` is a portable fallback |
| `wikitext` validation file (for `llama-perplexity`) | Optional post-quant sanity check | ✗ (not in repo) | — | Generate on-demand from `datasets/assembled/validation/` sample extracts; perplexity check is optional per CONTEXT.md discretion |

**Missing dependencies with no fallback:**
- None — everything has either a primary install path or a documented workaround.

**Missing dependencies with fallback:**
- `gguf` Python package — add to `requirements.txt`; pip install covers it.
- `llama.cpp` CLI tools — brew install (macOS) or source build (any platform), both documented in README.
- `convert_hf_to_gguf.py` script — installed via brew or source clone.

**Key pre-flight check for Plan 1:** Before Plan 2 execution is unblocked, verify:
```bash
command -v llama-quantize && llama-quantize --help 2>&1 | head -5 \
    && command -v gguf-dump \
    && python -c "import gguf; print(gguf.__version__)" \
    && [ -f "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" ]
```

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 7.x+ [VERIFIED: `pytest.ini`, `requirements.txt`] |
| Config file | `/Users/lakshman/Documents/Lyra/pytest.ini` (`testpaths = tests`) |
| Quick run command | `pytest tests/test_gguf_pipeline.py tests/test_release_artifacts.py -x` |
| Full suite command | `pytest` |
| Phase 10 test file (NEW) | `tests/test_gguf_pipeline.py` |
| Existing suite to extend | `tests/test_release_artifacts.py` (currently 1 RED test: `test_gitattributes_lfs`) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| REL-05 | `scripts/convert_gguf.sh` parses args and runs without error in dry-run mode | unit | `pytest tests/test_gguf_pipeline.py::test_convert_gguf_dry_run -x` | ❌ Wave 0 |
| REL-05 | `scripts/verify_gguf.py` exits 0 when GGUF has tokenizer.chat_template; exits 1 when missing | unit | `pytest tests/test_gguf_pipeline.py::test_verify_gguf_accepts_good_metadata -x` and `::test_verify_gguf_rejects_missing_template` | ❌ Wave 0 |
| REL-05 | GGUF produced by pipeline has `tokenizer.chat_template` field containing `<|im_start|>` | integration / manual smoke | Human runbook step (fits in Plan 4 UAT); pytest version gated on llama.cpp install and a real conversion (slow, `@pytest.mark.slow`) | ❌ Wave 0 (stub only) |
| REL-05 | GGUF loads in llama-cli and produces sensible output for a tool-call prompt | manual UAT | Human runbook; not automatable in CI | ❌ Wave 0 (runbook in Plan 4) |
| REL-05 | GGUF loads in LM Studio and produces sensible output for a tool-call prompt | manual UAT | Human runbook; not automatable | ❌ Wave 0 (runbook in Plan 4) |
| REL-07 | `.gitattributes` tracks `*.gguf` under LFS | unit | `pytest tests/test_release_artifacts.py::test_gitattributes_lfs -x` (extend existing test to check *.gguf too) | ✓ (extend existing RED test) |
| REL-07 | `.gitattributes` tracks `datasets/assembled/**/*.arrow` under LFS | unit | `pytest tests/test_release_artifacts.py::test_gitattributes_lfs_arrow -x` (new) | ❌ Wave 0 |
| REL-07 | `scripts/assemble_dataset.py stats --json --output` produces a JSON file conforming to `DatasetStatsSchema` (Pydantic) | unit | `pytest tests/test_assemble_dataset.py::test_stats_json_output -x` | ✓ (extend existing test_assemble_dataset.py) |
| REL-07 | Dataset version tag format (`dataset-vMAJOR.MINOR.PATCH`) is validated | unit | `pytest tests/test_gguf_pipeline.py::test_dataset_tag_format -x` (or a dedicated `test_versioning.py`) | ❌ Wave 0 |
| REL-07 | CHANGELOG.md exists and contains v1.0.0 section with `### Added` at repo root | unit | `pytest tests/test_release_artifacts.py::test_changelog_exists -x` (new) | ❌ Wave 0 |
| REL-07 | `datasets/README.md` has a "Dataset Versions" section with link to Releases | unit | Extend `tests/test_release_artifacts.py::test_dataset_card` | ✓ (extend existing) |
| REL-06 | Moved to Out of Scope per D-02 (no test; REQUIREMENTS.md change is a docs-edit task) | docs-only | Grep verification that REQUIREMENTS.md has REL-06 in Out of Scope and Traceability marked "Out of Scope" | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `pytest tests/test_gguf_pipeline.py tests/test_release_artifacts.py tests/test_assemble_dataset.py -x` (< 5 seconds)
- **Per wave merge:** `pytest` (full suite — ~15 seconds based on STATE.md Phase 8/9 history)
- **Phase gate:** Full suite green AND manual UAT runbook checklist complete before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `tests/test_gguf_pipeline.py` — new file, covers REL-05 unit-testable behaviors and the version-tag format validator. Uses mocks/fixtures for GGUF byte patterns rather than real conversion.
- [ ] `tests/test_release_artifacts.py` — extend with `test_gitattributes_lfs` (update to also require `*.gguf`), `test_gitattributes_lfs_arrow`, `test_changelog_exists`, and extend `test_dataset_card` for the Dataset Versions section.
- [ ] `tests/test_assemble_dataset.py` — extend with `test_stats_json_output` (use an in-memory fixture DatasetDict; do not require real `datasets/assembled/` on disk).
- [ ] `scripts/verify_gguf.py` — new script.
- [ ] `scripts/convert_gguf.sh` — new script (a shell script; tests should exercise its arg parsing via subprocess invocation or by extracting parse logic into a Python helper).
- [ ] `requirements.txt` — add `gguf==0.18.0`.

*(Wave 0 convention per STATE.md Phase 9 practice: tests are added in RED state in the prep plan and turned GREEN by the implementation tasks.)*

## Security Domain

> `security_enforcement` is not explicitly disabled in `.planning/config.json`; treating as enabled.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no | No user-facing auth surface in this phase |
| V3 Session Management | no | No sessions |
| V4 Access Control | no | No multi-user access |
| V5 Input Validation | **yes** | Any script that takes CLI input validates: model dir path (existing pattern in `scripts/eval_runner.py::_validate_model_path`), output filename, version string (SemVer regex) |
| V6 Cryptography | **yes (indirect)** | Release asset integrity — recommend publishing SHA256 of each GGUF in the Release notes; users verify with `shasum -a 256` |
| V10 Malicious Code | **yes** | llama.cpp binary must come from a trusted channel (brew or `https://github.com/ggml-org/llama.cpp` official); document the install source in README |
| V12 File Uploads / Files | **yes** | Subprocess calls to `llama-quantize`, `convert_hf_to_gguf.py`, `llama-perplexity` use list-form (no shell=True), consistent with `scripts/eval_runner.py` pattern T-03-05; output paths validated to prevent traversal |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Command injection via filename args | Tampering | Pass arguments as a list, never via shell; validate paths against `MODEL_PATH_PATTERN` regex (existing pattern in `scripts/eval_runner.py` T-03-07); reject paths containing shell metacharacters |
| Path traversal in output filename | Tampering | Validate output path is under `build/gguf/` or user-specified absolute path; reject `..` components |
| GGUF file with malicious metadata RCE | Tampering / Elevation | The "GGUF RCE" class [CITED: 0reg.dev/blog/from-gguf-model-format-metadata-rce-to-state-of-the-art-nlp-project-rces] was patched upstream; keep llama.cpp pinned to a recent stable build. Do not load GGUFs from untrusted sources. This phase only loads our own just-produced GGUFs. |
| Supply-chain attack on llama.cpp binary | Tampering | Document that users should `brew install llama.cpp` (brew formulae are signed) or clone from the official `ggml-org/llama.cpp` GitHub repo (not a fork) |
| Release asset tampering (published artifact swapped) | Tampering | Publish SHA256 of each GGUF in the Release notes; GitHub Release pages are maintainer-only edit, so attack surface is the maintainer's account (covered by 2FA outside this phase) |
| Chat-template tampering causing silent eval regression | Tampering | Post-conversion `verify_gguf.py` asserts template markers present (Pitfall 1 mitigation); CI test runs the verify script. This is also an **integrity** control for V6. |
| Denial of service via oversized input file | Denial of Service | `convert_hf_to_gguf.py` loads the safetensors into memory; if a malicious 100 GB safetensors is passed it could OOM. Current pattern validates directory existence; add an explicit size check: refuse if the safetensors exceeds a configured upper bound (e.g., 10 GB — Lyra's is 3.2 GB). |

**Threat-model cross-cuts specific to this phase:**
- **Subprocess surface:** The new `scripts/convert_gguf.sh` runs external binaries. CLAUDE.md doesn't mandate specific subprocess hardening, but the existing `scripts/eval_runner.py` already enforces list-form subprocess calls and path validation (T-03-05, T-03-07). Phase 10 scripts must follow the same pattern.
- **Dataset-bundle integrity:** The GitHub Release asset is the authoritative bundle; the git-tagged LFS-tracked files are the reproducibility guarantee. Publish SHA256 of both in the Release notes so a user can verify.

## Project Constraints (from CLAUDE.md)

The project CLAUDE.md contains no destructive/forbidden-patterns directives that constrain Phase 10 — it is primarily technology-stack and architecture guidance, most of which is already covered in the Standard Stack. Actionable constraints cross-referenced here:

- **Base model: SmolLM2-1.7B** — fixed. All GGUFs derive from a SmolLM2-1.7B-architecture merged model. [CONSTRAINT]
- **License: MIT** — all new scripts and artifacts carry `# SPDX-License-Identifier: MIT` headers per existing project pattern (visible in `scripts/assemble_dataset.py`, `scripts/eval_compare.py`). [CONSTRAINT]
- **Data format: TRL-native messages/role/content** — the dataset versioning bundle preserves this format; the DatasetDict is not transformed for release. [CONSTRAINT]
- **Scale strategy: iterative** — versioning scheme must support iterative dataset growth (SemVer supports this directly). [ALIGNED]
- **No parallel model evaluations** (from user memory) — the Validation Architecture section above does not run evaluations in Phase 10; it only references pinning existing eval outputs produced in Phase 9 / 09.1 / 09.2. [NO CONFLICT]
- **GSD workflow enforcement** — all file edits go through GSD commands. Phase 10 execution follows the standard phase workflow. [ALIGNED]

Additional convention cross-references:
- **Flat `scripts/` directory with argparse CLI entry points** — new scripts conform (`scripts/verify_gguf.py` uses argparse; `scripts/convert_gguf.sh` is a shell script with documented `$1/$2` positional args). [ALIGNED]
- **JSON for machine-readable outputs, Markdown for human-readable** — stats JSON + BENCHMARK snapshot MD bundle follows this split. [ALIGNED]
- **Lazy imports of heavy ML deps inside functions** — not directly relevant here (no ML deps in the verify or release scripts; `gguf` is lightweight). `scripts/assemble_dataset.py` already uses this pattern for `datasets`. [ALIGNED]
- **Reproducible-from-repo discipline** — every pipeline stage has a script. GGUF conversion is one script (`convert_gguf.sh`), release bundle assembly is another (`release_bundle.sh` — optional). README documents both. [ALIGNED]
- **Git tags + GitHub Releases as external distribution** — extends to both dataset and model releases per D-10. [ALIGNED]

## Sources

### Primary (HIGH confidence)
- `/ggml-org/llama.cpp` via Context7 — quantize commands, GGUFReader Python API, gguf-dump CLI, chat_template metadata key (`tokenizer.chat_template`) [VERIFIED]
- `https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_hf_to_gguf.py` (argparse tail) — exhaustive CLI argument list confirming `--outtype` enum and absence of `--chat-template` flag [VERIFIED]
- `https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github` — GitHub file size limits (100 MiB regular, 2 GiB Release asset) [VERIFIED]
- `https://keepachangelog.com/en/1.1.0/` — Keep-a-Changelog format spec [CITED]
- Lyra repo files (direct read) — `.gitattributes`, `.gitignore`, `models/lyra-merged/`, `scripts/assemble_dataset.py`, `scripts/eval_compare.py`, `scripts/eval_merge.py`, `scripts/eval_runner.py`, `BENCHMARK.md`, `datasets/README.md`, `README.md`, `pytest.ini`, `tests/test_release_artifacts.py` [VERIFIED via Read tool]
- Lyra venv state — `gguf 0.18.0`, `llama-cpp-python 0.3.20` (current PyPI) [VERIFIED via PyPI lookup]
- System state — `git lfs 3.7.1` installed; `llama.cpp` available via brew stable 8890; `models/lyra-merged/model.safetensors` is 3.2 GB [VERIFIED via Bash]

### Secondary (MEDIUM confidence)
- `https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF` — reference Q4_K_M size (1.06 GB) for SmolLM2-1.7B [CITED]
- `https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf` — Unsloth GGUF save path, chat-template warning [CITED]
- `https://github.com/ggml-org/llama.cpp/discussions/7927` — HF → GGUF tutorial (invocation syntax) [CITED]
- `https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md` — llama-quantize invocation pattern [CITED]
- `https://github.com/abetlen/llama-cpp-python/issues/1096` — chat_template metadata availability in GGUF [CITED]

### Tertiary (LOW confidence — verification-flagged)
- Assertion that brew ships `convert_hf_to_gguf.py` under `/opt/homebrew/share/llama.cpp/` — **verify at Plan 1 implementation**; fallback to source clone documented.
- `https://www.knightli.com/en/2026/04/12/llama-quantize-gguf-guide/` — third-party tutorial; cross-verified invocation against official llama.cpp quantize README.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries version-verified on PyPI; llama.cpp tooling confirmed via Context7 and direct argparse source read.
- Architecture: HIGH — patterns derive directly from existing Lyra conventions (flat scripts/, argparse CLIs, SPDX headers) and existing Phase 9 release infrastructure.
- Pitfalls: HIGH — Pitfall 1 (chat-template) backed by Phase 09.1 regression; Pitfalls 3 (size limits) and 5 (tag naming) backed by documentation; Pitfall 2 (outtype) verified directly in source; Pitfall 6 (LM Studio) backed by community bug reports.
- Security: MEDIUM — general ASVS mapping is standard; specific threat (GGUF RCE) referenced to a security blog; recommend planner reviews at plan-check time against current llama.cpp security advisories if any have been published since research date.

**Research date:** 2026-04-24
**Valid until:** 2026-05-24 (llama.cpp releases rapidly — build numbers may shift weekly; re-verify pinned version at release-cut time)
