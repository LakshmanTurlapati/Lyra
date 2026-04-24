# Phase 10: Community Release Enhancements - Context

**Gathered:** 2026-04-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Ship GGUF quantized variants of the released Lyra model for local runtimes (LM Studio, llama.cpp) and establish a versioning scheme for dataset releases with pinned per-version metrics. Covers REL-05 and REL-07 only — REL-06 (interactive Gradio demo Space on HuggingFace) is dropped to Out of Scope during this discussion.

Phase 10 blocks on Phase 09.1/09.2: the artifacts released here are the Lyra weights that satisfy the tool-call-format success criterion (beat base 0.4065 or documented revert-to-base). No community-facing release of a known-regressed model.

</domain>

<decisions>
## Implementation Decisions

### Publishing Scope
- **D-01:** GitHub + Git LFS is the only distribution channel. No HuggingFace Model Hub weights, no HF dataset repo, no HF Space, no HF-hosted GGUF. Extends Phase 9 D-07 ("No HuggingFace publishing for now") through the community release. All artifacts live on GitHub and are pulled via clone or GitHub Releases.
- **D-02:** REL-06 (interactive Gradio demo Space on HuggingFace) is removed from v1. It moves to the `Out of Scope` table in REQUIREMENTS.md with reason "Community release is GitHub-native; interactive demo not pursued in v1." A corresponding ROADMAP.md update drops the Gradio Space from Phase 10's goal and success criteria.
- **D-03:** HuggingFace namespace decision is not made here. If REL-06 ever returns, namespace (personal account vs `lyra` org) is decided at publish time, not during planning.

### GGUF Pipeline (REL-05)
- **D-04:** Conversion toolchain is canonical llama.cpp: `convert_hf_to_gguf.py` produces f16 GGUF from `models/lyra-merged/`, then `llama-quantize` produces the shipped variants. The full pipeline is wrapped in a reproducible shell script under `scripts/` so the conversion is one command from the repo.
- **D-05:** Quantization levels shipped: **Q4_K_M and Q8_0 only** — exactly what REL-05 specifies. No additional levels (Q5_K_M, Q6_K, etc.) in this phase; keeps the LFS footprint minimal and avoids a sprawling per-variant README matrix.
- **D-06:** Model variant quantized: the weights from Phase 09.1/09.2 that satisfy the tool-call-format success criterion (beats base SmolLM2-1.7B's 0.4065 on tool-call-format, or a documented revert-to-base release). Phase 10 does not begin until those weights exist. Current pre-09.1 `models/lyra-merged/` is not shipped to the community.
- **D-07:** SmolLM2 chat template is embedded into GGUF metadata at convert time via `convert_hf_to_gguf.py --chat-template-config` (or equivalent supported flag) pointing at `models/lyra-merged/chat_template.jinja`. llama.cpp / LM Studio then applies the exact template used during training and evaluation. This closes the same class of runtime template-drift bug that caused the Phase 09.1 D-03/D-04 regressions.

### Dataset Versioning (REL-07)
- **D-08:** Versioning scheme is **SemVer** (`major.minor.patch`). Major = schema / format-breaking change. Minor = dataset re-curation or domain re-balance with the same schema. Patch = sample-level fixes or metadata corrections. Reads like software and maps cleanly to git tags + GitHub Releases.
- **D-09:** Initial release is **v1.0.0 = the current 25K post-09.1 assembled dataset** — the dataset the released model trains on. No retroactive v0 for the original ~5K Phase 7 assembly; that's history, not a versioned release.
- **D-10:** Version bundle contents (full bundle): train/validation/test JSONL from `datasets/assembled/`, a dataset stats JSON (produced by `scripts/assemble_dataset.py`), a pinned snapshot of BENCHMARK.md for that version, and a CHANGELOG entry. All files are attached to a git tag + GitHub Release so the version is reproducible from the tag alone.
- **D-11:** Metrics per version are **pinned to the model evals of the model trained on that dataset version**. A new dataset version triggers a new training run + eval run; the resulting `results/eval_*.json` files and the generated BENCHMARK.md snapshot are attached to that version's release. "Metrics per version" = "how did the model trained on this dataset perform." Matches the project's existing eval → compare → BENCHMARK.md workflow.

### Claude's Discretion
- Exact shell script name and structure for the GGUF conversion pipeline (e.g., `scripts/make_gguf.sh` vs `scripts/convert_gguf.py`).
- GGUF file-naming convention (e.g., `lyra-v1.0-q4_k_m.gguf`, `lyra-1.0.0-Q4_K_M.gguf`, or similar).
- Whether each GGUF variant ships with its own short README or a single combined README for the GGUF directory.
- CHANGELOG.md file location and format — project has no CHANGELOG yet; structure (Keep-a-Changelog style vs freeform) is Claude's call.
- Dataset-version git tag naming (e.g., `dataset-v1.0.0`, `data-1.0.0`, `v1.0.0-dataset`). Must not collide with any future model-release tag scheme.
- Additions to `.gitattributes` to track `*.gguf` under Git LFS.
- Where inside `scripts/` the GGUF pipeline lives and how it's invoked from the repo README.
- llama.cpp version pinning strategy (specific commit SHA, release tag, or "latest supported").
- Optional post-quantization perplexity sanity check before publishing a GGUF.

### Folded Todos
None — no pending todos matched this phase's scope.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 10 scope source
- `.planning/REQUIREMENTS.md` §Release — REL-05, REL-07 (in scope); REL-06 (moving to Out of Scope via D-02).
- `.planning/ROADMAP.md` §"Phase 10: Community Release Enhancements" — goal + success criteria. Must be updated to drop the Gradio Space bullet per D-02.
- `.planning/PROJECT.md` §Constraints — MIT license, SmolLM2-1.7B base, GitHub-native distribution posture.

### Prior-phase publishing stance
- `.planning/phases/09-benchmarking-and-core-release/09-CONTEXT.md` — Phase 9 D-07 "No HuggingFace publishing for now" and D-10 "Use Git LFS for large files." D-01 here extends both through community release.

### Model weights to quantize (source artifacts)
- `models/lyra-merged/` — full merged safetensors model (the target of `convert_hf_to_gguf.py`). Exact contents will be the Phase 09.1/09.2 outcome weights, per D-06.
- `models/lyra-merged/chat_template.jinja` — SmolLM2 chat template embedded into GGUF metadata at convert time per D-07.
- `models/lyra-merged/tokenizer.json`, `models/lyra-merged/tokenizer_config.json`, `models/lyra-merged/config.json` — read by `convert_hf_to_gguf.py`.
- `models/lyra-adapter/` — LoRA adapter (reference only; GGUF is produced from the merged model, not the adapter).

### Phase 09.1/09.2 dependency
- `.planning/phases/09.1-tool-call-format-regression-fix/` — pending plans 09.1-05 (retrain) and 09.1-06 (re-eval) must produce weights meeting the tool-call-format success criterion, or corrective action per 09.2 must complete, before Phase 10 planning begins in earnest.
- `.planning/phases/09.2-tool-call-regression-diagnosis/` — diagnostic + corrective-action work that gates D-06.

### Dataset versioning source artifacts
- `datasets/assembled/` — the HuggingFace DatasetDict (train/validation/test) that becomes v1.0.0 per D-09.
- `datasets/README.md` — existing dataset card; update per version release.
- `scripts/assemble_dataset.py` — produces the dataset stats JSON that ships in the bundle per D-10.

### Metrics for dataset-version bundles
- `BENCHMARK.md` (repo root) — canonical benchmark report; a pinned snapshot ships with each dataset-version release per D-10/D-11.
- `scripts/eval_runner.py`, `scripts/eval_inference.py`, `scripts/eval_compare.py`, `scripts/eval_merge.py` — the eval toolchain that produces the metrics attached to each version per D-11.
- `configs/eval.yaml` — benchmark task lists / few-shot settings used for those metrics.

### Existing release infrastructure (Phase 9)
- `.gitattributes` — already configured for Git LFS tracking of safetensors and large JSONL; must be extended for `*.gguf` per D-05.
- `README.md` (repo root) — current model card; GGUF variant section added per D-04/D-05.
- `LICENSE` — MIT license to propagate to any new artifacts introduced in Phase 10.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/assemble_dataset.py` — already has a stats command; its output is the dataset-stats JSON in each version bundle (D-10).
- `scripts/eval_runner.py` + `scripts/eval_compare.py` + `scripts/eval_merge.py` — the eval pipeline that produces per-version model metrics (D-11).
- `models/lyra-merged/chat_template.jinja` — the exact training/eval chat template; GGUF embeds it at convert time (D-07).
- `.gitattributes` + Git LFS — already configured from Phase 9 D-10; extend with a `*.gguf` filter pattern.
- `BENCHMARK.md` generator from Phase 9 plan 09-03 (`eval_compare.py --markdown`) — produces the pinned metrics snapshot attached to each dataset-version bundle.

### Established Patterns
- Flat `scripts/` directory with argparse CLI entry points — the GGUF conversion pipeline follows this convention (a single invocable script, not a scattered workflow).
- JSON for machine-readable outputs, Markdown for human-readable ones.
- Reproducible-from-repo discipline: every pipeline stage has a script that a user can run. GGUF conversion must hold this standard (script under `scripts/`, not manual invocation).
- Git tags + GitHub Releases as the external-surface distribution channel (extends to dataset versioning per D-10).
- Lazy imports of heavy ML deps inside functions (established in Phase 3 / Phase 8 for eval + train scripts) — apply the same pattern to any Python wrapper around the GGUF pipeline.

### Integration Points
- Phase 09.1/09.2 is an upstream hard dependency: Phase 10 inputs are the weights Phase 09.1/09.2 produces. Planner must gate Phase 10 start on that outcome (D-06).
- The CHANGELOG.md file does not exist yet; Phase 10 creates it. Its placement (repo root) and format are Claude's discretion (captured above).
- `llama.cpp` is a new external toolchain not currently in the repo. Installation is a one-time developer setup step documented in the README under a "GGUF conversion" section; the repo itself does not vendor llama.cpp.
- The GGUF pipeline must not require network-hosted intermediaries (no HF gguf-my-repo Space) per D-01 and D-04.

</code_context>

<specifics>
## Specific Ideas

- GGUF metadata must carry the exact `chat_template.jinja` used during training — this is non-negotiable given the Phase 09.1 template-drift regression class. Document the embed command in a comment inside the conversion script so the coupling is visible.
- "v1.0.0 = the dataset the released model was trained on" is the anchor for dataset versioning — the version number of the dataset and the version number of the released model are distinct but co-published (tagged together at release time).
- BENCHMARK.md snapshots inside a dataset-version bundle must be immutable once a release is cut — the snapshot records how the model trained on that dataset performed, not a rolling view.
- Post-quantization sanity check (optional per Claude's discretion): a small perplexity delta between f16 GGUF and the Q4_K_M GGUF catches catastrophic quantization errors before publishing.

</specifics>

<deferred>
## Deferred Ideas

### Space Design (captured for future reference if REL-06 ever returns)
Although REL-06 is moving to Out of Scope, the Space UX direction discussed is preserved here in case a future project phase revisits an interactive demo:
- **UI layout:** Three tabs in a single Gradio `gr.Tabs` — one tab per capability area (tool calls, code, general knowledge). Mirrors the three-way dataset split the model was trained on.
- **Examples:** ~3 hand-picked example prompts per tab as clickable chips above the input box, drawn from the held-out test split.
- **Comparison UX:** Side-by-side panels running the same prompt through base SmolLM2-1.7B-Instruct and Lyra, with both outputs shown. Directly demonstrates capability delta.
- **Model load target:** The merged safetensors (not base + adapter, not GGUF) — simplest inference path via transformers `from_pretrained`.
- **Weight fetch mechanism:** Deferred to Claude's discretion if ever revived (candidates discussed: GitHub LFS pull on Space build vs HF dataset repo proxy).

### Planning-doc updates required before / during Phase 10 planning
- **REQUIREMENTS.md:** Move REL-06 from v1 Active to the `Out of Scope` table with reason "Community release is GitHub-native; interactive demo not pursued in v1." Update the Traceability table entry for REL-06. Planner applies the edits as part of Phase 10 plan 01.
- **ROADMAP.md:** Update Phase 10's goal statement and success criteria bullet list to remove the Gradio Space. Success criterion 2 (interactive Gradio demo Space) is removed; success criteria 1 (GGUF) and 3 (dataset versioning) remain.

### Out of Phase 10 scope
- HuggingFace Model Hub publishing of Lyra weights — not pursued in v1.
- HuggingFace dataset repo for Lyra — not pursued in v1.
- Additional GGUF quantization levels beyond Q4_K_M / Q8_0 (e.g., Q5_K_M, Q6_K) — can be added in a later patch release if community demand emerges.
- Continuous / automated release pipeline (GitHub Actions that cuts a release on every tag push) — manual release is sufficient for v1.
- Hosted inference / API / Ollama Modelfile publication — users deploy on their own infrastructure per PROJECT.md Out of Scope posture.

</deferred>

---

*Phase: 10-community-release-enhancements*
*Context gathered: 2026-04-24*
