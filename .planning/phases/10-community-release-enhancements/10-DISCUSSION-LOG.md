# Phase 10: Community Release Enhancements - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-24
**Phase:** 10-community-release-enhancements
**Areas discussed:** HuggingFace publishing scope, GGUF pipeline & model variant, Gradio demo Space design, Dataset versioning scheme

---

## HuggingFace Publishing Scope

### Q1: How far does HuggingFace publishing extend now that Phase 10 requires a Space?

| Option | Description | Selected |
|--------|-------------|----------|
| Space only (Recommended) | Gradio Space on HF; model weights / GGUF / dataset stay on GitHub. Space pulls weights from GitHub LFS. | |
| Model + Space on HF | Publish model weights + GGUF to HF Model Hub, Space loads from there. Dataset stays on GitHub. | |
| Everything on HF | Model, GGUF, dataset, Space all on HF. GitHub still source for scripts. | |
| HF canonical, GitHub mirror | HF becomes canonical for artifacts; GitHub keeps code. | |

**User's choice:** Free-text "Github LFS only for now." — interpreted as: no HuggingFace at all, including no Space. All artifacts on GitHub + Git LFS.
**Notes:** This pivot foreshadowed dropping REL-06 entirely, which came up two questions later.

### Q2: Under what HuggingFace namespace should Lyra live?

| Option | Description | Selected |
|--------|-------------|----------|
| Personal user account | `huggingface.co/{user}/lyra`. | |
| New "lyra" org | Create an HF org for the project. | |
| Skip now, decide at publish | Defer namespace decision to publish time. | ✓ |

**User's choice:** Skip now, decide at publish.
**Notes:** Consistent with the "no HF in this milestone" direction. If REL-06 is ever revived, namespace is picked then.

### Q3: What model weights does the Gradio Space load?

| Option | Description | Selected |
|--------|-------------|----------|
| Merged safetensors (Recommended) | `models/lyra-merged/` via transformers `from_pretrained`. | ✓ |
| Base + LoRA adapter | Base SmolLM2 from HF + adapter via peft at runtime. | |
| GGUF via llama-cpp-python | Space runs Q4_K_M GGUF on CPU. | |

**User's choice:** Merged safetensors.
**Notes:** Moot once REL-06 was dropped, but retained in CONTEXT.md deferred section as intent for any future Space work.

### Q4: Should the Space include a live comparison vs base SmolLM2-1.7B?

| Option | Description | Selected |
|--------|-------------|----------|
| Side-by-side (Recommended) | Same prompt, both outputs shown simultaneously. | ✓ |
| Lyra only, link to BENCHMARK.md | Single panel; static link for comparison. | |
| Toggle per example | Default Lyra-only, opt-in base comparison. | |

**User's choice:** Side-by-side.
**Notes:** Preserved as deferred Space-design note.

---

## GGUF Pipeline & Model Variant

### Q1: Which GGUF conversion toolchain should we commit to?

| Option | Description | Selected |
|--------|-------------|----------|
| llama.cpp `convert_hf_to_gguf.py` (Recommended) | Canonical llama.cpp path: convert → quantize via shell script. | ✓ |
| autogguf / ggml-org wrappers | Higher-level Python wrappers around same binaries. | |
| HF gguf-my-repo Space | One-click Space; conflicts with "GitHub only" decision. | |

**User's choice:** llama.cpp `convert_hf_to_gguf.py`.
**Notes:** Matches the project's script-first convention. Conflicts with HF gguf-my-repo were already ruled out by the HF scope answer.

### Q2: Which quantization levels should we ship?

| Option | Description | Selected |
|--------|-------------|----------|
| Required only: Q4_K_M + Q8_0 | Exact REL-05 spec. Minimum LFS footprint. | ✓ |
| Standard set: Q4_K_M + Q5_K_M + Q8_0 (Recommended) | Adds Q5_K_M sweet-spot. | |
| Full family: Q3/Q4/Q5/Q6/Q8 | Full spectrum; heavy LFS footprint. | |

**User's choice:** Required only: Q4_K_M + Q8_0.
**Notes:** Additional quants can be added in a patch release later if community asks.

### Q3: Which Lyra weights become the community-release artifact?

| Option | Description | Selected |
|--------|-------------|----------|
| Wait for 09.1/09.2 final (Recommended) | Block Phase 10 start until tool-call-format success criterion met. | ✓ |
| Ship current merged v2 now | Quantize regressed v2; document the regression. | |
| Ship base + Lyra adapter, swap later | Prove the pipeline on base, swap in Lyra fine-tune later. | |

**User's choice:** Wait for 09.1/09.2 final.
**Notes:** Creates an explicit dependency gate for Phase 10 planning.

### Q4: How do we handle SmolLM2's chat template in the GGUF files?

| Option | Description | Selected |
|--------|-------------|----------|
| Embed custom template at convert time (Recommended) | Bake `chat_template.jinja` into GGUF metadata via convert flag. | ✓ |
| Rely on llama.cpp built-in SmolLM2 template | Use arch-selected default; drift risk on template changes. | |
| Document the template, leave GGUF default | Document and expect users to override at runtime. | |

**User's choice:** Embed custom template at convert time.
**Notes:** Explicitly closes the Phase 09.1 class of runtime template-drift regression.

---

## Gradio Demo Space Design

### Q1: How should the three capability areas surface in the Space UI?

| Option | Description | Selected |
|--------|-------------|----------|
| Three tabs (Recommended) | `gr.Tabs` one per capability. | ✓ |
| Single unified chat | One chat, generic prompt, no routing. | |
| Landing page + deep links | Marketing-style landing page with example buttons. | |

**User's choice:** Three tabs.

### Q2: What inference backend does the Space use?

| Option | Description | Selected |
|--------|-------------|----------|
| Free CPU Space (Recommended) | HF free tier: 2 vCPU, 16GB RAM, ~5-15 tok/s. | |
| ZeroGPU (HF dynamic GPU) | Dynamic GPU allocation; needs HF Pro / grant. | |
| llama.cpp via Q4_K_M GGUF | Dogfood GGUF artifact inside the Space. | |

**User's choice:** Free-text "no spaces in this milestone" — REL-06 deferred entirely.
**Notes:** This is the pivot that dropped the whole sub-area. All other answers in this section are preserved as deferred Space-design intent in CONTEXT.md.

### Q3: How should we seed the Space with example prompts?

| Option | Description | Selected |
|--------|-------------|----------|
| 3 curated examples per tab (Recommended) | Hand-picked examples from test split as Gradio chips. | ✓ |
| Freeform only | Empty input; no presets. | |
| Full example gallery | 10-20 examples per tab from test split. | |

**User's choice:** 3 curated examples per tab.
**Notes:** Retained as deferred.

### Q4: How does the Space get the merged safetensors at build/boot time?

| Option | Description | Selected |
|--------|-------------|----------|
| Pull from GitHub LFS on Space build (Recommended) | `git lfs clone` against a pinned release tag; cache in /data. | |
| HuggingFace dataset as LFS proxy | Push weights to private HF dataset repo as bandwidth proxy. | |
| You decide | Claude picks a workable path at plan time. | ✓ |

**User's choice:** You decide (moot given REL-06 deferral, but captured in deferred design notes).

### Clarifying follow-up (freeform)

**Q:** The answer "no spaces in this milestone" is a pivot — which of three readings applies?
(1) Defer REL-06 entirely; (2) Ship Gradio code locally, no HF deployment; (3) Build code, defer only HF publish.
**User's choice:** 1 — Defer REL-06 entirely.
**Follow-up:** What should happen to REL-06 in the planning docs?

### Q5: What should happen to REL-06 in the project planning docs?

| Option | Description | Selected |
|--------|-------------|----------|
| Move to v2 Requirements (Recommended) | Relocate to v2 Requirements; update ROADMAP.md Phase 10. | |
| Drop entirely (Out of Scope) | Move to Out of Scope table; not coming back. | ✓ |
| Park in backlog | Backlog phase 999.x. | |

**User's choice:** Drop entirely (Out of Scope).

### Q6: Keep the three Space UX answers on record for v2?

| Option | Description | Selected |
|--------|-------------|----------|
| Keep as v2 notes (Recommended) | Preserve in CONTEXT.md deferred section. | ✓ |
| Drop | Don't record. | |

**User's choice:** Keep as v2 notes.
**Notes:** Mild tension with Q5 answer (Out of Scope vs v2 notes) — honored by keeping design intent in Deferred Ideas even though REL-06 is formally removed.

---

## Dataset Versioning Scheme

### Q1: What version scheme should dataset releases use?

| Option | Description | Selected |
|--------|-------------|----------|
| SemVer (1.0.0 / 1.1.0 / 2.0.0) (Recommended) | major.minor.patch with defined semantics. | ✓ |
| Date-based (2026.04) | Year.month tag. | |
| Integer v1/v2/v3 | Simple counter. | |

**User's choice:** SemVer.

### Q2: What is the v1.0 (initial) dataset release?

| Option | Description | Selected |
|--------|-------------|----------|
| Current 25K assembled (post-09.1) (Recommended) | The dataset the released model was trained on. | ✓ |
| Original ~5K v0 + 25K v1 | Retroactively version original Phase 7 data as v0.1.0. | |
| Whatever the released model was trained on | Define at release time; same as option 1 in practice. | |

**User's choice:** Current 25K assembled.

### Q3: What does a dataset-version bundle contain?

| Option | Description | Selected |
|--------|-------------|----------|
| Full bundle (Recommended) | JSONL + stats JSON + pinned BENCHMARK.md + CHANGELOG. | ✓ |
| Minimal: JSONL + CHANGELOG | Data + CHANGELOG only. | |
| JSONL + stats + CHANGELOG, metrics external | Metrics live in main-branch BENCHMARK.md. | |

**User's choice:** Full bundle.

### Q4: How do per-version dataset metrics get captured?

| Option | Description | Selected |
|--------|-------------|----------|
| Pinned to model evals (Recommended) | Dataset version attaches eval_results from the model trained on it. | ✓ |
| Dataset-intrinsic metrics only | Sample counts, token distributions, dedup retention only. | |
| Both: intrinsic + model metrics | Full capture, most ceremony. | |

**User's choice:** Pinned to model evals.
**Notes:** Matches the project's existing eval → compare → BENCHMARK.md workflow.

---

## Claude's Discretion

Captured in CONTEXT.md under `<decisions>` → `### Claude's Discretion`:
- GGUF conversion shell script name / structure
- GGUF file naming convention
- Per-variant README layout vs combined README
- CHANGELOG.md location and format (Keep-a-Changelog vs freeform)
- Dataset-version git tag naming
- `.gitattributes` GGUF LFS filter additions
- llama.cpp version pinning strategy
- Optional post-quantization perplexity sanity check

## Deferred Ideas

- **REL-06 Space design** preserved for future reference despite "Out of Scope" label — three tabs, 3 curated examples per tab, side-by-side base vs Lyra, merged safetensors via transformers.
- **Planning-doc updates required:** REQUIREMENTS.md (REL-06 → Out of Scope), ROADMAP.md (Phase 10 success criteria).
- **HuggingFace publishing** across all surfaces (model, dataset, GGUF, Space) deferred; GitHub + Git LFS is the only channel for v1.
- **Additional GGUF quant levels** (Q5_K_M, Q6_K, etc.) deferred to a potential patch release.
- **Continuous release automation** deferred — manual release is sufficient for v1.
