---
phase: 10-community-release-enhancements
plan: 01
subsystem: release-engineering
tags: [gguf, llama.cpp, git-lfs, keep-a-changelog, pydantic, semver, dataset-versioning, test-scaffolding]

# Dependency graph
requires:
  - phase: 07-dataset-assembly
    provides: scripts/assemble_dataset.py compute_stats() function extended for JSON envelope
  - phase: 09-benchmarking-and-core-release
    provides: .gitattributes LFS precedent for *.safetensors; tests/test_release_artifacts.py scaffold; README/datasets/README structure
provides:
  - ".gitattributes LFS rules for *.gguf, *.safetensors, *.arrow, datasets/assembled/**/*.jsonl (committed BEFORE any Phase 10 binary per Pitfall 4)"
  - "scripts/verify_gguf.py -- thin-CLI GGUF metadata verifier (tokenizer.chat_template + SmolLM2 <|im_start|> marker)"
  - "scripts/convert_gguf.sh -- reproducible HF->f16->Q4_K_M/Q8_0 pipeline with arg-regex validation + chat_template pre/postconditions"
  - "DatasetStats Pydantic model + build_dataset_stats_payload() for machine-readable dataset-stats JSON"
  - "scripts/assemble_dataset.py stats --json [--output PATH] [--dataset-version X.Y.Z] flags"
  - "CHANGELOG.md Keep-a-Changelog 1.1.0 scaffold with [Unreleased], [Dataset v1.0.0], [Model v1.0.0] headers"
  - "requirements.txt pin: gguf==0.18.0"
  - "README.md GGUF Variants section (Q4_K_M/Q8_0 scaffolded, SHA256/size filled by Plan 04)"
  - "datasets/README.md Dataset Versions section (v1.0.0 row scaffolded, values filled by Plan 04)"
  - "REQUIREMENTS.md: REL-06 moved to Out of Scope (D-02); Traceability + Coverage updated (32 -> 31)"
  - "ROADMAP.md: Phase 10 block rewritten -- Gradio dropped from goal/success criteria; UI hint: no"
  - "Wave 0 test scaffold: test_release_artifacts.py extended + test_gguf_conversion.py + test_dataset_versioning.py + test_assemble_dataset.py TestStatsJsonOutput (mix of GREEN + xfail stubs)"
affects: [10-02, 10-03, 10-04]

# Tech tracking
tech-stack:
  added:
    - gguf==0.18.0 (GGUF metadata inspection)
    - Keep-a-Changelog 1.1.0 (CHANGELOG format)
  patterns:
    - "Thin-CLI script (mirrors scripts/eval_merge.py): shebang + SPDX + docstring + argparse + main() -> int + sys.exit(main())"
    - "Lazy-import heavy deps inside _load_gguf_reader() so tests can MagicMock gguf via sys.modules"
    - "Positional-arg regex validation ^[a-zA-Z0-9._/~\\-]+$ in bash wrappers (mirrors scripts/eval_runner.py T-03-07)"
    - "Pydantic envelope at data-export trust boundary (DatasetStats.model_validate before write) -- mirrors scripts/eval_config.py EvalResult pattern"
    - "Wave 0 TDD discipline: xfail markers for downstream-plan implementations; tests land in Plan 01 and turn GREEN in Plans 02/03/04"

key-files:
  created:
    - scripts/verify_gguf.py
    - scripts/convert_gguf.sh
    - CHANGELOG.md
    - tests/test_gguf_conversion.py
    - tests/test_dataset_versioning.py
    - .planning/phases/10-community-release-enhancements/10-01-SUMMARY.md
  modified:
    - .gitattributes
    - requirements.txt
    - scripts/assemble_dataset.py
    - tests/test_assemble_dataset.py
    - tests/test_release_artifacts.py
    - README.md
    - datasets/README.md
    - .planning/REQUIREMENTS.md
    - .planning/ROADMAP.md

key-decisions:
  - "REL-06 (Gradio Space) formally moved to Out of Scope per CONTEXT.md D-02: Community release is GitHub-native; interactive demo not pursued in v1"
  - "LFS rules land BEFORE any GGUF binary is produced (RESEARCH.md Pitfall 4): .gitattributes committed in Task 1, Plan 02 produces binaries afterwards"
  - "DatasetStats schema_version = 1.0.0 separate from dataset_version = 1.0.0 so JSON shape can evolve independent of dataset content"
  - "Convert pipeline locates llama.cpp convert_hf_to_gguf.py via LLAMA_CPP_DIR env (default /opt/homebrew/share/llama.cpp for brew); not vendored in repo"
  - "Tasks 2/3 remove xfail markers one-to-one as implementations land; test_produced_gguf_has_chat_template + test_dataset_v100_release_exists remain xfail for Plan 02/03"

patterns-established:
  - "Pattern: Thin-CLI + lazy import for optional heavy deps -- applied to scripts/verify_gguf.py (gguf package)"
  - "Pattern: Bash arg-regex guard before any subprocess substitution -- applied to scripts/convert_gguf.sh (T-10-01)"
  - "Pattern: Path-traversal refusal via .parts check before .write_text() -- applied to stats --output flag (T-10-04)"
  - "Pattern: DatasetStats envelope as schema-validated trust-boundary output -- will be reused by Plan 03 for dataset-bundle stats.json"

requirements-completed: []  # None of REL-05/REL-06/REL-07 are fully complete from this plan alone. Plan 01 is infrastructure prep only. REL-06 is now Deferred (Out of Scope). REL-05 closes in Plan 02; REL-07 closes in Plan 03. Requirement checkboxes remain unchecked in REQUIREMENTS.md per this plan's scope.

# Metrics
duration: 9min
completed: 2026-04-24
---

# Phase 10 Plan 01: Non-Gated Community-Release Prep Summary

**GGUF conversion toolchain (convert_gguf.sh + verify_gguf.py), dataset-stats JSON via DatasetStats Pydantic model, Keep-a-Changelog scaffold, .gitattributes LFS rules landed before any binary, and REL-06 formally moved to Out of Scope -- all in place so Plans 02/03/04 can execute without any Plan-01 follow-up.**

## Performance

- **Duration:** ~9 min
- **Started:** 2026-04-24T18:36:08Z
- **Completed:** 2026-04-24T18:45:00Z (approx)
- **Tasks:** 3 / 3
- **Files modified:** 9 (5 created, 4 modified from worktree base)
- **Lines added (approx):** 295 Task 1 + 145 Task 2 + 76 Task 3 = ~516 net insertions

## Accomplishments

- LFS rules for `*.gguf`, `*.safetensors`, `*.arrow`, `datasets/assembled/**/*.jsonl` committed in Task 1 BEFORE any GGUF binary is produced (RESEARCH.md Pitfall 4 ordering honored).
- `scripts/verify_gguf.py` thin-CLI + lazy-imported `gguf.GGUFReader`: exits 0 when `tokenizer.chat_template` is present and contains the SmolLM2 `<|im_start|>` marker; exits 1 otherwise. Threat-mitigated per T-10-01/T-10-02.
- `scripts/convert_gguf.sh` chmod +x: validates args against `^[a-zA-Z0-9._/~\-]+$`; checks `tokenizer_config.json` chat_template precondition; runs HF->f16->Q4_K_M/Q8_0 via llama.cpp; invokes verify_gguf.py on both outputs as D-07 postcondition.
- `DatasetStats` + `DatasetSplitStats` Pydantic models added to `scripts/assemble_dataset.py` plus `build_dataset_stats_payload()`. CLI `stats --json [--output PATH] [--dataset-version X.Y.Z]` flags landed, with T-10-04 path-traversal guard.
- `CHANGELOG.md` Keep-a-Changelog 1.1.0 scaffold with `[Unreleased]`, `[Dataset v1.0.0] - TBD`, `[Model v1.0.0] - TBD` headers; shasum verification note in header.
- `REQUIREMENTS.md` D-02 edits applied: REL-06 bullet removed, Out of Scope row added with reason string, Traceability row changed to `Out of Scope | Deferred`, Coverage totals 32 -> 31.
- `ROADMAP.md` D-02 edits applied: goal rewritten, depends-on annotated with 09.1/09.2 gating, requirements list now `REL-05, REL-07`, Gradio success criterion removed, `UI hint: no`, all 4 Phase 10 plan rows enumerated. Also fixed the top-of-file summary bullet (originally also listed Gradio).
- Wave 0 test scaffold: 11 GREEN new tests + 6 xfail stubs for Task 2 / Task 3 / Plan 02 / Plan 03. Task 2 removed 4 xfails (-> GREEN), Task 3 removed 2 xfails (-> GREEN), leaving 2 xfails for downstream plans.

## Task Commits

Each task was committed atomically (worktree uses `--no-verify` per parallel-executor convention; hooks validated centrally after merge):

1. **Task 1: Wave 0 scaffold + LFS rules + REL-06 move** -- `115b93e` (feat)
2. **Task 2: scripts/verify_gguf.py + convert_gguf.sh (xfails -> GREEN)** -- `168307e` (feat)
3. **Task 3: DatasetStats Pydantic model + stats --json/--output flags (xfails -> GREEN)** -- `c3a7000` (feat)

All three commits build on base `81ff489` (docs(state): record phase 10 planning complete).

## Files Created/Modified

**Created:**
- `.planning/phases/10-community-release-enhancements/10-01-SUMMARY.md` — this file
- `CHANGELOG.md` — 21 lines; Keep-a-Changelog 1.1.0 scaffold
- `scripts/verify_gguf.py` — 80 lines; thin-CLI GGUF metadata verifier with lazy gguf import and path regex guard
- `scripts/convert_gguf.sh` — 65 lines (chmod +x); two-stage HF->f16->Q4_K_M/Q8_0 pipeline with pre/postcondition checks
- `tests/test_gguf_conversion.py` — 78 lines; 4 GREEN + 1 xfail for Plan 02
- `tests/test_dataset_versioning.py` — 45 lines; 4 GREEN tag-format + CHANGELOG test + 1 xfail for Plan 03

**Modified:**
- `.gitattributes` — rewritten: replaces 3-line "don't track binaries" comment with 8-line LFS rule block (D-05, D-10, Pitfall 4)
- `requirements.txt` — +3 lines: `gguf==0.18.0` pinned with comment (REL-05 dep)
- `scripts/assemble_dataset.py` — +66 lines: DatasetSplitStats + DatasetStats models, build_dataset_stats_payload(), 3 new argparse flags, JSON write handler with T-10-04 guard
- `tests/test_assemble_dataset.py` — +48 lines: TestStatsJsonOutput class (2 tests, GREEN after Task 3)
- `tests/test_release_artifacts.py` — +29 lines: 3 new tests (gitattributes GGUF LFS, gitattributes arrow LFS, CHANGELOG exists) + Dataset Versions assertion in test_dataset_card
- `README.md` — +31 lines: `## GGUF Variants` section with Q4_K_M/Q8_0 scaffolded table, conversion runbook, verify_gguf runbook
- `datasets/README.md` — +14 lines: `## Dataset Versions` section with v1.0.0 row scaffolded, bundle contents list
- `.planning/REQUIREMENTS.md` — D-02 edits: REL-06 removal, Out of Scope row, Traceability update, Coverage 32 -> 31
- `.planning/ROADMAP.md` — D-02 edits: top summary bullet + Phase 10 block rewrite

## Decisions Made

- **Scaffolded README.md / datasets/README.md sections in Task 1, not Task 3** (see Deviations below). The `must_haves` frontmatter and `test_dataset_card` assertion both required these sections to pass Task 1 acceptance_criteria `pytest -q` exit 0 overall. Task 3's Step 5/6 text is identical, so no content change; only task-boundary moved.
- **Arrow cache files in `datasets/assembled/` left unstaged.** The existing arrow files were committed before Plan 10-01 added LFS rules. `git lfs status` flags them as "needing LFS conversion," but staging them would invoke `git lfs migrate import`-style conversion which is out of Plan 10-01 scope (an orthogonal maintenance concern). Those files remain in-tree unchanged; a separate task (outside this plan) can run `git lfs migrate import --include="*.arrow"` if desired. No impact on functionality -- git tracks them as regular blobs now; they would be LFS-tracked if rewritten.
- **`convert_gguf.sh` is not unit-tested end-to-end (no llama.cpp CLI here).** Only the arg-validation error path is unit-tested (GREEN). The full two-stage pipeline is an integration-test surface for Plan 02 (which must run with real llama.cpp installed).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Plan task-boundary inconsistency: README scaffolding moved from Task 3 Step 5/6 into Task 1**
- **Found during:** Task 1 pytest validation (`test_dataset_card` failed because `## Dataset Versions` assertion ran against a datasets/README.md that Task 1 did not yet modify).
- **Issue:** Plan 10-01's frontmatter `must_haves.truths` explicitly claims Task 1 scaffolds the README GGUF Variants + datasets/README Dataset Versions sections, and Task 1 acceptance_criteria requires `pytest -q` exit 0 overall. But the per-task `<files>` lists put these writes in Task 3 Step 5/6, which would leave Task 1 acceptance RED. Internal inconsistency in the plan.
- **Fix:** Moved the README.md GGUF Variants section write + datasets/README.md Dataset Versions section write into Task 1 immediately after the test_release_artifacts.py extension. Content is verbatim from Task 3 Step 5/6 text.
- **Files modified:** README.md, datasets/README.md
- **Verification:** `pytest tests/test_release_artifacts.py tests/test_dataset_versioning.py tests/test_gguf_conversion.py -q` -> 11 passed, 6 xfailed; Task 1 acceptance greps all match.
- **Committed in:** 115b93e (Task 1 commit) -- documented in that commit's DEVIATION note.

**2. [Rule 3 - Blocking] .gitattributes whitespace normalized to single-space columns to match acceptance_criteria regex**
- **Found during:** Task 1 pytest + acceptance-grep sweep after first .gitattributes write.
- **Issue:** My initial write used column-aligned spacing (`*.gguf        filter=lfs...`) which reads nicely but does not match Task 1's acceptance_criteria literal: `grep -c '^\*\.gguf filter=lfs diff=lfs merge=lfs -text$' .gitattributes` returns 1 (expects single space).
- **Fix:** Rewrote .gitattributes with single-space columns. Still readable, still unambiguous, matches acceptance.
- **Files modified:** .gitattributes
- **Verification:** `grep -c '^\*\.gguf filter=lfs diff=lfs merge=lfs -text$' .gitattributes` returns 1.
- **Committed in:** 115b93e (Task 1 commit).

**3. [Rule 3 - Blocking] ROADMAP.md top-of-file Phase 10 summary bullet (line 24) also mentioned Gradio and needed D-02 rewrite**
- **Found during:** Task 1 acceptance sweep -- `grep -c "interactive Gradio demo Space" .planning/ROADMAP.md` returned 1 instead of the expected 0.
- **Issue:** The plan's Task 1 Step 9 only addresses the Phase 10 block at lines 190-199, but the top-of-file roadmap checklist (line 24) ALSO mentions "interactive Gradio demo Space." Per D-02, Gradio is dropped entirely from Phase 10 -- so this line must also be updated.
- **Fix:** Rewrote line 24 to drop the Gradio phrase: "GGUF quantized variants and versioned dataset releases."
- **Files modified:** .planning/ROADMAP.md (line 24, in addition to the Phase 10 block)
- **Verification:** `grep -c "interactive Gradio demo Space" .planning/ROADMAP.md` returns 0.
- **Committed in:** 115b93e (Task 1 commit).

---

**Total deviations:** 3 auto-fixed (all Rule 3 - Blocking; all plan-internal inconsistencies that acceptance_criteria made visible)
**Impact on plan:** All three fixes are necessary to make Task 1 acceptance_criteria pass as written. No scope creep -- the resolutions are all things the plan already *wanted* (per must_haves + D-02) but mis-routed across tasks. Total extra effort: ~5 lines of text edits.

## Issues Encountered

**Worktree base was out of date.** Worktree-agent-a1058651 was initially checked out at `5b9c69d1` (before Phase 10 artifacts existed). `git merge-base HEAD 81ff489e` returned `5b9c69d1`, so I hard-reset the worktree to the expected base `81ff489e` per `<worktree_branch_check>`. All 3 tasks and SUMMARY.md sit on top of that correct base.

**Arrow files show as "modified" under the new LFS rules.** After committing `.gitattributes`, `git status` reported 7 arrow files under `datasets/assembled/` as modified (Git wants to convert them to LFS pointers). These were pre-existing regular-blob commits; staging them here is explicitly out of scope for Plan 10-01 (which is about prep, not data migration). I did `git add` only the intended Plan 10-01 files by name (never `git add -A`), leaving the arrow files unstaged. A separate maintenance task can run `git lfs migrate import --include="*.arrow" HEAD` if desired.

## User Setup Required

None -- Plan 10-01 is infrastructure prep only. `llama.cpp` install is documented in the new README GGUF Variants section but is NOT required until Plan 02 (gated on Phase 09.1/09.2 weights).

## Next Phase Readiness

Plans 02, 03, 04 can start without any Plan-01 follow-up:

- **Plan 02 (GGUF conversion):** `scripts/convert_gguf.sh` + `scripts/verify_gguf.py` exist and are unit-tested for arg/mocked-metadata paths. `.gitattributes` has `*.gguf filter=lfs` so any committed binaries flow through LFS from the first `git add`. `requirements.txt` pins `gguf==0.18.0`. Only gate: Phase 09.1/09.2 weights meeting tool-call-format success criterion per D-06.
- **Plan 03 (dataset v1.0.0 release):** `DatasetStats` envelope + `stats --json --output PATH` machine-readable export ready. `CHANGELOG.md` has `[Dataset v1.0.0] - TBD` section to fill at release-cut time. `datasets/README.md` has scaffolded `## Dataset Versions` table.
- **Plan 04 (docs finalization + UAT):** README.md + datasets/README.md both have scaffolded sections with `TBD` placeholders for SHA256/size/date. Plan 04 fills in the values after Plans 02/03 produce actual artifacts.

## Threat Flags

None. All files created/modified fall within the plan's `<threat_model>` surface (T-10-01 through T-10-04). No new network endpoints, auth paths, or trust boundaries were introduced outside of what the plan enumerated. The `convert_gguf.sh` script is a developer-local tool (not a network service); `verify_gguf.py` is a passive metadata reader; `stats --output` already has T-10-04 path-traversal guard.

## Known Stubs

These placeholders are intentional; they will be filled at release-cut time:

- **README.md GGUF Variants table** — rows with `TBD` for Size, SHA256. File: `README.md`, lines ~97-99. Reason: Plan 04 Task 1 fills values after Plan 02 produces the GGUFs. Plan reference: acceptance_criteria `grep -c '^## GGUF Variants' README.md` returns 1; content completion is Plan 04's task.
- **datasets/README.md Dataset Versions table** — row with `TBD` for Date, Samples, SHA256. File: `datasets/README.md`, line ~47. Reason: Plan 04 Task 1 fills values after Plan 03 cuts the dataset-v1.0.0 release.
- **CHANGELOG.md [Dataset v1.0.0] / [Model v1.0.0] sections** — date `TBD` and single "Placeholder" bullet under ### Added. File: `CHANGELOG.md`, lines 13-20. Reason: Plans 02/03/04 replace these placeholders with real release-cut contents.

All stubs are documented in the file text as "filled at release-cut time by Plan 04" so downstream agents and human readers see the deferred-state intent.

## TDD Gate Compliance

Plan is not a plan-level `type: tdd` (frontmatter says `type: execute`), but per-task `tdd="true"` TDD discipline applied:

- **Task 1 (tdd=true):** Wave 0 RED tests committed alongside implementation. 11 GREEN-from-day-1 + 6 xfail placeholders for downstream tasks/plans. This is Wave-0 scaffold discipline (tests land before implementations), not classical RED->GREEN in a single task.
- **Task 2 (tdd=true):** 4 xfail markers REMOVED as implementation lands (xfail -> GREEN transition). scripts/verify_gguf.py + scripts/convert_gguf.sh written; tests go GREEN in same commit. The RED phase was already on disk from Task 1 (the xfails are the RED markers), so Task 2 is the GREEN commit.
- **Task 3 (tdd=true):** Same transition pattern. 2 xfail markers removed; DatasetStats + --json/--output flag code added in same commit.

No separate `test(...)` commit was made per task — the test files were all landed in Task 1, and Tasks 2/3 removed xfail markers as part of their `feat(...)` commits. This is acceptable for Wave-0 scaffold discipline; the RED->GREEN sequence is visible in xfail-count drop: 6 xfails (after Task 1) -> 2 xfails (after Task 2) -> 2 xfails remain (Plan 02 + Plan 03 targets).

## Self-Check: PASSED

Self-check performed in execution: all created files exist; all task commits land on HEAD branch descending from `81ff489`; full pytest suite passes (311 passed, 2 skipped, 2 xfailed); no files accidentally deleted by any commit (`git diff --diff-filter=D --name-only HEAD~N HEAD` returned empty for N in 1..3).

---
*Phase: 10-community-release-enhancements*
*Plan: 01*
*Completed: 2026-04-24*
