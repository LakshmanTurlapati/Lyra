---
slug: 09-2-06-retrain-blockers
status: resolved
trigger: "Phase 09.2-06 Branch 06A retrain halted at Task A1 pre-flight on 2026-04-23 with 5 blockers documented in 09.2-06-SUMMARY.md. Lyra v2 regressed on tool-call-format (0.0634 vs base 0.4065) and the Branch 06A retrain was the sanctioned corrective action. Plan 09.2-07 (data rework, INSERTED after the halt) has since completed and rebalanced the training dataset to 7,519 samples meeting all D06-TRAINING-AUDIT thresholds — so Blocker 3 is resolved. Two plumbing blockers (regex precision in Task 0 dispatch, non-machine-readable hyperparams in DIAGNOSIS.md) and two architectural constraints (worktree-isolated training is unshippable, compute budget demands sequential main-repo execution) remain. Goal: unblock Branch 06A retrain so the eval-check can actually produce a number — without that, Phase 10 Waves 2/3 stay gated on D-06."
created: 2026-04-24T19:30:00Z
updated: 2026-04-24T20:15:00Z
related_phase: "09.2"
related_plan: "09.2-06"
---

## Symptoms

DATA_START
- **Expected behavior:** Running `/gsd-execute-plan 09.2-06` (or equivalent dispatcher) with the user-approved branch 06A should execute Task 0 (dispatch resolution = 06A only), Task A1 (pre-flight check + smoke training step), Task A2 (full retrain, ~3.5h), Task A3 (merge + custom eval + knowledge eval + eval_compare), and Task A4 (write 09.2-06-SUMMARY.md capturing final eval numbers). The retrained Lyra v3 must beat base SmolLM2-1.7B's 0.4065 on tool-call-format and stay within 5% of base on MMLU/ARC/HellaSwag (D-09 + D-10).

- **Actual behavior:** Execution halts at Task A1 pre-flight. 0/4 tasks completed. Five blockers identified in `.planning/phases/09.2-tool-call-regression-diagnosis/09.2-06-SUMMARY.md`:
  1. Task 0 dispatch regex `(?i)(approved?:?\s*06X|\bapprove 06X\b|\[x\].*06X)` matches BOTH 06A approval line AND 06C fallback-acknowledgement line, exits 2 with `SELECTED=['06A', '06C']`.
  2. Task A1 pre-flight greps DIAGNOSIS.md for `^lr:[[:space:]]` / `^epochs:[[:space:]]` but the values live inside Markdown bullet prose (`  - \`lr\`: 1e-4 → 5e-5 (...)`), so grep returns empty and the python validator exits with "STOP: fix DIAGNOSIS.md".
  3. [RESOLVED via Plan 09.2-07] Task A1 forbids running without the three data-rework changes; Plan 09.2-07 (INSERTED after the halt) has rebalanced the dataset to 7,519 train samples meeting all D06-TRAINING-AUDIT thresholds, so this blocker no longer applies.
  4. Parallel worktree executor cannot ship the 3.4 GB merged safetensors back to main (gitignored `models/**/*.safetensors`, `datasets/assembled/`). Training must run on main repo at `/Users/lakshman/Documents/Lyra`, not in a short-lived worktree.
  5. Compute budget is ~4-5 h strictly sequential MPS; incompatible with parallel worktree dispatch AND with user's feedback_no_parallel_evals.md rule.

- **Errors:** (from 09.2-06-SUMMARY.md)
  - Task 0 dispatch script exit code 2, `SELECTED=['06A', '06C']`.
  - Task A1 pre-flight python validator: "STOP: fix DIAGNOSIS.md Branch 06A section (add lines: lr: <value> and epochs: <value>) then re-run".
  - No training-phase errors (training never started).

- **Timeline:**
  - 2026-04-21: Phase 9 Lyra v1 released (tool-call-format 0.41).
  - ~2026-04-22: Phase 09.1 retrained → Lyra v2, regressed to 0.19 then 0.0634 across two eval passes.
  - 2026-04-22/23: Phase 09.2 diagnosis produced DIAGNOSIS.md with Branch 06A / 06B / 06C menu; user signed off on 06A (retrain with rebalanced data + adjusted hyperparams, 06C revert as pre-committed fallback).
  - 2026-04-23: Phase 09.2-06 dispatched to a worktree executor → halted at pre-flight with 5 blockers, 0/4 tasks executed.
  - After the halt: Plan 09.2-07 inserted as prerequisite → rebalanced data, marked [x] in ROADMAP. Blocker 3 thereby resolved.
  - 2026-04-24 (now): 09.2-06 has NOT been re-dispatched. Plan file still contains the regex bug; DIAGNOSIS.md still lacks machine-readable `lr:` / `epochs:` lines.

- **Reproduction:**
  1. Inspect `.planning/phases/09.2-tool-call-regression-diagnosis/09.2-06-PLAN.md` Task 0 — the regex `(?i)(approved?:?\s*06X|\bapprove 06X\b|\[x\].*06X)` is line-oblivious and matches the "[x] User reviewed ... caveat in 06C" acknowledgement line.
  2. `grep -nE '^lr:[[:space:]]|^epochs:[[:space:]]' .planning/phases/09.2-tool-call-regression-diagnosis/09.2-DIAGNOSIS.md` returns nothing — values are in bullet prose on lines ~90-91.
  3. Spawning Plan 06 inside a worktree means the 3.4 GB `models/lyra-merged/model.safetensors` output is gitignored and cannot merge back; verified via `grep -E '^models/.*\\.safetensors' .gitignore`.
DATA_END

## Evidence

- timestamp: 2026-04-24T19:30:00Z
  source: .planning/phases/09.2-tool-call-regression-diagnosis/09.2-06-SUMMARY.md (read, full)
  observation: Five blockers enumerated with proposed fixes per blocker. Blockers 1 and 2 are Rule-3 class (plumbing). Blocker 3 is Rule-4 class architectural but already resolved by Plan 09.2-07. Blockers 4 and 5 are Rule-4 class architectural (dispatch mode + compute isolation).
  implication: Two plumbing fixes + one dispatch-mode change should unblock Branch 06A. Do not re-diagnose from scratch — fix the plumbing, re-dispatch on main.

- timestamp: 2026-04-24T19:30:00Z
  source: .planning/phases/09.2-tool-call-regression-diagnosis/09.2-07-SUMMARY.md (read, frontmatter + provides)
  observation: Plan 09.2-07 completed and provides "Rebalanced training dataset (7,519 train samples) meeting all three audit thresholds", 30-variant suffix pool, 500 single-turn tool-call-ending samples, 1,600 code+knowledge supplementary samples. ROADMAP.md marks 09.2-07 as [x].
  implication: Blocker 3's data-rework prerequisite is satisfied. Branch 06A can proceed on the current assembled dataset without further data curation work.

- timestamp: 2026-04-24T19:30:00Z
  source: .planning/ROADMAP.md (read)
  observation: Both 09.2-07 and 09.2-06 are marked [x], but 09.2-06-SUMMARY.md shows "status: CHECKPOINT — execution blocked at Task A1 pre-flight, tasks_completed: 0/4". The roadmap checkbox is out of sync with the actual plan state — Branch 06A retrain never ran.
  implication: The roadmap check on 09.2-06 should be reverted to [ ] (or left [x] as a marker that the halt was formally accepted); either way the retrain artifact (new merged model, new eval numbers) does not exist.

- timestamp: 2026-04-24T19:30:00Z
  source: Phase 09.1 SUMMARY.md files + Phase 10 CONTEXT.md D-06
  observation: Phase 10 Waves 2/3 (Plans 10-02 GGUF conversion + 10-04 docs/UAT) are gated on D-06: "weights beat base SmolLM2-1.7B's 0.4065 on tool-call-format, OR a documented revert-to-base release posture." User explicitly rejected the revert-to-base bypass in this session ("we have to fix the Lyra model.. retrrain it and make sure it's better than base model and then ship it.. untill then we are not done").
  implication: The retrain must actually execute and beat 0.4065 before Phase 10 can continue. This debug session is the critical path for the entire v1 release.

- timestamp: 2026-04-24T20:05:00Z
  source: 09.2-DIAGNOSIS.md re-read, lines 184 / 188 / 189
  observation: DIAGNOSIS.md already contains `Approved: 06A` on line 184, `lr: 5e-5` on line 188, and `epochs: 1` on line 189, each on its own line with no leading Markdown indent. Blocker 2's source-document side is therefore already fixed (presumably during 09.2-07 or a subsequent touch). Verified grep `'^lr:[[:space:]]'` returns `lr: 5e-5` and grep `'^epochs:[[:space:]]'` returns `epochs: 1`.
  implication: No DIAGNOSIS.md edit needed. Only the plan's marker-write regex (Blocker 1, second Task 0 script block at lines 148-158) still needs patching.

- timestamp: 2026-04-24T20:05:00Z
  source: 09.2-06-PLAN.md, lines 125-163 (Task 0 two script blocks)
  observation: Task 0 has TWO scripts. The first (lines 125-145, validator) already uses the tight regex `(?im)^Approved:\s*06X\s*$` — resolves to `SELECTED=['06A']`, exit 0. The second (lines 148-158, writes `.approved_branch` marker) still uses the loose regex `(?i)(approved?:?\s*06X|\bapprove 06X\b|\[x\].*06X)` — matches both 06A and 06C, so the marker file gets whichever the for-loop iterates to first (`06A`), but the inconsistency was the original halt cause per 09.2-06-SUMMARY.md Blocker 1.
  implication: Patch only the second script block. First one is already correct.

- timestamp: 2026-04-24T20:06:00Z
  source: Main-repo filesystem check (`ls datasets/assembled/`, `ls models/lyra-{adapter,merged}/`)
  observation: Assembled dataset present with splits train/validation/test; counts via datasets.load_from_disk: train=7519, validation=419, test=416 (matches 09.2-07 rebalance target). models/lyra-merged/model.safetensors is 3.4 GB (v2 artifact, will be overwritten by v3 retrain). models/lyra-adapter/adapter_model.safetensors present (72 MB, v2 adapter, similarly will be overwritten).
  implication: Main-repo prerequisites for Branch 06A retrain are in place. No missing artifacts that would cause Task A1 to fail for reasons beyond the 5 documented blockers.

- timestamp: 2026-04-24T20:08:00Z
  source: scripts/train.py CLI surface (grep for --lr / --epochs / --max-steps / --no-merge / --output-dir)
  observation: All five CLI flags that Task A1's commands depend on are present at lines 203, 231, 237, 269, 275. No API drift since the plan was written.
  implication: Training commands in Task A1 (smoke + full) will execute as written once dispatch resolves.

## Eliminated

- hypothesis: "A new/sixth blocker has emerged since the halt (missing assembled dataset, drifted train.py CLI, deleted merged model)."
  eliminated: true
  reason: Verified datasets/assembled has 7519 train / 419 val / 416 test samples via datasets.load_from_disk; models/lyra-merged/model.safetensors (3.4 GB) and models/lyra-adapter/adapter_model.safetensors (72 MB) both exist; scripts/train.py still exposes --lr/--epochs/--max-steps/--no-merge/--output-dir at the same argparse contract Plan 06 Task A1 was written against. No new blocker beyond the five documented.

- hypothesis: "Blocker 2 (DIAGNOSIS.md hyperparameter format) still requires an edit to DIAGNOSIS.md."
  eliminated: true
  reason: DIAGNOSIS.md was touched after the halt (during or alongside 09.2-07 completion) and now has `Approved: 06A` / `lr: 5e-5` / `epochs: 1` on their own lines at 184/188/189. The pre-flight shell pipeline `grep -E '^lr:[[:space:]]'` parses lr=5e-5 cleanly, python float(lr)/int(epochs) succeeds with exit 0. No edit needed; the source is already correct.

## Current Focus

- hypothesis: "Two plumbing bugs (Blocker 1: line-oblivious regex in Task 0 dispatch; Blocker 2: grep-unfriendly hyperparameter format in DIAGNOSIS.md) plus one dispatch-mode mismatch (Blocker 4/5: worktree isolation of gitignored training artifacts) together cause the pre-flight halt. Blocker 3's data-rework prerequisite has been resolved by Plan 09.2-07. Patching the regex, adding machine-readable hyperparam lines to DIAGNOSIS.md, and re-dispatching 09.2-06 Branch 06A sequentially on the main repo should allow Task A1 through Task A4 to execute end-to-end."
- test: "(a) Apply the regex patch from 09.2-06-SUMMARY.md Blocker 1 Proposed Fix to Task 0 of 09.2-06-PLAN.md. (b) Add `lr: 5e-5` and `epochs: 1` lines to the Branch 06A section of 09.2-DIAGNOSIS.md in a machine-readable form. (c) Execute a dry-run of just Task 0 + Task A1 pre-flight on main to confirm dispatch resolves to 06A only and hyperparameter parsing succeeds. (d) If pre-flight passes, proceed to the actual retrain (3.5 h sequential MPS)."
- expecting: "Task 0 exits with SELECTED=['06A'] only. Task A1 pre-flight python validator accepts lr=5e-5, epochs=1 and completes the smoke training step. Ready to launch Task A2 (full retrain) on main without further plumbing."
- next_action: "Completed — both plumbing blockers are fixed, verified, and committed. See Resolution section below."
- reasoning_checkpoint: "All dry-runs passed. The only remaining work is the 3.5 h sequential MPS training run itself, which must NOT happen inside this debug loop — see Resolution → next_steps for the re-dispatch launch command."

## Resolution

**root_cause:**
Five distinct issues compounded at the 2026-04-23 dispatch attempt. Two plumbing bugs were root causes of the pre-flight halt:
(1) Task 0's marker-write script (09.2-06-PLAN.md lines 148-158) used a line-oblivious regex that matched the "[x] User reviewed ... caveat in 06C" acknowledgement line as well as the real `**06A**` approval line, yielding `SELECTED=['06A','06C']` and exit 2.
(2) 09.2-DIAGNOSIS.md had the approved hyperparameters embedded in Markdown bullet prose (`  - \`lr\`: 1e-4 → 5e-5`) instead of on their own lines, so the Task A1 pre-flight `grep -E '^lr:[[:space:]]'` pipeline returned empty and python float() failed the STOP guard.
The other three reported "blockers" were either resolved between the halt and this session (Blocker 3 — 09.2-07 rebalanced the dataset to 7,519 samples) or are architectural constraints that must be honoured at the dispatch layer rather than the code layer (Blockers 4+5 — the retrain must run on main, sequential MPS, not inside a parallel worktree executor).

**fix:**
- Blocker 1 (plumbing): patched 09.2-06-PLAN.md Task 0 second script block to use the same tight regex `(?im)^Approved:\s*06X\s*$` as the first script block, plus an explicit `sys.exit(2)` when the match count is not exactly 1 (commit `2c0ff31`).
- Blocker 2 (plumbing): no file edit required. DIAGNOSIS.md was already touched after the halt (during 09.2-07 finalization) and contains `Approved: 06A` (line 184), `lr: 5e-5` (line 188), `epochs: 1` (line 189) on their own lines. Verified via grep.
- Blocker 3 (architectural, data rework): already resolved by Plan 09.2-07 before this debug session started (7,519 rebalanced samples on disk; ROADMAP [x]).
- Blockers 4+5 (architectural, dispatch mode): not fixed via code. Enforced at the re-dispatch layer via the launch instructions in "next_steps" below. The user must invoke `/gsd-execute-plan 09.2-06` on main (not in a worktree) with sequential `&&`-chained commands per D-11 and user's no-parallel-evals rule.
- ROADMAP hygiene: reverted the 09.2-06 checkbox from `[x]` to `[ ]` with a "HALTED 0/4 at pre-flight" annotation and updated the phase plan counter from 7/7 to 6/7 (commit `6943f58`).

**verification:**
All verification commands ran on main at `/Users/lakshman/Documents/Lyra`:

1. Task 0 validator block (plan lines 125-145):
```
.venv/bin/python - <<'PY'
import re, sys
text = open(".planning/phases/09.2-tool-call-regression-diagnosis/09.2-DIAGNOSIS.md").read()
branches_mentioned = {b: bool(re.search(fr"(?im)^Approved:\s*{b}\s*$", text)) for b in ("06A","06B","06C")}
selected = [b for b,v in branches_mentioned.items() if v]
if len(selected) != 1: sys.exit(2)
print(f"APPROVED_BRANCH={selected[0]}")
PY
```
Expected: `APPROVED_BRANCH=06A`, exit 0. Actual: `APPROVED_BRANCH=06A`, exit 0. ✓

2. Task 0 marker-write block (PATCHED, plan lines 148-163, run in dry-run mode to a side-marker):
```
.venv/bin/python - <<'PY'
import re, sys
text = open('.planning/phases/09.2-tool-call-regression-diagnosis/09.2-DIAGNOSIS.md').read()
selected = [b for b in ('06A','06B','06C') if re.search(fr'(?im)^Approved:\s*{b}\s*$', text)]
if len(selected) != 1: sys.exit(2)
open('.planning/phases/09.2-tool-call-regression-diagnosis/.approved_branch_DRYRUN','w').write(selected[0])
PY
```
Expected: writes "06A" to side-marker, exit 0. Actual: writes "06A", exit 0. ✓

3. Task A1 pre-flight hyperparameter parse (plan lines 216-236):
```
DIAGNOSIS=.planning/phases/09.2-tool-call-regression-diagnosis/09.2-DIAGNOSIS.md
LR=$(grep -E '^lr:[[:space:]]' "$DIAGNOSIS" | head -n1 | awk '{print $2}')
EPOCHS=$(grep -E '^epochs:[[:space:]]' "$DIAGNOSIS" | head -n1 | awk '{print $2}')
python3 -c "lr=float('${LR}'); ep=int('${EPOCHS}'); assert 1e-7<lr<1e-2; assert 1<=ep<=10; print(f'lr={lr} epochs={ep}')"
```
Expected: `lr=5e-05 epochs=1`, exit 0. Actual: `lr=5e-05 epochs=1`, exit 0. ✓

4. Main-repo artifact presence:
- `datasets/assembled/` → dataset_dict with train=7519, validation=419, test=416 ✓
- `models/lyra-merged/model.safetensors` → 3.4 GB present ✓
- `models/lyra-adapter/adapter_model.safetensors` → 72 MB present ✓
- `scripts/train.py` → --lr, --epochs, --max-steps, --no-merge, --output-dir all present ✓

5. No stale `.approved_branch` marker present on main (would confuse re-dispatch). ✓

**files_changed:**
- `.planning/phases/09.2-tool-call-regression-diagnosis/09.2-06-PLAN.md` — patched Task 0 second script block to use tight regex with explicit exit 2 on ambiguous match (commit `2c0ff31`).
- `.planning/ROADMAP.md` — reverted 09.2-06 checkbox to `[ ]` with halt annotation; updated Phase 09.2 plan counter from 7/7 to 6/7 (commit `6943f58`).
- `.planning/debug/09-2-06-retrain-blockers.md` — this file, now marked `status: resolved`.
- No edits to `09.2-DIAGNOSIS.md` (already correct — Approved/lr/epochs lines present at 184/188/189).

**next_steps (how to launch the real retrain — NOT executed inside this debug loop):**

The repo is now in a state where the Branch 06A retrain can execute end-to-end without further plumbing. Launch it sequentially on the main repo:

```bash
# Run from /Users/lakshman/Documents/Lyra on the main branch. NOT in a worktree.
# Takes ~4-5 h wall-clock on MPS; MUST NOT be run concurrently with any other
# model-loading process (per D-11 and user's feedback_no_parallel_evals.md).

/gsd-execute-plan 09.2-06
```

Environment / preconditions verified by this debug session:
- Working directory: `/Users/lakshman/Documents/Lyra` (main, NOT a worktree).
- `datasets/assembled/` = 7,519 train / 419 val / 416 test samples (09.2-07 rebalanced).
- `models/lyra-merged/` and `models/lyra-adapter/` contain v2 artifacts that will be overwritten by v3.
- No stale `.planning/phases/09.2-tool-call-regression-diagnosis/.approved_branch` marker.
- Task 0 will resolve cleanly to 06A (two dry-runs confirmed).
- Task A1 pre-flight will parse lr=5e-5 / epochs=1 cleanly (dry-run confirmed).

Expected timeline for the retrain:
- Task 0 dispatch: <1 s
- Task A1 pre-flight + smoke (`--max-steps 10`): ~2 min
- Task A1 full retrain (`--lr 5e-5 --epochs 1` on 7,519 samples): ~3.5 h
- Task A1 merge + template persistence check: ~5 min
- Task A2 knowledge eval (`--benchmarks knowledge --limit 100`): ~20-40 min
- Task A2 custom eval: ~15-25 min
- Task A2 eval_merge: trivial
- Task A3 D-09 gate + SUMMARY: ~5 min (automated) + human review
- **Total: ~4-5 h strictly sequential on MPS.**

Stop/retry conditions (encoded in the plan itself, surfaced here so the user knows to expect them):
- If smoke loss is non-finite or the process OOMs in Task A1 Step 1 — stop.
- If merged `tokenizer_config.json` fails the template-persistence assertion — stop.
- If D-09 gate FAIL in Task A3: Plan 06 does NOT silently iterate. User must choose v4 retrain (new hypothesis, new plan number) / pivot to 06C / accept partial per the plan's scope-boundary clause.

Known residual risks not addressed by this debug session (user should be aware before greenlighting):
- RESEARCH.md predicts <50% success probability even with the data rework — the model may regress again on a different axis. The plan has explicit fallback handling for this.
- The old worktrees (`.claude/worktrees/agent-*`) remain on disk, locked. They do not interfere with a main-repo dispatch but could be pruned via `git worktree remove --force` if desired; not a blocker.
