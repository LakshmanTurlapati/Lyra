# D-07 Reproducibility Check

Run date: 2026-04-24 (amended after debug session)
Exit status: PASS (amended from FAIL — see "Amendment: root cause" below)

## Per-benchmark delta (threshold: abs(new - old) < 0.02)

```
('custom', 'code-syntax')                old=0.2000 new=0.1875 delta=0.0125 PASS
('custom', 'tool-call-format')           old=0.4065 new=0.3422 delta=0.0643 FAIL (apparent) -> N/A (real)
('knowledge', 'arc_challenge')           old=0.4500 new=0.4500 delta=0.0000 PASS
('knowledge', 'hellaswag')               old=0.6600 new=0.6600 delta=0.0000 PASS
('knowledge', 'mmlu')                    old=0.5012 new=0.5012 delta=0.0000 PASS
```

## Commands executed

```
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m scripts.eval_inference \
    --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --dataset-dir datasets/assembled \
    --output results/base_custom.json \
  && PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m scripts.eval_runner \
       --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
       --benchmarks knowledge \
       --output results/base_knowledge_repro.json \
       --limit 100
```

Both invocations returned exit code 0. Total wall-clock: ~59 minutes on MPS
(eval_inference ~20 min, eval_runner ~39 min). Chain was `&&`-sequential per
D-11 — no parallel model loads.

## Interpretation (original, superseded by Amendment below)

- **Knowledge benchmarks reproduce exactly.** mmlu, arc_challenge, and
  hellaswag all match the canonical `results/base_knowledge.json` to four
  decimal places. lm-eval-harness + the base weights on HuggingFace are
  behaving identically across runs. Knowledge eval is NOT the drifting
  component.

- **code-syntax is within tolerance.** 0.0125 drift on a 32-sample benchmark
  is a single-sample swing (1/32 = 0.03125), which is consistent with stable
  behavior given greedy decoding and the small denominator.

- **tool-call-format appeared to drift -0.0643 (0.4065 -> 0.3422).** This was
  flagged as a D-07 FAIL because the delta exceeded the 0.02 threshold.
  Further investigation (see Amendment) showed the delta was not a
  reproducibility failure — it was a test-set change.

## Amendment: root cause (added 2026-04-24 after debug session `improve-tool-call-baseline`)

The initial "tool-call-format drift" finding was NOT a reproducibility failure. It was an apples-to-oranges comparison caused by a legitimate test-set change from an earlier phase.

**Timeline:**

- 2026-04-21 18:52:49 — `results/base_custom.json` produced (Phase 9-04), tool-call-format = 0.4065. Test set: Phase 9 original assembly (pre-9bec343).
- 2026-04-22 01:02:58 — Commit `9bec343 feat(09.1-04): curate all three domains at full scale from combined raw data` rewrote every curated source JSONL.
- 2026-04-23 20:53:59 — `datasets/assembled/` rebuilt automatically, now reflecting the 9bec343 re-curation.
- 2026-04-23 22:23:07 — D-07 repro run produced 0.3422. Test set: Phase 9.1-04 re-curated assembly.

**Why it looked like drift:** The 599 tool-calling test samples in the "original" and "re-curated" assemblies are different sample compositions (both totaling 599, but drawn from different curated source JSONLs via stratified split). Running the same base model against different samples naturally produces different scores.

**Why it was not an environment issue:**

- Package versions were identical across both runs (verified via the surviving `venv/` from 2026-04-20 22:54 which carries torch 2.11.0 / transformers 5.5.4 / tokenizers 0.22.2 — the exact set current in `.venv/`). No torch/transformers/tokenizers upgrade ever occurred between the two measurements.
- Only eval-code change between runs was `f59ca34` (defensive chat_template loading), which is a structural no-op for base SmolLM2 whose tokenizer already ships a chat_template.
- Knowledge benchmarks reproduced bit-exactly, confirming general inference stability.

**Resolution:**

- `results/base_custom.json` is now the 0.3422 measurement against the Phase 9.1-04 test split (authoritative baseline).
- `results/base_custom_phase09_original.json` preserves the 0.4065 historical measurement against the pre-9bec343 test split (archived, not used for regression math).
- v1 and v2 Lyra were both trained AND evaluated AFTER 9bec343, so their measurements (`lyra_custom.json`, `lyra_v2_custom.json`) are already consistent with the 0.3422 baseline.
- Phase 09.2 regression math is now internally consistent: base (post-9bec343) 0.3422 -> v2 0.0634 (delta -0.2788). This is the true v2 regression magnitude.

**Gate status:** PASS under the amended interpretation. All benchmarks either reproduce bit-exactly (knowledge), are within single-sample tolerance (code-syntax), or were measuring legitimately different test sets (tool-call-format). No environment or code reproducibility failure exists.

**Wave 0 test scaffolding (Task 1) is complete** — pytest + tests/test_phase_09_2/ with 2 RED stubs and 3 fixtures collect cleanly. Plans 02-06 are cleared to proceed.

## Debug session

Full investigation record: `.planning/debug/resolved/improve-tool-call-baseline.md`.
