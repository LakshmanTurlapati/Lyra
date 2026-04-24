# D-07 Reproducibility Check

Run date: 2026-04-24
Exit status: FAIL

## Per-benchmark delta (threshold: abs(new - old) < 0.02)

```
('custom', 'code-syntax')                old=0.2000 new=0.1875 delta=0.0125 PASS
('custom', 'tool-call-format')           old=0.4065 new=0.3422 delta=0.0643 FAIL
('knowledge', 'arc_challenge')           old=0.4500 new=0.4500 delta=0.0000 PASS
('knowledge', 'hellaswag')               old=0.6600 new=0.6600 delta=0.0000 PASS
('knowledge', 'mmlu')                    old=0.5012 new=0.5012 delta=0.0000 PASS
```

## Commands executed

```
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m scripts.eval_inference \
    --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --dataset-dir datasets/assembled \
    --output results/base_custom_repro.json \
  && PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m scripts.eval_runner \
       --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
       --benchmarks knowledge \
       --output results/base_knowledge_repro.json \
       --limit 100
```

Both invocations returned exit code 0. Total wall-clock: ~59 minutes on MPS
(eval_inference ~20 min, eval_runner ~39 min). Chain was `&&`-sequential per
D-11 — no parallel model loads.

## Interpretation

- **Knowledge benchmarks reproduce exactly.** mmlu, arc_challenge, and
  hellaswag all match the canonical `results/base_knowledge.json` to four
  decimal places. lm-eval-harness + the base weights on HuggingFace are
  behaving identically across runs. Knowledge eval is NOT the drifting
  component.

- **code-syntax is within tolerance.** 0.0125 drift on a 32-sample benchmark
  is a single-sample swing (1/32 = 0.03125), which is consistent with stable
  behavior given greedy decoding and the small denominator.

- **tool-call-format drifted -0.0643 (0.4065 → 0.3422).** This is the single
  failing benchmark and is more than 3x the threshold. Because greedy
  decoding with deterministic template rendering SHOULD be bitwise
  reproducible against a pinned HuggingFace model snapshot
  (`31b70e2e869a7173562077fd711b654946d38674`, confirmed in the eval logs),
  either (a) the HuggingFace base weights changed between Phase 9 run
  (2026-04-21) and this run (2026-04-23), or (b) something in the inference
  path (torch/transformers minor-version updates, tokenizer caching, MPS
  numerics) shifted behavior enough to nudge ~60 of the 599 tool-calling
  test samples across the `check_tool_call_format` boundary.

## Gate behavior (plan Task 3 CRITICAL)

Per 09.2-01-PLAN.md Task 3:

> If the Python script exits non-zero (any benchmark drifted ≥ 0.02), do
> NOT proceed to Plans 02-04. Write "Exit status: FAIL" into
> D07-REPRODUCIBILITY.md, commit, and return to the user with an explicit
> escalation message.

## Escalation message to user

> D-07 reproducibility FAILED on `tool-call-format` (delta 0.0643,
> threshold 0.02). The eval pipeline has drifted since Phase 9. Knowledge
> benchmarks reproduce bit-for-bit so the drift is isolated to the
> `scripts/eval_inference.py` path or its dependencies. Per D-07 in
> CONTEXT.md, deep model diagnosis is invalid until the eval is
> re-stabilized. Please review this file and decide whether to:
>
> 1. Investigate the eval drift (regenerate base_custom.json as the new
>    reference, or pin torch/transformers/tokenizers versions and re-run),
>    OR
> 2. Proceed with Plans 02-06 using the new 0.3422 number as the base
>    (knowing the 0.41 bar in D-09 may need renegotiation — or the v2
>    0.0634 score should be compared against 0.3422, not 0.4065, when
>    assessing the regression magnitude),
>    OR
> 3. Revert to the Phase 9 commit that produced the original base_custom.json
>    and regenerate the repro in that environment.

**Wave 0 test scaffolding (Task 1) is complete regardless of this gate.**
pytest + tests/test_phase_09_2/ with 2 RED stubs and 3 fixtures are in
place and collect cleanly. Plans 02 and 04 can still unblock those stubs
*after* the user resolves the D-07 gate.
