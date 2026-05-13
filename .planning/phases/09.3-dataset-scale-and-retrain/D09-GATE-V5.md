# D-09 Gate Verdict — v5

**Date:** 2026-05-13
**Model:** models/lyra-adapter-v5 (v5) / models/lyra-merged-v5
**Dataset:** datasets/assembled (test split: 272 tool-calling + 72 code + 72 knowledge = 416 samples)
**Device:** mps (Apple Silicon), float32, greedy decoding

## Verdict

**FAIL-below-parity** — v5 tool-call-format (0.2353) is significantly below both base (0.4065) and v4 (0.4044). This is a 17-point regression vs v4 and breaks the strict D-09 gate.

Per D-14, this is NOT an auto-pivot. User must pick ship-decision A / B / C.

## Strict D-09 Gate (tool-call-format > base 0.4065)

| Metric                | Value   | Target  | Result               |
|-----------------------|--------:|--------:|---------------------:|
| v5 tool-call-format   | 0.2353  | > 0.4065 | **FAIL-below-parity** |
| base tool-call-format | 0.4065  | reference | -                  |
| v4 tool-call-format   | 0.4044  | reference | -                  |

**v5 - base delta:** -0.1712 (-42% relative)
**v5 - v4 delta:** -0.1691 (-42% relative)

## D-10 Held (MMLU/ARC/HellaSwag within 5% of base)

| Benchmark | v5     | base   | delta   | within 5%? |
|-----------|-------:|-------:|--------:|:----------:|
| MMLU      | 0.4882 | 0.5012 | -0.0130 | ✓ (-2.6%)  |
| ARC       | 0.4600 | 0.4500 | +0.0100 | ✓ (+2.2%)  |
| HellaSwag | 0.6500 | 0.6600 | -0.0100 | ✓ (-1.5%)  |

**D-10 held:** YES

## D-13 Soft Target (code-syntax ≥ 0.40)

| Metric      | v5     | v4     | base   | ≥ 0.40? |
|-------------|-------:|-------:|-------:|:-------:|
| code-syntax | 0.5694 | 0.4861 | 0.2000 | ✓       |

**D-13 met:** YES (code-syntax improved +0.083 over v4, +0.369 over base)

## v5 vs base Comparison

```
Category        Benchmark         Metric    Baseline   Candidate    Delta
--------------- ----------------- --------- ---------- ----------- --------
knowledge       mmlu              acc         0.5012     0.4882    -0.0130
knowledge       arc_challenge     acc_norm    0.4500     0.4600    +0.0100
knowledge       hellaswag         acc_norm    0.6600     0.6500    -0.0100
custom          tool-call-format  pass@1      0.4065     0.2353    -0.1712
custom          code-syntax       pass@1      0.2000     0.5694    +0.3694
```

## v5 vs v4 Comparison

```
Category        Benchmark         Metric    Baseline   Candidate    Delta
--------------- ----------------- --------- ---------- ----------- --------
knowledge       mmlu              acc         0.4972     0.4882    -0.0089
knowledge       arc_challenge     acc_norm    0.4700     0.4600    -0.0100
knowledge       hellaswag         acc_norm    0.6600     0.6500    -0.0100
custom          tool-call-format  pass@1      0.4044     0.2353    -0.1691
custom          code-syntax       pass@1      0.4861     0.5694    +0.0833
```

## Overall Phase 09.3 Outcome

- Strict D-09 (tool-call-format > 0.4065): **FAIL-below-parity** (0.2353)
- D-10 held (knowledge within 5%):         **YES**
- D-13 soft (code-syntax ≥ 0.40):          **MET** (0.5694)

**Net read:** The continue-train on 42K rebalanced samples HURT tool-call-format despite improving code-syntax and holding knowledge. This is a targeted regression on the exact metric we were trying to fix, suggesting the 42K rebalanced dataset's tool-calling samples either:
1. Used a format inconsistent with the test split's expected format, or
2. Diluted tool-call signal vs other domains, or
3. Triggered catastrophic forgetting on the v4-learned tool-call behavior despite lr=1e-5.

## Next Steps — User Ship-Decision (per D-14)

Three options on the table:

**A. v6 retrain** — investigate WHY tool-call regressed (probe v5 outputs on the 272 tool-calling samples; compare formats), fix the dataset/training setup, retrain v6. Highest upside, longest path. Best if root cause is identifiable and fixable.

**B. Accept-partial ship** — ship v5 anyway as the "code-strong, tool-weak" variant. Code-syntax 0.5694 is a meaningful win. v4 already exists and is preserved as the tool-strong variant. Ship two adapters with documentation. Medium path.

**C. 06C revert-to-base** — abandon both v4 and v5 LoRA paths, ship the base SmolLM2-1.7B-Instruct with prompt-engineering fixes for tool calls. Lowest effort, lowest gain. Best if tool-call regression turns out to be a fundamental rebalanced-data problem we can't fix.

**Recommended first step before deciding:** probe v5's actual outputs on ~10 failing tool-call samples. If v5 is producing valid tool calls in a slightly different format (e.g. forgot `<tool_call>` wrapper, or used a different JSON shape), the "regression" may be a format mismatch fixable with a quick regex or training-data tweak. If v5 is producing genuinely wrong content (refusals, malformed JSON, no tool call at all), it's harder.
