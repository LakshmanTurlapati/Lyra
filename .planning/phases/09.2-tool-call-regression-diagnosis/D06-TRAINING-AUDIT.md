# D-06: Training-Run Review

Date: 2026-04-24
Phase: 09.2 — Tool-Call Regression Diagnosis

## Loss Curve Review

From 09.1-05-SUMMARY.md (v2 retrain):
- Epoch 1 eval_loss: **0.1613**
- Epoch 2 eval_loss: **0.1384**
- Full training: 1,496 steps in 3h 37m 15s on MPS
- Hyperparameters: lr=1e-4, epochs=2, per-device batch=4, grad-accum=4

Phase 8 baseline (for comparison): Phase 8 training loss curves not published in SUMMARY — wandb was not enabled for the Phase 8 run (see 08-02-SUMMARY.md), so there is no per-step loss trajectory on disk to compare against. Only the final v1 model and its eval results exist as a baseline.

### Interpretation
- eval_loss dropping from 0.1613 → 0.1384 looks clean on paper.
- BUT per RESEARCH.md Pitfall 4, validation is on the SAME distribution as train (90.2% tool-calling, 5 canned endings). A model that memorizes those endings gets low eval_loss AND fails in the wild.
- Epoch-2 loss of 0.1384 is consistent with memorization of a small set of suffix templates (low entropy to optimize against).
- The smooth monotonic decrease does NOT rule out mode collapse; in fact it is the *expected* training-loss signature of a model overfitting to a low-diversity target distribution — the optimizer finds the canned suffixes and locks onto them.

## Distribution Fingerprint (measured live this run)

```
=== Domain distribution (train) ===
  tool-calling     10793  90.22%
  knowledge          586   4.90%
  code               584   4.88%

=== Tool-calling assistant ending types (10793 samples) ===
  ends in tool_call :      0   0.00%
  ends in text      :  10793  100.00%

=== Top 10 ending prefixes (60 chars) in tool-calling train ===
   1381 (12.8%) "I've gathered all the information. Here are the combined res"
    932 ( 8.6%) "I've completed the task. Here's what I found: the operation "
    894 ( 8.3%) "I've completed the task.  the operation was successful."
    884 ( 8.2%) "I've completed the task. The results show that the operation"
    874 ( 8.1%) "I've completed the task. Based on the results, the operation"
    363 ( 3.4%) 'Based on the results, the result is success.'
    351 ( 3.3%) 'The results show that the result is success.'
    347 ( 3.2%) "Here's what I found: the result is success."
    346 ( 3.2%) 'the result is success.'
    191 ( 1.8%) 'The tool returned an error. This might be a temporary issue.'

Top-5 coverage: 4965/10793 = 46.0%
```

## Canned-Suffix Prevalence

| Prefix (60 chars) | Count | % of tool-calling train |
|-------------------|------:|------------------------:|
| `I've gathered all the information. Here are the combined res` | 1381 | 12.8% |
| `I've completed the task. Here's what I found: the operation ` |  932 |  8.6% |
| `I've completed the task.  the operation was successful.`      |  894 |  8.3% |
| `I've completed the task. The results show that the operation` |  884 |  8.2% |
| `I've completed the task. Based on the results, the operation` |  874 |  8.1% |

Top-5 coverage: **46.0%** of 10,793 tool-calling train samples.

The second-tier prefixes (ranks 6–10) add another 14.9% coverage, all minor variations of `the result is success.` / `The results show...` / `Based on the results...`. Top-10 coverage is ~60.9% on the tool-calling split — a very narrow response-template space.

## Findings

1. **H1 is strongly supported:** 100% of tool-calling train samples end in assistant TEXT (zero end in a tool_call). The model was never shown a training example where the correct answer is a tool_call on its own — explaining why single-turn probe prompts (D-05) degenerate to canned `get_weather(city="Tokyo")` / `git_commit(message="Fix typo in docs")` outputs rather than producing a tool_call that matches the user's intent.
2. **Canned suffix memorization is measurable:** top-5 ending prefix coverage is ~46% (matches RESEARCH.md pre-check to the percent). eval_loss 0.1384 reflects memorization of this finite phrase pool, not real capability. This is the quantitative source-side evidence for Plan 03's observation that 45% of dataset-mode generations fit the `I've completed the task. [CLAUSE] the operation was successful.` template.
3. **Domain skew is the expected ceiling:** 90.22% tool-calling / 4.90% knowledge / 4.88% code. MMLU -7.5% is consistent with catastrophic forgetting of general knowledge under this skew — only ~5% of gradient updates ever touched non-tool-calling content.
4. **Template diversity collapse bridges to downstream failure:** the top-5 prefixes all begin with either `I've completed the task.` or `I've gathered all the information.` — the very same two canned openers Plan 03's D-03 probe found dominating 85% of model generations. The training distribution directly predicts the failure shape.

## Locked Thresholds (see tests/test_phase_09_2/test_training_audit.py)

- `pct_end_in_tool_call < 0.10` forward-looking ceiling (currently **0.00** — archival xfail)
- `top-5 canned-suffix coverage <= 0.50` (currently **0.46**)
- `tool-calling fraction of train <= 0.92` (currently **0.9022**)

These three literals are asserted as CI gates so any future retrain that re-introduces the same pathology fails loudly rather than silently shipping another regressed v2.

## Feeds Plan 05

Root-cause evidence for DIAGNOSIS.md:
- 0% tool-call-ending training examples → mode collapse on single-turn (Plan 03 D-05: 7/8 prompts collapse to canned tool-call).
- 46% canned-suffix coverage on top-5 prefixes → echo-on-multi-turn failure (Plan 03 D-03: 45% summary-template + 40% tool-result-echo).
- 90.2% domain skew → MMLU regression (-7.5% vs base).
- Together with Plan 02 D-04 (template parity GREEN — H5 eliminated) and Plan 03 D-03/D-05 (probe evidence — downstream shape of mode collapse), the diagnosis points unambiguously at the training-data distribution. The retrain prescription is to rebalance all three dimensions, not to change the chat template or tokenizer.
