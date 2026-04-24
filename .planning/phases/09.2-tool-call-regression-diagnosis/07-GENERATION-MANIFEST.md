# 07-GENERATION-MANIFEST

Generation manifest for Phase 09.2 Plan 07 — dataset rework via parallel Claude Code subagents. Documents per-wave subagent counts, outputs, wall-clock, drop rates, and the downsample step that delivered threshold compliance.

## Architecture

- Orchestrator: main Claude Code session (this context)
- Workers: `general-purpose` subagents spawned via the `Agent` tool
- Parallelism per wave: 10 concurrent subagents
- Each subagent wrote its JSONL output directly to disk and returned a compact confirmation
- No Anthropic API key required — subagents execute inside the Claude Code harness
- No local ML model loads (respects "no parallel evals" constraint per user memory)

## Wave A — Suffix pool expansion

| Metric | Value |
|--------|-------|
| Subagents | 1 |
| Target | 30 diverse ending phrases |
| Accepted | 30 |
| Drop rate | 0% |
| Blacklist violations | 0 |
| Artifact | `07-SUFFIX-POOL.md` |

## Wave B — Single-turn tool-call-ending samples

Addresses D-06 pathology #1: `pct_end_in_tool_call == 0.0`.

Each sample: 3 messages — system, user, assistant with empty content and populated `tool_calls`. No tool-result follow-up. Validator (`scripts/validate_format.py`) updated to permit a final assistant with `tool_calls` to have zero subsequent tool-result turns (see regression test `test_final_assistant_tool_call_without_result_passes`).

| Subagent | Domain hint | Target | Accepted | Unique tools |
|----------|-------------|-------:|---------:|-------------:|
| 01 | weather / geo / calendar | 50 | 50 | 31 |
| 02 | email / messaging / chat | 50 | 50 | 37 |
| 03 | file / git / shell | 50 | 50 | 50 |
| 04 | web / search / scraping | 50 | 50 | 29 |
| 05 | database / SQL / analytics | 50 | 50 | 45 |
| 06 | code / build / test | 50 | 50 | 38 |
| 07 | tasks / project mgmt | 50 | 50 | 50 |
| 08 | finance / stocks / crypto | 50 | 50 | 50 |
| 09 | cloud / devops | 50 | 50 | 50 |
| 10 | IoT / media / e-commerce | 50 | 50 | 44 |
| **Total** | | **500** | **500** | **421 distinct** |

Drop rate: 0% after validator relaxation. `pct_end_in_tool_call` verified at 100%.

## Wave C — Multi-turn diverse-ending samples

Addresses D-06 pathology #2: top-5 canned-suffix coverage 46% → <=20%.

Each sample: 5 messages — system, user, assistant(tool_call), tool(result), assistant(summary). The final assistant summary opens with one of the 30 Wave A pool phrases; each phrase distributed ~10 times per 300-sample batch.

| Subagent | Domain hint | Target | Accepted | Unique tools | Suffix coverage |
|----------|-------------|-------:|---------:|-------------:|----------------:|
| 01 | weather / travel | 300 | 300 | 32 | 100% |
| 02 | email / messaging | 300 | 300 | 30 | 100% |
| 03 | file / git / shell | 300 | 300 | 39 | 100% |
| 04 | web / search | 300 | 300 | 37 | 100% |
| 05 | database / SQL | 300 | 300 | 32 | 100% |
| 06 | code / build | 300 | 300 | 36 | 100% |
| 07 | tasks / project mgmt | 300 | 300 | 39 | 100% |
| 08 | finance | 300 | 300 | 37 | 100% |
| 09 | cloud / devops | 300 | 300 | 52 | 100% |
| 10 | IoT / media | 300 | 300 | 32 | 100% |
| **Total** | | **3000** | **3000** | **365 distinct** | **97.33% aggregate** |

Drop rate: 0%. Blacklisted openers (the five canned prefixes): 0 hits.

## Wave D — Code and knowledge supplementary samples

Addresses D-06 pathology #3: tool-calling domain fraction 90.22% → <=80%.

Each sample: 3 messages (or 5 with a 20-30% follow-up rate). `tools: []`, `domain` in `{code, knowledge}`, no `tool_calls`.

| Subagent | Focus | Target | Accepted |
|----------|-------|-------:|---------:|
| code-01 | Python algorithms & data structures | 160 | 160 |
| code-02 | Python debugging & refactoring | 160 | 160 |
| code-03 | Python web / API | 160 | 160 |
| code-04 | Python testing & tooling | 160 | 160 |
| code-05 | Python data & scripts | 160 | 160 |
| knowledge-01 | Science & nature | 160 | 160 |
| knowledge-02 | History & world events | 160 | 160 |
| knowledge-03 | Math & reasoning | 160 | 160 |
| knowledge-04 | Technology & computing | 160 | 160 |
| knowledge-05 | Arts, culture, philosophy | 160 | 160 |
| **Total** | | **1600** | **1600** |

Drop rate: 0%. Two subagents (code-02, code-05) hit Anthropic `overloaded_error` 529 on first dispatch; both succeeded on retry.

## Wave totals

| Wave | Target | Accepted | Drop rate |
|------|-------:|---------:|----------:|
| A (suffix pool) | 30 | 30 | 0% |
| B (single-turn tool-call) | 500 | 500 | 0% |
| C (multi-turn diverse) | 3000 | 3000 | 0% |
| D (code + knowledge) | 1600 | 1600 | 0% |
| **Grand total (B+C+D)** | **5100** | **5100** | **0%** |

## Merge + Downsample

The naïve merge (new samples appended to curated, exact-content dedup) produced:

| Curated file | Before | After merge | Delta |
|--------------|-------:|------------:|------:|
| tool-calling-curated.jsonl | 11,992 | 15,492 | +3,500 |
| code-curated.jsonl | 649 | 1,449 | +800 |
| knowledge-curated.jsonl | 651 | 1,451 | +800 |

With 0 exact-match duplicates. Post-merge assembly produced a train split that still failed all three thresholds because the 10,793 legacy pathological tool-calling samples numerically dominated the 3,500 new ones:

| Metric | Post-merge | Target |
|--------|-----------:|-------:|
| tool-calling ratio | 0.8423 | <=0.80 FAIL |
| pct_end_in_tool_call | 0.0325 | >=0.05 FAIL |
| top_5_canned_suffix_coverage | 0.3557 | <=0.20 FAIL |

Applied Plan 07 deviation Rule 3 option (b) — **targeted downsample**. Dropped every legacy tool-calling sample whose final-assistant content opens with any of the 7 blacklisted/top-10 D-06 canned prefixes:

```
"I've gathered all the information"
"I've completed the task"
"Here's what I found:"
"Based on the results,"
"The results show that"
"the result is success"
"The tool returned an error"
```

| Group | Samples |
|-------|--------:|
| Dropped (canned-ending legacy) | 10,038 |
| Kept: single-turn ends-in-tool_call | 500 |
| Kept: multi-turn non-canned | 4,954 |
| tool-calling after downsample | **5,454** |

## Reassembly

`python scripts/assemble_dataset.py assemble --output-dir datasets/assembled --seed 42` on the post-downsample curated JSONLs.

| Split | Total | tool-calling | code | knowledge |
|-------|------:|-------------:|-----:|----------:|
| train | 7,519 | 4,909 (65.3%) | 1,304 (17.3%) | 1,306 (17.4%) |
| validation | 419 | 273 (65.2%) | 73 (17.4%) | 73 (17.4%) |
| test | 416 | 272 (65.4%) | 72 (17.3%) | 72 (17.3%) |

## Threshold verification (`tests/test_phase_09_2/test_training_audit.py`)

Test thresholds were tightened from the archival 09.1 ceilings to the Plan 07 rebalance targets.

| Metric | Before Plan 07 | Target | After Plan 07 | Status |
|--------|---------------:|-------:|--------------:|:------:|
| pct_end_in_tool_call | 0.0000 | >= 0.05 | **0.0937** | PASS |
| top_5_canned_suffix_coverage | 0.4597 | <= 0.20 | **0.1695** | PASS |
| tool-calling domain ratio | 0.9022 | <= 0.80 | **0.6529** | PASS |

`pytest tests/test_phase_09_2/test_training_audit.py -v` → 4 passed (ending floor, ending ceiling guardrail, canned-suffix coverage, domain skew).

## Handoff

Plan 06A retrain may now proceed — the rebalanced dataset lives at `datasets/assembled/`, the audit thresholds are locked as CI gates, and the canned-suffix pathology is both removed from training data and guarded against regression.
