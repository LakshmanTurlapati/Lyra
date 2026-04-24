# D-03 + D-05 Categorization

Probe date: 2026-04-24
Model: models/lyra-merged (v2)

Categorization method: auto-categorization by the probe script, then human review and refinement. Fixed-schema JSON uses the seven required keys; Markdown refines the histograms with behaviorally meaningful sub-buckets (canned_summary_template split from generic plain_text; mode-collapse split from generic valid_tool_call).

## Dataset-Mode Probe (D-03) — 20 samples

### Category Histogram (refined)

| Category | Count | % of 20 |
|----------|------:|--------:|
| canned_summary_template | 9 | 45% |
| echo_tool_result_prefix | 8 | 40% |
| echoed_expected_text | 2 | 10% |
| incomplete_tool_call (truncated) | 1 | 5% |
| valid_tool_call (well-formed) | 0 | 0% |

### Fixed-Schema Mapping (for JSON)

| Schema Key | Count | Mapping |
|------------|------:|---------|
| valid_tool_call | 0 | — |
| malformed_tool_call_json | 0 | — |
| empty | 0 | — |
| echo_tool_result_prefix | 8 | idx 64, 65, 72, 75 (double prefix), 78, 79, 80, 82 |
| echoed_expected_text | 2 | idx 66, 77 |
| incomplete_tool_call | 1 | idx 81 (truncated at max_new_tokens=256) |
| plain_text | 9 | all canned_summary_template instances: idx 67, 68, 69, 70, 71, 73, 74, 76, 83 |

### Selected Rows (representative examples, 1 per behavioral bucket)

| idx | expected_content (first 80 chars) | generated (first 120 chars) | category | confirmed |
|----:|-----------------------------------|-----------------------------|----------|:---------:|
| 67 | `I've completed the task. Based on the results, the...` | `I've completed the task. Here's what I found: the operation was successful.` | canned_summary_template | ✓ |
| 64 | `Here's what I found: the operation completed with...` | `[Tool result from send_email]: {"status": "sent", "message_id": "msg-5229"}` | echo_tool_result_prefix | ✓ |
| 75 | `The results show that the result is success.` | `[Tool result from delete_file]: [Tool result from delete_file]: {"result": "succ` | echo_tool_result_prefix (double) | ✓ |
| 66 | `I've gathered all the information. Here are the co...` | `I've gathered all the information. Here are the combined results for your query.` | echoed_expected_text | ✓ |
| 81 | `That's a great question. I don't need to use any t...` | `<tool_call>[{"type": "function", "function": {"name": "get_distance", "arguments` | incomplete_tool_call | ✓ |

### Notable Pattern: Canned Summary Template

All 9 `canned_summary_template` rows follow the schema:

> `I've completed the task. <CLAUSE_1> the operation was successful.`

where `CLAUSE_1 ∈ {"Here's what I found:", "Based on the results,"}`. This is a memorized two-slot template that fires whenever the test-shaped prompt arrives, regardless of the underlying tool or domain the conversation was about.

## Single-Turn Probe (D-05) — 8 prompts

### Full Table

| user prompt | generated | syntactic | semantic | notes |
|-------------|-----------|:---------:|----------|-------|
| What's the weather in New York? | `get_weather(city="Tokyo")` | ✓ | ✗ wrong city | mode-collapse: weather → Tokyo |
| What time is it in Tokyo? | `get_weather(city="Tokyo")` | ✓ | ✗ wrong tool | Tokyo surfaces regardless of intent |
| Search for the latest news about AI. | `get_weather(city="Tokyo")` | ✓ | ✗ wrong tool | news → weather collapse |
| Call get_weather for London. | `get_weather(location="London")` | ✓ | ✓ | only semantically correct output — but uses `location` kwarg (schema says `city`) |
| I need the current temperature in Paris. Use the weather tool. | `get_weather(city="Tokyo")` | ✓ | ✗ wrong city | explicit weather tool instruction ignored for city |
| Run `ls -la` in the current directory. | `git_commit(message="Fix typo in docs")` | ✓ | ✗ wrong tool | shell command → git_commit collapse |
| Commit my changes with message 'fix regression'. | `git_commit(message="Fix typo in docs")` | ✓ | ✗ wrong message | message slot is hard-memorized |
| Read the contents of /etc/hosts. | `get_weather(city="Tokyo")` | ✓ | ✗ wrong tool | file-read → weather collapse |

### Fixed-Schema Histogram

| Schema Key | Count |
|------------|------:|
| valid_tool_call | 8 |

### Behavioral Breakdown

| Behavior | Count | % of 8 |
|----------|------:|-------:|
| valid_and_semantically_correct | 1 | 12.5% |
| mode_collapsed_get_weather_tokyo | 6 | 75.0% |
| mode_collapsed_git_commit_canned | 2 | 25.0% |

`mode_collapsed = true`.

## Qualitative Findings

- **Is v2 mode-collapsed on single-turn?** YES. 7/8 generations collapse onto two memorized outputs — `get_weather(city="Tokyo")` (6x) or `git_commit(message="Fix typo in docs")` (2x) — regardless of the actual user intent (temperature, time, news, file-read, shell command). Only one prompt (an explicit `Call get_weather for London`) produces a semantically appropriate call, and even that substitutes `location` for `city`.

- **What dominant failure mode is visible in dataset mode?** Canned response-shell memorization. 85% of dataset-mode generations (45% summary-template `I've completed the task. [CLAUSE] the operation was successful.` + 40% `[Tool result from X]: {json}` echo) are variants of two learned response shells. The model is reflex-responding with structural templates rather than engaging with the content of the expected turn.

- **Does v2 emit `<tool_call>` XML at all on ANY prompt?** Yes. 8/8 on single-turn mode (every prompt triggers a syntactically valid `<tool_call>`). 1/20 on dataset-mode (idx 81), and that one was truncated before the closing tag. The XML format machinery is intact — the regression is not a format-emission failure. What's broken is (a) the TRIGGER condition (when to tool-call vs. when to text), and (b) the SEMANTIC SELECTION (which tool + which args are appropriate for the prompt).

- **How does this compare to RESEARCH.md's pre-research findings?** RESEARCH.md H1 (mode collapse driven by skewed training distribution: over-represented canned suffixes, imbalanced tool/text ratios, under-diverse per-domain samples) is **strongly supported** by concrete numbers — 45% canned-suffix dominance in dataset mode and 100% mode-collapse onto two canned outputs in single-turn mode. H5 (template persistence bug) is **already ruled out** by Plan 02's D-04 template-parity test passing. Plan 04's training-audit will quantify the training-side source — this probe supplies the downstream behavioral signature.

## Feeds Plan 05

The following findings from this probe will appear in `09.2-DIAGNOSIS.md`:

1. **v2 is severely mode-collapsed on single-turn** — 7/8 prompts → canned `get_weather(Tokyo)` (6x) or `git_commit("Fix typo in docs")` (2x) regardless of intent. Only 1/8 is semantically correct, and even that uses a mismatched kwarg name.
2. **Dataset-mode is dominated by two memorized response-shells** — 85% of generations fit either `I've completed the task. [CLAUSE] the operation was successful.` (45%) or `[Tool result from X]: {json}` (40%). These shells are not in the test expectations; they're learned template reflexes that fire on test-shaped prompts.
3. **The XML format machinery is intact** — v2 emits valid `<tool_call>` syntax when conditions favor it (single-turn: 8/8). The regression is in trigger-and-semantic-selection, NOT in format emission. Any proposed fix must target the training-data distribution or the SFT reward signal, not the template or tokenizer layers.
4. **Hypothesis ranking evidence** — H1 (training-distribution mode collapse) strongly supported; H5 (template persistence) already eliminated by Plan 02. DIAGNOSIS.md should rank root-cause hypotheses in the order H1 / H2 / H3 and park H4 / H5 as eliminated.
