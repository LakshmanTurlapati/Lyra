---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 09.1-01-PLAN.md
last_updated: "2026-04-22T05:14:52.065Z"
last_activity: 2026-04-22 -- Phase --phase execution started
progress:
  total_phases: 11
  completed_phases: 8
  total_plans: 32
  completed_plans: 26
  percent: 81
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-20)

**Core value:** Curate Opus-quality training data that makes a 1.7B parameter model practically useful for day-to-day development tasks -- tool calls, quick code, and general reasoning.
**Current focus:** Phase --phase — 09.1

## Current Position

Phase: --phase (09.1) — EXECUTING
Plan: 2 of 6
Status: Executing Phase 09.1
Last activity: 2026-04-22 -- Phase --phase execution started

Progress: [████████░░] 81%

## Performance Metrics

**Velocity:**

- Total plans completed: 21
- Average duration: --
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 3 | - | - |
| 02 | 2 | - | - |
| 03 | 2 | - | - |
| 04 | 4 | - | - |
| 05 | 3 | - | - |
| 06 | 3 | - | - |
| 07 | 2 | - | - |
| 08 | 2 | - | - |

**Recent Trend:**

- Last 5 plans: --
- Trend: --

*Updated after each plan completion*
| Phase 01 P01 | 4min | 2 tasks | 9 files |
| Phase 01 P02 | 8min | 2 tasks | 8 files |
| Phase 01 P03 | 3min | 2 tasks | 5 files |
| Phase 02 P01 | 4min | 2 tasks | 4 files |
| Phase 02 P02 | 6min | 2 tasks | 7 files |
| Phase 03 P01 | 2min | 2 tasks | 4 files |
| Phase 03 P02 | 4min | 2 tasks | 4 files |
| Phase 04 P01 | 10min | 2 tasks | 4 files |
| Phase 04 P02 | 4min | 2 tasks | 41 files |
| Phase 04 P03 | 2min | 2 tasks | 27 files |
| Phase 04 P04 | 7min | 2 tasks | 4 files |
| Phase 05 P01 | 8min | 2 tasks | 2 files |
| Phase 05 P02 | 1min | 2 tasks | 68 files |
| Phase 05 P03 | 7min | 2 tasks | 3 files |
| Phase 06 P01 | 36min | 1 tasks | 2 files |
| Phase 06 P02 | 2min | 2 tasks | 67 files |
| Phase 06 P03 | 9min | 2 tasks | 2 files |
| Phase 07 P01 | 6min | 1 tasks | 3 files |
| Phase 07 P02 | 2min | 1 tasks | 1 files |
| Phase 08 P01 | 6min | 2 tasks | 3 files |
| Phase 08 P02 | 2min | 2 tasks | 2 files |
| Phase 09-benchmarking-and-core-release P01 | 4min | 2 tasks | 5 files |
| Phase 09-benchmarking-and-core-release P02 | 3min | 2 tasks | 2 files |
| Phase 09-benchmarking-and-core-release P03 | 2min | 1 tasks | 1 files |
| Phase 09.1 P01 | 3min | 2 tasks | 4 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: Evaluation framework built before training (Phase 3 before Phase 8) per research recommendation
- [Roadmap]: Domain datasets generated independently (Phases 4/5/6 parallelizable) to isolate failures per domain
- [Roadmap]: Format validation is foundational -- Phase 1 locks the ShareGPT schema before any data generation at scale
- [Phase 01]: TRL-native messages/role/content format over classic ShareGPT from/value -- TRL SFTTrainer expects this natively
- [Phase 01]: Pydantic model_validator(mode=after) for structural rule enforcement across conversation messages
- [Phase 01]: Strict system-first rule to prevent SmolLM2 default system prompt injection
- [Phase 01]: Pre-process TRL-native tool_calls to SmolLM2 <tool_call> XML format before tokenization -- SmolLM2 chat template does not handle structured tool_calls
- [Phase 01]: Use return_dict=True for apply_chat_template on transformers 5.x -- return_tensors=None returns BatchEncoding not list
- [Phase 01]: EOS check strips trailing whitespace tokens -- SmolLM2 template ends with im_end+newline so EOS is second-to-last
- [Phase 01]: YAML over JSON for prompt templates -- multi-line strings for system prompts are more readable
- [Phase 01]: 7 shared system prompts cross-referenced by domain templates via system_prompt_ref field
- [Phase 02]: Response-scope dedup as default -- catches response homogeneity (mode collapse)
- [Phase 02]: Diversity signal as placeholder at sample level -- batch dedup handles actual diversity per D-03
- [Phase 02]: Meta-commentary patterns checked only in assistant messages -- user messages are not flagged
- [Phase 02]: Separated pipeline_config.py from curate_pipeline.py to avoid circular imports between style_validator tests and pipeline module
- [Phase 02]: Used or-fallback pattern for Pydantic model_dump() Optional None values in style config
- [Phase 03]: Plain f-string table formatting for compare output -- zero external dependencies, consistent with curate_pipeline.py
- [Phase 03]: EvalResult.model_validate_json for safe JSON loading at trust boundary per T-03-02
- [Phase 03]: Lazy imports for torch/lm_eval/transformers inside functions -- avoids import errors when packages not installed
- [Phase 03]: Model path validation via regex before subprocess/library calls per T-03-07
- [Phase 04]: 64 schemas total: 41 developer, 18 everyday, 3 MCP meta, 2 CLI (64% developer weight per D-04)
- [Phase 04]: Template-based query generation with placeholder filling for diversity and seeded Random per batch for reproducibility
- [Phase 04]: Edge case distribution uses explicit count allocation (22% no-tool + 4% error) rather than modulo-based splitting
- [Phase 04]: Multi-turn samples use exactly 2 tool call rounds per happy-path conversation (sufficient for 1750 token budget)
- [Phase 04]: MCP discovery pattern inserts user confirmation step between list_servers and list_tools for realism
- [Phase 04]: Per-domain dedup config: tool-calling uses full-scope dedup at 0.9 threshold instead of global response-scope at 0.7
- [Phase 04]: Dedup includes tool_calls serialization: function names + arguments included in comparison text for tool-calling domain
- [Phase 04]: Edge case no-tool samples retained (405 of 2470): teaches model when NOT to call tools
- [Phase 05]: Language-specific code pools with idiomatic patterns per language for code generation training data
- [Phase 05]: Debugging entries stored as explicit (query, response) tuples for exact Bug/Fix format control per D-06
- [Phase 05]: Sequential seed strategy (batch*100 + offset) for reproducibility across categories
- [Phase 05]: Force-add (git add -f) to override datasets/**/*.jsonl gitignore rule (same as Phase 4)
- [Phase 05]: Code domain uses user-response dedup scope (excludes system prompts) at 0.98 threshold for template-generated data
- [Phase 05]: max_prose_ratio relaxed from 0.4 to 0.6 for code domain to accommodate Bug/Fix debugging format
- [Phase 06]: Topic pool approach with 192 pre-written unique Q&A pairs for knowledge generation, cross-domain fallback for pool exhaustion
- [Phase 06]: Sequential seed strategy (batch*100 + category_offset) for reproducible knowledge batch generation
- [Phase 06]: Knowledge domain dedup tuned to 0.995 threshold with user-response scope; 560 samples maximum achievable from 192-topic pool generation
- [Phase 06]: Knowledge min_tokens relaxed from 200 to 120 for Q&A samples in 131-198 token range
- [Phase 07]: Manual stratified split over datasets library stratify_by_column -- ClassLabel type requirement incompatible with string domain values
- [Phase 08]: Lazy imports for torch/peft/trl inside functions rather than module-level -- enables testing without ML library installation
- [Phase 08]: Mock module injection pattern (sys.modules) for test environment -- tests create lightweight mock torch/peft/trl when real packages unavailable
- [Phase 08]: Move argparse before heavy imports in main() so --help works without torch/peft/trl installed
- [Phase 08]: Disable epoch-based save/eval strategies when max_steps is active to avoid trainer errors on early exit
- [Phase 08]: Subprocess-based integration test runs train.py as subprocess to validate actual CLI entry point end-to-end
- git-lfs installed via brew (system tool) not pip -- .gitattributes committed before any model files staged per T-09-01 threat mitigation
- Wave 0 test scaffolding written in RED state intentionally -- Plans 02, 03, 05 will turn them GREEN
- lm-eval pinned at 0.4.11 with [hf] extra for HuggingFace model backend support in knowledge benchmarks
- Module-level None references for load_from_disk/run_inference_on_sample/_load_model_and_tokenizer assigned lazily in run_custom_eval() body enable pytest monkeypatching without import-time side effects
- eval_merge.py uses model metadata from first file and appends second file categories -- asymmetric merge is intentional per Pattern 3 in RESEARCH.md
- write_benchmark_md uses plain f-string table formatting consistent with format_compare_table (no external deps, project convention)
- format_mermaid_bar_chart emits two bar series (baseline first, candidate second) per xychart-beta specification
- Jinja fallback reads chat_template.jinja when tokenizer.chat_template is None (D-03 fix for eval regression)
- Template persistence writes inline to tokenizer_config.json after merge with generation markers stripped (D-04)

### Pending Todos

None yet.

### Roadmap Evolution

- Phase 09.1 inserted after Phase 9: Tool-Call Format Regression Fix (URGENT) — tool-call-format score regressed from 0.41 to 0.19 after fine-tuning, requires root cause analysis, potential data scaling to 25K samples, retrain, and re-evaluation

### Blockers/Concerns

- SmolLM2 chat template specifics for tool calls need hands-on verification in Phase 1
- Optimal LoRA hyperparameters for 3-way multi-task at 1.7B need empirical sweep in Phase 8
- **CRITICAL:** Lyra tool-call-format score regressed -0.22 vs base model — Phase 09.1 inserted to investigate and fix before release

## Session Continuity

Last session: 2026-04-22T05:14:52.060Z
Stopped at: Completed 09.1-01-PLAN.md
Resume file: None

**Planned Phase:** 09.1 (tool-call-format-regression-fix) — 6 plans — 2026-04-22T05:04:14.737Z
