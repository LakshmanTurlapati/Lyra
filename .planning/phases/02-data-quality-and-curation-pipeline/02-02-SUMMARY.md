---
phase: 02-data-quality-and-curation-pipeline
plan: 02
subsystem: data-quality
tags: [pydantic, yaml, pipeline, style-validation, curation, cli]

requires:
  - phase: 02-data-quality-and-curation-pipeline/01
    provides: quality_scorer.py (score_sample), dedup.py (deduplicate_batch)
  - phase: 01-data-format-and-pipeline-foundation
    provides: validate_format.py (Conversation model), templates/*.yaml
provides:
  - End-to-end curation pipeline (curate_pipeline.py) wiring 4 stages
  - Domain-specific style validation (style_validator.py)
  - Pydantic-validated YAML pipeline config (pipeline_config.py + configs/pipeline.yaml)
  - CLI entry point for batch curation with per-domain thresholds
affects: [data-generation, training, evaluation]

tech-stack:
  added: [pyyaml (safe_load for config)]
  patterns: [Pydantic config validation with per-domain overrides, 4-stage pipeline orchestrator, domain-specific style heuristics]

key-files:
  created:
    - scripts/style_validator.py
    - scripts/pipeline_config.py
    - scripts/curate_pipeline.py
    - configs/pipeline.yaml
    - tests/test_style_validator.py
    - tests/test_pipeline_config.py
    - tests/test_curate_pipeline.py
  modified: []

key-decisions:
  - "Separated pipeline_config.py from curate_pipeline.py to avoid circular imports -- style_validator and test modules both need config types"
  - "Used `or` fallback instead of dict.get() defaults for Pydantic model_dump() None values -- model_dump() serializes Optional[int]=None as {key: None} which defeats get() defaults"

patterns-established:
  - "Pydantic config with get_domain_config(): global defaults deep-merged with domain overrides for style sub-config"
  - "Pipeline stage pattern: format -> quality -> dedup -> style, each stage filters and the orchestrator tracks counts"
  - "Style config via `or` fallback: use `style.get('max_tokens') or 600` not `style.get('max_tokens', 600)` when config comes from Pydantic model_dump()"

requirements-completed: [DATA-03, DATA-04, DATA-05]

duration: 6min
completed: 2026-04-20
---

# Phase 02 Plan 02: Style Validator, Pipeline Config, and Orchestrator Summary

**End-to-end curation pipeline with YAML config, Pydantic validation, domain-specific style checks, and CLI entry point wiring quality_scorer + dedup + style_validator into 4-stage filtering**

## Performance

- **Duration:** 6 min
- **Started:** 2026-04-20T10:46:28Z
- **Completed:** 2026-04-20T10:52:42Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- Style validator enforces domain-specific output styles: code=terse (short, code-heavy), knowledge=detailed (long, reasoning markers), tool-calling=moderate length
- Pipeline config (configs/pipeline.yaml) provides full control over all thresholds with per-domain overrides for 3 domains
- Pipeline orchestrator wires all 4 stages (format, quality, dedup, style) with per-sample quality metadata in output JSONL
- CLI entry point: `python -m scripts.curate_pipeline --input <file> --domain <domain>` with --config and --output options

## Task Commits

Each task was committed atomically:

1. **Task 1: Style validator and pipeline config** - `a92b611` (test) + `91757a1` (feat)
2. **Task 2: Pipeline orchestrator wiring all stages** - `56f0eae` (test) + `b0e1427` (feat)

_TDD tasks have separate test (RED) and implementation (GREEN) commits._

## Files Created/Modified
- `scripts/style_validator.py` - Domain-specific style validation (terse/detailed/moderate) with 4 exported functions
- `scripts/pipeline_config.py` - Pydantic models (StyleConfig, DomainConfig, PipelineConfig) + load_config + get_domain_config
- `scripts/curate_pipeline.py` - Pipeline orchestrator: 4 stages, CLI entry point, per-domain config, quality metadata output
- `configs/pipeline.yaml` - Full pipeline configuration with defaults, 3 domain overrides, topic distribution, template paths
- `tests/test_style_validator.py` - 16 tests covering code/knowledge/tool-calling domains and edge cases
- `tests/test_pipeline_config.py` - 9 tests for YAML loading, domain merging, Pydantic validation rejection
- `tests/test_curate_pipeline.py` - 8 integration tests for all 4 pipeline stages and CLI

## Decisions Made
- Separated pipeline_config.py from curate_pipeline.py to avoid circular imports between style_validator tests and the pipeline module
- Used `or` fallback pattern for Pydantic-serialized Optional values that may be None in model_dump() output

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed None-handling for Pydantic model_dump() style config values**
- **Found during:** Task 2 (Pipeline orchestrator integration tests)
- **Issue:** When PipelineConfig is created without explicit style overrides, model_dump() serializes Optional[int]=None as {"max_tokens": None}. dict.get("max_tokens", 600) returns None (key exists) instead of fallback 600, causing TypeError in comparisons.
- **Fix:** Changed style.get("max_tokens", 600) to style.get("max_tokens") or 600 across all 3 domain branches in style_validator.py
- **Files modified:** scripts/style_validator.py
- **Verification:** All 33 tests pass including integration tests that exercise Pydantic-generated configs
- **Committed in:** b0e1427 (part of Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix)
**Impact on plan:** Bug fix necessary for correctness when style config comes from Pydantic models. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviation above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 02 is now complete: quality scoring, deduplication, style validation, and pipeline orchestration are all implemented and tested
- 98 total tests pass across the entire project with zero regressions
- The curation pipeline is ready to process JSONL files generated in future data generation phases (Phases 4/5/6)
- Key integration: `python -m scripts.curate_pipeline --input <raw.jsonl> --domain <code|knowledge|tool-calling>` produces curated output

## Self-Check: PASSED

- All 7 created files verified on disk
- All 4 task commits verified in git history (a92b611, 91757a1, 56f0eae, b0e1427)

---
*Phase: 02-data-quality-and-curation-pipeline*
*Completed: 2026-04-20*
