---
phase: 02-data-quality-and-curation-pipeline
plan: 01
subsystem: data-quality
tags: [quality-scoring, deduplication, heuristics, jaccard, ngram, pydantic, sharegpt]

# Dependency graph
requires:
  - phase: 01-data-format-and-pipeline-foundation
    provides: Conversation Pydantic model from validate_format.py for format signal
provides:
  - quality_scorer.py: 4-signal heuristic scoring (format, completeness, naturalness, diversity placeholder)
  - dedup.py: N-gram Jaccard deduplication with configurable scope and threshold
  - Both modules importable by curate_pipeline.py in Plan 02
affects: [02-02-pipeline-orchestrator, phase-04-tool-calling, phase-05-code, phase-06-knowledge]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Config-driven thresholds via dict with config.get() defaults -- no hardcoded magic numbers"
    - "Signal-based scoring: each signal returns {signal, score, pass, issues} for granular diagnostics"
    - "Stdlib-only deduplication: zero external dependencies for n-gram Jaccard"

key-files:
  created:
    - scripts/quality_scorer.py
    - scripts/dedup.py
    - tests/test_quality_scorer.py
    - tests/test_dedup.py
  modified: []

key-decisions:
  - "Response-scope dedup as default (catches response homogeneity, the more dangerous failure mode)"
  - "Diversity signal as placeholder at sample level -- actual diversity handled by dedup at batch level per D-03"
  - "Meta-commentary detection in assistant messages only -- user messages mentioning AI are not flagged"

patterns-established:
  - "Signal-based scoring: independent {signal, score, pass, issues} dicts composable by pipeline"
  - "Config-driven thresholds: all scoring/dedup parameters via config.get() with documented defaults"
  - "Stdlib-only batch processing: no external ML dependencies for dedup per D-04"

requirements-completed: [DATA-03]

# Metrics
duration: 4min
completed: 2026-04-20
---

# Phase 02 Plan 01: Quality Scoring and Deduplication Summary

**4-signal heuristic quality scorer and stdlib-only n-gram Jaccard deduplication with config-driven thresholds, 38 tests passing**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-20T10:39:43Z
- **Completed:** 2026-04-20T10:44:01Z
- **Tasks:** 2
- **Files created:** 4

## Accomplishments
- Quality scorer with 4 independent heuristic signals (format, completeness, naturalness, diversity) producing structured pass/fail results per D-01/D-03
- N-gram Jaccard deduplication with configurable scope (response/prompt/full), threshold, and n-gram size using only Python stdlib per D-04
- All thresholds configurable via config dict -- zero hardcoded magic numbers
- 38 tests total (17 quality scorer + 21 dedup), all passing with deterministic assertions

## Task Commits

Each task was committed atomically:

1. **Task 1: Quality scorer with 4 heuristic signals** - `7a65c28` (test: RED) -> `84f94bd` (feat: GREEN)
2. **Task 2: N-gram Jaccard deduplication module** - `56b0d53` (test: RED) -> `589280d` (feat: GREEN)

_TDD flow: failing tests committed first, then implementation making all tests pass._

## Files Created/Modified
- `scripts/quality_scorer.py` - Tier 1 heuristic scoring: score_format (Pydantic validation), score_completeness (code blocks, truncation, length), score_naturalness (meta-commentary, turn balance), score_sample (aggregate)
- `scripts/dedup.py` - N-gram Jaccard deduplication: extract_ngrams, jaccard_similarity, get_dedup_text (scope extraction), deduplicate_batch (O(n^2) all-pairs)
- `tests/test_quality_scorer.py` - 17 tests covering all 4 signals, config-driven thresholds, edge cases
- `tests/test_dedup.py` - 21 tests covering n-gram extraction, Jaccard computation, scope extraction, batch dedup

## Decisions Made
- Response-scope dedup as default: catches response homogeneity (mode collapse) which is more dangerous than prompt repetition
- Diversity signal is a placeholder at sample level -- real diversity detection happens at batch level via dedup.py, avoiding redundant per-sample computation
- Meta-commentary patterns checked only in assistant messages -- user messages saying "as an AI" are legitimate prompts, not quality issues
- Truncation detection covers both ASCII "..." and Unicode ellipsis char

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Both modules are importable: `from scripts.quality_scorer import score_sample` and `from scripts.dedup import deduplicate_batch`
- Ready for Plan 02 (pipeline orchestrator) to compose these into curate_pipeline.py
- Config dict interface allows pipeline.yaml values to be passed directly

## Self-Check: PASSED

- All 4 files exist on disk
- All 4 commit hashes found in git log
- 38/38 tests passing

---
*Phase: 02-data-quality-and-curation-pipeline*
*Completed: 2026-04-20*
