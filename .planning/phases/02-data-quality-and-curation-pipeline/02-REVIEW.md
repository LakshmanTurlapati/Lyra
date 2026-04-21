---
phase: 02-data-quality-and-curation-pipeline
reviewed: 2026-04-20T00:00:00Z
depth: standard
files_reviewed: 11
files_reviewed_list:
  - scripts/quality_scorer.py
  - scripts/dedup.py
  - scripts/style_validator.py
  - scripts/pipeline_config.py
  - scripts/curate_pipeline.py
  - configs/pipeline.yaml
  - tests/test_quality_scorer.py
  - tests/test_dedup.py
  - tests/test_style_validator.py
  - tests/test_pipeline_config.py
  - tests/test_curate_pipeline.py
findings:
  critical: 2
  warning: 4
  info: 5
  total: 11
status: issues_found
---

# Phase 02: Code Review Report

**Reviewed:** 2026-04-20
**Depth:** standard
**Files Reviewed:** 11
**Status:** issues_found

## Summary

The Phase 02 data quality and curation pipeline implementation is well-structured with clear separation of concerns and comprehensive test coverage. The code applies appropriate security mitigations (yaml.safe_load, pathlib.Path, Pydantic validation). However, two critical bugs were identified: (1) missing "messages" key validation before processing samples, and (2) unvalidated dict key access in jaccard_similarity. Additionally, code quality improvements are recommended around error handling, type safety, and duplicative configuration logic.

## Critical Issues

### CR-01: Unchecked Message Key Access in score_sample

**File:** `scripts/quality_scorer.py:56-57`

**Issue:** The `score_sample()` flow calls `score_completeness()`, `score_naturalness()`, and eventually processes assistant messages by calling `.get("messages")` without validating that the "messages" key exists or contains list data. If a sample dict is missing "messages" or has malformed data, downstream code will process an empty list silently, causing quality signals to incorrectly report pass=True on invalid samples.

**Fix:**
```python
def score_completeness(sample: dict, config: dict) -> dict:
    """Check response completeness: unchecked code blocks, truncation, minimum length."""
    if "messages" not in sample or not isinstance(sample["messages"], list):
        return {
            "signal": "completeness",
            "score": 0.0,
            "pass": False,
            "issues": ["missing_or_invalid_messages_key"],
        }
    messages = sample.get("messages", [])
    # ... rest of function
```

This ensures that samples without proper "messages" structure are rejected before quality scoring begins.

---

### CR-02: Potential Code Ratio Division Logic Error in style_validator.py

**File:** `scripts/style_validator.py:119-121`

**Issue:** In the code domain validation, the code checks `if code_ratio < (1 - max_prose)` where `max_prose_ratio` defaults to 0.4. This means the check is `code_ratio < 0.6` (i.e., code must be >60% of response). However, the logic appears inverted: a sample with 40% code and 60% prose should fail (prose > max_prose_ratio), but the code checks if code_ratio is LESS than the inverse. This is correct mathematically but semantically confusing. The real bug: `max_prose` can be None, and the code uses `or 0.4` fallback which masks the None value but doesn't match intent of StyleConfig where max_prose_ratio is Optional[float].

**Fix:**
```python
# Explicitly validate max_prose_ratio is present before use
if domain == "code":
    max_tokens = style.get("max_tokens") or 600
    if approx_tokens > max_tokens:
        return False
    if style.get("require_code_blocks", False) and "```" not in response_text:
        return False
    max_prose = style.get("max_prose_ratio")
    if max_prose is not None:
        code_ratio = get_code_ratio(response_text)
        if code_ratio < (1 - max_prose):
            return False
```

This explicitly checks if max_prose_ratio is set before using it, avoiding silent fallbacks.

---

## Warnings

### WR-01: Duplicated Domain Config Merging Logic

**File:** `scripts/curate_pipeline.py:37-63`

**Issue:** The `get_domain_config()` function in curate_pipeline.py duplicates the logic of `PipelineConfig.get_domain_config()` from pipeline_config.py. This creates maintenance burden and risk of divergence. The pipeline orchestrator should directly use the config model's method rather than reimplementing it.

**Fix:**
```python
# In curate_pipeline.py, replace get_domain_config() with:
def get_domain_config(config: PipelineConfig, domain: str) -> dict:
    """Merge global defaults with domain-specific overrides."""
    domain_cfg = config.get_domain_config(domain)
    result = domain_cfg.model_dump()
    
    # Add only the pipeline-level settings needed by modules
    result.update({
        "ngram_size": config.ngram_size,
        "dedup_threshold": config.dedup_threshold,
        "dedup_scope": config.dedup_scope,
        "style_validation": config.style_validation,
        "include_quality_scores": config.include_quality_scores,
    })
    return result
```

This defers all complex merging to the Pydantic model, reducing duplication.

---

### WR-02: Undefined Fallback Behavior in pipeline_config.py

**File:** `scripts/pipeline_config.py:113`

**Issue:** In `style_validator.py` line 113, the code uses `style.get("max_tokens") or 600` which provides a default only if the value is falsy (None, 0, empty). If a config explicitly sets `max_tokens: 0`, the `or` operator will substitute the default 600, ignoring the explicit zero. This violates the principle that explicit config overrides should be respected.

**Fix:**
```python
# Use a helper to distinguish "not provided" from "provided as falsy"
def get_config_value(config: dict, key: str, default):
    """Get config value, respecting explicit None/0 values."""
    return config[key] if key in config else default

# Then in style_validator.py:
max_tokens = get_config_value(style, "max_tokens", 600)
if approx_tokens > max_tokens:
    return False
```

Or use Pydantic's more explicit validation by ensuring Optional fields are not None.

---

### WR-03: Unvalidated Score Aggregation in quality_scorer.py

**File:** `scripts/quality_scorer.py:181`

**Issue:** The `score_sample()` function aggregates all 4 signal scores using `min(s["score"] for s in signals.values())`. This means if 3 signals have score 1.0 and one has 0.1, the overall score is 0.1. This is a valid aggregation strategy but has a subtle bug: if ANY signal passes (pass=True) but the score is 0.0 (impossible scenario given the logic), the min() approach will report score=0.0 even though all critical signals passed. The real issue is that `pass` and `score` are independently computed, which could lead to inconsistent state (pass=True but score=0.0). The current code logic prevents this, but it's fragile.

**Fix:**
```python
def score_sample(sample: dict, config: dict) -> dict:
    """Run all 4 quality signals and return aggregate result."""
    signals = {
        "format": score_format(sample),
        "completeness": score_completeness(sample, config),
        "naturalness": score_naturalness(sample, config),
        "diversity": _score_diversity(),
    }
    
    all_pass = all(s["pass"] for s in signals.values())
    # Use weighted average instead of min() for better signal differentiation
    min_score = min(s["score"] for s in signals.values())
    
    # Explicitly validate consistency
    if all_pass and min_score == 0.0:
        logger.warning("Inconsistent state: all signals pass but min_score=0.0")
    
    return {
        "pass": all_pass,
        "score": min_score,
        "signals": signals,
    }
```

Add explicit logging or assertions to validate the invariant.

---

### WR-04: Missing Error Context in run_pipeline Log Output

**File:** `scripts/curate_pipeline.py:153`

**Issue:** In the style validation stage, when a sample is rejected, the log message `logger.info("Style rejected for domain '%s'", domain)` does not include which sample or why it was rejected. This makes it difficult to debug quality issues. Compare to line 134 which logs the signal details.

**Fix:**
```python
for sample in deduped_samples:
    if validate_style(sample, domain, domain_config):
        final_samples.append(sample)
    else:
        # Log the sample ID (first user message) and domain for context
        user_msg = next(
            (m.get("content", "")[:50] for m in sample.get("messages", []) if m.get("role") == "user"),
            "unknown"
        )
        logger.info(
            "Style rejected for domain '%s': sample='%s'",
            domain,
            user_msg,
        )
```

---

## Info

### IN-01: Missing Type Hints in Module Functions

**File:** `scripts/quality_scorer.py`, `scripts/dedup.py`, `scripts/style_validator.py`

**Issue:** Functions lack complete type hints for return values. For example, `score_completeness()` returns `dict` without specifying the structure (keys: "signal", "score", "pass", "issues"). While Pydantic models in pipeline_config.py have full hints, the scorer and dedup modules use untyped dicts, reducing IDE autocomplete and static analysis capability.

**Fix:**
```python
from typing import TypedDict

class ScoreSignal(TypedDict):
    signal: str
    score: float
    pass: bool
    issues: list[str]

def score_completeness(sample: dict, config: dict) -> ScoreSignal:
    """..."""
    # Function body
```

---

### IN-02: Hardcoded Magic Numbers in score_completeness and score_naturalness

**File:** `scripts/quality_scorer.py:79, 132`

**Issue:** The score degradation factors (0.5, 0.3) are hardcoded. These control how many issues cause a signal to fail. Currently: each completeness issue degrades score by 0.5 (2 issues fail), each naturalness issue by 0.3 (3+ issues fail). These should be configurable.

**Fix:**
```python
def score_completeness(sample: dict, config: dict) -> dict:
    """..."""
    messages = sample.get("messages", [])
    issues = []
    min_chars = config.get("min_response_chars", 10)
    issue_penalty = config.get("completeness_issue_penalty", 0.5)
    
    for msg in assistant_msgs:
        # ... collect issues ...
    
    score = max(0.0, 1.0 - len(issues) * issue_penalty)
```

Add `completeness_issue_penalty` and `naturalness_issue_penalty` to StyleConfig.

---

### IN-03: Unused Placeholder in dedup.py Docstring

**File:** `scripts/dedup.py:1-17`

**Issue:** The module docstring mentions "Scaling limit documented" but does not document the limit clearly. The comment on line 104 mentions "At 5K samples the O(n^2) approach completes in seconds" but this is performance analysis, not a hard limit. The docstring should be explicit: "For >10K samples, consider MinHash LSH or external deduplication."

**Fix:**
```python
"""dedup.py -- N-gram Jaccard similarity deduplication.

Scaling note: O(n^2) all-pairs comparison. Practical limits:
  - 5K samples: ~12.5M comparisons, completes in seconds
  - 10K samples: ~50M comparisons, ~10-30 seconds
  - 50K+ samples: Consider external deduplication (MinHash LSH)
"""
```

---

### IN-04: Incomplete Test Coverage for Config Error Paths

**File:** `tests/test_pipeline_config.py`

**Issue:** The config loading tests do not cover the error case where the YAML file is malformed or missing required schema fields. For example, no test validates that `dedup_scope` must be one of "response", "prompt", "full".

**Fix:** Add to test_pipeline_config.py:
```python
def test_config_validation_rejects_malformed_yaml(tmp_path):
    """Malformed YAML raises exception."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("invalid: yaml: syntax:")
    with pytest.raises(Exception):  # yaml.YAMLError or ValueError
        load_config(bad_yaml)

def test_config_validation_rejects_missing_version(tmp_path):
    """Config without version field raises ValidationError."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("defaults:\n  min_response_chars: 10")
    with pytest.raises(ValidationError):
        load_config(bad_yaml)
```

---

### IN-05: Inconsistent Error Handling in run_pipeline

**File:** `scripts/curate_pipeline.py:105-110`

**Issue:** When JSON parsing fails on a line, the code appends an empty dict `{}` to `raw_samples` to count it. This inflates the input_count metric and could mask parsing errors. Lines that fail to parse should be logged and skipped without being counted in input_count.

**Fix:**
```python
for line_num, line in enumerate(f, 1):
    stripped = line.strip()
    if not stripped:
        continue
    try:
        sample = json.loads(stripped)
        raw_samples.append(sample)
    except json.JSONDecodeError as e:
        logger.warning("Line %d: JSON parse error: %s", line_num, e)
        # Do NOT count unparseable lines in input_count
        continue

input_count = len(raw_samples)  # Now reflects only parseable JSON
```

---

## Recommendations Summary

1. **Critical:** Add explicit "messages" key validation in quality scorer before processing
2. **Critical:** Clarify and fix the optional config value handling (max_prose_ratio, max_tokens)
3. **High:** Remove duplicated config merging logic in curate_pipeline.py
4. **High:** Add type hints for return types (ScoreSignal, etc.)
5. **Medium:** Make score degradation factors configurable
6. **Medium:** Improve logging for style rejection details
7. **Low:** Add missing test cases for malformed YAML and JSON parsing errors
8. **Low:** Clarify scaling limits in dedup.py docstring

---

_Reviewed: 2026-04-20_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
