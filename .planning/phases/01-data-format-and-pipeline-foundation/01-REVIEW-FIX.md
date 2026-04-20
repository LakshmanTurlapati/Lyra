---
phase: 01-data-format-and-pipeline-foundation
fixed_at: 2026-04-20T12:15:00Z
review_path: .planning/phases/01-data-format-and-pipeline-foundation/01-REVIEW.md
iteration: 1
findings_in_scope: 4
fixed: 4
skipped: 0
status: all_fixed
---

# Phase 01: Code Review Fix Report

**Fixed at:** 2026-04-20T12:15:00Z
**Source review:** .planning/phases/01-data-format-and-pipeline-foundation/01-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 4
- Fixed: 4
- Skipped: 0

## Fixed Issues

### WR-01: EOS Detection Mutates Input Token List

**Files modified:** `scripts/validate_tokenizer.py`
**Commit:** 1ce08a6
**Applied fix:** Replaced the while-loop that stripped trailing whitespace tokens (which could incorrectly remove the EOS token itself if it decoded to whitespace) with a simpler and more robust check that verifies EOS is present among the last 3 tokens of the sequence. This handles SmolLM2's template pattern where EOS appears as second-to-last token before a trailing newline.

### WR-02: Missing File Encoding Specification

**Files modified:** `scripts/validate_format.py`, `scripts/validate_tokenizer.py`, `scripts/generate_sample.py`
**Commit:** e3ac3dd
**Applied fix:** Added explicit `encoding="utf-8"` to all `open()` calls across the three scripts. This prevents silent data corruption of non-ASCII characters (mathematical symbols, accented names) on systems where the default locale encoding is not UTF-8 (e.g., Windows cp1252).

### WR-03: Format Validator Does Not Check Tool Message Following Non-Tool-Call Assistant

**Files modified:** `scripts/validate_format.py`
**Commit:** 6551ab3
**Status:** fixed: requires human verification
**Applied fix:** Restructured the tool message validation to first check `i == 0` (tool at start), then explicitly check whether the preceding assistant message has `tool_calls` before allowing the tool message. A tool message following an assistant without `tool_calls` now raises a clear ValueError. The original check only verified role was "assistant" or "tool" but did not verify the assistant actually made tool calls.

### WR-04: Token Limit Mismatch Between Spec and Implementation

**Files modified:** `scripts/validate_tokenizer.py`
**Commit:** 5be9b02
**Applied fix:** Added a detailed comment block explaining the rationale for the 2048 limit (conservative vs 8192 context, matches native training sequence length, faster training). Added `max_tokens` parameter to `validate_conversation()` and `validate_file()` with default of `MAX_TOKENS` for backward compatibility. Added `--max-tokens` CLI argument to allow overriding the limit for future phases with longer samples.

## Skipped Issues

None -- all in-scope findings were fixed.

---

_Fixed: 2026-04-20T12:15:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
