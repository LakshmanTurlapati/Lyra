---
phase: 01-data-format-and-pipeline-foundation
plan: 02
subsystem: data-validation
tags: [tokenizer, smollm2, validation, sharegpt, sample-generator, chat-template]

# Dependency graph
requires:
  - phase: 01-01
    provides: Pydantic validation schema (Conversation, Message, ToolCall, ToolSchema models), validate_file(), shared pytest fixtures
provides:
  - SmolLM2-1.7B tokenizer alignment validation (validate_conversation, validate_file, load_tokenizer)
  - TRL-native to SmolLM2 message pre-processing (_prepare_messages_for_template)
  - Sample conversation generator with 10 samples across 3 domains
  - Dataset directory structure (datasets/tool-calling, datasets/code, datasets/knowledge)
affects: [01-03, phase-02, phase-04, phase-05, phase-06, phase-08]

# Tech tracking
tech-stack:
  added: [transformers 5.5.4, jinja2 3.1.6]
  patterns: [tokenizer-round-trip-validation, tool-call-xml-conversion, return-dict-pattern]

key-files:
  created:
    - scripts/validate_tokenizer.py
    - scripts/generate_sample.py
    - tests/test_tokenizer_validator.py
    - datasets/tool-calling/.gitkeep
    - datasets/code/.gitkeep
    - datasets/knowledge/.gitkeep
  modified:
    - .gitignore
    - pytest.ini

key-decisions:
  - "Pre-process TRL-native tool_calls to SmolLM2 <tool_call> XML format before tokenization -- SmolLM2 chat template does not handle structured tool_calls"
  - "Use return_dict=True for apply_chat_template on transformers 5.x -- return_tensors=None returns BatchEncoding not list"
  - "EOS check strips trailing whitespace tokens -- SmolLM2 template ends with <|im_end|>\\n so EOS is second-to-last"

patterns-established:
  - "Tokenizer validation pattern: pre-process messages -> apply_chat_template -> check token count, EOS, markers, default prompt"
  - "Tool call conversion: TRL-native tool_calls field -> <tool_call>[{name, arguments}]</tool_call> content string"
  - "Tool schema injection: tools column -> JSON injected into system prompt content"
  - "Sample generator pattern: hardcoded domain-grouped samples with Pydantic + tokenizer validation"

requirements-completed: [DATA-02]

# Metrics
duration: 8min
completed: 2026-04-20
---

# Phase 01 Plan 02: Tokenizer Alignment Validation and Sample Generator Summary

**SmolLM2-1.7B tokenizer validation with tool call pre-processing, 10 sample conversations, and dataset directory structure**

## Performance

- **Duration:** 8 min
- **Started:** 2026-04-20T09:30:23Z
- **Completed:** 2026-04-20T09:38:34Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- Tokenizer validation script that catches oversized conversations (>2048 tokens), missing EOS tokens, default system prompt injection, and missing chat template markers
- Pre-processing layer that converts TRL-native structured tool_calls to SmolLM2's native <tool_call> XML format, enabling tokenizer validation of tool-calling conversations
- Sample generator with 10 conversations across 3 domains (3 tool-calling, 4 code, 3 knowledge) -- all pass both Pydantic format and tokenizer validation
- Dataset directory structure with .gitignore configured to exclude generated data while preserving .gitkeep markers

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tokenizer tests** - `1b3be94` (test)
2. **Task 1 GREEN: Tokenizer validation implementation** - `ed8b090` (feat)
3. **Task 2: Sample generator and dataset directories** - `f76589a` (feat)

## Files Created/Modified
- `scripts/validate_tokenizer.py` - SmolLM2 tokenizer alignment validation with CLI (exports: validate_conversation, validate_file, load_tokenizer)
- `scripts/generate_sample.py` - 10 sample conversations across 3 domains with format and tokenizer validation (exports: generate_samples, write_samples, validate_samples)
- `tests/test_tokenizer_validator.py` - 8 unit tests covering token count, EOS, markers, default prompt, file validation
- `datasets/tool-calling/.gitkeep` - Directory marker for tool-calling training data
- `datasets/code/.gitkeep` - Directory marker for code training data
- `datasets/knowledge/.gitkeep` - Directory marker for knowledge training data
- `.gitignore` - Updated to exclude generated JSONL/JSON while preserving .gitkeep files
- `pytest.ini` - Added slow marker registration for tokenizer tests

## Decisions Made
- SmolLM2's chat template is a simple concatenation (`<|im_start|>{role}\n{content}<|im_end|>\n`) that does NOT handle structured tool_calls or tools columns -- required building a pre-processing layer to convert TRL-native format to SmolLM2's `<tool_call>` XML format before tokenization
- Used `return_dict=True` instead of `return_tensors=None` for `apply_chat_template` on transformers 5.5.4 -- the latter returns a BatchEncoding object instead of a flat list of token IDs
- EOS token check accounts for trailing whitespace tokens -- SmolLM2's template appends `\n` after `<|im_end|>`, placing EOS as second-to-last token

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed jinja2 dependency**
- **Found during:** Task 1 (tokenizer validation implementation)
- **Issue:** transformers.apply_chat_template requires jinja2 for template rendering, but it was not installed
- **Fix:** Installed jinja2 3.1.6 via pip
- **Files modified:** None (system package)
- **Verification:** apply_chat_template succeeds after installation
- **Committed in:** ed8b090 (Task 1 GREEN commit)

**2. [Rule 1 - Bug] Fixed apply_chat_template return type handling**
- **Found during:** Task 1 (tokenizer validation implementation)
- **Issue:** transformers 5.5.4 apply_chat_template with return_tensors=None returns BatchEncoding object, not a flat list of ints -- caused len() to return number of sequences (2) instead of token count
- **Fix:** Changed to return_dict=True and access encoded["input_ids"] for the flat token ID list
- **Files modified:** scripts/validate_tokenizer.py
- **Verification:** Token counts are correct (26 for simple conversation, not 2)
- **Committed in:** ed8b090 (Task 1 GREEN commit)

**3. [Rule 1 - Bug] Fixed EOS token position check**
- **Found during:** Task 1 (tokenizer validation implementation)
- **Issue:** SmolLM2 chat template ends with <|im_end|>\n, making the last token a newline (id=198), not EOS (id=2). Direct last-token check always failed for valid conversations.
- **Fix:** Strip trailing whitespace tokens before checking for EOS, then verify the last meaningful token is EOS
- **Files modified:** scripts/validate_tokenizer.py
- **Verification:** Valid conversations pass EOS check; test_detects_eos_token passes
- **Committed in:** ed8b090 (Task 1 GREEN commit)

**4. [Rule 2 - Missing Critical] Added message pre-processing for tool calls**
- **Found during:** Task 1 (tokenizer validation implementation)
- **Issue:** SmolLM2's chat template does not handle TRL-native structured tool_calls field (requires message['content'] to be a string). Tool-calling conversations caused template errors.
- **Fix:** Added _prepare_messages_for_template() that converts tool_calls to <tool_call> XML content, injects tool schemas into system prompt, and ensures all content fields are strings
- **Files modified:** scripts/validate_tokenizer.py
- **Verification:** Tool-calling conversation fixtures pass tokenizer validation; all 10 samples pass
- **Committed in:** ed8b090 (Task 1 GREEN commit)

---

**Total deviations:** 4 auto-fixed (2 bugs, 1 missing critical, 1 blocking)
**Impact on plan:** All auto-fixes necessary for correct operation with SmolLM2's actual tokenizer behavior. The pre-processing layer is essential infrastructure for all future tokenizer validation. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviations above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Tokenizer validation and sample generator are complete and tested
- Plan 01-03 (prompt templates) can use the sample generator as a reference for conversation structure
- Phases 4-6 (data generation) have both format AND tokenizer validation to validate output
- The _prepare_messages_for_template function establishes the canonical conversion from TRL-native to SmolLM2-native format

## Self-Check: PASSED

All 8 created/modified files verified present on disk. All 3 task commits verified in git history.

---
*Phase: 01-data-format-and-pipeline-foundation*
*Completed: 2026-04-20*
