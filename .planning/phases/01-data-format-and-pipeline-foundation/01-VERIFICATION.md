---
phase: 01-data-format-and-pipeline-foundation
verified: 2026-04-20T10:15:00Z
status: passed
score: 4/4 must-haves verified
overrides_applied: 1
gaps: []
overrides:
  - truth: "User can generate a small batch (10-50 samples) via the Anthropic API and receive JSONL output on disk"
    resolution: "ROADMAP SC-4 updated to reflect D-05 decision. generate_sample.py provides 10 local test samples validated end-to-end. Anthropic API generation deferred to Phase 4-6."
    date: 2026-04-20
human_verification:
  - test: "Run tokenizer validation on generated samples"
    expected: "All 10 samples pass tokenizer check with token counts under 2048"
    why_human: "Requires loading SmolLM2-1.7B-Instruct tokenizer from HuggingFace (network + PyTorch dependency not available in this environment)"
---

# Phase 1: Data Format and Pipeline Foundation Verification Report

**Phase Goal:** Users can generate correctly-formatted ShareGPT conversations that are validated against SmolLM2-1.7B's tokenizer and chat template
**Verified:** 2026-04-20T10:15:00Z
**Status:** gaps_found
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | User can run a generation command that produces ShareGPT-format conversations with correct role ordering | VERIFIED | `python3 -m scripts.generate_sample --validate-only` produces 10/10 valid samples. Format enforces system-first, user/assistant alternation, tool ordering via Pydantic model_validator. 11 tests pass. |
| 2 | User can validate any generated conversation against SmolLM2-1.7B's tokenizer and chat template and get pass/fail with error details | VERIFIED | `scripts/validate_tokenizer.py` exports validate_conversation() and validate_file() with token_count, errors list, decoded_preview. 8 tokenizer tests exist. Checks: token limit (2048), EOS presence, chat template markers, default system prompt injection. |
| 3 | User can browse a prompt template library organized by category (tool call, code, general knowledge) with documented system prompts | VERIFIED | 4 YAML template files in templates/ directory. tool-calling.yaml (5 categories), code.yaml (3 categories), knowledge.yaml (3 categories), system-prompts.yaml (7 prompts). All 8 template tests pass. Cross-references validated. |
| 4 | User can generate a small batch (10-50 samples) via the Anthropic API and receive JSONL output on disk | FAILED | No Anthropic API client exists anywhere in the codebase. D-05 explicitly chose "No Anthropic API SDK or external API pipeline." generate_sample.py uses hardcoded data, not API calls. `anthropic` is not in requirements.txt. |

**Score:** 3/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `specs/sharegpt-format.md` | Canonical format specification | VERIFIED | 280+ lines. Covers TRL-native format, 9 role ordering rules, all 5 tool call patterns with JSON examples, 2048 token budget, storage conventions. |
| `scripts/validate_format.py` | Pydantic-based JSONL format validation | VERIFIED | 197 lines. Exports Conversation, Message, ToolCall, ToolSchema, validate_file. model_validator enforces all structural rules. CLI entry point with argparse. |
| `scripts/validate_tokenizer.py` | SmolLM2 tokenizer alignment validation | VERIFIED | 301 lines. Exports validate_conversation, validate_file, load_tokenizer. Checks token count, EOS, markers, default prompt injection. Pre-processes tool_calls to SmolLM2 XML format. |
| `scripts/generate_sample.py` | Sample conversation generator | VERIFIED | 289 lines. 10 hardcoded samples across 3 domains. Exports generate_samples, write_samples, validate_samples. CLI with --validate-only and --tokenizer-check flags. |
| `tests/conftest.py` | Shared pytest fixtures | VERIFIED | 113 lines. 6 fixtures: valid_conversation, valid_tool_call_conversation, valid_parallel_tool_call_conversation, invalid_no_system, invalid_orphan_tool, invalid_undefined_tool. |
| `tests/test_format_validator.py` | Unit tests for format validation | VERIFIED | 125 lines. 11 tests covering DATA-01. All pass. |
| `tests/test_tokenizer_validator.py` | Unit tests for tokenizer validation | VERIFIED | 138 lines. 8 tests covering DATA-02. |
| `tests/test_templates.py` | Unit tests for template library | VERIFIED | 128 lines. 8 tests covering DATA-06. All pass. |
| `templates/tool-calling.yaml` | Tool calling prompt templates | VERIFIED | Contains single_call, multi_turn, parallel_calls, mcp_patterns, cli_commands categories. |
| `templates/code.yaml` | Code generation prompt templates | VERIFIED | Contains utility_functions, file_operations, debugging categories. |
| `templates/knowledge.yaml` | General knowledge prompt templates | VERIFIED | Contains reasoning_chains, factual_qa, explanations categories. |
| `templates/system-prompts.yaml` | Shared system prompt variants | VERIFIED | 7 prompts: tool_assistant, code_assistant, code_debugger, knowledge_assistant, knowledge_reasoning, mcp_assistant, cli_assistant. |
| `datasets/tool-calling/.gitkeep` | Directory structure | VERIFIED | Exists. |
| `datasets/code/.gitkeep` | Directory structure | VERIFIED | Exists. |
| `datasets/knowledge/.gitkeep` | Directory structure | VERIFIED | Exists. |
| `requirements.txt` | Python dependencies | VERIFIED | Contains pydantic, transformers, pyyaml, pytest. |
| `pytest.ini` | Pytest configuration | VERIFIED | testpaths = tests, slow marker registered. |
| `.gitignore` | Exclude generated data | VERIFIED | Excludes datasets/**/*.jsonl, __pycache__/, .pytest_cache/, .cache/. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| tests/test_format_validator.py | scripts/validate_format.py | `from scripts.validate_format import Conversation, validate_file` | WIRED | Import at line 13, used in all 11 tests |
| tests/conftest.py | scripts/validate_format.py | Fixtures return dicts validated by Conversation model | WIRED | Fixtures produce dicts consumed by test_format_validator.py tests |
| tests/test_tokenizer_validator.py | scripts/validate_tokenizer.py | `from scripts.validate_tokenizer import validate_conversation, validate_file, load_tokenizer` | WIRED | Import at line 12, used in all 8 tests |
| scripts/validate_tokenizer.py | HuggingFaceTB/SmolLM2-1.7B-Instruct | `AutoTokenizer.from_pretrained(MODEL_ID)` | WIRED | MODEL_ID constant at line 18, used in load_tokenizer() |
| scripts/generate_sample.py | scripts/validate_format.py | `from scripts.validate_format import Conversation` | WIRED | Import at line 16, used in validate_samples() |
| scripts/generate_sample.py | scripts/validate_tokenizer.py | `from scripts.validate_tokenizer import load_tokenizer, validate_conversation` | WIRED | Lazy import at line 249, used in --tokenizer-check flow |
| templates/tool-calling.yaml | specs/sharegpt-format.md | Templates produce data matching format spec (tool_calls field) | WIRED | tool-calling.yaml references format spec in description, uses tool_calls structure |
| templates/system-prompts.yaml | templates/tool-calling.yaml | system_prompt_ref cross-references | WIRED | All 5 tool-calling categories reference valid prompt IDs. Test validates this. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| scripts/generate_sample.py | SAMPLES_BY_DOMAIN | Hardcoded dict literals | Yes -- 10 complete conversations | FLOWING |
| scripts/validate_format.py | validate_file() results | JSONL file read + Pydantic validation | Yes -- returns actual pass/fail counts | FLOWING |
| scripts/validate_tokenizer.py | validate_conversation() results | tokenizer.apply_chat_template() | Yes -- returns token_count, errors from real tokenizer | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Format validator imports correctly | `python3 -c "from scripts.validate_format import Conversation, Message, ToolCall, ToolSchema, validate_file"` | "All exports available" | PASS |
| Tokenizer validator imports correctly | `python3 -c "from scripts.validate_tokenizer import validate_conversation, validate_file, load_tokenizer"` | "All tokenizer exports available" (PyTorch warning, non-blocking) | PASS |
| Sample generator produces valid samples | `python3 -m scripts.generate_sample --validate-only` | "Generated 10 samples across 3 domains; Format validation: 10/10 valid" | PASS |
| Format validation tests pass | `python3 -m pytest tests/test_format_validator.py -v` | 11 passed in 0.04s | PASS |
| Template tests pass | `python3 -m pytest tests/test_templates.py -v` | 8 passed in 0.08s | PASS |
| Tokenizer tests exist and structured | `grep -c "def test_" tests/test_tokenizer_validator.py` | 8 test methods | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-----------|-------------|--------|----------|
| DATA-01 | 01-01 | User can generate ShareGPT-format conversation datasets with strict role ordering | SATISFIED | Pydantic schema enforces 9 role ordering rules; 11 tests pass; generate_sample.py produces valid conversations |
| DATA-02 | 01-02 | User can validate conversation format alignment with SmolLM2-1.7B tokenizer and chat template | SATISFIED | validate_tokenizer.py with validate_conversation() and validate_file(); checks token count, EOS, markers, default prompt; 8 tests exist |
| DATA-06 | 01-03 | Prompt template library organized by category with documented system prompts | SATISFIED | 4 YAML templates (tool-calling, code, knowledge, system-prompts); 11 categories total; 7 system prompts; 8 tests pass |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | - | - | - | No anti-patterns detected |

No TODOs, FIXMEs, placeholders, or stub implementations found in any production file.

### Human Verification Required

1. **Tokenizer validation end-to-end test**

**Test:** Run `python3 -m pytest tests/test_tokenizer_validator.py -v` with PyTorch installed
**Expected:** All 8 tests pass, including oversized conversation rejection and EOS detection
**Why human:** Requires PyTorch + transformers with full model download from HuggingFace Hub (network-dependent, ~500MB tokenizer files)

### Gaps Summary

One gap found: ROADMAP Success Criterion 4 ("User can generate a small batch (10-50 samples) via the Anthropic API and receive JSONL output on disk") is not implemented. No Anthropic API client exists in the codebase.

**Root cause analysis:** This is a deliberate design decision (D-05 in 01-CONTEXT.md), not an oversight. The discussion phase decided that data generation happens in Claude Code sessions (Claude Opus writes JSONL directly), making an API client unnecessary. However, the ROADMAP was never updated to reflect this decision.

**Resolution options:**
1. **Update ROADMAP SC-4** to match D-05: "User can generate a small batch of 10 sample conversations via generate_sample.py and validate them end-to-end" -- reflecting the actual approach.
2. **Build the Anthropic API client** as originally specified -- add scripts/generate_batch.py that calls Claude Opus via the Message Batches API to produce JSONL.

**This looks intentional.** The codebase explicitly documents (D-05) that no API pipeline is needed because Claude Code generates data directly. To accept this deviation, add to VERIFICATION.md frontmatter:

```yaml
overrides:
  - must_have: "User can generate a small batch (10-50 samples) via the Anthropic API and receive JSONL output on disk"
    reason: "Design decision D-05: Data generation happens in Claude Code sessions, not via API client. generate_sample.py provides 10 pipeline-test samples. Actual Opus-quality data generation occurs directly in Phases 4-6 via Claude Code."
    accepted_by: "{your name}"
    accepted_at: "2026-04-20T10:15:00Z"
```

---

_Verified: 2026-04-20T10:15:00Z_
_Verifier: Claude (gsd-verifier)_
