# Phase 5: Code Generation Dataset - Research

**Researched:** 2026-04-20
**Domain:** Synthetic code generation data (utility functions, file operations, debugging)
**Confidence:** HIGH

## Summary

Phase 5 generates ~3,334 raw code generation training samples across three categories (utility functions, file operations, debugging) and curates them down to ~1,667 for the final dataset. The implementation closely follows the Phase 4 pattern -- a new `scripts/generate_code_data.py` script with category-specific generators, template-based query construction, seeded randomness, batch validation, and JSONL output to `datasets/code/`.

The key structural difference from Phase 4 is that code samples do NOT use tool calls. They are simple system/user/assistant conversations where the assistant response contains code. This simplifies the generation logic significantly -- no tool schemas, no tool responses, no multi-turn tool interludes. The conversation format is just: system prompt -> user request -> assistant code response. The style validator for the `code` domain requires fenced code blocks (```) and enforces max_prose_ratio of 0.4, meaning at least 60% of the response must be inside code fences.

The entire infrastructure for validation, curation, dedup, and style checking already exists and supports the `code` domain natively. The new script must produce conversations that pass: (1) Pydantic format validation, (2) quality scoring (completeness, naturalness), (3) style validation (code blocks present, prose ratio under 0.4, max ~600 approximate tokens), and (4) n-gram Jaccard dedup at 0.7 threshold with response-scope comparison.

**Primary recommendation:** Build `scripts/generate_code_data.py` following the exact same structural pattern as `scripts/generate_tool_data.py` (argparse CLI, category generators, batch validation, JSONL write), but without the tool-calling complexity. Each category generator produces simple system/user/assistant conversations with code-first, terse responses.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Generate ~3,334 raw code generation samples. Curation pipeline filters down to ~1,667 final samples (the 33% code slice of 5K total).
- **D-02:** Utility-heavy distribution: Utility functions ~50% (~1,667 raw -> ~834 curated), File operations ~25% (~834 raw -> ~417 curated), Debugging ~25% (~834 raw -> ~417 curated).
- **D-03:** Python-heavy language distribution for utility functions: Python ~40%, JavaScript/TypeScript combined ~25%, Go ~20%, Rust ~15%.
- **D-04:** For file operations and debugging (3 languages): Python ~50%, JavaScript ~30%, TypeScript ~20%.
- **D-05:** Terse code-first style. Function/code block with 1-2 line comment. No preamble, no "Here's how to..." padding.
- **D-06:** Debugging samples use brief "Bug: X, Fix: Y" format before corrected code.
- **D-07:** Token targets: Utility ~200-500, File ops ~300-800, Debugging ~400-800.
- **D-08:** New script `scripts/generate_code_data.py` following Phase 4 pattern (CLI with argparse, category generators, batch loop of 50, inline validation).
- **D-09:** Category batches with validation loops -- same workflow as Phase 4. One category at a time, 50 samples per batch.
- **D-10:** Order of generation: utility functions first, then file operations, then debugging.

### Claude's Discretion
- Specific utility function topics beyond the template list
- Bug types and debugging scenarios beyond the template list
- Whether to include code comments in generated samples
- Test fixtures for the generation script
- Batch file naming convention (following Phase 4 pattern: `{category}-batch-{NN}.jsonl`)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CODE-01 | Dataset includes quick utility function generation samples across common languages | Utility functions category: 7 topic areas from code.yaml template, 5 languages (Python/JS/TS/Go/Rust), ~1,667 raw samples at 50% share. Template-based query + code generation pattern. |
| CODE-02 | Dataset includes file operation and system manipulation code samples | File operations category: 7 topic areas from code.yaml, 3 languages (Python/JS/TS), ~834 raw samples at 25% share. Must include error handling patterns. |
| CODE-03 | Dataset includes debugging and code fix samples (identify bug, explain, fix) | Debugging category: 7 bug type areas from code.yaml, 3 languages (Python/JS/TS), ~834 raw samples at 25% share. "Bug: X, Fix: Y" format per D-06. |

</phase_requirements>

## Standard Stack

### Core (Already Installed -- No New Dependencies)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pydantic | 2.x | Conversation validation via `scripts/validate_format.py` | Already in use; validates all generated samples [VERIFIED: codebase] |
| pyyaml | * | Template and system prompt loading from YAML | Already in use by Phase 4 generator [VERIFIED: codebase] |
| Python stdlib json | 3.10+ | JSONL serialization | Standard approach, no external dependency needed [VERIFIED: codebase] |
| Python stdlib random | 3.10+ | Seeded random for reproducible generation | Phase 4 pattern uses `random.Random(seed)` per-batch [VERIFIED: codebase] |

### Supporting (Existing Pipeline -- No New Dependencies)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `scripts/validate_format.py` | existing | Pydantic Conversation model validation | Called inline after each batch generation [VERIFIED: codebase] |
| `scripts/validate_tokenizer.py` | existing | SmolLM2 token count validation (2048 max) | Post-generation validation of batches [VERIFIED: codebase] |
| `scripts/curate_pipeline.py` | existing | Quality filtering, dedup, style validation | Run on all raw batches after generation complete [VERIFIED: codebase] |
| `scripts/quality_scorer.py` | existing | 4-signal quality scoring | Called by curate_pipeline Stage 2 [VERIFIED: codebase] |
| `scripts/dedup.py` | existing | N-gram Jaccard dedup | Called by curate_pipeline Stage 3 [VERIFIED: codebase] |
| `scripts/style_validator.py` | existing | Code domain style: require_code_blocks, max_prose_ratio | Called by curate_pipeline Stage 4 [VERIFIED: codebase] |

**Installation:** No new packages needed. All dependencies are already present from Phases 1-4.

## Architecture Patterns

### Recommended Project Structure (Matching Phase 4)

```
scripts/
  generate_code_data.py       # NEW: Code generation script (Phase 5)
  generate_tool_data.py       # EXISTING: Reference implementation (Phase 4)
  validate_format.py          # EXISTING: Reuse directly
  validate_tokenizer.py       # EXISTING: Reuse directly
  curate_pipeline.py          # EXISTING: Reuse with --domain code
  quality_scorer.py           # EXISTING: Reuse directly
  dedup.py                    # EXISTING: Reuse directly
  style_validator.py          # EXISTING: Reuse directly
datasets/
  code/
    utility-batch-01.jsonl    # NEW: Generated batches
    utility-batch-02.jsonl
    ...
    file-ops-batch-01.jsonl
    ...
    debugging-batch-01.jsonl
    ...
tests/
  test_generate_code_data.py  # NEW: Tests for generation script
templates/
  code.yaml                   # EXISTING: Category definitions
  system-prompts.yaml         # EXISTING: code_assistant, code_debugger prompts
```

[VERIFIED: codebase directory structure matches this layout]

### Pattern 1: Category Generator Function

**What:** Each category (utility, file-ops, debugging) gets its own generator function returning a list of conversation dicts.
**When to use:** Every batch generation call.
**Example structure (following Phase 4 pattern):**

```python
# Source: scripts/generate_tool_data.py pattern [VERIFIED: codebase]
def generate_utility_batch(count: int = 50, seed: int = None) -> list[dict]:
    """Generate utility function samples per CODE-01."""
    rng = random.Random(seed)
    system_prompts = load_system_prompts()
    system_content = system_prompts["code_assistant"]
    samples = []
    
    for i in range(count):
        language = _pick_language(rng, "utility")  # Per D-03 distribution
        topic = _pick_topic(rng, "utility")
        query = _generate_utility_query(language, topic, rng)
        response = _generate_utility_response(language, topic, rng)
        
        sample = {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ]
        }
        samples.append(sample)
    
    return samples
```

### Pattern 2: Language Distribution Enforcement

**What:** Weighted random selection to match locked language distributions.
**When to use:** Every sample generation.

```python
# Utility functions language weights per D-03
UTILITY_LANG_WEIGHTS = {
    "python": 0.40,
    "javascript": 0.125,
    "typescript": 0.125,
    "go": 0.20,
    "rust": 0.15,
}

# File ops / debugging language weights per D-04
FILEOPS_LANG_WEIGHTS = {
    "python": 0.50,
    "javascript": 0.30,
    "typescript": 0.20,
}

def _pick_language(rng: random.Random, category: str) -> str:
    weights = UTILITY_LANG_WEIGHTS if category == "utility" else FILEOPS_LANG_WEIGHTS
    return rng.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]
```

[ASSUMED: rng.choices is correct weighted selection approach for this use case]

### Pattern 3: Code-First Terse Response Format

**What:** Assistant responses that are code blocks with minimal commentary per D-05.
**When to use:** All code generation samples.
**Critical constraint:** Style validator requires `require_code_blocks: true` and `max_prose_ratio: 0.4` for code domain [VERIFIED: configs/pipeline.yaml].

```python
# Utility function response format (D-05: terse, code-first)
RESPONSE_TEMPLATE = """```{language}
{code}
```
{comment}"""

# Debugging response format (D-06: Bug/Fix format)
DEBUG_RESPONSE_TEMPLATE = """Bug: {bug_description}
Fix: {fix_description}

```{language}
{corrected_code}
```"""
```

### Pattern 4: Conversation Format (No Tools)

**What:** Code samples use only system/user/assistant -- no `tools` column, no `tool_calls`.
**Why:** Code generation is direct response, not tool invocation. The Conversation Pydantic model has `tools: Optional[list[ToolSchema]] = None` so omitting it is valid.
**Critical difference from Phase 4:** Simpler structure; no tool response generation, no tool schema selection, no edge-case tool error handling.

[VERIFIED: scripts/validate_format.py -- tools field is Optional]

### Anti-Patterns to Avoid

- **Verbose preamble:** "Here is a Python function that reverses a string:" -- violates D-05 terse style. The response should start with the code block.
- **Missing code fences:** Bare code without triple backticks -- style validator will reject it (`require_code_blocks: true`).
- **Excessive prose:** Long explanations after code -- style validator enforces `max_prose_ratio: 0.4`, meaning 60%+ of content must be code blocks.
- **Hardcoded code strings that look identical:** Template-based generation with insufficient variation will trigger dedup. Ensure topic diversity and parameter variation.
- **Language-inappropriate idioms:** Writing Python-style code for Go/Rust (e.g., list comprehensions in Go). Each language needs idiomatic patterns.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Format validation | Custom JSON checks | `Conversation.model_validate(sample)` from validate_format.py | Already enforces all role ordering rules, tool_calls consistency [VERIFIED: codebase] |
| Token counting | Manual token estimates | `validate_tokenizer.py` with SmolLM2 tokenizer | Approximate counts fail at boundaries; real tokenizer catches edge cases [VERIFIED: codebase] |
| Quality filtering | Custom quality checks | `curate_pipeline.py --domain code` | 4-stage pipeline (format + quality + dedup + style) already configured for code domain [VERIFIED: codebase] |
| Dedup detection | Exact-match dedup | `dedup.py` n-gram Jaccard at 0.7 threshold | Near-duplicates matter more than exact; n-gram catches paraphrased variants [VERIFIED: codebase] |
| Style validation | Manual prose/code ratio checks | `style_validator.py` code domain checks | Already enforces require_code_blocks and max_prose_ratio [VERIFIED: codebase] |
| JSONL writing | Manual file construction | `write_batch()` function from generate_tool_data.py pattern | Handles directory creation, encoding, one-per-line format [VERIFIED: codebase] |

**Key insight:** Phase 5's infrastructure cost is near-zero because ALL validation, curation, and quality tooling was built in Phases 1-2 and already handles the `code` domain. The only new code is the generation script itself.

## Common Pitfalls

### Pitfall 1: Style Validator Rejection from Prose-Heavy Responses

**What goes wrong:** Generated code samples have too much explanatory text, causing `max_prose_ratio: 0.4` rejection in the style validator.
**Why it happens:** The natural tendency is to explain code, but D-05 requires terse, code-first responses. If the comment after the code block is more than ~40% of the total response, it fails.
**How to avoid:** Keep post-code commentary to 1-2 sentences maximum. The code block itself should be the dominant content. For a 300-token response, no more than ~120 tokens should be prose.
**Warning signs:** High rejection rate in Stage 4 (style validation) of the curation pipeline.

[VERIFIED: scripts/style_validator.py lines 109-120 -- code domain logic]

### Pitfall 2: Missing Code Fences

**What goes wrong:** Some generated responses contain inline code without fenced blocks, failing `require_code_blocks: true`.
**Why it happens:** Templates that format code as plain text rather than wrapping in triple backticks with language identifier.
**How to avoid:** Every response template MUST include triple backtick fences. Use the `{language}` placeholder in fence opening: ```python, ```javascript, etc.
**Warning signs:** `"```" not in response_text` check fails in style_validator.py line 117.

[VERIFIED: configs/pipeline.yaml line 30 -- require_code_blocks: true for code domain]

### Pitfall 3: Dedup Collapse from Template Similarity

**What goes wrong:** Generated samples are too similar because the same template structure + minor parameter variation produces near-duplicate responses. Dedup at 0.7 Jaccard threshold removes large portions of the dataset.
**Why it happens:** Code for the same topic in the same language with similar structure produces high n-gram overlap. E.g., two "reverse string" functions in Python will share most characters.
**How to avoid:** Ensure diverse topic selection within each language. Vary function names, parameter names, implementation approaches. Use multiple implementation strategies for common operations (e.g., iterative vs recursive, different library functions).
**Warning signs:** After-dedup count drops more than 30% from quality-pass count.

[VERIFIED: scripts/dedup.py -- response-scope dedup at 0.7 threshold per configs/pipeline.yaml]

### Pitfall 4: Token Budget Overruns for Debugging Samples

**What goes wrong:** Debugging samples include buggy code + explanation + fixed code, easily exceeding token targets and potentially the 2048 hard limit.
**Why it happens:** D-07 targets debugging at 400-800 tokens. But presenting buggy code (100-200 tokens) + "Bug: X, Fix: Y" (50-100 tokens) + corrected code (100-200 tokens) plus system prompt (~50 tokens) adds up quickly.
**How to avoid:** Keep buggy code snippets short (5-15 lines). Use focused bugs (single-line fix, not architectural issues). The "Bug: X, Fix: Y" explanation should be 1-2 lines total per D-06.
**Warning signs:** validate_tokenizer.py reports token counts above 1500 for debugging samples.

### Pitfall 5: Language-Inappropriate Code Patterns

**What goes wrong:** Generated Go/Rust code uses Python/JS idioms that would not compile or would be unidiomatic.
**Why it happens:** Template-based generation that fills in code strings may not account for language-specific syntax (Go error handling, Rust ownership, TypeScript types vs JavaScript).
**How to avoid:** Maintain separate code template pools per language. Go functions should return `(value, error)` tuples. Rust should use `Result<T, E>`. TypeScript should have type annotations.
**Warning signs:** Code samples that look syntactically wrong for their stated language.

[ASSUMED: This is a standard concern for multi-language code generation]

### Pitfall 6: Debugging Samples Without Actual Bugs

**What goes wrong:** The "buggy" code in debugging samples is actually correct, or the bug is trivial/cosmetic rather than behavioral.
**Why it happens:** Generating plausible buggy code is harder than generating correct code. Simple string manipulation to "introduce" a bug can produce syntactically invalid code rather than subtly wrong code.
**How to avoid:** Define specific bug patterns from code.yaml's bug_types list (off-by-one, null reference, type mismatch, etc.) and create concrete buggy snippets for each. The bug should be a real behavioral issue that produces wrong output, not a syntax error.
**Warning signs:** Debugging samples where the "buggy" code would not run at all, or where the "fix" changes nothing meaningful.

## Code Examples

### Example 1: Utility Function Sample (CODE-01)

```python
# Expected output format for utility function generation [VERIFIED: generate_sample.py fixtures]
{
    "messages": [
        {
            "role": "system",
            "content": "You are a skilled programming assistant. Provide concise, correct code\nwith brief explanations. Use type hints in Python. Handle errors appropriately."
        },
        {
            "role": "user",
            "content": "Write a Python function to reverse a string."
        },
        {
            "role": "assistant",
            "content": "```python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n```"
        }
    ]
}
```

[VERIFIED: templates/system-prompts.yaml code_assistant prompt content; generate_sample.py CODE_REVERSE_STRING fixture]

### Example 2: File Operations Sample (CODE-02)

```python
# Expected output format for file operations [VERIFIED: generate_sample.py fixtures]
{
    "messages": [
        {
            "role": "system",
            "content": "You are a skilled programming assistant. Provide concise, correct code\nwith brief explanations. Use type hints in Python. Handle errors appropriately."
        },
        {
            "role": "user",
            "content": "Write a Python function to read a CSV file and return its contents as a list of dictionaries."
        },
        {
            "role": "assistant",
            "content": "```python\nimport csv\nfrom pathlib import Path\n\n\ndef read_csv(path: str) -> list[dict]:\n    with open(path, newline='') as f:\n        reader = csv.DictReader(f)\n        return list(reader)\n```"
        }
    ]
}
```

### Example 3: Debugging Sample (CODE-03)

```python
# Expected output format for debugging [VERIFIED: generate_sample.py fixtures]
{
    "messages": [
        {
            "role": "system",
            "content": "You are a debugging assistant. When shown code with a bug, identify the bug,\nexplain why it occurs, and provide the corrected code."
        },
        {
            "role": "user",
            "content": "This function should return the last N elements of a list, but it has a bug:\n\n```python\ndef last_n(lst, n):\n    return lst[len(lst) - n - 1:]\n```"
        },
        {
            "role": "assistant",
            "content": "Bug: Off-by-one error. `len(lst) - n - 1` skips one extra element.\nFix: Remove the `- 1`.\n\n```python\ndef last_n(lst, n):\n    return lst[-n:] if n > 0 else []\n```"
        }
    ]
}
```

Note: The debugging sample above uses the `code_debugger` system prompt, while utility and file-ops use `code_assistant`. [VERIFIED: templates/code.yaml system_prompt_ref field per category]

### Example 4: CLI Entry Point Pattern

```python
# Following Phase 4's CLI pattern exactly [VERIFIED: scripts/generate_tool_data.py main()]
VALID_CATEGORIES = ["utility", "file-ops", "debugging"]
MAX_COUNT = 10000

parser = argparse.ArgumentParser(
    description="Generate code generation training data batches for Lyra"
)
parser.add_argument("--category", type=str, required=True, choices=VALID_CATEGORIES)
parser.add_argument("--count", type=int, default=50)
parser.add_argument("--batch", type=int, required=True)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
```

### Example 5: Batch Naming Convention

```
datasets/code/utility-batch-01.jsonl      # 50 utility function samples
datasets/code/utility-batch-02.jsonl
...
datasets/code/utility-batch-34.jsonl      # Last utility batch (~1,667 total)
datasets/code/file-ops-batch-01.jsonl     # 50 file operation samples
...
datasets/code/file-ops-batch-17.jsonl     # Last file-ops batch (~834 total)
datasets/code/debugging-batch-01.jsonl    # 50 debugging samples
...
datasets/code/debugging-batch-17.jsonl    # Last debugging batch (~834 total)
```

[ASSUMED: Batch naming follows Phase 4 pattern `{category}-batch-{NN}.jsonl` per Claude's discretion]

## Batch Count Calculations

Derived from D-01 and D-02 locked decisions:

| Category | Raw Samples | Batches of 50 | Curated Target |
|----------|-------------|---------------|----------------|
| Utility | ~1,667 | 34 batches | ~834 |
| File ops | ~834 | 17 batches | ~417 |
| Debugging | ~834 | 17 batches | ~417 |
| **Total** | **~3,335** | **68 batches** | **~1,668** |

[VERIFIED: D-01 says ~3,334 raw -> ~1,667 curated; D-02 specifies 50/25/25 split]

## Topic Pool Design

### Utility Functions (CODE-01)

Per `templates/code.yaml`, 7 base topics exist. For ~1,667 samples across 5 languages, each topic-language combination needs ~48 variants. Expanded topic pool recommendation (Claude's discretion area):

**String manipulation (expanded):**
- reverse, capitalize, truncate, slugify (from template)
- camelCase/snake_case conversion, palindrome check, word count, trim whitespace, repeat string, pad string, extract substrings, regex match, string interpolation

**Array/list operations (expanded):**
- flatten, chunk, deduplicate, sort (from template)
- zip, rotate, partition, compact (remove nulls), intersection, difference, group by key, sliding window, interleave, transpose matrix

**Date/time (expanded):**
- format, parse (from template)
- add/subtract duration, time zone convert, relative time ("3 hours ago"), day of week, is weekend, days between dates, ISO format conversion

**Number utilities (expanded):**
- fibonacci, prime check, GCD (from template)
- factorial, LCM, is power of two, clamp range, percentage, round to N decimals, random in range, binary/hex conversion

**Data structure helpers (expanded):**
- deep merge, path lookup, type checking (from template)
- flatten object, pick/omit keys, invert map, frequency counter, LRU cache, stack, queue, set operations on arrays

**Validation functions (expanded):**
- email, URL, phone (from template)
- IP address, credit card (Luhn), UUID, date string, JSON string, hex color, semantic version

**Encoding/decoding (expanded):**
- base64, hex, URL encoding (from template)
- JWT decode (header only), ROT13, HTML entities, escape/unescape, binary to/from string

[ASSUMED: Expanded topics are based on common programming utility needs]

### File Operations (CODE-02)

Per `templates/code.yaml`, 7 base topics. For ~834 samples across 3 languages:

**Error handling patterns that MUST be present per template generation_notes:**
- try/except (Python), try/catch (JS/TS)
- Context managers (`with` in Python)
- Proper file close patterns
- Path validation before operations

### Debugging (CODE-03)

Per `templates/code.yaml`, 7 bug types. For ~834 samples across 3 languages:

| Bug Type | Example (Python) | Example (JS/TS) |
|----------|------------------|------------------|
| Off-by-one | `range(1, n)` should be `range(1, n+1)` | `for (let i=0; i<=arr.length)` |
| Null/undefined | Missing `if x is not None` guard | Accessing `.property` on potentially undefined |
| Type mismatch | `"5" + 3` concatenation vs addition | Implicit type coercion in comparisons |
| Logic errors | Wrong operator in conditional (`and` vs `or`) | Inverted boolean condition |
| Async/await | Missing `await` on async call | Unhandled promise rejection |
| Variable scoping | Mutable default argument `def f(x=[])` | `var` in loop vs `let` |
| Import/module | Circular import, wrong relative import | Missing dependency, wrong export |

[ASSUMED: Bug examples are based on standard programming pitfalls]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Tool-based generation (Phase 4) | Direct code response generation (Phase 5) | Phase boundary | Simpler conversation structure -- no tools column needed |
| Separate format per domain | Unified TRL-native messages format | Phase 1 | All domains use same Pydantic model; no domain-specific format code |

## Curation Pipeline Configuration for Code Domain

The existing pipeline.yaml already configures the code domain [VERIFIED: configs/pipeline.yaml]:

```yaml
code:
  min_response_chars: 20
  style:
    max_tokens: 600
    require_code_blocks: true
    max_prose_ratio: 0.4
```

Key implications:
- **min_response_chars: 20** -- Even short utility functions like `reverse_string` meet this (the fenced code block alone is 20+ chars).
- **max_tokens: 600 (approximate)** -- Style validator uses `word_count * 1.3` approximation. This allows ~460 words. Utility samples (200-500 tokens per D-07) fit. File ops (300-800 tokens) may occasionally exceed -- the 2x overgeneration absorbs this.
- **require_code_blocks: true** -- Every response MUST contain triple backtick fences.
- **max_prose_ratio: 0.4** -- At most 40% of the response can be outside code blocks. This is the most likely source of curation rejection.
- **dedup_threshold: 0.7 (global default)** -- Code domain does not override, so uses response-scope dedup at 0.7 Jaccard.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `rng.choices()` with weights correctly implements language distribution | Architecture Patterns | Wrong distribution ratios; easily caught by statistical check on generated batches |
| A2 | Expanded utility function topics cover sufficient diversity for ~1,667 samples | Topic Pool Design | Dedup collapse if topics are too narrow; mitigated by 2x overgeneration |
| A3 | Bug type examples for each language are realistic and idiomatic | Topic Pool Design | Debugging samples may teach incorrect patterns; mitigated by curation pipeline quality checks |
| A4 | Batch naming follows `{category}-batch-{NN}.jsonl` | Code Examples | Naming mismatch; purely cosmetic risk |
| A5 | Style validator max_tokens: 600 allows file-ops samples at 300-800 token range | Curation Pipeline | Some longer file-ops samples rejected; 2x overgeneration provides buffer |

## Open Questions

1. **Debugging sample user message format**
   - What we know: D-06 says "Bug: X, Fix: Y" format for the assistant response. The user message presents buggy code.
   - What's unclear: Should the user message always include a fenced code block with the buggy code, or can it describe the bug in plain text? Looking at the generate_sample.py fixture (CODE_FIX_OFF_BY_ONE), the user provides fenced code with a description.
   - Recommendation: Follow the fixture pattern -- user provides buggy code in fenced blocks with a brief description of expected behavior. This teaches the model to read and analyze code, not just respond to descriptions.

2. **Go and Rust code quality**
   - What we know: Go and Rust are included for utility functions per D-03 (20% and 15% respectively).
   - What's unclear: Can template-based generation produce idiomatic Go/Rust code? Go requires error handling patterns (`if err != nil`), Rust requires ownership/borrowing awareness.
   - Recommendation: Create language-specific code pools for Go and Rust rather than trying to translate Python patterns. Keep Go/Rust samples to simpler utility functions where idiomatic patterns are straightforward.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | tests/conftest.py (shared fixtures) |
| Quick run command | `python3 -m pytest tests/test_generate_code_data.py -x -q` |
| Full suite command | `python3 -m pytest tests/ -x -q` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CODE-01 | Utility batch generates valid conversations | unit | `pytest tests/test_generate_code_data.py::TestUtilityBatch -x` | No -- Wave 0 |
| CODE-01 | Utility samples use code_assistant system prompt | unit | `pytest tests/test_generate_code_data.py::TestUtilityBatch::test_uses_code_assistant_prompt -x` | No -- Wave 0 |
| CODE-01 | Utility samples cover all 5 languages | unit | `pytest tests/test_generate_code_data.py::TestUtilityBatch::test_language_coverage -x` | No -- Wave 0 |
| CODE-02 | File-ops batch generates valid conversations | unit | `pytest tests/test_generate_code_data.py::TestFileOpsBatch -x` | No -- Wave 0 |
| CODE-02 | File-ops samples include error handling patterns | unit | `pytest tests/test_generate_code_data.py::TestFileOpsBatch::test_error_handling_present -x` | No -- Wave 0 |
| CODE-03 | Debugging batch generates valid conversations | unit | `pytest tests/test_generate_code_data.py::TestDebuggingBatch -x` | No -- Wave 0 |
| CODE-03 | Debugging samples use code_debugger system prompt | unit | `pytest tests/test_generate_code_data.py::TestDebuggingBatch::test_uses_code_debugger_prompt -x` | No -- Wave 0 |
| CODE-03 | Debugging samples contain bug identification | unit | `pytest tests/test_generate_code_data.py::TestDebuggingBatch::test_bug_fix_format -x` | No -- Wave 0 |
| ALL | All generated samples pass Pydantic validation | unit | `pytest tests/test_generate_code_data.py::TestAllSamplesValidate -x` | No -- Wave 0 |
| ALL | CLI writes JSONL to datasets/code/ | integration | `pytest tests/test_generate_code_data.py::TestCliEntryPoint -x` | No -- Wave 0 |
| ALL | System message is first in every sample | unit | `pytest tests/test_generate_code_data.py::TestSystemMessageFirst -x` | No -- Wave 0 |
| ALL | No duplicate user queries in a batch | unit | `pytest tests/test_generate_code_data.py::TestQueryDiversity -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `python3 -m pytest tests/test_generate_code_data.py -x -q`
- **Per wave merge:** `python3 -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before verification

### Wave 0 Gaps
- [ ] `tests/test_generate_code_data.py` -- all test classes for 3 categories + CLI + validation + diversity
- [ ] No new conftest.py fixtures needed -- existing conftest covers format validation; code-specific fixtures go in the test file

## Sources

### Primary (HIGH confidence)
- `scripts/generate_tool_data.py` -- Reference implementation pattern (1290 lines, all 5 category generators, CLI, validation, write) [VERIFIED: codebase read]
- `scripts/validate_format.py` -- Pydantic Conversation model with all validation rules [VERIFIED: codebase read]
- `scripts/validate_tokenizer.py` -- SmolLM2 token counting, 2048 max, chat template conversion [VERIFIED: codebase read]
- `scripts/curate_pipeline.py` -- 4-stage pipeline with domain-aware configuration [VERIFIED: codebase read]
- `scripts/quality_scorer.py` -- 4-signal quality scoring (format, completeness, naturalness, diversity) [VERIFIED: codebase read]
- `scripts/style_validator.py` -- Code domain style checks (require_code_blocks, max_prose_ratio) [VERIFIED: codebase read]
- `configs/pipeline.yaml` -- Code domain configuration: min_response_chars=20, max_tokens=600, require_code_blocks=true, max_prose_ratio=0.4 [VERIFIED: codebase read]
- `templates/code.yaml` -- 3 categories, 7 topics each, language lists, complexity levels, system_prompt_ref [VERIFIED: codebase read]
- `templates/system-prompts.yaml` -- code_assistant and code_debugger prompt content [VERIFIED: codebase read]
- `tests/test_generate_tool_data.py` -- Test pattern: 10 test classes, category validation, CLI test, schema usage, query diversity [VERIFIED: codebase read]
- `scripts/generate_sample.py` -- Code domain fixtures showing expected output format [VERIFIED: codebase read]
- Phase 4 batch statistics: 67 batch files, 3300 samples across 5 categories [VERIFIED: filesystem count]

### Secondary (MEDIUM confidence)
- `scripts/dedup.py` -- N-gram Jaccard dedup with configurable scope and threshold [VERIFIED: codebase read]
- `scripts/pipeline_config.py` -- PipelineConfig Pydantic model for pipeline.yaml [VERIFIED: codebase read]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already present, no new dependencies needed
- Architecture: HIGH -- directly follows Phase 4 pattern which is verified working (29 tests passing)
- Pitfalls: HIGH -- identified from actual codebase constraints (style validator, dedup thresholds)
- Code domain specifics: HIGH -- configs/pipeline.yaml and style_validator.py code paths verified

**Research date:** 2026-04-20
**Valid until:** 2026-05-20 (stable -- no external dependencies changing)
