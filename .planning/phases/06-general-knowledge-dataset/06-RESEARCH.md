# Phase 6: General Knowledge Dataset - Research

**Researched:** 2026-04-20
**Domain:** Synthetic general knowledge dataset generation (reasoning, Q&A, explanations)
**Confidence:** HIGH

## Summary

Phase 6 generates ~3,334 raw general knowledge training samples across three categories (factual Q&A, explanations, reasoning chains) that the existing curation pipeline filters to ~1,667 curated samples. This is the third and final data generation phase, following the proven pattern established in Phases 4-5: standalone Python script with category batch generators, CLI entry point, Pydantic validation, and JSONL output to `datasets/knowledge/`.

The critical lesson from Phase 5 is that template-based generation with a small topic pool (Phase 5 had ~20 templates per code topic area) produces high structural similarity that collapses under the dedup pipeline -- Phase 5 achieved only 17.6% retention (600/3,400). Phase 6 MUST use a large topic pool (200+ unique questions) with varied phrasing across how/what/why/explain/compare forms. Knowledge questions are inherently more diverse than code templates because each question addresses different factual content, which should naturally produce lower n-gram Jaccard similarity. However, the response structure (especially reasoning chains with repeated "Step 1...", "Step 2...", "Therefore..." patterns) can trigger the style validator's reasoning_markers check AND the dedup system simultaneously if not carefully varied.

**Primary recommendation:** Build `scripts/generate_knowledge_data.py` following the Phase 4-5 pattern exactly, but with a 200+ question topic pool organized by domain (tech, math, science, other) and category (qa, explanation, reasoning). Each question must have a pre-written response that varies in structure and vocabulary. Target 50%+ curation retention by ensuring question-response pairs have genuinely diverse content, not just template substitutions.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Generate ~3,334 raw knowledge samples. Curation pipeline filters down to ~1,667 final samples (the 33% knowledge slice of 5K total).
- **D-02:** Q&A-heavy distribution:
  - Factual Q&A: ~40% (~1,334 raw -> ~667 curated)
  - Explanations: ~35% (~1,167 raw -> ~584 curated)
  - Reasoning chains: ~25% (~834 raw -> ~417 curated)
- **D-03:** Category-adaptive response depth:
  - Reasoning chains: always show numbered steps + explicit conclusion. Target ~500-1200 tokens.
  - Factual Q&A: concise 1-5 sentences for simple facts. Target ~200-600 tokens.
  - Explanations: structured with examples/analogies, adaptive to topic complexity. Target ~400-1000 tokens.
- **D-04:** This is the "detailed" domain per Phase 2 style validation (D-09). Reasoning chains and explanations use chain-of-thought. Q&A stays concise but factually complete.
- **D-05:** Tech and STEM weighted topic distribution:
  - Technology/computing: ~40%
  - Math/logic: ~25%
  - Science: ~20%
  - Other (history, geography, everyday): ~15%
- **D-06:** Aligns with Lyra's developer audience -- practical knowledge a developer would query a small model about.
- **D-07:** Large topic pool of 200+ unique question topics (vs Phase 5's ~20 templates that caused 78% dedup rejection). Each batch uses different topic subsets.
- **D-08:** Varied question phrasing across how/what/why/explain/compare forms. Natural language diversity prevents n-gram overlap at the 0.7 Jaccard threshold.
- **D-09:** Generation script must produce samples with unique enough content that the curation pipeline retains 50%+ (vs Phase 5's 17.6% retention rate).
- **D-10:** New script `scripts/generate_knowledge_data.py` following the same pattern as Phases 4-5 (CLI, category generators, batch of 50, inline validation).
- **D-11:** Category batches with validation loops. Order: factual Q&A first (simplest, largest volume), then explanations, then reasoning chains (most complex).

### Claude's Discretion
- Specific question topics within the 200+ pool
- Exact phrasing variations per topic
- Whether to include follow-up questions in multi-turn knowledge conversations
- Batch file naming convention (following pattern: `{category}-batch-{NN}.jsonl`)
- Test fixtures for the generation script

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| KNOW-01 | Dataset includes reasoning chain samples with explicit chain-of-thought | Reasoning chain category (~25% of 3,334 raw). Uses `knowledge_reasoning` system prompt. Style validator requires `has_reasoning_markers()` (2+ patterns from: "step N", "because", "therefore", "first,", "finally,", etc.). Response structure: numbered steps followed by explicit conclusion. |
| KNOW-02 | Dataset includes factual Q&A samples across diverse domains | Factual Q&A category (~40% of 3,334 raw). Uses `knowledge_assistant` system prompt. 200+ unique questions spanning technology (~40%), math (~25%), science (~20%), other (~15%). Concise 1-5 sentence responses. |
| KNOW-03 | Dataset includes explanation and teaching samples with adaptive detail level | Explanation category (~35% of 3,334 raw). Uses `knowledge_assistant` system prompt. How/what/why/compare/explain-to-beginner question forms. Response depth adapts to topic complexity per D-03. |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- **Base model**: SmolLM2-1.7B -- all training data must fit within 2048 token limit after tokenization
- **Data format**: TRL-native ShareGPT messages/role/content format (NOT classic from/value)
- **License**: MIT -- all generated data must be original, not copied from copyrighted sources
- **Data source**: All training data patterns are pre-written in the script, no live Opus API calls during generation
- **Scale strategy**: ~5K total samples, 33% knowledge = ~1,667 curated
- **Token limits**: Practical max per training sample ~4000-6000 tokens total raw, fitting 2048 after tokenization
- **GSD Workflow**: All work through GSD commands, no direct repo edits

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python stdlib (json, random, argparse) | 3.10+ | Script infrastructure | Same as Phase 4-5 scripts [VERIFIED: existing codebase pattern] |
| PyYAML | installed | Template and system prompt loading | Already a dependency, used by all generation scripts [VERIFIED: generate_code_data.py imports yaml] |
| Pydantic (via validate_format) | 2.x | Conversation format validation | Existing Conversation model validates all samples [VERIFIED: validate_format.py] |

### Supporting (reuse, no new installs)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scripts.validate_format.Conversation | existing | Pydantic validation per sample | validate_batch() calls Conversation.model_validate() [VERIFIED: generate_code_data.py line 737] |
| scripts.curate_pipeline | existing | Post-generation curation (format + quality + dedup + style) | Run after all raw batches generated [VERIFIED: curate_pipeline.py] |
| scripts.dedup | existing | N-gram Jaccard deduplication | Knowledge domain uses response-scope at 0.7 threshold (defaults) [VERIFIED: pipeline.yaml] |
| scripts.style_validator | existing | Knowledge domain: min_tokens=200, require_reasoning_markers=true | Style stage of curation pipeline [VERIFIED: pipeline.yaml knowledge config] |
| scripts.quality_scorer | existing | Format compliance, completeness, naturalness, diversity checks | Quality stage of curation pipeline [VERIFIED: quality_scorer.py] |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Pre-written responses | Anthropic Batch API for Opus-generated responses | Higher quality but adds API cost and latency. D-10 says follow Phase 4-5 pattern (pre-written). Opus generation deferred to v2 data improvement. |
| Flat topic pool | Topic hierarchy with template substitution | Hierarchy adds complexity but no diversity gain if the atomic topics are already unique |

**Installation:**
No new packages required. All dependencies already installed from Phases 1-5.

## Architecture Patterns

### Recommended Project Structure
```
scripts/
  generate_knowledge_data.py    # NEW: Phase 6 generation script
  generate_code_data.py         # Reference: Phase 5 (same pattern)
  generate_tool_data.py         # Reference: Phase 4 (same pattern)
  validate_format.py            # Reuse: Pydantic validation
  curate_pipeline.py            # Reuse: Post-generation curation
  dedup.py                      # Reuse: N-gram Jaccard dedup
  quality_scorer.py             # Reuse: 4-signal quality scoring
  style_validator.py            # Reuse: Knowledge domain style checks
datasets/
  knowledge/
    qa-batch-01.jsonl           # Raw Q&A batches
    qa-batch-02.jsonl
    ...
    explanation-batch-01.jsonl  # Raw explanation batches
    ...
    reasoning-batch-01.jsonl    # Raw reasoning chain batches
    ...
    curated/
      knowledge-curated.jsonl   # Final curated output
templates/
  knowledge.yaml                # Category definitions (already exists)
  system-prompts.yaml           # System prompts (already exists)
tests/
  test_generate_knowledge_data.py  # NEW: Tests for generation script
configs/
  pipeline.yaml                 # Curation config (already has knowledge domain)
```

### Pattern 1: Category Batch Generator (from Phase 4-5)
**What:** Each category (qa, explanation, reasoning) has its own generator function that produces a list of conversation dicts.
**When to use:** Every batch generation call.
**Example:**
```python
# Source: generate_code_data.py (verified pattern)
def generate_qa_batch(count: int = 50, seed: int = None) -> list[dict]:
    """Generate factual Q&A samples per KNOW-02."""
    rng = random.Random(seed)
    prompts = load_system_prompts()
    system_content = prompts["knowledge_assistant"]

    samples = []
    used_queries = set()

    for _ in range(count):
        topic = _pick_weighted_topic(QA_TOPICS, DOMAIN_WEIGHTS, rng)
        query, response = _get_unique_qa(topic, used_queries, rng)

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

### Pattern 2: Large Topic Pool with Pre-Written Responses (Phase 6 specific)
**What:** Unlike Phase 5's template substitution ({lang} placeholders), Phase 6 uses unique question-response pairs where each pair has genuinely different content.
**When to use:** All three categories.
**Key difference from Phase 5:** Phase 5 had ~80 utility query templates * 5 languages = ~400 combinations, but the responses were structurally identical (just different language fences). Phase 6 has 200+ unique questions where BOTH the question and response content differ.

```python
# Knowledge domain: each entry is a unique question + unique response
# This is the critical diversity strategy per D-07
QA_TOPICS = {
    "technology": [
        {
            "q": "What is a DNS server and what does it do?",
            "a": "A DNS (Domain Name System) server translates human-readable domain names like example.com into IP addresses that computers use to communicate. When you type a URL in your browser, your device queries a DNS server to find the corresponding IP address, then connects to that IP to load the website."
        },
        {
            "q": "How does HTTPS differ from HTTP?",
            "a": "HTTPS adds a TLS/SSL encryption layer on top of HTTP. With HTTP, data travels in plaintext -- anyone intercepting the traffic can read it. HTTPS encrypts all data between your browser and the server using public-key cryptography, preventing eavesdropping and tampering."
        },
        # ... 80+ technology topics
    ],
    "math": [
        # ... 50+ math topics
    ],
    # ...
}
```

### Pattern 3: Reasoning Chain Structure
**What:** Reasoning samples must pass `has_reasoning_markers()` in the style validator, which checks for 2+ matches from: "step N", numbered lists ("1.", "2."), "because", "therefore", "first,", "second,", "finally,", "let me", "this means".
**When to use:** All reasoning chain samples.
**Key insight:** Every reasoning response MUST include at least 2 of these markers to pass style validation. The response should show explicit numbered steps followed by a conclusion.

```python
# Reasoning chain response structure that passes style validation
REASONING_TOPICS = {
    "math": [
        {
            "q": "If a train travels 120 km in 2 hours, then speeds up to cover the next 180 km in 2 hours, what is its average speed for the entire trip?",
            "a": "Let me work through this step by step.\n\n1. Total distance traveled: 120 km + 180 km = 300 km\n2. Total time taken: 2 hours + 2 hours = 4 hours\n3. Average speed = Total distance / Total time = 300 km / 4 hours = 75 km/h\n\nTherefore, the average speed for the entire trip is 75 km/h. Note that this is different from simply averaging the two speeds (60 km/h and 90 km/h), which would give 75 km/h in this case but would be incorrect in general when the time intervals differ."
        },
        # ...
    ],
}
```

### Anti-Patterns to Avoid
- **Template substitution for diversity:** Do NOT use `{placeholder}` patterns that produce structurally identical responses (Phase 5 lesson). Each question-response pair must be genuinely unique content.
- **Shared response prefixes:** Do NOT start every Q&A response with the same phrase (e.g., "Great question!"). This inflates n-gram similarity. Vary openings.
- **Uniform response lengths:** Do NOT target exact token counts. Let responses vary naturally within the D-03 ranges. Fixed-length responses increase dedup similarity.
- **Reasoning markers only at end:** Do NOT put "therefore" only as the last word. Spread reasoning markers throughout the response to pass the 2+ marker threshold reliably.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Format validation | Custom JSON schema checks | `Conversation.model_validate()` from `scripts/validate_format.py` | Already handles all role ordering rules, tool call consistency, system-first enforcement |
| Quality scoring | Custom quality checks | `scripts/quality_scorer.score_sample()` | 4-signal scoring already handles completeness, naturalness, meta-commentary, format |
| Deduplication | Custom similarity detection | `scripts/dedup.deduplicate_batch()` | N-gram Jaccard with configurable scope and threshold |
| Style validation | Custom response style checks | `scripts/style_validator.validate_style()` with domain="knowledge" | Already enforces min_tokens=200, require_reasoning_markers=true |
| Curation pipeline | Manual multi-step filtering | `scripts/curate_pipeline.run_pipeline()` with domain="knowledge" | 4-stage pipeline already configured for knowledge domain in pipeline.yaml |
| Token counting | Manual word counting | `scripts/validate_tokenizer.py` | Exact SmolLM2 tokenizer counts vs approximate word*1.3 |

**Key insight:** The entire curation infrastructure is built and tested (204 tests passing). Phase 6 only needs to produce raw JSONL that feeds into the existing pipeline. The generation script is the only new code.

## Common Pitfalls

### Pitfall 1: Dedup Collapse from Structural Similarity (CRITICAL -- Phase 5 Lesson)
**What goes wrong:** Template-generated data shares structural patterns (same sentence openings, same explanation frameworks, same reasoning step format) that produce high n-gram Jaccard similarity even when the factual content differs.
**Why it happens:** The dedup pipeline uses character n-grams at n=3, so shared phrases like "Step 1:", "Let me explain", "The answer is" create large overlapping n-gram sets. At the default 0.7 threshold, two responses that share 70%+ of their 3-character subsequences are considered duplicates.
**How to avoid:**
  1. Use 200+ unique question-response pairs (not templates)
  2. Vary response openings -- do not start every Q&A with the same phrase
  3. Vary reasoning step phrasing -- alternate between "Step 1:", "First,", "1.", "To begin,"
  4. Vary explanation structures -- some use analogies, some use examples, some use comparisons
  5. Keep responses at naturally varying lengths within D-03 ranges
**Warning signs:** If a test batch of 50 samples deduplicates to fewer than 35, the topic pool needs more diversity.
**Phase 5 numbers for reference:** 3,400 raw -> 762 unique after exact dedup -> 600 after full pipeline (17.6% retention). Phase 6 target: 50%+ retention = 3,334 raw -> 1,667+ curated.

### Pitfall 2: Style Validator Rejection for Q&A Samples
**What goes wrong:** Factual Q&A responses that are too short (below 200 approximate tokens) fail the knowledge domain style check.
**Why it happens:** The knowledge domain config requires `min_tokens: 200` (approximately 154 words at 1.3x ratio). Short factual Q&A like "The capital of France is Paris" would fail.
**How to avoid:** Even concise Q&A responses must meet the 200 approximate token minimum. Add brief context or explanation to reach the threshold. D-03 specifies Q&A targets ~200-600 tokens, so the lower bound aligns with the style validator.
**Warning signs:** Q&A samples that are single sentences will fail style validation.

### Pitfall 3: Reasoning Marker Check Failure
**What goes wrong:** Responses that explain reasoning but don't use the specific marker patterns checked by `has_reasoning_markers()` fail style validation.
**Why it happens:** The validator requires 2+ matches from a specific pattern list: `step \d`, `\d+\.`, `because`, `therefore`, `first,`, `second,`, `finally,`, `let me`, `this means`. If a response uses different phrasing (e.g., "consequently" instead of "therefore"), it may not match.
**How to avoid:** Always include at least 2-3 of the recognized patterns in reasoning chain AND explanation responses. Use numbered lists ("1.", "2.", "3.") as the most reliable trigger (matches `\d+\.` pattern).
**Warning signs:** The `require_reasoning_markers: true` config applies to ALL knowledge domain samples. Both reasoning chains AND explanations must pass this check, not just reasoning chains.

### Pitfall 4: Token Budget Overflow
**What goes wrong:** Samples that exceed the 2048 token limit after SmolLM2 tokenization are rejected during training.
**Why it happens:** The knowledge domain targets longer responses (500-1200 tokens for reasoning chains). Combined with system prompt (~200 tokens) and user question (~50 tokens), it is easy to exceed 2048.
**How to avoid:** D-03 token targets (200-600 for Q&A, 400-1000 for explanations, 500-1200 for reasoning) are designed to fit. Keep individual responses within these ranges. The system prompts are short (~40-50 words each).
**Warning signs:** Reasoning chains with elaborate multi-step solutions that exceed 1200 tokens.

### Pitfall 5: Knowledge Domain min_response_chars = 100
**What goes wrong:** Samples with assistant responses shorter than 100 characters fail the quality scorer's completeness check for the knowledge domain.
**Why it happens:** The pipeline.yaml knowledge domain override sets `min_response_chars: 100` (vs default 10). This is more aggressive than other domains.
**How to avoid:** Ensure every response is at least 100 characters. Even the shortest Q&A responses should be 2-3 sentences minimum.
**Warning signs:** Any Q&A response under ~20 words will fail this check.

## Code Examples

Verified patterns from the existing codebase:

### Three-Message Knowledge Conversation (Basic Structure)
```python
# Source: generate_code_data.py verified pattern + knowledge system prompts
# Knowledge domain uses simple system -> user -> assistant structure
sample = {
    "messages": [
        {"role": "system", "content": "You are a knowledgeable assistant. Provide accurate, clear answers.\nFor complex topics, break your explanation into steps.\nFor simple factual questions, be concise."},
        {"role": "user", "content": "What is the difference between TCP and UDP?"},
        {"role": "assistant", "content": "TCP (Transmission Control Protocol) and UDP (User Datagram Protocol) are both transport layer protocols, but they differ in key ways.\n\n1. TCP is connection-oriented -- it establishes a connection before sending data and guarantees delivery through acknowledgments and retransmission. UDP is connectionless and sends data without establishing a connection first.\n\n2. TCP provides ordered delivery, meaning packets arrive in the sequence they were sent. UDP does not guarantee ordering.\n\n3. TCP is slower due to the overhead of connection management and error checking. UDP is faster because it skips these steps.\n\nTCP is used for web browsing, email, and file transfers where reliability matters. UDP is used for video streaming, gaming, and DNS queries where speed matters more than perfect delivery."},
    ]
}
```

### Category Generator with Weighted Topic Selection
```python
# Source: generate_code_data.py _pick_language pattern adapted for topic domains
DOMAIN_WEIGHTS = {
    "technology": 0.40,
    "math": 0.25,
    "science": 0.20,
    "other": 0.15,
}

def _pick_weighted_domain(rng: random.Random) -> str:
    """Pick a topic domain using weighted distribution per D-05."""
    domains = list(DOMAIN_WEIGHTS.keys())
    weights = list(DOMAIN_WEIGHTS.values())
    return rng.choices(domains, weights=weights, k=1)[0]
```

### Unique Query Selection (Anti-Dedup Pattern)
```python
# Source: generate_code_data.py _get_unique_query adapted for knowledge
# Key difference: no template substitution, each entry is a unique dict
def _get_unique_qa(domain_pool: list[dict], used: set, rng: random.Random) -> tuple[str, str]:
    """Select a unique Q&A pair, avoiding duplicates within a batch."""
    max_attempts = 100
    for _ in range(max_attempts):
        entry = rng.choice(domain_pool)
        if entry["q"] not in used:
            used.add(entry["q"])
            return entry["q"], entry["a"]
    # Fallback: use entry with variant marker
    entry = rng.choice(domain_pool)
    q = f"{entry['q']} (variant {len(used)})"
    used.add(q)
    return q, entry["a"]
```

### Batch Validation and Writing (Exact Reuse)
```python
# Source: generate_code_data.py lines 735-845 (verified)
def validate_batch(samples: list[dict]) -> dict:
    """Validate all samples via Conversation.model_validate()."""
    results = {"total": len(samples), "valid": 0, "invalid": 0, "errors": []}
    for i, sample in enumerate(samples):
        try:
            Conversation.model_validate(sample)
            results["valid"] += 1
        except Exception as e:
            results["invalid"] += 1
            results["errors"].append({"index": i, "error": str(e)})
    return results

def write_batch(samples: list[dict], output_path: Path) -> Path:
    """Write samples as JSONL to output_path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    return output_path
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Template substitution ({lang} placeholders) | Unique question-response pairs | Phase 6 (lesson from Phase 5) | Prevents dedup collapse, targets 50%+ retention vs 17.6% |
| Response-scope dedup at 0.7 | Same (default for knowledge) | Phase 2 | Knowledge domain uses defaults; code/tool-calling have custom overrides |
| Code-first terse style | Detailed chain-of-thought style | Phase 2 | Knowledge domain requires min_tokens=200, reasoning markers |

**Key Phase 5 takeaway:** Template-generated data produced 17.6% retention (600/3,400). The root cause was not the dedup threshold but the generation approach -- templates produce structurally identical outputs. Phase 6's unique Q&A pairs should naturally achieve higher diversity because the factual content differs per question.

## Assumptions Log

> List all claims tagged [ASSUMED] in this research.

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Knowledge Q&A pairs with unique factual content will naturally achieve lower Jaccard similarity than code templates | Pitfall 1 | If wrong, may need to raise dedup threshold for knowledge domain (like code domain at 0.98) or add dedup scope override |
| A2 | 200+ pre-written Q&A pairs is sufficient for 3,334 raw samples at 50%+ retention | Architecture | If wrong, need more entries (each entry used ~16x on average; dedup may still flag reuse within domain subsets) |
| A3 | The style validator's require_reasoning_markers applies to ALL knowledge domain samples equally | Pitfall 3 | If wrong (only reasoning chains), then Q&A samples don't need markers. But code shows it applies per-domain, not per-category [VERIFIED: style_validator.py line 124-128 checks domain=="knowledge" not category] |

**Clarification on A3:** This is actually VERIFIED. The style_validator.py `validate_style()` function checks `domain == "knowledge"` (line 124) and applies `require_reasoning_markers` to ALL samples in that domain. This means even factual Q&A responses must contain at least 2 reasoning marker patterns. This is a significant constraint -- short factual answers like "The speed of light is 299,792,458 m/s" would fail. All Q&A responses need some structural markers (numbered points, "because", etc.) to pass.

## Open Questions

1. **Should knowledge domain dedup config be tuned?**
   - What we know: Knowledge domain uses defaults (response-scope, 0.7 threshold). Code domain needed 0.98 threshold. Tool-calling uses full-scope at 0.9.
   - What's unclear: Whether 200+ unique Q&A pairs at 0.7 threshold will achieve 50%+ retention. The threshold may need adjustment if responses share too many structural patterns.
   - Recommendation: Generate a test batch of 50 Q&A samples, run through dedup, and check retention before committing to full generation. If retention is below 50%, raise threshold to 0.8-0.9 in pipeline.yaml for knowledge domain.

2. **Should require_reasoning_markers be disabled for Q&A?**
   - What we know: The style validator applies require_reasoning_markers=true to ALL knowledge domain samples. Q&A responses are supposed to be concise (1-5 sentences per D-03).
   - What's unclear: Whether concise Q&A can reliably include 2+ reasoning markers without feeling forced.
   - Recommendation: Design Q&A responses to naturally include markers like numbered points ("1.", "2.") or "because" explanations. This is compatible with concise answers -- "X happens because Y" is still concise. If this proves too restrictive, the pipeline.yaml could split knowledge into sub-domains, but that requires pipeline changes outside this phase's scope.

3. **Single-turn vs multi-turn knowledge conversations?**
   - What we know: D-10 says "follow Phase 4-5 pattern." Phase 5 (code) uses single-turn only (system -> user -> assistant). Phase 4 (tool-calling) uses multi-turn. Claude's Discretion includes "whether to include follow-up questions."
   - What's unclear: Whether multi-turn adds value for knowledge (e.g., user asks follow-up after an explanation).
   - Recommendation: Keep single-turn for v1. Multi-turn adds complexity (more messages, higher token count, harder to deduplicate) without clear benefit for a 1.7B model learning factual content. Single-turn is simpler and proven.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing, 204 tests passing) |
| Config file | existing pytest configuration |
| Quick run command | `python3 -m pytest tests/test_generate_knowledge_data.py -x -q` |
| Full suite command | `python3 -m pytest -x -q` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| KNOW-01 | Reasoning chain samples have numbered steps + conclusion + reasoning markers | unit | `pytest tests/test_generate_knowledge_data.py::TestReasoningBatch -x` | No -- Wave 0 |
| KNOW-02 | Q&A samples span diverse domains with factual responses | unit | `pytest tests/test_generate_knowledge_data.py::TestQABatch -x` | No -- Wave 0 |
| KNOW-03 | Explanation samples use adaptive detail with analogies/examples | unit | `pytest tests/test_generate_knowledge_data.py::TestExplanationBatch -x` | No -- Wave 0 |
| D-02 | Distribution weights match Q&A 40%, explanation 35%, reasoning 25% | unit | `pytest tests/test_generate_knowledge_data.py::TestDistribution -x` | No -- Wave 0 |
| D-05 | Topic domain weights: tech 40%, math 25%, science 20%, other 15% | unit | `pytest tests/test_generate_knowledge_data.py::TestDomainWeights -x` | No -- Wave 0 |
| D-07 | 200+ unique topics produce high diversity (dedup retention >50%) | integration | `pytest tests/test_generate_knowledge_data.py::TestDiversity -x` | No -- Wave 0 |
| D-09 | Curation retention rate exceeds 50% on test batch | integration | Manual: generate batch + run pipeline | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `python3 -m pytest tests/test_generate_knowledge_data.py -x -q`
- **Per wave merge:** `python3 -m pytest -x -q`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_generate_knowledge_data.py` -- covers KNOW-01, KNOW-02, KNOW-03, D-02, D-05, D-07
- [ ] No new conftest.py needed (existing test infrastructure sufficient)
- [ ] No framework install needed (pytest already configured)

## Security Domain

Not applicable for this phase. Phase 6 generates static training data with no external service calls, no user input processing, no authentication, and no network I/O. All code is deterministic batch generation using Python stdlib + existing project dependencies.

The only security-relevant pattern is the existing threat mitigation for YAML loading (T-02-04: yaml.safe_load only), which is already enforced in the codebase and does not need new code.

## Sources

### Primary (HIGH confidence)
- `scripts/generate_code_data.py` -- Phase 5 generation script pattern (verified in codebase)
- `scripts/generate_tool_data.py` -- Phase 4 generation script pattern (verified in codebase)
- `scripts/curate_pipeline.py` -- Curation pipeline implementation (verified in codebase)
- `scripts/dedup.py` -- N-gram Jaccard dedup with configurable scope/threshold (verified in codebase)
- `scripts/style_validator.py` -- Knowledge domain style checks: min_tokens=200, require_reasoning_markers=true (verified in codebase)
- `scripts/quality_scorer.py` -- 4-signal quality scoring (verified in codebase)
- `configs/pipeline.yaml` -- Knowledge domain config: min_response_chars=100, response-scope dedup at 0.7 (verified in codebase)
- `templates/knowledge.yaml` -- Category definitions with topics and generation_notes (verified in codebase)
- `templates/system-prompts.yaml` -- knowledge_assistant and knowledge_reasoning system prompts (verified in codebase)
- `specs/sharegpt-format.md` -- TRL-native format specification (verified in codebase)
- `.planning/phases/05-code-generation-dataset/05-03-SUMMARY.md` -- Phase 5 dedup lessons: 17.6% retention, template diversity root cause (verified in codebase)

### Secondary (MEDIUM confidence)
- `.planning/phases/06-general-knowledge-dataset/06-CONTEXT.md` -- User decisions D-01 through D-11 (verified in codebase)
- `.planning/REQUIREMENTS.md` -- KNOW-01, KNOW-02, KNOW-03 requirement definitions (verified in codebase)

### Tertiary (LOW confidence)
None -- all findings verified against codebase sources.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in use, no new dependencies
- Architecture: HIGH -- follows proven Phase 4-5 pattern exactly, with modifications for diversity
- Pitfalls: HIGH -- critical dedup lesson from Phase 5 is well-documented with specific numbers
- Topic pool design: MEDIUM -- the 200+ unique pairs assumption (A2) needs validation during generation

**Research date:** 2026-04-20
**Valid until:** 2026-05-20 (stable -- no external dependencies to change)
