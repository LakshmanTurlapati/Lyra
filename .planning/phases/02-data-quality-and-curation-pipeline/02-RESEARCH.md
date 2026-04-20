# Phase 02: Data Quality and Curation Pipeline - Research

**Researched:** 2026-04-20
**Domain:** Data quality filtering, deduplication, and pipeline configuration for LLM training data
**Confidence:** HIGH

## Summary

This phase builds a curation pipeline that takes raw JSONL conversations (generated in Claude Code sessions) and produces filtered, scored, deduplicated output ready for training. The pipeline must implement: (1) automated heuristic quality scoring with pass/fail signals, (2) n-gram Jaccard deduplication without external ML dependencies, (3) YAML-based configuration for thresholds and reuse, and (4) adaptive output style validation ensuring domain-appropriate response styles.

The implementation is constrained to stdlib + pydantic + pyyaml -- no datasketch, sentence-transformers, or Anthropic SDK. All components live as standalone scripts in the flat `scripts/` directory, consistent with Phase 1 patterns. The existing `validate_format.py` Pydantic models are reused directly for format compliance checking.

**Primary recommendation:** Build a `scripts/curate_pipeline.py` entry point that orchestrates four stages (format validation, completeness/naturalness scoring, deduplication, style validation) driven by a single `configs/pipeline.yaml` file with per-domain overrides. Each stage is also importable as a module for testing and independent use.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Two-tier scoring system. Tier 1: automated Python heuristics (pass/fail). Tier 2: manual LLM-as-judge via Claude Code sessions for a calibration subset.
- **D-02:** No Anthropic API SDK introduced for scoring -- judging happens in Claude Code sessions, consistent with D-05 from Phase 1.
- **D-03:** Automated heuristic signals (all four apply to every sample):
  - Format compliance (reuse validate_format.py Pydantic models)
  - Response completeness (no truncation, closed code blocks, conclusions reached)
  - Content diversity (near-duplicate detection within batch)
  - Conversation naturalness (balanced turn lengths, no copy-paste artifacts, no unintended meta-commentary)
- **D-04:** N-gram overlap strategy (3-gram or 4-gram Jaccard similarity). No external dependencies (datasketch, sentence-transformers). Configurable similarity threshold.
- **D-05:** Deduplication scope is Claude's discretion.
- **D-06:** Single YAML config file (pipeline.yaml) with sections for quality thresholds, dedup settings, topic distribution targets, and prompt template paths.
- **D-07:** Per-domain overrides within the single file -- global defaults with optional domain-specific sections.
- **D-08:** Consistent with Phase 1's YAML choice for templates.
- **D-09:** Template-driven style enforcement. Style instructions baked into prompt templates. Code templates guide terse responses, knowledge templates guide detailed chain-of-thought.
- **D-10:** Specific style validation heuristics are Claude's discretion.

### Claude's Discretion
- Deduplication comparison scope (prompt-only, response-only, full conversation, or combination)
- Specific style validation heuristics (token ranges, structural markers, or hybrid approach)
- Default threshold values in pipeline.yaml
- N-gram size (3 vs 4) and similarity threshold default
- Quality score output format (numeric, categorical, or both)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DATA-03 | User can filter generated data through deduplication, format validation, and quality scoring pipeline | N-gram Jaccard dedup implementation, Pydantic format reuse, heuristic scoring functions -- all documented in Architecture Patterns and Code Examples |
| DATA-04 | User can configure and reuse the generation pipeline with custom prompt templates, topic distributions, and quality thresholds via config files | YAML config structure with per-domain overrides, Pydantic config validation -- documented in Architecture Patterns |
| DATA-05 | Training data includes adaptive output styles -- terse responses for code tasks, detailed chain-of-thought for reasoning tasks | Style validation heuristics (token count ranges, structural markers) -- documented in Architecture Patterns and Code Examples |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pydantic | 2.12.5 | Config validation + reuse of existing Conversation models | Already a project dependency; provides type-safe validation of pipeline config and conversation format |
| pyyaml | 6.0+ | YAML config file parsing | Already a project dependency; used for templates in Phase 1 |
| Python stdlib (collections, json, pathlib, re) | 3.10+ | N-gram generation, Jaccard computation, file I/O, regex patterns | Zero new dependencies; D-04 mandates no external dedup libraries |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | 7.0+ | Unit testing for all pipeline stages | Already installed; test infrastructure established in Phase 1 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom n-gram Jaccard | datasketch (MinHash LSH) | Much faster at scale (O(1) vs O(n^2)), but D-04 explicitly forbids external dedup dependencies. Custom implementation is fine for 5K samples. |
| Custom n-gram Jaccard | sentence-transformers (semantic dedup) | Catches semantic duplicates that n-gram misses, but requires GPU and large model download. Explicitly excluded by D-04. |
| Pydantic config validation | dataclasses | Pydantic already in the project and provides richer validation (ranges, defaults, custom validators). No reason to use a weaker option. |

**Installation:**
```bash
# No new packages needed. All dependencies are already in requirements.txt:
# pydantic==2.12.5, pyyaml>=6.0, pytest>=7.0
```

**Version verification:** All packages already pinned in project requirements.txt. No new additions required. [VERIFIED: project requirements.txt]

## Architecture Patterns

### Recommended Project Structure
```
scripts/
  curate_pipeline.py     # Main entry point: orchestrates all stages
  quality_scorer.py      # Tier 1 heuristic scoring (4 signals)
  dedup.py               # N-gram Jaccard deduplication
  style_validator.py     # Adaptive output style checks
  validate_format.py     # (existing) Pydantic format validation
  validate_tokenizer.py  # (existing) Token count validation
configs/
  pipeline.yaml          # Pipeline configuration with per-domain overrides
datasets/
  {domain}/
    raw/                 # Input: raw JSONL from generation
    curated/             # Output: filtered, scored, deduplicated JSONL
```

### Pattern 1: Pipeline Orchestrator with Stage Composition
**What:** A single script reads pipeline.yaml, runs each stage in sequence (validate -> score -> dedup -> style-check), writes curated output with per-sample quality metadata.
**When to use:** Every pipeline run.
**Example:**
```python
# scripts/curate_pipeline.py
"""Pipeline orchestrator -- reads config, runs stages, writes output."""
import argparse
import json
from pathlib import Path

import yaml

from scripts.validate_format import Conversation
from scripts.quality_scorer import score_sample
from scripts.dedup import deduplicate_batch
from scripts.style_validator import validate_style


def load_config(config_path: Path) -> dict:
    """Load pipeline.yaml and return parsed config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_pipeline(input_path: Path, output_path: Path, config: dict, domain: str):
    """Run full curation pipeline on a JSONL file."""
    # Merge global defaults with domain-specific overrides
    domain_config = {**config.get("defaults", {}), **config.get("domains", {}).get(domain, {})}

    samples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    # Stage 1: Format validation (reuse existing validator)
    valid_samples = []
    for sample in samples:
        try:
            Conversation.model_validate(sample)
            valid_samples.append(sample)
        except Exception:
            pass  # Log rejection

    # Stage 2: Quality scoring
    scored = []
    for sample in valid_samples:
        score = score_sample(sample, domain_config)
        sample["_quality"] = score
        if score["pass"]:
            scored.append(sample)

    # Stage 3: Deduplication
    deduped = deduplicate_batch(scored, domain_config)

    # Stage 4: Style validation
    final = []
    for sample in deduped:
        if validate_style(sample, domain, domain_config):
            final.append(sample)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in final:
            f.write(json.dumps(sample) + "\n")

    return {"input": len(samples), "output": len(final)}
```

### Pattern 2: Heuristic Quality Scoring with Pass/Fail + Numeric Score
**What:** Each sample gets scored on 4 signals. Each signal produces a numeric score (0.0-1.0) and a pass/fail boolean. The overall score is the minimum of all signals (fail-fast: any single failure fails the sample).
**When to use:** Stage 2 of every pipeline run.
**Example:**
```python
# scripts/quality_scorer.py
"""Tier 1 automated quality scoring with 4 heuristic signals."""

def score_completeness(sample: dict, config: dict) -> dict:
    """Check response completeness: no truncation, closed code blocks, conclusions."""
    messages = sample["messages"]
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    issues = []

    for msg in assistant_msgs:
        content = msg.get("content") or ""
        # Check unclosed code blocks
        if content.count("```") % 2 != 0:
            issues.append("unclosed_code_block")
        # Check truncation indicators
        if content.rstrip().endswith(("...", "…")):
            issues.append("possible_truncation")
        # Check minimum length for assistant content (not tool-call-only)
        if not msg.get("tool_calls") and len(content) < config.get("min_response_chars", 10):
            issues.append("response_too_short")

    score = 1.0 - (len(issues) * 0.5)
    return {"signal": "completeness", "score": max(0.0, score), "pass": len(issues) == 0, "issues": issues}


def score_naturalness(sample: dict, config: dict) -> dict:
    """Check conversation naturalness: balanced turns, no copy-paste artifacts."""
    messages = sample["messages"]
    issues = []

    # Check for meta-commentary (model talking about being a model)
    meta_patterns = ["as an ai", "i cannot", "i'm an ai", "as a language model"]
    for msg in messages:
        content = (msg.get("content") or "").lower()
        for pattern in meta_patterns:
            if pattern in content:
                issues.append(f"meta_commentary:{pattern}")
                break

    # Check turn length balance
    user_lengths = [len(m.get("content") or "") for m in messages if m["role"] == "user"]
    assistant_lengths = [len(m.get("content") or "") for m in messages if m["role"] == "assistant" and m.get("content")]

    if user_lengths and assistant_lengths:
        ratio = max(assistant_lengths) / max(1, min(user_lengths))
        if ratio > config.get("max_turn_ratio", 50):
            issues.append("extreme_turn_imbalance")

    score = 1.0 - (len(issues) * 0.3)
    return {"signal": "naturalness", "score": max(0.0, score), "pass": len(issues) == 0, "issues": issues}
```

### Pattern 3: N-gram Jaccard Deduplication (Zero Dependencies)
**What:** Extract character n-grams from the target text, compute Jaccard similarity between all pairs, remove items above threshold. O(n^2) but fine for 5K samples.
**When to use:** Stage 3 of pipeline.
**Example:**
```python
# scripts/dedup.py
"""N-gram Jaccard similarity deduplication -- no external dependencies."""
from collections import Counter


def extract_ngrams(text: str, n: int = 3) -> set[str]:
    """Extract character n-grams from text."""
    text = text.lower().strip()
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def get_dedup_text(sample: dict, scope: str = "response") -> str:
    """Extract text for deduplication comparison based on scope."""
    messages = sample.get("messages", [])
    if scope == "response":
        # Compare assistant responses only (most discriminative)
        return " ".join(
            m.get("content") or ""
            for m in messages
            if m["role"] == "assistant" and m.get("content")
        )
    elif scope == "prompt":
        return " ".join(m.get("content") or "" for m in messages if m["role"] == "user")
    else:  # "full"
        return " ".join(m.get("content") or "" for m in messages if m.get("content"))


def deduplicate_batch(samples: list[dict], config: dict) -> list[dict]:
    """Remove near-duplicates from a batch using n-gram Jaccard similarity."""
    n = config.get("ngram_size", 3)
    threshold = config.get("dedup_threshold", 0.7)
    scope = config.get("dedup_scope", "response")

    # Pre-compute n-gram sets
    ngram_sets = []
    for sample in samples:
        text = get_dedup_text(sample, scope)
        ngram_sets.append(extract_ngrams(text, n))

    # Mark duplicates (keep first occurrence)
    keep = [True] * len(samples)
    for i in range(len(samples)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(samples)):
            if not keep[j]:
                continue
            sim = jaccard_similarity(ngram_sets[i], ngram_sets[j])
            if sim >= threshold:
                keep[j] = False

    return [s for s, k in zip(samples, keep) if k]
```

### Pattern 4: YAML Config with Per-Domain Overrides
**What:** A single YAML file with global defaults and optional domain-specific sections that override defaults.
**When to use:** Every pipeline run reads this config.
**Example:**
```yaml
# configs/pipeline.yaml
version: 1

defaults:
  # Quality thresholds
  min_response_chars: 10
  max_turn_ratio: 50
  min_assistant_messages: 1

  # Deduplication
  ngram_size: 3
  dedup_threshold: 0.7
  dedup_scope: "response"  # response | prompt | full

  # Style validation
  style_validation: true

  # Output
  include_quality_scores: true
  output_suffix: "_curated"

domains:
  tool-calling:
    # Stricter format requirements for tool calling
    min_response_chars: 5  # Tool call responses can be short JSON
    max_turn_ratio: 100  # Tool responses can be much longer than user prompts
    style:
      max_tokens: 800
      require_tool_calls: true
      allow_terse: true

  code:
    # Code domain: terse responses expected
    min_response_chars: 20
    style:
      max_tokens: 600
      require_code_blocks: true
      max_prose_ratio: 0.4  # At most 40% prose, rest is code

  knowledge:
    # Knowledge domain: detailed chain-of-thought expected
    min_response_chars: 100
    style:
      min_tokens: 200
      require_reasoning_markers: true  # "Step", numbered lists, "because", etc.
      min_prose_ratio: 0.7  # At least 70% prose/explanation

# Topic distribution targets (for reporting, not filtering)
topic_distribution:
  tool-calling: 0.33
  code: 0.33
  knowledge: 0.34

# Prompt template paths (reference only -- generation uses these)
template_paths:
  tool-calling: "templates/tool-calling.yaml"
  code: "templates/code.yaml"
  knowledge: "templates/knowledge.yaml"
```

### Pattern 5: Style Validation with Domain-Specific Heuristics
**What:** Validates that generated samples exhibit the correct output style for their domain. Code should be terse (short, code-heavy). Knowledge should be detailed (long, prose-heavy with reasoning markers).
**When to use:** Stage 4 of pipeline, after deduplication.
**Example:**
```python
# scripts/style_validator.py
"""Adaptive output style validation per domain."""
import re


def count_tokens_approx(text: str) -> int:
    """Approximate token count (words * 1.3 for English text)."""
    return int(len(text.split()) * 1.3)


def get_code_ratio(text: str) -> float:
    """Calculate ratio of text inside code blocks vs total."""
    code_blocks = re.findall(r"```[\s\S]*?```", text)
    code_chars = sum(len(block) for block in code_blocks)
    return code_chars / max(1, len(text))


def has_reasoning_markers(text: str) -> bool:
    """Check for explicit reasoning/chain-of-thought markers."""
    markers = [
        r"step \d",
        r"\d+\.",
        r"because",
        r"therefore",
        r"first,",
        r"second,",
        r"finally,",
        r"let me",
        r"this means",
    ]
    text_lower = text.lower()
    matches = sum(1 for m in markers if re.search(m, text_lower))
    return matches >= 2


def validate_style(sample: dict, domain: str, config: dict) -> bool:
    """Validate that a sample matches expected style for its domain."""
    style_config = config.get("style", {})
    if not config.get("style_validation", True):
        return True

    # Get assistant response text
    messages = sample.get("messages", [])
    response_text = " ".join(
        m.get("content") or ""
        for m in messages
        if m["role"] == "assistant" and m.get("content")
    )

    if not response_text:
        return True  # Tool-call-only responses pass by default

    approx_tokens = count_tokens_approx(response_text)

    if domain == "code":
        max_tokens = style_config.get("max_tokens", 600)
        if approx_tokens > max_tokens:
            return False
        if style_config.get("require_code_blocks", False):
            if "```" not in response_text:
                return False
        max_prose = style_config.get("max_prose_ratio", 0.4)
        code_ratio = get_code_ratio(response_text)
        if code_ratio < (1 - max_prose):
            return False

    elif domain == "knowledge":
        min_tokens = style_config.get("min_tokens", 200)
        if approx_tokens < min_tokens:
            return False
        if style_config.get("require_reasoning_markers", False):
            if not has_reasoning_markers(response_text):
                return False

    elif domain == "tool-calling":
        max_tokens = style_config.get("max_tokens", 800)
        if approx_tokens > max_tokens:
            return False

    return True
```

### Anti-Patterns to Avoid
- **Embedding-based dedup for 5K samples:** Overkill, adds GPU dependency, and D-04 explicitly forbids it. N-gram Jaccard is sufficient at this scale.
- **Single monolithic scoring function:** Each heuristic signal should be independent and testable. Do not collapse all checks into one function.
- **Hardcoded thresholds in code:** All thresholds must come from pipeline.yaml. The code should have NO magic numbers that cannot be overridden via config.
- **Mutating input files:** The pipeline reads from raw/ and writes to curated/. Never modify input files in-place.
- **Filtering without logging:** Every rejected sample must have a logged reason. Without rejection logs, you cannot improve prompt templates.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Format validation | Custom JSON schema checks | Existing `validate_format.py` Pydantic models | Already built, tested, and correct. Reuse directly via import. |
| Token counting | Manual character-based estimation | `validate_tokenizer.py` for exact counts, word*1.3 for fast approximation | Exact tokenizer counts require model download; use approximation for style checks, exact for final validation. |
| YAML parsing | Custom config parser | PyYAML `safe_load` | Standard, already in requirements. |
| Config validation | Manual dict key checking | Pydantic model for config schema | Catches misconfigured pipeline.yaml early with clear error messages. |

**Key insight:** The project already has strong format validation from Phase 1. This phase ADDS quality scoring and deduplication on top of format validation -- it does not replace or rebuild what exists.

## Common Pitfalls

### Pitfall 1: O(n^2) Dedup at Scale
**What goes wrong:** N-gram Jaccard with all-pairs comparison is O(n^2). At 5K samples, that is 12.5M comparisons. At 25K, it is 312M.
**Why it happens:** The decision to avoid external dedup libraries (D-04) limits optimization options.
**How to avoid:** For Phase 2 (5K samples), O(n^2) is acceptable (seconds on modern hardware with set operations). Document the scaling limit. If the project scales beyond 10K, consider MinHash LSH as a future optimization (separate from this phase).
**Warning signs:** Pipeline taking more than 60 seconds on the dedup stage.

### Pitfall 2: Overly Strict Style Validation
**What goes wrong:** Style thresholds set too tight reject legitimate samples. Code samples with extended explanations get rejected; knowledge samples with code examples get rejected.
**Why it happens:** Domain boundaries are fuzzy. A code debugging sample might have detailed explanation (knowledge-like). A knowledge sample about programming might include code blocks.
**How to avoid:** Start with permissive defaults. Use style validation as a soft signal (score contribution) rather than hard rejection. Log borderline cases for manual review. The config file should allow per-domain override of every threshold.
**Warning signs:** Style validation rejecting more than 15% of format-valid, quality-scored samples.

### Pitfall 3: Meta-Commentary Detection False Positives
**What goes wrong:** Naturalness checks flag legitimate conversational patterns. "I cannot help with that" in a refusal training example gets flagged. "As a language model" in a meta-discussion about AI gets flagged.
**Why it happens:** Simple string matching does not understand context. Refusal examples are valid training data. Meta-discussion examples might be intentional.
**How to avoid:** Make meta-commentary patterns configurable in pipeline.yaml. Add an allowlist mechanism for intentional meta-commentary. Only flag patterns in assistant messages (not system or user). Consider frequency -- a single "I cannot" is fine; 5 instances suggests copy-paste.
**Warning signs:** High rejection rate on refusal/safety samples.

### Pitfall 4: Dedup Scope Mismatch
**What goes wrong:** Deduplication on prompts removes legitimately different conversations that start with similar questions. Deduplication on responses removes conversations where different prompts lead to similar answers.
**Why it happens:** No single dedup scope is universally correct. Identical prompts with different responses are fine (diversity). Identical responses to different prompts indicate mode collapse.
**How to avoid:** Recommend response-scope dedup as default (catches the more dangerous failure mode: response homogeneity). Document that prompt-scope dedup is an alternative for detecting prompt repetition. The config supports switching scope without code changes.
**Warning signs:** Dedup removing more than 20% of samples (threshold too low or scope too broad).

### Pitfall 5: Quality Score Inflation from Circular Evaluation
**What goes wrong:** All 4 heuristic signals pass on 95%+ of Opus-generated data because Opus naturally produces well-formatted, complete, balanced responses. The quality filter adds no value.
**Why it happens:** Pitfall 5 from PITFALLS.md -- Opus generates data, heuristics tuned to Opus output. The filter catches only extreme failures.
**How to avoid:** Accept that Tier 1 heuristics are a coarse filter -- they catch mechanical failures (truncation, unclosed blocks, extreme imbalance). Tier 2 (manual LLM-as-judge in Claude Code sessions) provides the deeper quality signal. Document expected pass rates in pipeline output so users know the filter is working as designed.
**Warning signs:** 99%+ pass rate with no rejections logged.

## Code Examples

Verified patterns from existing codebase and standard Python:

### Reusing Existing Format Validator
```python
# Source: scripts/validate_format.py (existing Phase 1 code)
from scripts.validate_format import Conversation

def validate_format(sample: dict) -> tuple[bool, str]:
    """Validate sample against Lyra format spec. Returns (valid, error_message)."""
    try:
        Conversation.model_validate(sample)
        return True, ""
    except Exception as e:
        return False, str(e)
```

### Pydantic Config Model for pipeline.yaml
```python
# Source: standard Pydantic v2 pattern [VERIFIED: pydantic docs]
from pydantic import BaseModel, Field
from typing import Optional


class StyleConfig(BaseModel):
    """Style validation thresholds for a specific domain."""
    max_tokens: Optional[int] = None
    min_tokens: Optional[int] = None
    require_code_blocks: bool = False
    require_reasoning_markers: bool = False
    max_prose_ratio: Optional[float] = None
    min_prose_ratio: Optional[float] = None
    require_tool_calls: bool = False
    allow_terse: bool = False


class DomainConfig(BaseModel):
    """Per-domain configuration overrides."""
    min_response_chars: int = 10
    max_turn_ratio: float = 50.0
    min_assistant_messages: int = 1
    style: StyleConfig = StyleConfig()


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration schema."""
    version: int = 1
    defaults: DomainConfig = DomainConfig()
    domains: dict[str, DomainConfig] = Field(default_factory=dict)
    ngram_size: int = Field(default=3, ge=2, le=5)
    dedup_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    dedup_scope: str = Field(default="response", pattern="^(response|prompt|full)$")
    style_validation: bool = True
    include_quality_scores: bool = True
    output_suffix: str = "_curated"

    def get_domain_config(self, domain: str) -> DomainConfig:
        """Merge global defaults with domain-specific overrides."""
        base = self.defaults.model_dump()
        override = self.domains.get(domain, DomainConfig()).model_dump(exclude_unset=True)
        # Deep merge style sub-config
        if "style" in override:
            base["style"] = {**base.get("style", {}), **override["style"]}
            del override["style"]
        base.update(override)
        return DomainConfig(**base)
```

### N-gram Extraction and Jaccard (stdlib only)
```python
# Source: standard algorithm implementation [VERIFIED: GeeksforGeeks, TheAlgorithms/Python]
def extract_ngrams(text: str, n: int = 3) -> set[str]:
    """Extract character n-grams from normalized text.

    Uses character n-grams (not word n-grams) for better detection of
    paraphrased content with similar character sequences.
    """
    # Normalize: lowercase, collapse whitespace
    text = " ".join(text.lower().split())
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard index: |A intersect B| / |A union B|."""
    if not set_a and not set_b:
        return 1.0  # Two empty sets are identical
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0
```

### Quality Score Output Format
```python
# Per-sample quality metadata appended to the JSONL record
{
    "messages": [...],  # Original conversation
    "tools": [...],      # If present
    "_quality": {
        "pass": True,
        "score": 0.85,  # Minimum of all signal scores
        "signals": {
            "format": {"score": 1.0, "pass": True, "issues": []},
            "completeness": {"score": 0.85, "pass": True, "issues": []},
            "naturalness": {"score": 1.0, "pass": True, "issues": []},
            "diversity": {"score": 0.9, "pass": True, "issues": []}
        }
    }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| MinHash LSH for all dedup | Exact hash + n-gram Jaccard for small datasets, MinHash for large | 2024-2025 | For datasets under 10K, exact computation is fast enough and avoids false negatives from LSH approximation |
| Single quality score | Multi-signal scoring with per-signal pass/fail | 2024-2025 (NeMo Curator, distilabel) | Allows targeted improvement: if completeness fails, fix truncation; if diversity fails, vary prompts |
| Post-hoc quality filtering | Config-driven pipelines with thresholds | 2025 | Reproducibility and iteration: change a YAML value, rerun, compare results |
| Semantic dedup only | Lexical + semantic hybrid | 2024-2025 | N-gram catches surface duplicates fast; semantic catches paraphrases. For small datasets with single source (Opus), lexical is usually sufficient. |

**Deprecated/outdated:**
- Classic ShareGPT from/value format: Lyra uses TRL-native messages/role/content (Phase 1 decision)
- datasketch for small datasets: Overkill under 10K samples; stdlib set operations are faster for exact Jaccard

## Assumptions Log

> List all claims tagged [ASSUMED] in this research.

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | N-gram size 3 is sufficient for detecting near-duplicates in conversational text | Architecture Patterns | Low -- config allows changing to 4; 3-grams are well-established for short text similarity |
| A2 | Jaccard threshold 0.7 is appropriate for near-duplicate detection | Architecture Patterns | Low -- config allows tuning; 0.7 is standard in literature but may need calibration on Opus output |
| A3 | Response-scope dedup is more effective than prompt-scope for catching mode collapse | Architecture Patterns | Medium -- depends on how diverse prompts are vs responses; may need empirical validation |
| A4 | O(n^2) all-pairs comparison completes in under 60 seconds for 5K samples | Common Pitfalls | Low -- Python set operations on 3-gram sets are fast; 12.5M comparisons with small sets is trivial |
| A5 | Word count * 1.3 approximates token count adequately for style thresholds | Code Examples | Low -- only used for soft style checks, not hard limits; exact tokenizer used for 2048 enforcement |
| A6 | Meta-commentary patterns ("as an ai", "i cannot") are reliable indicators of unnatural text | Code Examples | Medium -- may need tuning; refusal examples are valid training data |

**If this table is empty:** N/A -- assumptions documented above.

## Open Questions

1. **Optimal n-gram size for Opus-generated text**
   - What we know: 3-grams are standard for general text similarity; 4-grams reduce false positives but increase false negatives
   - What's unclear: Whether Opus's specific output style (longer sentences, consistent formatting) works better with 3 or 4
   - Recommendation: Default to 3, provide config option, calibrate on first batch of real data

2. **Dedup threshold calibration**
   - What we know: 0.7 Jaccard is standard in NLP dedup literature
   - What's unclear: What the actual duplicate rate in Opus-generated conversation data is
   - Recommendation: Run the dedup with threshold logging (report similarity distribution) so users can calibrate. Start at 0.7, adjust based on observed distribution.

3. **Style validation boundary cases**
   - What we know: Code should be terse, knowledge should be detailed
   - What's unclear: How to handle hybrid samples (debugging explanation with code = code or knowledge?)
   - Recommendation: Domain is determined by which directory the file lives in, not by content analysis. Templates already enforce style at generation time; validation confirms it.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.10+ | All scripts | Yes | 3.14.2 | -- |
| pydantic | Config validation, format validation | Yes | 2.12.5 (in requirements.txt) | -- |
| pyyaml | Config loading | Yes | 6.0+ (in requirements.txt) | -- |
| pytest | Testing | Yes | 7.0+ (in requirements.txt) | -- |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:** None.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 7.0+ |
| Config file | pytest.ini (exists, configured with testpaths=tests) |
| Quick run command | `pytest tests/ -x --timeout=30` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-03 | Pipeline filters raw JSONL through dedup + format validation + quality scoring | integration | `pytest tests/test_curate_pipeline.py -x` | No -- Wave 0 |
| DATA-03 | N-gram dedup removes near-duplicates above threshold | unit | `pytest tests/test_dedup.py -x` | No -- Wave 0 |
| DATA-03 | Quality scorer produces pass/fail with 4 signals | unit | `pytest tests/test_quality_scorer.py -x` | No -- Wave 0 |
| DATA-04 | Pipeline reads config from pipeline.yaml with per-domain overrides | unit | `pytest tests/test_pipeline_config.py -x` | No -- Wave 0 |
| DATA-04 | Config validates with Pydantic schema | unit | `pytest tests/test_pipeline_config.py::test_config_validation -x` | No -- Wave 0 |
| DATA-05 | Code domain samples validated as terse (short, code-heavy) | unit | `pytest tests/test_style_validator.py::test_code_style -x` | No -- Wave 0 |
| DATA-05 | Knowledge domain samples validated as detailed (long, reasoning markers) | unit | `pytest tests/test_style_validator.py::test_knowledge_style -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/ -x --timeout=30`
- **Per wave merge:** `pytest tests/ -v`
- **Phase gate:** Full suite green before /gsd-verify-work

### Wave 0 Gaps
- [ ] `tests/test_dedup.py` -- covers DATA-03 deduplication logic
- [ ] `tests/test_quality_scorer.py` -- covers DATA-03 quality scoring signals
- [ ] `tests/test_style_validator.py` -- covers DATA-05 adaptive style validation
- [ ] `tests/test_pipeline_config.py` -- covers DATA-04 YAML config loading and validation
- [ ] `tests/test_curate_pipeline.py` -- covers DATA-03 end-to-end pipeline integration

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | N/A -- local CLI pipeline, no auth |
| V3 Session Management | No | N/A -- stateless pipeline |
| V4 Access Control | No | N/A -- single-user local tool |
| V5 Input Validation | Yes | Pydantic validation of pipeline.yaml config; JSONL input validated via existing Conversation model |
| V6 Cryptography | No | N/A -- no secrets or encryption needed |

### Known Threat Patterns for This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malicious YAML (yaml.load vulnerability) | Tampering | Use `yaml.safe_load()` exclusively -- never `yaml.load()` |
| Path traversal in config file paths | Tampering | Validate that configured paths resolve within project root |
| Large input causing OOM | Denial of Service | Document maximum expected input size; pipeline processes line-by-line for format validation |

## Sources

### Primary (HIGH confidence)
- Project codebase: `scripts/validate_format.py`, `scripts/validate_tokenizer.py`, `scripts/generate_sample.py` -- existing format validation, token validation, and sample generation code
- Project specs: `specs/sharegpt-format.md` -- canonical format specification
- Project research: `.planning/research/PITFALLS.md` -- pitfalls 1 (homogeneity) and 5 (circular evaluation) directly relevant
- Project research: `.planning/research/ARCHITECTURE.md` -- curation layer architecture and data flow patterns
- Project templates: `templates/*.yaml` -- existing YAML template patterns

### Secondary (MEDIUM confidence)
- [NeMo Curator Heuristic Filtering documentation](https://docs.nvidia.com/nemo/curator/25.09/curate-text/process-data/quality-assessment/heuristic.html) -- 30+ heuristic filter patterns, RepeatingTopNGramsFilter, PunctuationFilter
- [Synthetic Data Generation survey (arXiv 2503.14023)](https://arxiv.org/html/2503.14023v1) -- curation pipeline patterns for synthetic data
- [GeeksforGeeks: Jaccard Similarity](https://www.geeksforgeeks.org/python/jaccard-similarity/) -- standard Jaccard implementation reference
- [TheAlgorithms/Python: Jaccard Similarity](https://github.com/TheAlgorithms/Python/blob/master/maths/jaccard_similarity.py) -- reference implementation

### Tertiary (LOW confidence)
- [Document Deduplication with LSH](https://mattilyra.github.io/2017/05/23/document-deduplication-with-lsh.html) -- LSH scaling context (not implemented here but informs scaling limits)
- [Data Quality Is All You Need](https://www.shashankshekhar.com/blog/data-quality) -- general data quality philosophy

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies, all existing libraries reused
- Architecture: HIGH -- patterns follow established Phase 1 conventions and ARCHITECTURE.md recommendations
- Pitfalls: HIGH -- directly informed by project-specific PITFALLS.md research
- Dedup implementation: MEDIUM -- algorithm is standard but threshold calibration needs empirical data
- Style validation: MEDIUM -- heuristics are reasonable but may need tuning on real Opus output

**Research date:** 2026-04-20
**Valid until:** 2026-05-20 (stable domain; no fast-moving dependencies)
