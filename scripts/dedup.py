#!/usr/bin/env python3
"""dedup.py -- N-gram Jaccard similarity deduplication.

Implements near-duplicate detection per D-04 and D-05:
  - Character n-gram extraction with text normalization
  - Jaccard similarity computation
  - Configurable text scope (response, prompt, full)
  - Batch deduplication with O(n^2) all-pairs comparison

Uses only Python stdlib -- no datasketch, sentence-transformers, or external ML libraries.
At 5K samples the O(n^2) approach completes in seconds. Scaling limit documented.

All thresholds are configurable via config dict:
  - ngram_size (default 3): character n-gram size
  - dedup_threshold (default 0.7): Jaccard similarity threshold for duplicate marking
  - dedup_scope (default "response"): which message roles to compare
"""


def extract_ngrams(text: str, n: int = 3) -> set[str]:
    """Extract character n-grams from normalized text.

    Normalization: lowercase, collapse whitespace.
    If text length < n after normalization, returns {text} if non-empty, else empty set.

    Args:
        text: Input text to extract n-grams from.
        n: N-gram size (default 3).

    Returns:
        Set of character n-gram strings.
    """
    # Normalize: lowercase, collapse whitespace
    normalized = " ".join(text.lower().split())
    if not normalized:
        return set()
    if len(normalized) < n:
        return {normalized}
    return {normalized[i:i + n] for i in range(len(normalized) - n + 1)}


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity index between two sets.

    Jaccard(A, B) = |A intersect B| / |A union B|.
    Two empty sets are considered identical (returns 1.0).

    Args:
        set_a: First set.
        set_b: Second set.

    Returns:
        Float in [0.0, 1.0].
    """
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _serialize_tool_calls(msg: dict) -> str:
    """Serialize tool_calls from a message into a comparable text representation.

    Includes function names and arguments so that samples calling different
    tools or with different arguments are distinguished during dedup.

    Args:
        msg: Message dict that may contain a "tool_calls" field.

    Returns:
        String representation of tool calls, or empty string if none.
    """
    tool_calls = msg.get("tool_calls")
    if not tool_calls:
        return ""
    import json
    parts = []
    for tc in tool_calls:
        func = tc.get("function", {})
        parts.append(f"{func.get('name', '')}({json.dumps(func.get('arguments', {}), sort_keys=True)})")
    return " ".join(parts)


def get_dedup_text(sample: dict, scope: str = "response") -> str:
    """Extract text for deduplication comparison based on scope.

    For "response" scope, includes both assistant content AND serialized
    tool_calls data. This ensures tool-calling samples that differ in which
    tools they call (but have similar boilerplate text) are not falsely
    deduplicated.

    Args:
        sample: Conversation dict with "messages" key.
        scope: One of "response", "prompt", "full".
            - "response": join all assistant messages' content + tool_calls
            - "prompt": join all user messages' content
            - "full": join all messages' content + tool_calls (skip None)

    Returns:
        Concatenated text string for comparison.
    """
    messages = sample.get("messages", [])

    if scope == "response":
        parts = []
        for m in messages:
            if m.get("role") == "assistant":
                if m.get("content"):
                    parts.append(m["content"])
                tc_text = _serialize_tool_calls(m)
                if tc_text:
                    parts.append(tc_text)
        return " ".join(parts)
    elif scope == "prompt":
        return " ".join(
            m.get("content")
            for m in messages
            if m.get("role") == "user" and m.get("content")
        )
    else:  # "full"
        parts = []
        for m in messages:
            if m.get("content"):
                parts.append(m["content"])
            if m.get("role") == "assistant":
                tc_text = _serialize_tool_calls(m)
                if tc_text:
                    parts.append(tc_text)
        return " ".join(parts)


def deduplicate_batch(samples: list[dict], config: dict) -> list[dict]:
    """Remove near-duplicates from a batch using n-gram Jaccard similarity.

    O(n^2) all-pairs comparison. For each pair (i, j) where i < j,
    if jaccard_similarity >= threshold, mark j for removal (keep first occurrence).

    Scaling note: At 5K samples, this performs ~12.5M comparisons which completes
    in seconds with Python set operations. For >10K samples, consider MinHash LSH.

    Args:
        samples: List of conversation dicts.
        config: Dict with optional keys:
            - ngram_size (int, default 3)
            - dedup_threshold (float, default 0.7)
            - dedup_scope (str, default "response")

    Returns:
        Filtered list with near-duplicates removed, preserving original order.
    """
    if not samples:
        return []

    n = config.get("ngram_size", 3)
    threshold = config.get("dedup_threshold", 0.7)
    scope = config.get("dedup_scope", "response")

    # Pre-compute n-gram sets for all samples
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
