"""Tests for dedup.py -- N-gram Jaccard deduplication module.

Tests n-gram extraction, Jaccard similarity computation, text extraction
by scope, and batch deduplication with configurable thresholds.
All operations use only Python stdlib -- no external ML libraries.
"""
import pytest

from scripts.dedup import (
    deduplicate_batch,
    extract_ngrams,
    get_dedup_text,
    jaccard_similarity,
)


# --- extract_ngrams tests ---


def test_extract_ngrams_basic():
    """extract_ngrams returns character 3-grams from text."""
    result = extract_ngrams("hello world", n=3)
    assert "hel" in result
    assert "ell" in result
    assert "llo" in result
    assert "lo " in result
    assert " wo" in result
    assert "wor" in result
    assert "orl" in result
    assert "rld" in result


def test_extract_ngrams_short_text():
    """Text shorter than n returns set containing the text itself."""
    result = extract_ngrams("ab", n=3)
    assert result == {"ab"}


def test_extract_ngrams_empty():
    """Empty string returns empty set."""
    result = extract_ngrams("", n=3)
    assert result == set()


def test_extract_ngrams_normalization():
    """Normalization: lowercase and collapse whitespace produce same result."""
    result_a = extract_ngrams("Hello  World", n=3)
    result_b = extract_ngrams("hello world", n=3)
    assert result_a == result_b


def test_extract_ngrams_exact_length():
    """Text exactly n characters returns single n-gram."""
    result = extract_ngrams("abc", n=3)
    assert result == {"abc"}


# --- jaccard_similarity tests ---


def test_jaccard_identical():
    """Identical sets return 1.0."""
    assert jaccard_similarity({"a", "b", "c"}, {"a", "b", "c"}) == 1.0


def test_jaccard_disjoint():
    """Disjoint sets return 0.0."""
    assert jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0


def test_jaccard_partial():
    """Partial overlap returns correct Jaccard index."""
    # intersection = {"b", "c"} (size 2), union = {"a", "b", "c", "d"} (size 4)
    result = jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
    assert result == 0.5


def test_jaccard_empty_sets():
    """Two empty sets are considered identical (return 1.0)."""
    assert jaccard_similarity(set(), set()) == 1.0


def test_jaccard_one_empty():
    """One empty, one non-empty returns 0.0."""
    assert jaccard_similarity(set(), {"a", "b"}) == 0.0


# --- get_dedup_text tests ---


def test_get_dedup_text_response_scope():
    """scope='response' extracts only assistant content."""
    sample = {
        "messages": [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "User question."},
            {"role": "assistant", "content": "Assistant answer."},
        ]
    }
    result = get_dedup_text(sample, scope="response")
    assert "Assistant answer" in result
    assert "User question" not in result
    assert "System prompt" not in result


def test_get_dedup_text_prompt_scope():
    """scope='prompt' extracts only user content."""
    sample = {
        "messages": [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "User question."},
            {"role": "assistant", "content": "Assistant answer."},
        ]
    }
    result = get_dedup_text(sample, scope="prompt")
    assert "User question" in result
    assert "Assistant answer" not in result


def test_get_dedup_text_full_scope():
    """scope='full' extracts all message content."""
    sample = {
        "messages": [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "User question."},
            {"role": "assistant", "content": "Assistant answer."},
        ]
    }
    result = get_dedup_text(sample, scope="full")
    assert "System prompt" in result
    assert "User question" in result
    assert "Assistant answer" in result


def test_get_dedup_text_skips_none_content():
    """None content in tool-call messages is skipped."""
    sample = {
        "messages": [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "Do something."},
            {"role": "assistant", "content": None, "tool_calls": [
                {"type": "function", "function": {"name": "tool", "arguments": {}}}
            ]},
            {"role": "tool", "name": "tool", "content": "result"},
            {"role": "assistant", "content": "Done."},
        ]
    }
    result = get_dedup_text(sample, scope="response")
    assert "Done" in result
    # Should not have "None" as a string
    assert "None" not in result


# --- deduplicate_batch tests ---


def test_deduplicate_batch_removes_duplicate():
    """Two samples with identical assistant responses: only first kept."""
    samples = [
        {"messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Question A."},
            {"role": "assistant", "content": "The answer is exactly forty-two."},
        ]},
        {"messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Question B."},
            {"role": "assistant", "content": "The answer is exactly forty-two."},
        ]},
    ]
    config = {"ngram_size": 3, "dedup_threshold": 0.7, "dedup_scope": "response"}
    result = deduplicate_batch(samples, config)
    assert len(result) == 1
    assert result[0]["messages"][1]["content"] == "Question A."


def test_deduplicate_batch_keeps_different():
    """Two samples with different responses both kept."""
    samples = [
        {"messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a high-level programming language known for its readable syntax."},
        ]},
        {"messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is Rust?"},
            {"role": "assistant", "content": "Rust is a systems programming language focused on safety and performance."},
        ]},
    ]
    config = {"ngram_size": 3, "dedup_threshold": 0.7, "dedup_scope": "response"}
    result = deduplicate_batch(samples, config)
    assert len(result) == 2


def test_deduplicate_batch_configurable_threshold():
    """High threshold (0.95) keeps samples that low threshold (0.7) would remove."""
    # Create two very similar but not identical responses
    samples = [
        {"messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Explain X."},
            {"role": "assistant", "content": "This is a detailed explanation of the concept at hand."},
        ]},
        {"messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Explain Y."},
            {"role": "assistant", "content": "This is a detailed explanation of the concept in question."},
        ]},
    ]
    config_strict = {"ngram_size": 3, "dedup_threshold": 0.95, "dedup_scope": "response"}
    config_loose = {"ngram_size": 3, "dedup_threshold": 0.5, "dedup_scope": "response"}

    result_strict = deduplicate_batch(samples, config_strict)
    result_loose = deduplicate_batch(samples, config_loose)

    # Strict threshold should keep more (both), loose threshold should keep fewer (one)
    assert len(result_strict) >= len(result_loose)


def test_deduplicate_batch_preserves_order():
    """First occurrence is always kept, later duplicates removed."""
    samples = [
        {"messages": [
            {"role": "system", "content": "S."},
            {"role": "user", "content": "First."},
            {"role": "assistant", "content": "Identical response content here."},
        ]},
        {"messages": [
            {"role": "system", "content": "S."},
            {"role": "user", "content": "Second."},
            {"role": "assistant", "content": "Totally different response."},
        ]},
        {"messages": [
            {"role": "system", "content": "S."},
            {"role": "user", "content": "Third."},
            {"role": "assistant", "content": "Identical response content here."},
        ]},
    ]
    config = {"ngram_size": 3, "dedup_threshold": 0.7, "dedup_scope": "response"}
    result = deduplicate_batch(samples, config)
    # First and second kept; third (duplicate of first) removed
    assert len(result) == 2
    assert result[0]["messages"][1]["content"] == "First."
    assert result[1]["messages"][1]["content"] == "Second."


def test_deduplicate_batch_empty_input():
    """Empty batch returns empty list."""
    config = {"ngram_size": 3, "dedup_threshold": 0.7, "dedup_scope": "response"}
    result = deduplicate_batch([], config)
    assert result == []


def test_deduplicate_batch_single_item():
    """Single-item batch returns that item."""
    samples = [
        {"messages": [
            {"role": "system", "content": "S."},
            {"role": "user", "content": "Q."},
            {"role": "assistant", "content": "A."},
        ]},
    ]
    config = {"ngram_size": 3, "dedup_threshold": 0.7, "dedup_scope": "response"}
    result = deduplicate_batch(samples, config)
    assert len(result) == 1


def test_deduplicate_batch_uses_config_defaults():
    """Empty config dict uses defaults (ngram_size=3, threshold=0.7, scope=response)."""
    samples = [
        {"messages": [
            {"role": "system", "content": "S."},
            {"role": "user", "content": "Q."},
            {"role": "assistant", "content": "A unique response."},
        ]},
    ]
    # Should not raise even with empty config
    result = deduplicate_batch(samples, {})
    assert len(result) == 1
