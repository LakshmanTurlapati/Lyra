"""Tests for quality_scorer.py -- Tier 1 heuristic quality scoring.

Tests 4 independent signals: format, completeness, naturalness, diversity.
Each signal returns {"signal": str, "score": float, "pass": bool, "issues": list}.
"""
import pytest

from scripts.quality_scorer import (
    score_completeness,
    score_format,
    score_naturalness,
    score_sample,
)


# --- Format signal tests ---


def test_score_format_valid(valid_conversation):
    """Valid conversation returns pass=True with score 1.0."""
    result = score_format(valid_conversation)
    assert result["signal"] == "format"
    assert result["score"] == 1.0
    assert result["pass"] is True
    assert result["issues"] == []


def test_score_format_invalid():
    """Conversation missing system message returns pass=False."""
    sample = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }
    result = score_format(sample)
    assert result["signal"] == "format"
    assert result["score"] == 0.0
    assert result["pass"] is False
    assert len(result["issues"]) > 0


# --- Completeness signal tests ---


def test_score_completeness_pass(valid_conversation):
    """Complete conversation with no issues returns pass=True."""
    result = score_completeness(valid_conversation, {})
    assert result["signal"] == "completeness"
    assert result["pass"] is True
    assert result["issues"] == []


def test_score_completeness_unclosed_code_block():
    """Assistant message with odd number of ``` returns unclosed_code_block issue."""
    sample = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Write code."},
            {"role": "assistant", "content": "Here is code:\n```python\nprint('hello')"},
        ]
    }
    result = score_completeness(sample, {})
    assert result["pass"] is False
    assert "unclosed_code_block" in result["issues"]


def test_score_completeness_truncation():
    """Assistant message ending with '...' returns possible_truncation issue."""
    sample = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Explain something."},
            {"role": "assistant", "content": "This is a long explanation that ends with..."},
        ]
    }
    result = score_completeness(sample, {})
    assert result["pass"] is False
    assert "possible_truncation" in result["issues"]


def test_score_completeness_truncation_ellipsis_unicode():
    """Assistant message ending with unicode ellipsis returns possible_truncation."""
    sample = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Tell me more."},
            {"role": "assistant", "content": "This explanation trails off\u2026"},
        ]
    }
    result = score_completeness(sample, {})
    assert result["pass"] is False
    assert "possible_truncation" in result["issues"]


def test_score_completeness_short_response():
    """Response below min_response_chars returns response_too_short issue."""
    sample = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
    }
    result = score_completeness(sample, {"min_response_chars": 10})
    assert result["pass"] is False
    assert "response_too_short" in result["issues"]


def test_score_completeness_tool_call_no_content_skipped():
    """Assistant with tool_calls and no content should not trigger short response."""
    sample = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Get the weather."},
            {"role": "assistant", "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "NYC"}}}
            ]},
            {"role": "tool", "name": "get_weather", "content": '{"temp": 70}'},
            {"role": "assistant", "content": "The current temperature in New York City is 70 degrees Fahrenheit."},
        ],
        "tools": [{"type": "function", "function": {
            "name": "get_weather", "description": "Get weather", "parameters": {}
        }}],
    }
    # min_response_chars=50 should pass: tool-call message has no content (skipped),
    # final assistant message is 65 chars (above threshold)
    result = score_completeness(sample, {"min_response_chars": 50})
    assert result["pass"] is True


# --- Naturalness signal tests ---


def test_score_naturalness_pass(valid_conversation):
    """Normal conversation returns pass=True."""
    result = score_naturalness(valid_conversation, {})
    assert result["signal"] == "naturalness"
    assert result["pass"] is True
    assert result["issues"] == []


def test_score_naturalness_meta_commentary():
    """Assistant saying 'as an ai' returns pass=False with meta_commentary issue."""
    sample = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Can you help me?"},
            {"role": "assistant", "content": "As an AI, I would be happy to help you with that."},
        ]
    }
    result = score_naturalness(sample, {})
    assert result["pass"] is False
    assert any("meta_commentary" in issue for issue in result["issues"])


def test_score_naturalness_meta_commentary_case_insensitive():
    """Meta-commentary detection is case insensitive."""
    sample = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Help me."},
            {"role": "assistant", "content": "I'm A Language Model and I can help."},
        ]
    }
    result = score_naturalness(sample, {})
    assert result["pass"] is False


def test_score_naturalness_meta_in_user_ignored():
    """Meta-commentary patterns in user messages should NOT trigger failure."""
    sample = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "As an AI, what can you do?"},
            {"role": "assistant", "content": "I can help with coding, answering questions, and more."},
        ]
    }
    result = score_naturalness(sample, {})
    assert result["pass"] is True


def test_score_naturalness_extreme_imbalance():
    """Extreme turn length ratio returns pass=False."""
    sample = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "x" * 5000},
        ]
    }
    result = score_naturalness(sample, {"max_turn_ratio": 50})
    assert result["pass"] is False
    assert any("extreme_turn_imbalance" in issue for issue in result["issues"])


# --- score_sample integration tests ---


def test_score_sample_all_pass(valid_conversation):
    """Valid sample returns pass=True with 4 signal entries."""
    result = score_sample(valid_conversation, {})
    assert result["pass"] is True
    assert isinstance(result["score"], float)
    assert len(result["signals"]) == 4
    assert "format" in result["signals"]
    assert "completeness" in result["signals"]
    assert "naturalness" in result["signals"]
    assert "diversity" in result["signals"]


def test_score_sample_one_fails():
    """Sample with one failing signal returns pass=False."""
    sample = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Help."},
            {"role": "assistant", "content": "As an AI, I will help you with this task."},
        ]
    }
    result = score_sample(sample, {})
    assert result["pass"] is False
    assert result["signals"]["naturalness"]["pass"] is False


def test_all_thresholds_from_config():
    """Config dict with min_response_chars=50 causes short response to fail."""
    sample = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "A programming language."},
        ]
    }
    result = score_sample(sample, {"min_response_chars": 50})
    assert result["pass"] is False
    assert result["signals"]["completeness"]["pass"] is False
    assert "response_too_short" in result["signals"]["completeness"]["issues"]


def test_score_sample_diversity_always_passes():
    """Diversity signal at sample level always returns pass=True."""
    sample = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello."},
            {"role": "assistant", "content": "Hi there! How can I help?"},
        ]
    }
    result = score_sample(sample, {})
    assert result["signals"]["diversity"]["pass"] is True
    assert result["signals"]["diversity"]["score"] == 1.0
