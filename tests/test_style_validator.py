#!/usr/bin/env python3
"""Tests for scripts/style_validator.py -- domain-specific style validation."""
import pytest

from scripts.style_validator import (
    count_tokens_approx,
    get_code_ratio,
    has_reasoning_markers,
    validate_style,
)


# --- Helper: build a minimal conversation sample ---

def _make_sample(assistant_content: str, system: str = "You are a helpful assistant.") -> dict:
    """Build a minimal valid conversation with one user + one assistant message."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "Help me."},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def _make_tool_only_sample() -> dict:
    """Build a sample where assistant has tool_calls but no text content."""
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Get the weather."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "London"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": '{"temp": 15}', "name": "get_weather"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ],
    }


# --- Code domain tests ---

class TestCodeStyle:
    """Test code domain style validation (terse, code-heavy)."""

    def test_code_style_pass(self):
        """Code domain sample with short response containing code blocks passes."""
        # ~50 words -> ~65 tokens. Under 600. Code ratio high.
        code_response = (
            "Here is the function:\n\n"
            "```python\n"
            "def fibonacci(n):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n"
            "```\n"
        )
        sample = _make_sample(code_response)
        config = {
            "style_validation": True,
            "style": {
                "max_tokens": 600,
                "require_code_blocks": True,
                "max_prose_ratio": 0.4,
            },
        }
        assert validate_style(sample, "code", config) is True

    def test_code_style_fail_no_code_blocks(self):
        """Code domain sample with no code blocks fails when require_code_blocks=true."""
        sample = _make_sample("Just use the sorted() function with a key parameter.")
        config = {
            "style_validation": True,
            "style": {
                "max_tokens": 600,
                "require_code_blocks": True,
                "max_prose_ratio": 0.4,
            },
        }
        assert validate_style(sample, "code", config) is False

    def test_code_style_fail_too_verbose(self):
        """Code domain sample with 1000+ approx tokens fails when max_tokens=600."""
        # Generate a long response (~800 words -> ~1040 tokens)
        long_prose = " ".join(["word"] * 800)
        sample = _make_sample(long_prose)
        config = {
            "style_validation": True,
            "style": {
                "max_tokens": 600,
                "require_code_blocks": False,
                "max_prose_ratio": 1.0,
            },
        }
        assert validate_style(sample, "code", config) is False

    def test_code_style_fail_too_much_prose(self):
        """Code domain sample with <60% code ratio fails when max_prose_ratio=0.4."""
        # Lots of prose, tiny code block
        prose_heavy = (
            "Let me explain in detail how to approach this problem. "
            "You need to consider the edge cases carefully. "
            "First, handle the null case. Then handle the empty array. "
            "After that, iterate through each element and compare. "
            "Make sure to handle the boundary conditions properly. "
            "Here is a tiny example:\n\n"
            "```python\npass\n```\n"
            "That should work for most cases. "
            "Remember to test with edge cases. "
            "Consider performance implications for large inputs."
        )
        sample = _make_sample(prose_heavy)
        config = {
            "style_validation": True,
            "style": {
                "max_tokens": 2000,
                "require_code_blocks": False,
                "max_prose_ratio": 0.4,
            },
        }
        assert validate_style(sample, "code", config) is False


# --- Knowledge domain tests ---

class TestKnowledgeStyle:
    """Test knowledge domain style validation (detailed, reasoning-heavy)."""

    def test_knowledge_style_pass(self):
        """Knowledge domain sample with 300+ tokens and reasoning markers passes."""
        # ~250 words -> ~325 tokens. Has reasoning markers.
        detailed_response = (
            "Let me explain how TCP/IP works step by step.\n\n"
            "1. First, the application layer prepares the data for transmission. "
            "This means the data is formatted according to the protocol being used.\n\n"
            "2. The transport layer then segments the data and adds port numbers. "
            "TCP ensures reliable delivery because it uses acknowledgments.\n\n"
            "3. Therefore, the network layer adds IP addresses and routes the packet "
            "through intermediate routers toward the destination.\n\n"
            "4. Finally, the data link layer frames the packet for physical transmission "
            "over the network medium.\n\n"
            "This means that each layer adds its own header information, creating "
            "what we call encapsulation. The receiving host then strips these headers "
            "in reverse order, which is called decapsulation. "
            "Each layer only communicates with its peer layer on the other end, "
            "making the architecture modular and maintainable. "
            "The key advantage of this layered approach is that changes in one layer "
            "do not affect the others, as long as the interfaces remain consistent."
        )
        sample = _make_sample(detailed_response)
        config = {
            "style_validation": True,
            "style": {
                "min_tokens": 200,
                "require_reasoning_markers": True,
            },
        }
        assert validate_style(sample, "knowledge", config) is True

    def test_knowledge_style_fail_too_short(self):
        """Knowledge sample under min_tokens=200 fails."""
        short_response = "TCP/IP is a network protocol."
        sample = _make_sample(short_response)
        config = {
            "style_validation": True,
            "style": {
                "min_tokens": 200,
                "require_reasoning_markers": False,
            },
        }
        assert validate_style(sample, "knowledge", config) is False

    def test_knowledge_style_fail_no_reasoning(self):
        """Knowledge sample without reasoning markers fails when require_reasoning_markers=true."""
        # Long enough but no reasoning markers
        no_markers = (
            "TCP/IP is a set of communication protocols used to interconnect "
            "network devices on the internet. It stands for Transmission Control "
            "Protocol and Internet Protocol. TCP handles data segmentation and "
            "reassembly while IP handles addressing and routing. The protocol "
            "stack has four layers that handle different aspects of communication. "
            "Data flows down through the layers on the sending side and up "
            "through the layers on the receiving side. Each layer adds headers "
            "to the data as it passes through, and these headers are removed "
            "by the corresponding layer on the receiving end. The protocols "
            "are designed to be independent of the underlying hardware, "
            "which allows them to work across different types of networks."
        )
        sample = _make_sample(no_markers)
        config = {
            "style_validation": True,
            "style": {
                "min_tokens": 200,
                "require_reasoning_markers": True,
            },
        }
        assert validate_style(sample, "knowledge", config) is False


# --- Tool-calling domain tests ---

class TestToolCallingStyle:
    """Test tool-calling domain style validation."""

    def test_tool_calling_style_pass(self):
        """Tool-calling sample under max_tokens=800 passes."""
        sample = _make_sample("I will fetch the weather data for you.")
        config = {
            "style_validation": True,
            "style": {
                "max_tokens": 800,
            },
        }
        assert validate_style(sample, "tool-calling", config) is True


# --- Misc tests ---

class TestStyleValidationMisc:
    """Test edge cases and config-based behavior."""

    def test_style_validation_disabled(self):
        """style_validation=false causes all samples to pass regardless."""
        # This sample would normally fail code style (no code blocks)
        sample = _make_sample("Just plain text with no code.")
        config = {
            "style_validation": False,
            "style": {
                "require_code_blocks": True,
                "max_tokens": 5,
            },
        }
        assert validate_style(sample, "code", config) is True

    def test_tool_only_response_passes(self):
        """Sample with tool_calls but no text content passes style validation."""
        sample = _make_tool_only_sample()
        config = {
            "style_validation": True,
            "style": {
                "max_tokens": 10,
            },
        }
        assert validate_style(sample, "tool-calling", config) is True


# --- Utility function tests ---

class TestUtilityFunctions:
    """Test count_tokens_approx, get_code_ratio, has_reasoning_markers."""

    def test_count_tokens_approx(self):
        """Approximate token count is word_count * 1.3."""
        text = "one two three four five"
        # 5 words * 1.3 = 6.5, int = 6
        assert count_tokens_approx(text) == 6

    def test_count_tokens_approx_empty(self):
        assert count_tokens_approx("") == 0

    def test_get_code_ratio_all_code(self):
        text = "```python\nprint('hi')\n```"
        ratio = get_code_ratio(text)
        assert ratio > 0.9

    def test_get_code_ratio_no_code(self):
        text = "Just some plain text with no code blocks at all."
        ratio = get_code_ratio(text)
        assert ratio == 0.0

    def test_has_reasoning_markers_true(self):
        text = "Step 1: do this. Because it is important, therefore we proceed."
        assert has_reasoning_markers(text) is True

    def test_has_reasoning_markers_false(self):
        text = "Just a simple answer."
        assert has_reasoning_markers(text) is False
