#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""style_validator.py -- Adaptive output style validation per domain.

Implements per-domain style checks per D-09 and D-10:
  - Code: terse responses (short, code-heavy, minimal prose)
  - Knowledge: detailed responses (long, reasoning markers, prose-heavy)
  - Tool-calling: moderate length, tool-call presence expected

Each domain's thresholds are driven by the ``style`` section of the
domain config.  When ``style_validation`` is False the entire module
becomes a no-op (returns True for every sample).
"""
import re


# --- Reasoning marker patterns (checked in lowercase text) ---
_REASONING_PATTERNS: list[str] = [
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


def count_tokens_approx(text: str) -> int:
    """Approximate token count for English text.

    Uses word_count * 1.3 as a fast estimate.  Not used for hard limits --
    exact tokenizer counts are in validate_tokenizer.py.

    Args:
        text: Input text.

    Returns:
        Approximate token count (int).
    """
    return int(len(text.split()) * 1.3)


def get_code_ratio(text: str) -> float:
    """Calculate the ratio of text inside fenced code blocks to total text.

    Args:
        text: Full response text.

    Returns:
        Float in [0.0, 1.0].
    """
    code_blocks = re.findall(r"```[\s\S]*?```", text)
    code_chars = sum(len(block) for block in code_blocks)
    return code_chars / max(1, len(text))


def has_reasoning_markers(text: str) -> bool:
    """Check for explicit reasoning / chain-of-thought markers.

    Requires at least 2 distinct marker matches.

    Args:
        text: Full response text.

    Returns:
        True if >= 2 reasoning markers found.
    """
    text_lower = text.lower()
    matches = sum(1 for pattern in _REASONING_PATTERNS if re.search(pattern, text_lower))
    return matches >= 2


def validate_style(sample: dict, domain: str, config: dict) -> bool:
    """Validate that a sample matches the expected style for its domain.

    Args:
        sample: Conversation dict with ``messages`` key.
        domain: One of ``"code"``, ``"knowledge"``, ``"tool-calling"``.
        config: Domain-merged config dict.  Looks for:
            - ``style_validation`` (bool): master switch (default True).
            - ``style`` (dict): per-domain style thresholds.

    Returns:
        True if the sample passes style checks (or checks are disabled).
    """
    # Master switch -- skip all checks if disabled
    if not config.get("style_validation", True):
        return True

    style = config.get("style", {})

    # Extract assistant response text
    messages = sample.get("messages", [])
    response_text = " ".join(
        m.get("content") or ""
        for m in messages
        if m.get("role") == "assistant" and m.get("content")
    )

    # Tool-call-only responses (no text content) always pass
    if not response_text:
        return True

    approx_tokens = count_tokens_approx(response_text)

    # --- Code domain: terse, code-heavy ---
    if domain == "code":
        # Use ``or`` fallback because Pydantic model_dump() serializes
        # Optional[int]=None as {"max_tokens": None} -- dict.get() returns
        # None (key present) rather than the default.
        max_tokens = style.get("max_tokens") or 600
        if approx_tokens > max_tokens:
            return False
        if style.get("require_code_blocks", False) and "```" not in response_text:
            return False
        max_prose = style.get("max_prose_ratio") or 0.4
        code_ratio = get_code_ratio(response_text)
        if code_ratio < (1 - max_prose):
            return False

    # --- Knowledge domain: detailed, reasoning-heavy ---
    elif domain == "knowledge":
        min_tokens = style.get("min_tokens") or 200
        if approx_tokens < min_tokens:
            return False
        if style.get("require_reasoning_markers", False) and not has_reasoning_markers(response_text):
            return False

    # --- Tool-calling domain: moderate length ---
    elif domain == "tool-calling":
        max_tokens = style.get("max_tokens") or 800
        if approx_tokens > max_tokens:
            return False

    return True
