#!/usr/bin/env python3
"""quality_scorer.py -- Tier 1 automated heuristic quality scoring.

Implements 4 independent quality signals per D-01 and D-03:
  1. Format compliance (reuses Pydantic Conversation model from validate_format.py)
  2. Response completeness (unclosed code blocks, truncation, minimum length)
  3. Conversation naturalness (meta-commentary patterns, turn length balance)
  4. Content diversity (placeholder at sample level; actual diversity handled by dedup.py at batch level)

Each signal returns:
  {"signal": str, "score": float, "pass": bool, "issues": list[str]}

score_sample() aggregates all 4 signals into a single pass/fail result.
All thresholds are configurable via config dict -- no hardcoded magic numbers.
"""
from scripts.validate_format import Conversation


# --- Meta-commentary patterns checked in assistant messages only ---
_META_PATTERNS = [
    "as an ai",
    "i cannot",
    "i'm an ai",
    "as a language model",
    "i'm a language model",
]


def score_format(sample: dict) -> dict:
    """Validate sample against Lyra ShareGPT format using Pydantic model.

    Args:
        sample: Conversation dict with "messages" key (and optional "tools").

    Returns:
        Signal dict with score 1.0 (valid) or 0.0 (invalid).
    """
    try:
        Conversation.model_validate(sample)
        return {"signal": "format", "score": 1.0, "pass": True, "issues": []}
    except Exception as e:
        return {"signal": "format", "score": 0.0, "pass": False, "issues": [str(e)]}


def score_completeness(sample: dict, config: dict) -> dict:
    """Check response completeness: unclosed code blocks, truncation, minimum length.

    Args:
        sample: Conversation dict.
        config: Dict with optional keys:
            - min_response_chars (int, default 10): minimum assistant response length.

    Returns:
        Signal dict. Score = max(0.0, 1.0 - len(issues) * 0.5).
    """
    messages = sample.get("messages", [])
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    issues = []

    min_chars = config.get("min_response_chars", 10)

    for msg in assistant_msgs:
        content = msg.get("content") or ""
        has_tool_calls = bool(msg.get("tool_calls"))

        # Check unclosed code blocks (odd count of ```)
        if content.count("```") % 2 != 0:
            issues.append("unclosed_code_block")

        # Check truncation indicators
        stripped = content.rstrip()
        if stripped.endswith("...") or stripped.endswith("\u2026"):
            issues.append("possible_truncation")

        # Check minimum response length -- skip messages that have tool_calls and no content
        if not has_tool_calls and len(content) < min_chars:
            issues.append("response_too_short")

    score = max(0.0, 1.0 - len(issues) * 0.5)
    return {
        "signal": "completeness",
        "score": score,
        "pass": len(issues) == 0,
        "issues": issues,
    }


def score_naturalness(sample: dict, config: dict) -> dict:
    """Check conversation naturalness: meta-commentary and turn length balance.

    Args:
        sample: Conversation dict.
        config: Dict with optional keys:
            - max_turn_ratio (float, default 50): max ratio of longest assistant
              message to shortest user message.

    Returns:
        Signal dict. Score = max(0.0, 1.0 - len(issues) * 0.3).
    """
    messages = sample.get("messages", [])
    issues = []

    max_turn_ratio = config.get("max_turn_ratio", 50)

    # Check meta-commentary in assistant messages only
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = (msg.get("content") or "").lower()
        for pattern in _META_PATTERNS:
            if pattern in content:
                issues.append(f"meta_commentary:{pattern}")
                break  # One match per message is enough

    # Check turn length balance
    user_lengths = [
        len(m.get("content") or "")
        for m in messages
        if m.get("role") == "user"
    ]
    assistant_lengths = [
        len(m.get("content") or "")
        for m in messages
        if m.get("role") == "assistant" and m.get("content")
    ]

    if user_lengths and assistant_lengths:
        ratio = max(assistant_lengths) / max(1, min(user_lengths))
        if ratio > max_turn_ratio:
            issues.append("extreme_turn_imbalance")

    score = max(0.0, 1.0 - len(issues) * 0.3)
    return {
        "signal": "naturalness",
        "score": score,
        "pass": len(issues) == 0,
        "issues": issues,
    }


def _score_diversity() -> dict:
    """Diversity placeholder at sample level.

    Actual diversity is handled by dedup.py at batch level per D-03
    ("Content diversity: near-duplicate detection within batch").
    At the individual sample level, diversity always passes.

    Returns:
        Signal dict with score 1.0 and pass True.
    """
    return {"signal": "diversity", "score": 1.0, "pass": True, "issues": []}


def score_sample(sample: dict, config: dict) -> dict:
    """Run all 4 quality signals and return aggregate result.

    Args:
        sample: Conversation dict.
        config: Dict with threshold overrides for individual signals.

    Returns:
        {
            "pass": bool (True only if ALL signals pass),
            "score": float (minimum of all signal scores),
            "signals": {
                "format": {...},
                "completeness": {...},
                "naturalness": {...},
                "diversity": {...},
            }
        }
    """
    signals = {
        "format": score_format(sample),
        "completeness": score_completeness(sample, config),
        "naturalness": score_naturalness(sample, config),
        "diversity": _score_diversity(),
    }

    all_pass = all(s["pass"] for s in signals.values())
    min_score = min(s["score"] for s in signals.values())

    return {
        "pass": all_pass,
        "score": min_score,
        "signals": signals,
    }
