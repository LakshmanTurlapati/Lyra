# SPDX-License-Identifier: MIT
"""D-06: Training-data distribution audit — locks three thresholds that any
future retrain's curated dataset MUST satisfy. Phase 09.2 Plan 04."""
from collections import Counter
import pytest


@pytest.fixture(scope="module")
def assembled_train():
    from datasets import load_from_disk
    ds = load_from_disk("datasets/assembled")
    return ds["train"]


@pytest.fixture(scope="module")
def tool_calling_train(assembled_train):
    return [s for s in assembled_train if s.get("domain") == "tool-calling"]


def _pct_end_in_tool_call(tool_calling_train):
    n = len(tool_calling_train)
    assert n > 0, "no tool-calling samples found in train split"
    ends = 0
    for s in tool_calling_train:
        last = s["messages"][-1]
        if last.get("role") == "assistant" and last.get("tool_calls"):
            ends += 1
    return ends / n


@pytest.mark.slow
def test_tool_calling_assistant_ending_floor(tool_calling_train):
    """Plan 07 rebalance target: pct_end_in_tool_call >= 0.05.

    Trains the model that a tool_call on its own can be a complete response
    (prevents the mode collapse documented in D-06 where the 09.1 training
    set had 0.00% of tool-calling samples ending in a tool_call).
    """
    pct = _pct_end_in_tool_call(tool_calling_train)
    assert pct >= 0.05, (
        f"pct_end_in_tool_call = {pct:.4f} below 5% floor. "
        f"Plan 07 Wave B produces 500 single-turn tool-call-ending samples; "
        f"if this fails, the rebalance was undone. See D-06 + 09.2-07-SUMMARY."
    )


@pytest.mark.slow
def test_tool_calling_assistant_ending_ceiling(tool_calling_train):
    """Guardrail against over-correction: pct_end_in_tool_call stays
    below 0.30 so text-ending summaries remain the majority pattern.
    """
    pct = _pct_end_in_tool_call(tool_calling_train)
    assert pct < 0.30, (
        f"pct_end_in_tool_call = {pct:.4f} exceeds 30% ceiling — retrain has "
        f"over-corrected toward tool-call-ending samples."
    )


@pytest.mark.slow
def test_top5_canned_suffix_coverage_is_bounded(tool_calling_train):
    """Plan 07 rebalance target: top-5 60-char ending prefix coverage <= 20%.

    Before Plan 07: 46% of tool-calling samples shared one of 5 canned
    summary templates. Plan 07 Wave C + downsample of canned-ending legacy
    samples drops this to ~17%. A regression past 20% means the canned
    suffix cluster is returning.
    """
    prefixes = Counter()
    for s in tool_calling_train:
        last = s["messages"][-1]
        if last.get("role") == "assistant" and last.get("content"):
            prefixes[last["content"][:60]] += 1
    n = len(tool_calling_train)
    top5_total = sum(c for _, c in prefixes.most_common(5))
    coverage = top5_total / n if n else 0.0
    assert coverage <= 0.20, (
        f"Top-5 canned-suffix coverage = {coverage:.4f} of {n} tool-calling "
        f"train samples exceeds 20% threshold. See D-06 + 07-SUFFIX-POOL.md. "
        f"Top 5 prefixes: {prefixes.most_common(5)}"
    )


@pytest.mark.slow
def test_domain_skew_bounds(assembled_train):
    """Plan 07 rebalance target: tool-calling domain ratio <= 80% of train.

    Before Plan 07: tool-calling was 90.22% of train (4.9% knowledge,
    4.9% code) — documented as the MMLU -7.5% catastrophic-forgetting
    driver. Plan 07 Wave D + downsample brings tool-calling to ~65%.
    """
    domains = Counter(s.get("domain", "unknown") for s in assembled_train)
    n = sum(domains.values())
    tc_fraction = domains.get("tool-calling", 0) / n if n else 0.0
    assert tc_fraction <= 0.80, (
        f"tool-calling domain = {tc_fraction:.4f} of {n} train samples, "
        f"exceeds 80% target. Rebalance code/knowledge supplementary data."
    )
