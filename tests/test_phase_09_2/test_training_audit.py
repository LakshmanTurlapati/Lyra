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
@pytest.mark.xfail(
    strict=False,
    reason=(
        "Archival: documents the 09.1 dataset pathology where "
        "pct_end_in_tool_call == 0.0. Plan 06A retrain deliberately "
        "rebalances this; once that lands the xfail will XPASS->FAIL "
        "silently (strict=False) and the forward-looking ceiling test "
        "(below) is the active gate."
    ),
)
def test_tool_calling_assistant_ending_type_distribution_archival(tool_calling_train):
    """Per RESEARCH.md H1: on the 09.1 dataset, 100% of train tool-calling
    samples end in text, never in a tool_call. This xfail captures that
    state-of-the-world as an archival finding for Plan 05 to cite.
    """
    pct = _pct_end_in_tool_call(tool_calling_train)
    assert pct == 0.0, (
        f"Archival test: expected exactly 0.0 tool-call-ending fraction "
        f"(the 09.1 pathology), got {pct:.4f}."
    )


@pytest.mark.slow
def test_tool_calling_assistant_ending_type_ceiling_is_bounded(tool_calling_train):
    """Forward-looking ceiling. Passes BEFORE and AFTER Plan 06A's retrain.
    The intended remediation (adding a small fraction of tool-call-ending
    samples, target ~4%) must still satisfy pct_end_in_tool_call < 0.10.
    Only a pathological over-correction (>=10% tool-call-ending) fails.
    """
    pct = _pct_end_in_tool_call(tool_calling_train)
    assert pct < 0.10, (
        f"pct_end_in_tool_call = {pct:.1%} exceeds 10% forward-looking ceiling. "
        f"A retrain has over-corrected toward tool-call-ending samples. "
        f"Rebalance back toward text-ending majority; see RESEARCH.md H1 remediation."
    )


@pytest.mark.slow
def test_top5_canned_suffix_coverage_is_bounded(tool_calling_train):
    """Per RESEARCH.md H1: ~46% of training tool-calling samples share one
    of 5 ending prefixes (60-char). Retrains MUST keep this <= 50%.
    """
    prefixes = Counter()
    for s in tool_calling_train:
        last = s["messages"][-1]
        if last.get("role") == "assistant" and last.get("content"):
            prefixes[last["content"][:60]] += 1
    n = len(tool_calling_train)
    top5_total = sum(c for _, c in prefixes.most_common(5))
    coverage = top5_total / n if n else 0.0
    assert coverage <= 0.50, (
        f"Top-5 canned-suffix coverage = {coverage:.1%} of {n} tool-calling train "
        f"samples exceeds 50% threshold. Retrain must diversify endings "
        f"(see RESEARCH.md Pitfall 3 remediation). Top 5 prefixes: "
        f"{prefixes.most_common(5)}"
    )


@pytest.mark.slow
def test_domain_skew_bounds(assembled_train):
    """Per 09.1-04: tool-calling domain is 90.2% of train. Retrains MUST
    keep it <= 92% to avoid further skew (ideally rebalance DOWN per
    RESEARCH.md H1 remediation).
    """
    domains = Counter(s.get("domain", "unknown") for s in assembled_train)
    n = sum(domains.values())
    tc_fraction = domains.get("tool-calling", 0) / n if n else 0.0
    assert tc_fraction <= 0.92, (
        f"tool-calling domain = {tc_fraction:.1%} of {n} train samples, "
        f"exceeds 92% ceiling. Rebalancing required."
    )
