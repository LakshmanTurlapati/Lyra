#!/usr/bin/env python3
"""Unit tests for scripts/eval_compare.py -- comparison logic and table formatting."""
import pytest

from scripts.eval_compare import compare_results, format_compare_table
from scripts.eval_config import BenchmarkResult, CategoryResult, CompareResult, EvalResult


def _make_eval_result(model_name: str, scores: dict) -> EvalResult:
    """Factory function creating EvalResult with given benchmark scores.

    Args:
        model_name: Name for the model.
        scores: Dict mapping (category, benchmark, metric) -> score.
            Format: {("knowledge", "mmlu", "acc"): 0.25, ...}

    Returns:
        EvalResult with the specified benchmark scores.
    """
    # Group scores by category
    categories_map: dict[str, list[BenchmarkResult]] = {}
    for (category, benchmark, metric), score in scores.items():
        if category not in categories_map:
            categories_map[category] = []
        categories_map[category].append(
            BenchmarkResult(benchmark=benchmark, metric=metric, score=score)
        )

    categories = [
        CategoryResult(category=cat, benchmarks=benchmarks)
        for cat, benchmarks in categories_map.items()
    ]

    return EvalResult(
        model_path=f"models/{model_name}",
        model_name=model_name,
        timestamp="2026-04-20T12:00:00",
        device="cpu",
        categories=categories,
    )


def test_compare_computes_delta():
    """compare_results returns correct delta (candidate - baseline)."""
    baseline = _make_eval_result(
        "baseline",
        {("knowledge", "mmlu", "acc"): 0.25},
    )
    candidate = _make_eval_result(
        "candidate",
        {("knowledge", "mmlu", "acc"): 0.35},
    )

    results = compare_results(baseline, candidate)
    assert len(results) == 1
    assert results[0].delta == pytest.approx(0.10)
    assert results[0].baseline_score == pytest.approx(0.25)
    assert results[0].candidate_score == pytest.approx(0.35)


def test_compare_handles_missing_benchmark():
    """compare_results only matches benchmarks present in both results."""
    baseline = _make_eval_result(
        "baseline",
        {("knowledge", "mmlu", "acc"): 0.25},
    )
    candidate = _make_eval_result(
        "candidate",
        {
            ("knowledge", "mmlu", "acc"): 0.35,
            ("knowledge", "arc_challenge", "acc_norm"): 0.40,
        },
    )

    results = compare_results(baseline, candidate)
    # Only mmlu should be compared (present in both)
    assert len(results) == 1
    assert results[0].benchmark == "mmlu"


def test_compare_negative_delta():
    """compare_results returns negative delta when candidate is worse."""
    baseline = _make_eval_result(
        "baseline",
        {("code", "humaneval", "pass@1"): 0.20},
    )
    candidate = _make_eval_result(
        "candidate",
        {("code", "humaneval", "pass@1"): 0.15},
    )

    results = compare_results(baseline, candidate)
    assert len(results) == 1
    assert results[0].delta == pytest.approx(-0.05)


def test_compare_multiple_categories():
    """compare_results handles benchmarks across multiple categories."""
    baseline = _make_eval_result(
        "baseline",
        {
            ("knowledge", "mmlu", "acc"): 0.25,
            ("code", "humaneval", "pass@1"): 0.10,
        },
    )
    candidate = _make_eval_result(
        "candidate",
        {
            ("knowledge", "mmlu", "acc"): 0.30,
            ("code", "humaneval", "pass@1"): 0.18,
        },
    )

    results = compare_results(baseline, candidate)
    assert len(results) == 2
    deltas = {r.benchmark: r.delta for r in results}
    assert deltas["mmlu"] == pytest.approx(0.05)
    assert deltas["humaneval"] == pytest.approx(0.08)


def test_format_compare_table_header():
    """format_compare_table returns string with expected column headers."""
    results = [
        CompareResult(
            category="knowledge",
            benchmark="mmlu",
            metric="acc",
            baseline_score=0.25,
            candidate_score=0.35,
            delta=0.10,
        )
    ]
    output = format_compare_table(results)
    assert "Category" in output
    assert "Benchmark" in output
    assert "Delta" in output
    assert "Baseline" in output
    assert "Candidate" in output


def test_format_compare_table_alignment():
    """format_compare_table output lines have consistent formatting."""
    results = [
        CompareResult(
            category="knowledge",
            benchmark="mmlu",
            metric="acc",
            baseline_score=0.25,
            candidate_score=0.35,
            delta=0.10,
        ),
        CompareResult(
            category="code",
            benchmark="humaneval",
            metric="pass@1",
            baseline_score=0.10,
            candidate_score=0.15,
            delta=0.05,
        ),
    ]
    output = format_compare_table(results)
    lines = output.strip().splitlines()
    # Header + separator + 2 data rows = at least 4 lines
    assert len(lines) >= 4
    # All data lines should contain score values
    assert "0.25" in lines[2] or "0.2500" in lines[2]
    assert "0.10" in lines[3] or "0.1000" in lines[3]


def test_format_compare_table_empty():
    """format_compare_table handles empty results gracefully."""
    output = format_compare_table([])
    assert "No matching benchmarks" in output


def test_format_compare_table_positive_delta_prefix():
    """format_compare_table prefixes positive deltas with '+'."""
    results = [
        CompareResult(
            category="knowledge",
            benchmark="mmlu",
            metric="acc",
            baseline_score=0.25,
            candidate_score=0.35,
            delta=0.10,
        )
    ]
    output = format_compare_table(results)
    assert "+0.1000" in output


# ---------------------------------------------------------------------------
# Phase 9: Markdown + Mermaid output tests (EVAL-02 / D-04, D-05, D-06)
# ---------------------------------------------------------------------------

def test_write_benchmark_md(tmp_path):
    """write_benchmark_md writes BENCHMARK.md with summary table to given path."""
    from scripts.eval_compare import write_benchmark_md

    results = [
        CompareResult(
            category="knowledge",
            benchmark="mmlu",
            metric="acc",
            baseline_score=0.25,
            candidate_score=0.35,
            delta=0.10,
        ),
        CompareResult(
            category="custom",
            benchmark="tool-call-format",
            metric="pass@1",
            baseline_score=0.40,
            candidate_score=0.72,
            delta=0.32,
        ),
    ]

    output_path = tmp_path / "BENCHMARK.md"
    write_benchmark_md(results, output_path, "SmolLM2-1.7B-Instruct", "lyra-merged")

    assert output_path.exists()
    text = output_path.read_text()
    assert "# Benchmark Results" in text
    assert "SmolLM2-1.7B-Instruct" in text
    assert "lyra-merged" in text
    assert "mmlu" in text
    assert "tool-call-format" in text
    assert "+0.1000" in text or "+0.10" in text


def test_mermaid_chart_present(tmp_path):
    """write_benchmark_md includes a Mermaid xychart-beta block in BENCHMARK.md."""
    from scripts.eval_compare import write_benchmark_md

    results = [
        CompareResult(
            category="knowledge",
            benchmark="mmlu",
            metric="acc",
            baseline_score=0.25,
            candidate_score=0.35,
            delta=0.10,
        )
    ]

    output_path = tmp_path / "BENCHMARK.md"
    write_benchmark_md(results, output_path, "base", "lyra")
    text = output_path.read_text()
    assert "```mermaid" in text, "BENCHMARK.md does not contain a mermaid code block"
    assert "xychart-beta" in text, "BENCHMARK.md mermaid block does not use xychart-beta"
