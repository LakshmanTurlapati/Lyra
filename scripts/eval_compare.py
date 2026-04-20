#!/usr/bin/env python3
"""eval_compare.py -- Compare two eval result JSON files and print delta table (D-10).

Reads a baseline and candidate EvalResult JSON file, matches benchmarks by
(category, benchmark) pair, computes score deltas, and prints a formatted
comparison table to stdout.

Usage:
  python3 -m scripts.eval_compare --baseline results/base.json \\
    --candidate results/finetuned.json [--output results/compare.json]

Threat mitigations:
  T-03-02: EvalResult.model_validate_json rejects malformed input
  T-03-03: Error messages print file path and validation error to stderr
"""
import argparse
import json
import sys
from pathlib import Path

from scripts.eval_config import CompareResult, EvalResult


def compare_results(
    baseline: EvalResult, candidate: EvalResult
) -> list[CompareResult]:
    """Compute deltas between baseline and candidate eval results.

    Matches benchmarks by (category, benchmark) pair.  Only benchmarks
    present in both results are compared.

    Args:
        baseline: Evaluation result for the baseline model.
        candidate: Evaluation result for the candidate model.

    Returns:
        List of CompareResult objects with delta = candidate - baseline.
    """
    # Build lookup: (category, benchmark) -> BenchmarkResult
    baseline_lookup: dict[tuple[str, str], tuple[str, float]] = {}
    for cat in baseline.categories:
        for bench in cat.benchmarks:
            baseline_lookup[(cat.category, bench.benchmark)] = (
                bench.metric,
                bench.score,
            )

    results: list[CompareResult] = []
    for cat in candidate.categories:
        for bench in cat.benchmarks:
            key = (cat.category, bench.benchmark)
            if key in baseline_lookup:
                b_metric, b_score = baseline_lookup[key]
                results.append(
                    CompareResult(
                        category=cat.category,
                        benchmark=bench.benchmark,
                        metric=bench.metric,
                        baseline_score=b_score,
                        candidate_score=bench.score,
                        delta=bench.score - b_score,
                    )
                )

    return results


def format_compare_table(results: list[CompareResult]) -> str:
    """Format comparison results as a plain-text aligned table.

    Columns: Category, Benchmark, Metric, Baseline, Candidate, Delta.
    Delta prefix: "+" for positive, "-" for negative, " " for zero.

    Args:
        results: List of CompareResult objects to display.

    Returns:
        Multi-line string with fixed-width column table.
    """
    if not results:
        return "No matching benchmarks found between baseline and candidate."

    # Column headers
    header = (
        f"{'Category':<16}"
        f"{'Benchmark':<16}"
        f"{'Metric':<10}"
        f"{'Baseline':>9}"
        f"{'Candidate':>11}"
        f"{'Delta':>8}"
    )
    separator = (
        f"{'-' * 15} "
        f"{'-' * 15} "
        f"{'-' * 9} "
        f"{'-' * 9} "
        f"{'-' * 10} "
        f"{'-' * 7}"
    )

    lines = [header, separator]
    for r in results:
        if r.delta > 0:
            delta_str = f"+{r.delta:.4f}"
        elif r.delta < 0:
            delta_str = f"{r.delta:.4f}"
        else:
            delta_str = f" {r.delta:.4f}"

        lines.append(
            f"{r.category:<16}"
            f"{r.benchmark:<16}"
            f"{r.metric:<10}"
            f"{r.baseline_score:>9.4f}"
            f"{r.candidate_score:>11.4f}"
            f"{delta_str:>8}"
        )

    return "\n".join(lines)


def main() -> int:
    """CLI entry point for eval comparison.

    Returns:
        0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description="Compare two eval result JSON files and print delta table."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to baseline model EvalResult JSON file.",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Path to candidate model EvalResult JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write CompareResult list as JSON.",
    )
    args = parser.parse_args()

    # Validate file existence
    if not args.baseline.exists():
        print(f"Error: baseline file not found: {args.baseline}", file=sys.stderr)
        return 1
    if not args.candidate.exists():
        print(f"Error: candidate file not found: {args.candidate}", file=sys.stderr)
        return 1

    # Load and validate JSON files
    try:
        baseline = EvalResult.model_validate_json(args.baseline.read_text())
    except Exception as exc:
        print(
            f"Error: failed to parse baseline file {args.baseline}: {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        candidate = EvalResult.model_validate_json(args.candidate.read_text())
    except Exception as exc:
        print(
            f"Error: failed to parse candidate file {args.candidate}: {exc}",
            file=sys.stderr,
        )
        return 1

    # Compare and display
    deltas = compare_results(baseline, candidate)
    print(format_compare_table(deltas))

    # Optionally write JSON output
    if args.output is not None:
        output_data = [r.model_dump() for r in deltas]
        args.output.write_text(json.dumps(output_data, indent=2))
        print(f"\nComparison written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
