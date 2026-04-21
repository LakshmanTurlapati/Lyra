#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""eval_merge.py -- Merge two EvalResult JSON files into a single combined result.

Used to combine:
  - results/{model}_knowledge.json  (from eval_runner.py --benchmarks knowledge)
  - results/{model}_custom.json     (from eval_inference.py)
  -> results/{model}_full.json      (input to eval_compare.py --markdown)

Uses EvalResult.model_validate_json at the trust boundary to reject malformed
input JSON before combining (T-09-04 mitigation).

Usage:
  python3 -m scripts.eval_merge \\
    --first results/lyra_knowledge.json \\
    --second results/lyra_custom.json \\
    --output results/lyra_full.json
"""
import argparse
import sys
from pathlib import Path

from scripts.eval_config import EvalResult


def merge_eval_results(
    first_path: Path, second_path: Path, output_path: Path
) -> EvalResult:
    """Merge two EvalResult JSON files by combining their categories lists.

    Uses model metadata (model_path, model_name, timestamp, device) from the
    first file. Appends all categories from the second file to the first.

    Input files are validated via EvalResult.model_validate_json at the trust
    boundary to reject malformed or tampered JSON (T-09-04).

    Args:
        first_path: Path to first EvalResult JSON (e.g. knowledge results).
        second_path: Path to second EvalResult JSON (e.g. custom eval results).
        output_path: Destination path for merged EvalResult JSON.

    Returns:
        The merged EvalResult (also written to output_path).

    Raises:
        FileNotFoundError: If either input file does not exist.
        pydantic.ValidationError: If either file contains invalid EvalResult JSON.
    """
    first = EvalResult.model_validate_json(first_path.read_text())
    second = EvalResult.model_validate_json(second_path.read_text())

    merged = EvalResult(
        model_path=first.model_path,
        model_name=first.model_name,
        timestamp=first.timestamp,
        device=first.device,
        categories=first.categories + second.categories,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(merged.model_dump_json(indent=2))
    return merged


def main() -> int:
    """CLI entry point for eval result merging."""
    parser = argparse.ArgumentParser(
        description="Merge two EvalResult JSON files into a single combined result."
    )
    parser.add_argument(
        "--first",
        type=Path,
        required=True,
        help="First EvalResult JSON file (knowledge results).",
    )
    parser.add_argument(
        "--second",
        type=Path,
        required=True,
        help="Second EvalResult JSON file (custom eval results).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for merged EvalResult JSON.",
    )
    args = parser.parse_args()

    if not args.first.exists():
        print(f"Error: file not found: {args.first}", file=sys.stderr)
        return 1
    if not args.second.exists():
        print(f"Error: file not found: {args.second}", file=sys.stderr)
        return 1

    try:
        merged = merge_eval_results(args.first, args.second, args.output)
    except Exception as exc:
        print(f"Error merging results: {exc}", file=sys.stderr)
        return 1

    n_categories = len(merged.categories)
    n_benchmarks = sum(len(c.benchmarks) for c in merged.categories)
    print(f"Merged {n_categories} categories ({n_benchmarks} benchmarks) -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
