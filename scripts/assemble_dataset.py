#!/usr/bin/env python3
"""assemble_dataset.py -- Merge curated domain JSONL into a HuggingFace DatasetDict.

Loads tool-calling, code, and knowledge curated JSONL files, strips internal
_quality metadata, adds domain labels, performs stratified 90/5/5 splits,
and saves as an Arrow-backed DatasetDict ready for fine-tuning.

Usage:
    python -m scripts.assemble_dataset assemble [--output-dir DIR] [--seed N] [--validate]
    python -m scripts.assemble_dataset stats [--dataset-dir DIR]
"""
import argparse
import json
import logging
import random
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)

# Hardcoded source paths relative to project root (per project convention)
DOMAIN_SOURCES = {
    "tool-calling": "datasets/tool-calling/curated/tool-calling-curated.jsonl",
    "code": "datasets/code/curated/code-curated.jsonl",
    "knowledge": "datasets/knowledge/curated/knowledge-curated.jsonl",
}

DEFAULT_OUTPUT_DIR = "datasets/assembled"
DEFAULT_SEED = 42


def load_domain_jsonl(path: Path, domain: str) -> list[dict]:
    """Load a JSONL file, strip _quality metadata, add domain label.

    Args:
        path: Path to the JSONL file.
        domain: Domain label to add (e.g., 'tool-calling', 'code', 'knowledge').

    Returns:
        List of sample dicts with domain added and _quality removed.
    """
    samples = []
    path = Path(path)

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(
                    "Skipping malformed JSON at %s:%d: %s", path, line_num, e
                )
                continue

            # Strip internal quality metadata (T-07-02)
            sample.pop("_quality", None)

            # Add domain label
            sample["domain"] = domain

            # Ensure tools key exists (None for non-tool-calling)
            sample.setdefault("tools", None)

            samples.append(sample)

    logger.info("Loaded %d samples from %s (domain=%s)", len(samples), path, domain)
    return samples


def assemble(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    seed: int = DEFAULT_SEED,
    base_dir: str = ".",
) -> DatasetDict:
    """Load all domains, merge, stratified split, and save.

    Args:
        output_dir: Directory to save the DatasetDict.
        seed: Random seed for reproducible splits.
        base_dir: Base directory for resolving relative DOMAIN_SOURCES paths.

    Returns:
        The assembled DatasetDict with train/validation/test splits.
    """
    base = Path(base_dir)
    all_samples = []

    for domain, rel_path in DOMAIN_SOURCES.items():
        full_path = base / rel_path
        if not full_path.exists():
            logger.error("Source file not found: %s", full_path)
            sys.exit(1)
        samples = load_domain_jsonl(full_path, domain)
        all_samples.extend(samples)

    logger.info("Total samples after merge: %d", len(all_samples))

    # Stratified split: equivalent to stratify_by_column="domain" but handles
    # string domain values without requiring ClassLabel cast. Groups by domain,
    # then splits each group 90/5/5 to preserve domain proportions.
    rng = random.Random(seed)

    # Group indices by domain
    domain_indices: dict[str, list[int]] = {}
    for idx, sample in enumerate(all_samples):
        domain = sample["domain"]
        domain_indices.setdefault(domain, []).append(idx)

    train_indices = []
    val_indices = []
    test_indices = []

    for domain, indices in sorted(domain_indices.items()):
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        # 10% held out, then split held-out 50/50 for val/test
        n_held = max(2, round(n * 0.1))  # at least 2 for val+test
        n_test = max(1, n_held // 2)
        n_val = n_held - n_test

        test_indices.extend(shuffled[:n_test])
        val_indices.extend(shuffled[n_test : n_test + n_val])
        train_indices.extend(shuffled[n_test + n_val :])

    # Build split datasets from indices
    train_samples = [all_samples[i] for i in train_indices]
    val_samples = [all_samples[i] for i in val_indices]
    test_samples = [all_samples[i] for i in test_indices]

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(train_samples, on_mixed_types="use_json"),
            "validation": Dataset.from_list(val_samples, on_mixed_types="use_json"),
            "test": Dataset.from_list(test_samples, on_mixed_types="use_json"),
        }
    )

    # Save to disk
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))
    logger.info("Saved DatasetDict to %s", output_path)

    return dataset_dict


def compute_stats(dataset_dict: DatasetDict) -> dict:
    """Compute per-split domain distribution statistics.

    Args:
        dataset_dict: The assembled DatasetDict.

    Returns:
        Nested dict with per-split total, domain counts, and percentages.
    """
    stats = {}

    for split_name in dataset_dict:
        split_data = dataset_dict[split_name]
        total = len(split_data)
        domain_counts = {}

        for domain_val in split_data["domain"]:
            domain_counts[domain_val] = domain_counts.get(domain_val, 0) + 1

        domains_stats = {}
        for domain, count in sorted(domain_counts.items()):
            pct = (count / total * 100) if total > 0 else 0.0
            domains_stats[domain] = {"count": count, "percent": round(pct, 1)}

        stats[split_name] = {"total": total, "domains": domains_stats}

    return stats


def print_stats(dataset_dict: DatasetDict) -> None:
    """Print formatted domain distribution table to stdout.

    Args:
        dataset_dict: The assembled DatasetDict.
    """
    stats = compute_stats(dataset_dict)

    print(f"\n{'Split':<12} {'Total':<8} {'Domain':<15} {'Count':<8} {'Percent':<8}")
    print("-" * 55)

    for split_name in ["train", "validation", "test"]:
        if split_name not in stats:
            continue
        split_stats = stats[split_name]
        first = True
        for domain, domain_data in sorted(split_stats["domains"].items()):
            if first:
                print(
                    f"{split_name:<12} {split_stats['total']:<8} "
                    f"{domain:<15} {domain_data['count']:<8} {domain_data['percent']:.1f}%"
                )
                first = False
            else:
                print(
                    f"{'':12} {'':8} "
                    f"{domain:<15} {domain_data['count']:<8} {domain_data['percent']:.1f}%"
                )
        print()

    # Summary
    total_all = sum(stats[s]["total"] for s in stats)
    print(f"Total samples: {total_all}")
    print(
        f"Split ratio: "
        f"train={stats.get('train', {}).get('total', 0)} "
        f"({stats.get('train', {}).get('total', 0) / total_all * 100:.1f}%) | "
        f"validation={stats.get('validation', {}).get('total', 0)} "
        f"({stats.get('validation', {}).get('total', 0) / total_all * 100:.1f}%) | "
        f"test={stats.get('test', {}).get('total', 0)} "
        f"({stats.get('test', {}).get('total', 0) / total_all * 100:.1f}%)"
    )


def validate_assembled(dataset_dict: DatasetDict) -> dict:
    """Validate all samples in the DatasetDict against the Conversation schema.

    Args:
        dataset_dict: The assembled DatasetDict to validate.

    Returns:
        Dict with total, valid, invalid counts and list of errors.
    """
    from scripts.validate_format import Conversation

    results = {"total": 0, "valid": 0, "invalid": 0, "errors": []}

    for split_name in dataset_dict:
        for i in range(len(dataset_dict[split_name])):
            row = dataset_dict[split_name][i]
            results["total"] += 1

            # Build the conversation dict for validation
            conv_dict = {"messages": row["messages"]}
            if row.get("tools") is not None:
                conv_dict["tools"] = row["tools"]

            try:
                Conversation.model_validate(conv_dict)
                results["valid"] += 1
            except Exception as e:
                results["invalid"] += 1
                results["errors"].append(
                    {"split": split_name, "index": i, "error": str(e)}
                )

    return results


def main():
    """CLI entry point with assemble and stats subcommands."""
    parser = argparse.ArgumentParser(
        description="Assemble Lyra curated datasets into HuggingFace DatasetDict"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # assemble subcommand
    assemble_parser = subparsers.add_parser(
        "assemble", help="Merge domains, split, and save dataset"
    )
    assemble_parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    assemble_parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducible splits (default: {DEFAULT_SEED})",
    )
    assemble_parser.add_argument(
        "--validate",
        action="store_true",
        help="Run Pydantic validation on assembled dataset",
    )
    assemble_parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Base directory for resolving source paths (default: .)",
    )

    # stats subcommand
    stats_parser = subparsers.add_parser(
        "stats", help="Print domain distribution statistics"
    )
    stats_parser.add_argument(
        "--dataset-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Path to saved DatasetDict (default: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.command == "assemble":
        dd = assemble(
            output_dir=args.output_dir,
            seed=args.seed,
            base_dir=args.base_dir,
        )
        print_stats(dd)

        if args.validate:
            logger.info("Running validation...")
            results = validate_assembled(dd)
            print(
                f"\nValidation: {results['valid']}/{results['total']} valid, "
                f"{results['invalid']} invalid"
            )
            if results["invalid"] > 0:
                for err in results["errors"][:10]:
                    logger.error(
                        "  %s[%d]: %s", err["split"], err["index"], err["error"]
                    )
                sys.exit(1)

    elif args.command == "stats":
        dataset_dir = Path(args.dataset_dir)
        if not dataset_dir.exists():
            logger.error("Dataset directory not found: %s", dataset_dir)
            sys.exit(1)
        dd = DatasetDict.load_from_disk(str(dataset_dir))
        print_stats(dd)


if __name__ == "__main__":
    main()
