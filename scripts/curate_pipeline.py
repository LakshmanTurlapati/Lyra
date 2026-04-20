#!/usr/bin/env python3
"""curate_pipeline.py -- Pipeline orchestrator for Lyra dataset curation.

Reads pipeline.yaml configuration, runs 4 sequential stages, and writes
curated output with per-sample quality metadata:

  Stage 1 (Format):  Validate each JSONL record via Pydantic Conversation model
  Stage 2 (Quality): Score using 4 heuristic signals; reject failing samples
  Stage 3 (Dedup):   Remove near-duplicates via n-gram Jaccard similarity
  Stage 4 (Style):   Validate domain-specific output style (terse/detailed)

Usage:
  python -m scripts.curate_pipeline --input datasets/code/raw/samples.jsonl \\
    --domain code --config configs/pipeline.yaml

Threat mitigations applied:
  T-02-04: yaml.safe_load only (never yaml.load)
  T-02-05: Pydantic PipelineConfig validation before use
  T-02-07: Line-by-line JSONL parsing in Stage 1
  T-02-08: pathlib.Path for all file operations
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from scripts.dedup import deduplicate_batch
from scripts.pipeline_config import PipelineConfig, load_config
from scripts.quality_scorer import score_sample
from scripts.style_validator import validate_style
from scripts.validate_format import Conversation

logger = logging.getLogger(__name__)


def get_domain_config(config: PipelineConfig, domain: str) -> dict:
    """Merge global defaults with domain-specific overrides.

    Returns a plain dict suitable for passing to scorer, dedup, and
    style modules (which use ``config.get(key, default)``).

    Deep-merges the ``style`` sub-dict so domain-level style keys
    overlay defaults rather than replacing the entire block.

    Args:
        config: Validated PipelineConfig instance.
        domain: Domain name (e.g. "code", "knowledge", "tool-calling").

    Returns:
        Flat dict with merged configuration.
    """
    domain_cfg = config.get_domain_config(domain)
    result = domain_cfg.model_dump()

    # Propagate top-level pipeline settings that modules need
    result["ngram_size"] = config.ngram_size
    result["dedup_threshold"] = config.dedup_threshold
    result["dedup_scope"] = config.dedup_scope
    result["style_validation"] = config.style_validation
    result["include_quality_scores"] = config.include_quality_scores

    return result


def run_pipeline(
    input_path: Path,
    output_path: Path,
    config: PipelineConfig,
    domain: str,
) -> dict:
    """Execute the full 4-stage curation pipeline on a JSONL file.

    Args:
        input_path: Path to input JSONL file (one conversation per line).
        output_path: Path to write curated output JSONL.
        config: Validated PipelineConfig instance.
        domain: Domain name for per-domain thresholds.

    Returns:
        Stats dict with counts at each stage::

            {
                "input_count": int,
                "format_valid": int,
                "quality_pass": int,
                "after_dedup": int,
                "after_style": int,
                "output_count": int,
            }
    """
    domain_config = get_domain_config(config, domain)

    # ------------------------------------------------------------------
    # Stage 1: Format validation (line-by-line per T-02-07)
    # ------------------------------------------------------------------
    raw_samples: list[dict] = []
    valid_samples: list[dict] = []

    with open(input_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                sample = json.loads(stripped)
            except json.JSONDecodeError as e:
                logger.warning("Line %d: JSON parse error: %s", line_num, e)
                raw_samples.append({})  # count toward input
                continue

            raw_samples.append(sample)

            try:
                Conversation.model_validate(sample)
                valid_samples.append(sample)
            except Exception as e:
                logger.info("Line %d: format rejected: %s", line_num, e)

    input_count = len(raw_samples)
    format_valid = len(valid_samples)

    # ------------------------------------------------------------------
    # Stage 2: Quality scoring
    # ------------------------------------------------------------------
    scored_samples: list[dict] = []

    for sample in valid_samples:
        result = score_sample(sample, domain_config)
        sample["_quality"] = result
        if result["pass"]:
            scored_samples.append(sample)
        else:
            logger.info("Quality rejected: %s", result.get("signals", {}))

    quality_pass = len(scored_samples)

    # ------------------------------------------------------------------
    # Stage 3: Deduplication
    # ------------------------------------------------------------------
    deduped_samples = deduplicate_batch(scored_samples, domain_config)
    after_dedup = len(deduped_samples)

    # ------------------------------------------------------------------
    # Stage 4: Style validation
    # ------------------------------------------------------------------
    final_samples: list[dict] = []

    for sample in deduped_samples:
        if validate_style(sample, domain, domain_config):
            final_samples.append(sample)
        else:
            logger.info("Style rejected for domain '%s'", domain)

    after_style = len(final_samples)

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in final_samples:
            f.write(json.dumps(sample) + "\n")

    stats = {
        "input_count": input_count,
        "format_valid": format_valid,
        "quality_pass": quality_pass,
        "after_dedup": after_dedup,
        "after_style": after_style,
        "output_count": after_style,
    }

    return stats


def main() -> None:
    """CLI entry point for the curation pipeline."""
    parser = argparse.ArgumentParser(
        description="Lyra dataset curation pipeline -- filter, score, deduplicate, and validate JSONL samples",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output JSONL file (default: input path with _curated suffix)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pipeline.yaml"),
        help="Path to pipeline.yaml config (default: configs/pipeline.yaml)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=["tool-calling", "code", "knowledge"],
        help="Domain name for per-domain thresholds",
    )
    args = parser.parse_args()

    # Validate input file
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Validate config file
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Load and validate config (T-02-05: Pydantic validation)
    config = load_config(args.config)

    # Determine output path
    if args.output is None:
        stem = args.input.stem + config.output_suffix
        output_path = args.input.parent / f"{stem}{args.input.suffix}"
    else:
        output_path = args.output

    # Run pipeline
    stats = run_pipeline(args.input, output_path, config, args.domain)

    # Print summary
    print(f"Pipeline complete for domain '{args.domain}':")
    print(f"  Input samples:     {stats['input_count']}")
    print(f"  Format valid:      {stats['format_valid']}")
    print(f"  Quality pass:      {stats['quality_pass']}")
    print(f"  After dedup:       {stats['after_dedup']}")
    print(f"  After style:       {stats['after_style']}")
    print(f"  Output written:    {stats['output_count']}")
    print(f"  Output file:       {output_path}")


if __name__ == "__main__":
    main()
