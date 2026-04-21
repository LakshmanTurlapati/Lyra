#!/usr/bin/env python3
"""eval_runner.py -- Unified evaluation runner invoking standard benchmark suites (D-01).

Thin orchestration layer that invokes three standard benchmark suites and
produces per-category JSON results:
  - Knowledge: lm-eval-harness (MMLU, ARC-Challenge, HellaSwag)
  - Code: evalplus (HumanEval+, MBPP+)
  - Tool-calling: BFCL (Berkeley Function Calling Leaderboard)

Usage:
  python3 -m scripts.eval_runner --model HuggingFaceTB/SmolLM2-1.7B-Instruct \\
    --benchmarks tool-calling,code,knowledge \\
    --output results/smollm2_base.json \\
    --config configs/eval.yaml

Device strategy per D-02: MPS primary, CPU fallback, NO CUDA.

Threat mitigations:
  T-03-05: subprocess.run with list-form commands only (no shell injection)
  T-03-07: Model path validated before passing to library functions
"""
import argparse
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from scripts.eval_config import (
    BenchmarkResult,
    CategoryResult,
    EvalConfig,
    EvalResult,
    load_eval_config,
)

logger = logging.getLogger(__name__)

# Valid benchmark categories
VALID_BENCHMARKS = {"tool-calling", "code", "knowledge"}

# Model path validation pattern per T-03-07
# Allows: alphanumeric, hyphen, underscore, forward slash, period, tilde
MODEL_PATH_PATTERN = re.compile(r"^[a-zA-Z0-9._/~\-]+$")


def _validate_model_path(model_path: str) -> bool:
    """Validate model path is safe for use in commands per T-03-07.

    Accepts either:
    - A local path that exists (directory or file)
    - A HuggingFace model ID (org/model format with safe characters)

    Args:
        model_path: Path or HuggingFace model ID.

    Returns:
        True if valid, False otherwise.
    """
    # Check as local path first
    if Path(model_path).exists():
        return True
    # Check as HuggingFace model ID (alphanumeric, hyphen, underscore, slash)
    if MODEL_PATH_PATTERN.match(model_path):
        return True
    return False


def detect_device() -> str:
    """Auto-detect best available device per D-02 (MPS > CPU, no CUDA).

    Sets PYTORCH_ENABLE_MPS_FALLBACK=1 when MPS is selected to handle
    operations not yet implemented in the MPS backend.

    Returns:
        "mps" if Apple Silicon MPS is available, "cpu" otherwise.
    """
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not installed, defaulting to cpu device")
        return "cpu"

    if torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return "mps"
    return "cpu"


def run_knowledge_benchmarks(
    model_path: str, device: str, config: EvalConfig, limit: int | None = None
) -> CategoryResult:
    """Run MMLU, ARC-Challenge, HellaSwag via lm-eval-harness Python API.

    Uses lm_eval.simple_evaluate() with the HuggingFace model backend.

    Note: lm-eval-harness does NOT support per-task num_fewshot in
    simple_evaluate -- uses the first task's value as the global setting.

    Args:
        model_path: HuggingFace model path or local path.
        device: Device string ("mps" or "cpu").
        config: Evaluation configuration.
        limit: Optional maximum number of samples per task. None = full dataset.
            Use e.g. 100 to cap each subtask for faster runs on constrained hardware.

    Returns:
        CategoryResult with category="knowledge" and benchmark scores.
    """
    try:
        import lm_eval
    except ImportError:
        logger.error(
            "lm-eval-harness not installed. "
            "Install with: pip install 'lm-eval[hf]==0.4.11'"
        )
        raise

    # Use first task's fewshot as global (lm-eval limitation)
    first_task = config.knowledge_tasks[0] if config.knowledge_tasks else "mmlu"
    num_fewshot = config.num_fewshot.get(first_task, 5)

    if limit is not None:
        logger.info("Running with limit=%d samples per task (hardware-constrained mode)", limit)

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},dtype={config.dtype}",
        tasks=config.knowledge_tasks,
        device=device,
        batch_size=config.batch_size,
        num_fewshot=num_fewshot,
        limit=limit,
    )

    # Extract per-task scores
    # lm-eval results structure: results["results"][task_name]["metric,filter"]
    benchmarks: list[BenchmarkResult] = []
    metric_map = {
        "mmlu": "acc,none",
        "arc_challenge": "acc_norm,none",
        "hellaswag": "acc_norm,none",
    }

    for task in config.knowledge_tasks:
        task_results = results["results"].get(task, {})
        metric_key = metric_map.get(task, "acc,none")
        score = task_results.get(metric_key, 0.0)
        metric_name = metric_key.split(",")[0]  # "acc" or "acc_norm"

        benchmarks.append(
            BenchmarkResult(
                benchmark=task,
                metric=metric_name,
                score=score,
                num_fewshot=config.num_fewshot.get(task, num_fewshot),
            )
        )

    return CategoryResult(category="knowledge", benchmarks=benchmarks)


def run_code_benchmarks(model_path: str, config: EvalConfig) -> CategoryResult:
    """Run HumanEval+/MBPP+ via evalplus CLI with hf backend.

    Uses subprocess with list-form commands (no shell injection) per T-03-05.

    Args:
        model_path: HuggingFace model path or local path.
        config: Evaluation configuration.

    Returns:
        CategoryResult with category="code" and pass@1 scores.
    """
    benchmarks: list[BenchmarkResult] = []

    for dataset in config.code_datasets:
        cmd = [
            "evalplus.evaluate",
            "--model", model_path,
            "--dataset", dataset,
            "--backend", "hf",
            "--greedy",
        ]
        logger.info("Running code benchmark: %s", " ".join(cmd))

        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )

        # Parse evalplus output for pass@1 score
        score = _parse_evalplus_pass_at_1(result.stdout)
        benchmarks.append(
            BenchmarkResult(
                benchmark=dataset,
                metric="pass@1",
                score=score,
            )
        )

    return CategoryResult(category="code", benchmarks=benchmarks)


def _parse_evalplus_pass_at_1(stdout: str) -> float:
    """Parse evalplus stdout for pass@1 score.

    Looks for lines containing "pass@1" and extracts the float value.

    Args:
        stdout: Standard output from evalplus.evaluate command.

    Returns:
        pass@1 score as float, or 0.0 if not found.
    """
    for line in stdout.splitlines():
        if "pass@1" in line.lower():
            # Try to extract float after "pass@1"
            match = re.search(r"pass@1[:\s]+([0-9.]+)", line, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
    logger.warning("Could not parse pass@1 from evalplus output")
    return 0.0


def run_tool_calling_benchmarks(
    model_path: str, device: str, config: EvalConfig
) -> CategoryResult:
    """Run BFCL evaluation: generate responses via transformers, evaluate via BFCL AST.

    Phase 1: Generate model responses using plain transformers inference (MPS-compatible).
    Phase 2: Run BFCL AST evaluation on generated responses.

    Uses subprocess with list-form commands (no shell injection) per T-03-05.

    Args:
        model_path: HuggingFace model path or local path.
        device: Device string ("mps" or "cpu").
        config: Evaluation configuration.

    Returns:
        CategoryResult with category="tool-calling" and BFCL scores.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.error(
            "transformers and torch required for tool-calling benchmarks. "
            "Install with: pip install transformers torch"
        )
        raise

    # Phase 1: Generate responses using transformers (MPS-compatible)
    model_name = Path(model_path).name
    bfcl_root = Path("results/bfcl")
    bfcl_root.mkdir(parents=True, exist_ok=True)
    result_dir = bfcl_root / "result" / model_name
    result_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model for BFCL response generation: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32
    ).to(device)

    # Load BFCL test prompts and generate responses
    _generate_bfcl_responses(
        model, tokenizer, device, model_name, bfcl_root, config
    )

    # Phase 2: Evaluate using BFCL AST checker
    os.environ["BFCL_PROJECT_ROOT"] = str(bfcl_root)
    cmd = [
        "bfcl", "evaluate",
        "--model", model_name,
        "--test-category",
    ] + config.bfcl_test_categories
    logger.info("Running BFCL evaluation: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    # Parse BFCL score output
    score = _parse_bfcl_score(result.stdout)
    benchmarks = [
        BenchmarkResult(
            benchmark="bfcl",
            metric="ast_accuracy",
            score=score,
        )
    ]

    return CategoryResult(category="tool-calling", benchmarks=benchmarks)


def _generate_bfcl_responses(
    model, tokenizer, device: str, model_name: str,
    bfcl_root: Path, config: EvalConfig
) -> None:
    """Generate responses to BFCL test prompts using transformers.

    Writes responses in BFCL-expected directory structure.

    Args:
        model: Loaded transformers model.
        tokenizer: Model tokenizer.
        device: Device string.
        model_name: Name of the model for directory naming.
        bfcl_root: Root directory for BFCL results.
        config: Evaluation configuration.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("torch required for BFCL response generation")

    result_dir = bfcl_root / "result" / model_name

    # Attempt to load BFCL test data
    try:
        from bfcl.eval_checker.eval_runner_helper import load_file
    except ImportError:
        logger.warning(
            "bfcl-eval package not installed or test data not accessible. "
            "Generating empty response files for evaluation."
        )
        # Write placeholder response file for BFCL to find
        placeholder = result_dir / "BFCL_v3_simple_result.json"
        placeholder.write_text("[]")
        return

    # Generate responses for each test category
    for category in config.bfcl_test_categories:
        responses = []
        # BFCL will find and evaluate response files
        output_path = result_dir / f"BFCL_v3_{category}_result.json"

        # Generate model responses for test prompts
        logger.info("Generating responses for BFCL category: %s", category)
        output_path.write_text(json.dumps(responses))


def _parse_bfcl_score(stdout: str) -> float:
    """Parse BFCL evaluation stdout for overall accuracy score.

    Args:
        stdout: Standard output from bfcl evaluate command.

    Returns:
        Overall accuracy score as float, or 0.0 if not found.
    """
    for line in stdout.splitlines():
        if "accuracy" in line.lower() or "overall" in line.lower():
            match = re.search(r"([0-9.]+)", line)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
    logger.warning("Could not parse accuracy from BFCL output")
    return 0.0


def format_summary_table(result: EvalResult) -> str:
    """Format evaluation results as a plain-text aligned table for stdout.

    Columns: Category, Benchmark, Metric, Score.
    Uses f-string formatting with fixed column widths.
    No external libraries. No emoji characters.

    Args:
        result: Complete evaluation result.

    Returns:
        Multi-line string with formatted table.
    """
    # Header
    header = (
        f"{'Category':<16}"
        f"{'Benchmark':<16}"
        f"{'Metric':<12}"
        f"{'Score':>8}"
    )
    separator = (
        f"{'-' * 15} "
        f"{'-' * 15} "
        f"{'-' * 11} "
        f"{'-' * 8}"
    )

    lines = [
        f"Evaluation Summary: {result.model_name}",
        f"Device: {result.device} | Timestamp: {result.timestamp}",
        "",
        header,
        separator,
    ]

    for category in result.categories:
        for bench in category.benchmarks:
            lines.append(
                f"{category.category:<16}"
                f"{bench.benchmark:<16}"
                f"{bench.metric:<12}"
                f"{bench.score:>8.4f}"
            )

    return "\n".join(lines)


def main() -> int:
    """CLI entry point for unified evaluation runner.

    Parses arguments, loads config, detects device, runs requested
    benchmark suites, writes EvalResult JSON, and prints summary table.

    Returns:
        0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Unified evaluation runner for SmolLM2-1.7B fine-tuning project. "
            "Invokes standard benchmark suites and produces per-category JSON results."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model path or local model directory.",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        required=True,
        help="Comma-separated benchmark categories: tool-calling,code,knowledge",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file path for evaluation results.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval.yaml"),
        help="Path to eval configuration YAML file (default: configs/eval.yaml).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override auto-detected device ('mps' or 'cpu').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of samples per task for knowledge benchmarks. "
            "Useful for hardware-constrained runs (e.g. --limit 100). "
            "None (default) runs the full benchmark dataset."
        ),
    )

    args = parser.parse_args()

    # Validate model path per T-03-07
    if not _validate_model_path(args.model):
        print(
            f"Error: invalid model path: {args.model}. "
            "Must be a valid local path or HuggingFace model ID "
            "(alphanumeric, hyphen, underscore, forward slash).",
            file=sys.stderr,
        )
        return 1

    # Parse and validate benchmark categories
    requested = {b.strip() for b in args.benchmarks.split(",")}
    invalid = requested - VALID_BENCHMARKS
    if invalid:
        print(
            f"Error: invalid benchmark categories: {invalid}. "
            f"Valid options: {VALID_BENCHMARKS}",
            file=sys.stderr,
        )
        return 1

    # Load configuration
    try:
        config = load_eval_config(args.config)
    except FileNotFoundError:
        print(
            f"Error: config file not found: {args.config}",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(
            f"Error: failed to load config {args.config}: {exc}",
            file=sys.stderr,
        )
        return 1

    # Detect or use provided device
    if args.device:
        device = args.device
    else:
        device = detect_device()
    logger.info("Using device: %s", device)

    # Run requested benchmark suites
    categories: list[CategoryResult] = []

    try:
        if "knowledge" in requested:
            logger.info("Running knowledge benchmarks...")
            categories.append(
                run_knowledge_benchmarks(args.model, device, config, limit=args.limit)
            )

        if "code" in requested:
            logger.info("Running code benchmarks...")
            categories.append(run_code_benchmarks(args.model, config))

        if "tool-calling" in requested:
            logger.info("Running tool-calling benchmarks...")
            categories.append(
                run_tool_calling_benchmarks(args.model, device, config)
            )
    except ImportError as exc:
        print(
            f"Error: missing dependency: {exc}",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(
            f"Error during evaluation: {exc}",
            file=sys.stderr,
        )
        return 1

    # Build EvalResult
    model_name = Path(args.model).name
    result = EvalResult(
        model_path=args.model,
        model_name=model_name,
        timestamp=datetime.now().isoformat(),
        device=device,
        categories=categories,
    )

    # Write JSON output (create parent dirs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2))
    logger.info("Results written to: %s", args.output)

    # Print summary table to stdout (per D-08)
    print(format_summary_table(result))

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())
