#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""eval_inference.py -- Custom inference eval for tool-calling and code domains (D-02).

Runs model inference on the assembled test split and evaluates outputs using
format-checking functions:
  - check_tool_call_format: validates SmolLM2 <tool_call>JSON</tool_call> format
  - check_code_syntax: validates Python code blocks via ast.parse (no execution)

Device strategy per D-02: MPS primary (PYTORCH_ENABLE_MPS_FALLBACK=1 set by
detect_device), CPU fallback. No CUDA required.

Tool-call format (Phase 1 decision): <tool_call>{"name": "...", "arguments": {...}}</tool_call>

Usage:
  python3 -m scripts.eval_inference \\
    --model models/lyra-merged \\
    --dataset-dir datasets/assembled \\
    --output results/lyra_custom.json

Threat mitigations:
  T-09-05: ast.parse() only -- never exec() or eval(); parse-only, no execution
  T-09-06: re.DOTALL regex match on model output; untrusted text not executed
"""
import argparse
import ast
import gc
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

from scripts.eval_config import BenchmarkResult, CategoryResult, EvalResult, load_eval_config

logger = logging.getLogger(__name__)

# Module-level references for monkeypatching in tests.
# These are assigned lazily inside run_custom_eval() on first call.
load_from_disk = None
run_inference_on_sample = None
_load_model_and_tokenizer = None


def check_tool_call_format(output: str) -> bool:
    """Return True if output contains a valid SmolLM2-format tool call.

    Checks for the <tool_call>JSON</tool_call> XML wrapper (Phase 1 decision)
    and validates that the interior JSON contains both "name" and "arguments" keys.

    The model output is treated as untrusted text; it is never executed (T-09-06).

    Args:
        output: Raw model generation output string.

    Returns:
        True if output matches the SmolLM2 tool-call format, False otherwise.
    """
    match = re.search(r"<tool_call>(.*?)</tool_call>", output, re.DOTALL)
    if not match:
        return False
    try:
        obj = json.loads(match.group(1))
        return "name" in obj and "arguments" in obj
    except (json.JSONDecodeError, KeyError):
        return False


def check_code_syntax(output: str) -> bool:
    """Return True if output contains a syntactically valid Python code block.

    Extracts content from ```python or ```py fenced code blocks and validates
    via ast.parse(). This is a parse-only check -- no code is executed (T-09-05).

    Args:
        output: Raw model generation output string.

    Returns:
        True if a valid Python code block is found, False otherwise.
    """
    match = re.search(r"```(?:python|py)\n(.*?)```", output, re.DOTALL)
    if not match:
        return False
    try:
        ast.parse(match.group(1))
        return True
    except SyntaxError:
        return False


def _do_load_model_and_tokenizer(model_path: str, device: str):
    """Load model and tokenizer from a local or HuggingFace path.

    Heavy imports (torch, transformers) are lazy to support testing without
    those packages installed (Phase 8 pattern).

    Args:
        model_path: Local path or HuggingFace model ID.
        device: Target device string ("mps" or "cpu").

    Returns:
        Tuple of (model, tokenizer).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading model from %s onto %s", model_path, device)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def _run_inference_on_sample(
    model, tokenizer, sample: dict, device: str, max_new_tokens: int = 256
) -> str:
    """Run model inference on one test split sample.

    Builds prompt by excluding the last assistant turn (which is the label),
    applies the SmolLM2 chat template, and generates greedily for determinism
    (do_sample=False per Pitfall 4).

    Args:
        model: Loaded transformers model.
        tokenizer: Corresponding tokenizer.
        sample: Dataset sample dict with "messages" list and "domain" key.
        device: Device string ("mps" or "cpu").
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Decoded model output string (new tokens only).
    """
    import torch

    messages = sample["messages"]
    # Find last assistant turn index -- that is the label to predict
    last_asst_idx = max(
        i for i, m in enumerate(messages) if m["role"] == "assistant"
    )
    prompt_messages = messages[:last_asst_idx]  # everything before the label

    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding for determinism (Pitfall 4)
        )
    # Decode only the newly generated tokens (not the prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_custom_eval(model_path: str, device: str, dataset_dir: str) -> CategoryResult:
    """Run custom inference eval on the assembled test split.

    Iterates the test split (181 samples), skips knowledge-domain samples
    (covered by lm-eval-harness), and evaluates:
      - tool-calling samples via check_tool_call_format
      - code samples via check_code_syntax

    Frees MPS memory via del model + gc.collect() after inference loop
    to avoid OOM when running multiple models sequentially (Pitfall 2).

    Args:
        model_path: Local path or HuggingFace model ID.
        device: Target device string ("mps" or "cpu").
        dataset_dir: Path to assembled dataset directory (load_from_disk).

    Returns:
        CategoryResult with category="custom" and benchmark scores for
        "tool-call-format" and "code-syntax".
    """
    global load_from_disk, run_inference_on_sample, _load_model_and_tokenizer  # noqa: PLW0603

    # Assign module-level references if not already patched by tests
    if load_from_disk is None:
        from datasets import load_from_disk as _lfd
        load_from_disk = _lfd
    if run_inference_on_sample is None:
        run_inference_on_sample = _run_inference_on_sample
    if _load_model_and_tokenizer is None:
        _load_model_and_tokenizer = _do_load_model_and_tokenizer

    logger.info("Loading dataset from %s", dataset_dir)
    ds = load_from_disk(dataset_dir)
    test_split = ds["test"]

    logger.info("Loading model and tokenizer: %s", model_path)
    model, tokenizer = _load_model_and_tokenizer(model_path, device)

    tool_results: list[bool] = []
    code_results: list[bool] = []

    for sample in test_split:
        domain = sample.get("domain", "")
        if domain == "knowledge":
            # Knowledge samples are covered by lm-eval-harness benchmarks (D-02)
            continue

        output = run_inference_on_sample(model, tokenizer, sample, device)

        if domain == "tool-calling":
            tool_results.append(check_tool_call_format(output))
        elif domain == "code":
            code_results.append(check_code_syntax(output))

    # Free MPS memory between model loads (Pitfall 2 mitigation)
    del model
    gc.collect()

    tool_score = sum(tool_results) / len(tool_results) if tool_results else 0.0
    code_score = sum(code_results) / len(code_results) if code_results else 0.0

    logger.info(
        "Custom eval complete -- tool-call-format: %.4f (%d samples), "
        "code-syntax: %.4f (%d samples)",
        tool_score, len(tool_results),
        code_score, len(code_results),
    )

    return CategoryResult(
        category="custom",
        benchmarks=[
            BenchmarkResult(
                benchmark="tool-call-format",
                metric="pass@1",
                score=tool_score,
            ),
            BenchmarkResult(
                benchmark="code-syntax",
                metric="pass@1",
                score=code_score,
            ),
        ],
    )


# Model path validation (mirrors eval_runner.py per T-03-07)
_MODEL_PATH_PATTERN = re.compile(r"^[a-zA-Z0-9._/~\-]+$")


def _validate_model_path(model_path: str) -> bool:
    """Validate model path is safe to pass to library functions (T-03-07).

    Accepts local paths that exist or HuggingFace model IDs with safe chars.

    Args:
        model_path: Path or HuggingFace model ID.

    Returns:
        True if valid, False otherwise.
    """
    if Path(model_path).exists():
        return True
    if _MODEL_PATH_PATTERN.match(model_path):
        return True
    return False


def main() -> int:
    """CLI entry point for custom inference evaluation.

    Runs the custom eval on the assembled test split and writes an EvalResult
    JSON to the specified output path.

    Returns:
        0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Custom inference eval for tool-calling and code domains (D-02). "
            "Loads the assembled test split, runs transformers inference, "
            "and checks outputs for correct SmolLM2 tool-call format and "
            "valid Python code syntax."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Local model path or HuggingFace model ID.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="datasets/assembled",
        help="Path to assembled dataset directory (default: datasets/assembled).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file path for EvalResult.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override auto-detected device ('mps' or 'cpu').",
    )

    args = parser.parse_args()

    # Validate model path per T-03-07
    if not _validate_model_path(args.model):
        print(
            f"Error: invalid model path: {args.model}. "
            "Must be a valid local path or HuggingFace model ID.",
            file=sys.stderr,
        )
        return 1

    # Detect device (lazy import mirrors eval_runner.py pattern)
    if args.device:
        device = args.device
    else:
        try:
            import os
            import torch
            if torch.backends.mps.is_available():
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                device = "mps"
            else:
                device = "cpu"
        except ImportError:
            logger.warning("PyTorch not installed, defaulting to cpu")
            device = "cpu"

    logger.info("Using device: %s", device)

    try:
        category_result = run_custom_eval(args.model, device, args.dataset_dir)
    except Exception as exc:
        print(f"Error during custom eval: {exc}", file=sys.stderr)
        return 1

    model_name = Path(args.model).name
    result = EvalResult(
        model_path=args.model,
        model_name=model_name,
        timestamp=datetime.now().isoformat(),
        device=device,
        categories=[category_result],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2))
    logger.info("Results written to: %s", args.output)

    # Print summary table (mirrors eval_runner.py pattern)
    from scripts.eval_runner import format_summary_table
    print(format_summary_table(result))

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())
