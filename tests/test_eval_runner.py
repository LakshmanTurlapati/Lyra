#!/usr/bin/env python3
"""Unit tests for scripts/eval_runner.py -- eval orchestration with mocked benchmarks."""
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.eval_config import (
    BenchmarkResult,
    CategoryResult,
    EvalConfig,
    EvalResult,
)
from scripts.eval_runner import (
    detect_device,
    format_summary_table,
    main,
    run_code_benchmarks,
    run_knowledge_benchmarks,
)


# --- Device Detection Tests ---


def test_detect_device_mps():
    """detect_device returns 'mps' when MPS is available."""
    mock_torch = MagicMock()
    mock_torch.backends.mps.is_available.return_value = True
    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = detect_device()
    assert result == "mps"


def test_detect_device_cpu():
    """detect_device returns 'cpu' when MPS is unavailable."""
    mock_torch = MagicMock()
    mock_torch.backends.mps.is_available.return_value = False
    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = detect_device()
    assert result == "cpu"


def test_detect_device_sets_fallback_env():
    """detect_device sets PYTORCH_ENABLE_MPS_FALLBACK=1 when MPS available."""
    mock_torch = MagicMock()
    mock_torch.backends.mps.is_available.return_value = True

    with patch.dict("sys.modules", {"torch": mock_torch}):
        # Clear the env var first
        os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
        detect_device()
        assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"

    # Clean up
    os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)


# --- Knowledge Benchmarks Tests ---


def test_knowledge_benchmarks_calls_lm_eval():
    """run_knowledge_benchmarks calls lm_eval.simple_evaluate with correct params."""
    mock_lm_eval = MagicMock()
    mock_lm_eval.simple_evaluate.return_value = {
        "results": {
            "mmlu": {"acc,none": 0.25},
            "arc_challenge": {"acc_norm,none": 0.30},
            "hellaswag": {"acc_norm,none": 0.35},
        }
    }

    config = EvalConfig(
        knowledge_tasks=["mmlu", "arc_challenge", "hellaswag"],
        num_fewshot={"mmlu": 5, "arc_challenge": 25, "hellaswag": 10},
    )

    with patch.dict("sys.modules", {"lm_eval": mock_lm_eval}):
        result = run_knowledge_benchmarks("test-model", "cpu", config)

    assert result.category == "knowledge"
    assert len(result.benchmarks) == 3

    # Verify scores extracted correctly
    scores = {b.benchmark: b.score for b in result.benchmarks}
    assert scores["mmlu"] == pytest.approx(0.25)
    assert scores["arc_challenge"] == pytest.approx(0.30)
    assert scores["hellaswag"] == pytest.approx(0.35)

    # Verify metrics extracted correctly
    metrics = {b.benchmark: b.metric for b in result.benchmarks}
    assert metrics["mmlu"] == "acc"
    assert metrics["arc_challenge"] == "acc_norm"
    assert metrics["hellaswag"] == "acc_norm"

    # Verify lm_eval.simple_evaluate was called
    mock_lm_eval.simple_evaluate.assert_called_once()
    call_kwargs = mock_lm_eval.simple_evaluate.call_args
    assert call_kwargs[1]["model"] == "hf" or call_kwargs[0][0] == "hf"


# --- Code Benchmarks Tests ---


@patch("scripts.eval_runner.subprocess.run")
def test_code_benchmarks_calls_subprocess(mock_run):
    """run_code_benchmarks calls subprocess.run with evalplus command."""
    mock_result = MagicMock()
    mock_result.stdout = "Results:\n  pass@1: 0.12\n"
    mock_result.returncode = 0
    mock_run.return_value = mock_result

    config = EvalConfig(code_datasets=["humaneval", "mbpp"])

    result = run_code_benchmarks("test-model", config)

    assert result.category == "code"
    assert len(result.benchmarks) == 2

    # Verify subprocess was called for each dataset
    assert mock_run.call_count == 2

    # Verify list-form commands (no shell injection)
    for call in mock_run.call_args_list:
        cmd = call[0][0]
        assert isinstance(cmd, list)
        assert "evalplus.evaluate" in cmd
        assert "--model" in cmd
        assert "--backend" in cmd


@patch("scripts.eval_runner.subprocess.run")
def test_code_benchmarks_parses_pass_at_1(mock_run):
    """run_code_benchmarks correctly parses pass@1 from evalplus output."""
    mock_result = MagicMock()
    mock_result.stdout = "humaneval (base tests)\n  pass@1: 0.1829\n"
    mock_result.returncode = 0
    mock_run.return_value = mock_result

    config = EvalConfig(code_datasets=["humaneval"])

    result = run_code_benchmarks("test-model", config)
    assert result.benchmarks[0].score == pytest.approx(0.1829)
    assert result.benchmarks[0].metric == "pass@1"


# --- Summary Table Tests ---


def test_format_summary_table_output():
    """format_summary_table returns table with category names and scores."""
    result = EvalResult(
        model_path="models/test-model",
        model_name="test-model",
        timestamp="2026-04-20T12:00:00",
        device="cpu",
        categories=[
            CategoryResult(
                category="knowledge",
                benchmarks=[
                    BenchmarkResult(benchmark="mmlu", metric="acc", score=0.25),
                ],
            ),
            CategoryResult(
                category="code",
                benchmarks=[
                    BenchmarkResult(
                        benchmark="humaneval", metric="pass@1", score=0.12
                    ),
                ],
            ),
        ],
    )

    output = format_summary_table(result)
    assert "knowledge" in output
    assert "mmlu" in output
    assert "0.2500" in output
    assert "code" in output
    assert "humaneval" in output
    assert "0.1200" in output
    # Verify header elements present
    assert "Category" in output
    assert "Benchmark" in output
    assert "Metric" in output
    assert "Score" in output


def test_format_summary_table_no_emoji():
    """format_summary_table output contains no emoji characters."""
    result = EvalResult(
        model_path="models/test",
        model_name="test",
        timestamp="2026-04-20T12:00:00",
        device="cpu",
        categories=[
            CategoryResult(
                category="knowledge",
                benchmarks=[
                    BenchmarkResult(benchmark="mmlu", metric="acc", score=0.25),
                ],
            ),
        ],
    )

    output = format_summary_table(result)
    # Check no characters outside basic ASCII + common punctuation
    for char in output:
        assert ord(char) < 128, f"Non-ASCII character found: {char!r} (ord {ord(char)})"


def test_format_summary_table_model_info():
    """format_summary_table includes model name and device info."""
    result = EvalResult(
        model_path="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        model_name="SmolLM2-1.7B-Instruct",
        timestamp="2026-04-20T12:00:00",
        device="mps",
        categories=[],
    )

    output = format_summary_table(result)
    assert "SmolLM2-1.7B-Instruct" in output
    assert "mps" in output


# --- Main CLI Tests ---


def test_main_writes_json_output(tmp_path):
    """main() writes valid EvalResult JSON to output path."""
    output_file = tmp_path / "results" / "out.json"

    # Mock all benchmark functions to return fixed results
    mock_knowledge = CategoryResult(
        category="knowledge",
        benchmarks=[
            BenchmarkResult(benchmark="mmlu", metric="acc", score=0.25),
        ],
    )

    with (
        patch(
            "scripts.eval_runner.run_knowledge_benchmarks",
            return_value=mock_knowledge,
        ),
        patch("scripts.eval_runner.detect_device", return_value="cpu"),
        patch("scripts.eval_runner._validate_model_path", return_value=True),
        patch(
            "sys.argv",
            [
                "eval_runner",
                "--model", "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                "--benchmarks", "knowledge",
                "--output", str(output_file),
                "--config", "configs/eval.yaml",
            ],
        ),
    ):
        exit_code = main()

    assert exit_code == 0
    assert output_file.exists()

    # Parse output and validate
    data = json.loads(output_file.read_text())
    result = EvalResult.model_validate(data)
    assert result.model_name == "SmolLM2-1.7B-Instruct"
    assert result.device == "cpu"
    assert len(result.categories) == 1
    assert result.categories[0].category == "knowledge"


def test_main_invalid_benchmark_category(tmp_path, capsys):
    """main() returns 1 for invalid benchmark category."""
    output_file = tmp_path / "out.json"

    with patch(
        "sys.argv",
        [
            "eval_runner",
            "--model", "test-model",
            "--benchmarks", "invalid-category",
            "--output", str(output_file),
        ],
    ), patch("scripts.eval_runner._validate_model_path", return_value=True):
        exit_code = main()

    assert exit_code == 1


def test_main_invalid_model_path(tmp_path, capsys):
    """main() returns 1 for invalid model path."""
    output_file = tmp_path / "out.json"

    with patch(
        "sys.argv",
        [
            "eval_runner",
            "--model", "../../etc/passwd; rm -rf /",
            "--benchmarks", "knowledge",
            "--output", str(output_file),
        ],
    ):
        exit_code = main()

    assert exit_code == 1


# --- Integration Tests (marked slow, skip if deps not installed) ---


@pytest.mark.slow
def test_lm_eval_import():
    """Verify lm-eval-harness can be imported."""
    try:
        import lm_eval
        assert hasattr(lm_eval, "simple_evaluate")
    except ImportError:
        pytest.skip("lm-eval not installed")


@pytest.mark.slow
def test_evalplus_import():
    """Verify evalplus can be imported."""
    try:
        import evalplus
    except ImportError:
        pytest.skip("evalplus not installed")
