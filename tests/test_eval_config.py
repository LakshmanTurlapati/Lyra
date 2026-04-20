#!/usr/bin/env python3
"""Unit tests for scripts/eval_config.py -- Pydantic schemas and config loading."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.eval_config import (
    BenchmarkResult,
    CategoryResult,
    CompareResult,
    EvalConfig,
    EvalResult,
    load_eval_config,
)


def test_benchmark_result_valid():
    """BenchmarkResult validates correctly with all required fields."""
    result = BenchmarkResult(benchmark="mmlu", metric="acc", score=0.25)
    assert result.benchmark == "mmlu"
    assert result.metric == "acc"
    assert result.score == 0.25
    assert result.num_fewshot is None


def test_benchmark_result_with_fewshot():
    """BenchmarkResult accepts optional num_fewshot field."""
    result = BenchmarkResult(
        benchmark="arc_challenge", metric="acc_norm", score=0.30, num_fewshot=25
    )
    assert result.num_fewshot == 25


def test_eval_result_schema():
    """EvalResult validates correctly formed JSON and roundtrips."""
    result = EvalResult(
        model_path="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        model_name="SmolLM2-1.7B-Instruct",
        timestamp="2026-04-20T12:00:00",
        device="mps",
        categories=[
            CategoryResult(
                category="knowledge",
                benchmarks=[
                    BenchmarkResult(benchmark="mmlu", metric="acc", score=0.25),
                    BenchmarkResult(
                        benchmark="arc_challenge", metric="acc_norm", score=0.30
                    ),
                ],
            )
        ],
    )
    # Roundtrip via JSON
    json_str = result.model_dump_json()
    restored = EvalResult.model_validate_json(json_str)
    assert restored.model_path == "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    assert restored.model_name == "SmolLM2-1.7B-Instruct"
    assert restored.device == "mps"
    assert len(restored.categories) == 1
    assert len(restored.categories[0].benchmarks) == 2


def test_eval_result_rejects_missing_model():
    """EvalResult rejects creation without required model_path field."""
    with pytest.raises(ValidationError):
        EvalResult(
            model_name="test",
            timestamp="2026-04-20T12:00:00",
            device="cpu",
            categories=[],
        )


def test_category_result_valid():
    """CategoryResult holds multiple BenchmarkResults."""
    cat = CategoryResult(
        category="code",
        benchmarks=[
            BenchmarkResult(benchmark="humaneval", metric="pass@1", score=0.12),
            BenchmarkResult(benchmark="mbpp", metric="pass@1", score=0.15),
        ],
    )
    assert cat.category == "code"
    assert len(cat.benchmarks) == 2


def test_compare_result_fields():
    """CompareResult stores delta between baseline and candidate."""
    cr = CompareResult(
        category="knowledge",
        benchmark="mmlu",
        metric="acc",
        baseline_score=0.25,
        candidate_score=0.35,
        delta=0.10,
    )
    assert cr.delta == pytest.approx(0.10)


def test_eval_config_loads_yaml():
    """load_eval_config loads configs/eval.yaml and returns valid EvalConfig."""
    config = load_eval_config(Path("configs/eval.yaml"))
    assert config.knowledge_tasks == ["mmlu", "arc_challenge", "hellaswag"]
    assert config.code_datasets == ["humaneval", "mbpp"]
    assert config.batch_size == "auto"
    assert config.dtype == "float32"
    assert config.num_fewshot["mmlu"] == 5


def test_eval_config_defaults():
    """EvalConfig has correct default values when constructed without args."""
    config = EvalConfig()
    assert config.batch_size == "auto"
    assert config.dtype == "float32"
    assert config.version == 1


def test_eval_config_rejects_invalid_yaml(tmp_path):
    """load_eval_config raises on invalid YAML content."""
    bad_file = tmp_path / "bad.yaml"
    # version must be int, passing a list should fail validation
    bad_file.write_text("version: [1, 2, 3]\nknowledge_tasks: not-a-list\n")
    with pytest.raises((ValidationError, Exception)):
        load_eval_config(bad_file)


def test_eval_config_file_not_found():
    """load_eval_config raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        load_eval_config(Path("/nonexistent/path/eval.yaml"))
