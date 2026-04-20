#!/usr/bin/env python3
"""eval_config.py -- Pydantic schemas and config loader for the evaluation framework.

Defines typed data models for benchmark results, per-category aggregation,
cross-model comparison deltas, and the YAML-based eval configuration.

Schemas:
  BenchmarkResult  -- score for a single benchmark (e.g. mmlu acc)
  CategoryResult   -- scores for all benchmarks in one category
  EvalResult       -- complete evaluation output for one model (D-07)
  CompareResult    -- delta between baseline and candidate for one benchmark
  EvalConfig       -- configuration schema for configs/eval.yaml

Loader:
  load_eval_config(path) -- parse and validate eval.yaml

Threat mitigations:
  T-03-01: yaml.safe_load only (never yaml.load)
  T-03-02: Pydantic model_validate for all external data
"""
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class BenchmarkResult(BaseModel):
    """Score for a single benchmark."""

    benchmark: str  # e.g. "mmlu", "arc_challenge", "humaneval"
    metric: str  # e.g. "acc", "acc_norm", "pass@1"
    score: float  # 0.0 to 1.0
    num_fewshot: Optional[int] = None


class CategoryResult(BaseModel):
    """Scores for all benchmarks in one category."""

    category: str  # "tool-calling", "code", "knowledge"
    benchmarks: list[BenchmarkResult]


class EvalResult(BaseModel):
    """Complete evaluation result for one model (JSON output per D-07)."""

    model_path: str
    model_name: str
    timestamp: str  # ISO format
    device: str  # "mps" or "cpu"
    categories: list[CategoryResult]


class CompareResult(BaseModel):
    """Delta between baseline and candidate for one benchmark."""

    category: str
    benchmark: str
    metric: str
    baseline_score: float
    candidate_score: float
    delta: float  # candidate - baseline


class EvalConfig(BaseModel):
    """Configuration schema for configs/eval.yaml."""

    version: int = 1
    knowledge_tasks: list[str] = Field(
        default_factory=lambda: ["mmlu", "arc_challenge", "hellaswag"]
    )
    code_datasets: list[str] = Field(
        default_factory=lambda: ["humaneval", "mbpp"]
    )
    bfcl_test_categories: list[str] = Field(default_factory=lambda: ["all"])
    batch_size: str = "auto"
    num_fewshot: dict[str, int] = Field(
        default_factory=lambda: {"mmlu": 5, "arc_challenge": 25, "hellaswag": 10}
    )
    dtype: str = "float32"


def load_eval_config(path: Path) -> EvalConfig:
    """Load and validate eval configuration from a YAML file.

    Uses yaml.safe_load exclusively per T-03-01 (never yaml.load).

    Args:
        path: Path to the eval.yaml file.

    Returns:
        Validated EvalConfig instance.

    Raises:
        FileNotFoundError: If path does not exist.
        pydantic.ValidationError: If YAML content fails schema validation.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return EvalConfig.model_validate(raw)
