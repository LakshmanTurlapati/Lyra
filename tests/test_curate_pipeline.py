#!/usr/bin/env python3
"""Integration tests for scripts/curate_pipeline.py -- the pipeline orchestrator."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.curate_pipeline import load_config, run_pipeline
from scripts.pipeline_config import PipelineConfig


# --- Fixtures: reusable sample data ---

def _valid_code_sample() -> dict:
    """A valid code-domain sample that passes all 4 pipeline stages."""
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Write a Python function to reverse a string."},
            {
                "role": "assistant",
                "content": (
                    "Here is the function:\n\n"
                    "```python\n"
                    "def reverse_string(s: str) -> str:\n"
                    '    """Return the reverse of the input string."""\n'
                    "    return s[::-1]\n"
                    "```\n"
                ),
            },
        ]
    }


def _valid_knowledge_sample() -> dict:
    """A valid knowledge-domain sample with reasoning markers."""
    return {
        "messages": [
            {"role": "system", "content": "You are a knowledgeable assistant."},
            {"role": "user", "content": "How does photosynthesis work?"},
            {
                "role": "assistant",
                "content": (
                    "Let me explain photosynthesis step by step.\n\n"
                    "1. First, plants absorb sunlight through chlorophyll pigments "
                    "in their leaves. This light energy drives the entire process.\n\n"
                    "2. The light reactions occur in the thylakoid membranes, where "
                    "water molecules are split into oxygen, hydrogen ions, and electrons. "
                    "This is why plants release oxygen because the water splitting "
                    "liberates O2 as a byproduct.\n\n"
                    "3. Therefore, the energy from light is stored in ATP and NADPH "
                    "molecules, which then power the Calvin cycle.\n\n"
                    "4. Finally, in the Calvin cycle (dark reactions), carbon dioxide "
                    "from the air is fixed into glucose using the ATP and NADPH. "
                    "This means plants essentially convert light energy into chemical "
                    "energy stored in sugar molecules.\n\n"
                    "The overall equation is: 6CO2 + 6H2O + light -> C6H12O6 + 6O2. "
                    "This process is fundamental to virtually all life on Earth "
                    "because it produces both the oxygen we breathe and the sugars "
                    "that form the base of food chains."
                ),
            },
        ]
    }


def _malformed_sample() -> dict:
    """A sample missing the required system message (fails format validation)."""
    return {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }


def _truncated_sample() -> dict:
    """A sample with a truncated assistant response (fails quality scoring)."""
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing."},
            {
                "role": "assistant",
                "content": "Quantum computing is a type of computation that uses...",
            },
        ]
    }


def _near_duplicate_pair() -> tuple[dict, dict]:
    """Two samples with near-identical assistant responses."""
    base = {
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Write a function to add two numbers."},
            {
                "role": "assistant",
                "content": (
                    "Here is the function:\n\n"
                    "```python\n"
                    "def add(a: int, b: int) -> int:\n"
                    "    return a + b\n"
                    "```\n"
                ),
            },
        ]
    }
    dup = {
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Write a function to add two integers."},
            {
                "role": "assistant",
                "content": (
                    "Here is the function:\n\n"
                    "```python\n"
                    "def add(a: int, b: int) -> int:\n"
                    "    return a + b\n"
                    "```\n"
                ),
            },
        ]
    }
    return base, dup


def _no_code_sample() -> dict:
    """A code-domain sample that has NO code blocks (fails style validation)."""
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Write a function to sort a list."},
            {
                "role": "assistant",
                "content": "You can use the built-in sorted function to sort a list.",
            },
        ]
    }


def _write_jsonl(path: Path, samples: list[dict]) -> None:
    """Write samples to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    """Read samples from a JSONL file."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


@pytest.fixture
def code_config() -> PipelineConfig:
    """A config for code domain testing."""
    return PipelineConfig(
        style_validation=True,
        dedup_threshold=0.7,
        ngram_size=3,
        dedup_scope="response",
    )


# --- Pipeline stage tests ---

class TestPipelineStages:
    """Test individual pipeline stage behavior via run_pipeline."""

    def test_pipeline_rejects_invalid_format(self, tmp_path, code_config):
        """JSONL with malformed conversation produces empty output."""
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        _write_jsonl(input_path, [_malformed_sample()])

        stats = run_pipeline(input_path, output_path, code_config, "code")
        assert stats["format_valid"] == 0
        assert stats["output_count"] == 0

    def test_pipeline_rejects_low_quality(self, tmp_path, code_config):
        """JSONL with truncated assistant response produces empty output."""
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        _write_jsonl(input_path, [_truncated_sample()])

        stats = run_pipeline(input_path, output_path, code_config, "code")
        assert stats["quality_pass"] == 0
        assert stats["output_count"] == 0

    def test_pipeline_removes_duplicates(self, tmp_path, code_config):
        """JSONL with 2 near-identical samples produces output with 1 sample."""
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        base, dup = _near_duplicate_pair()
        _write_jsonl(input_path, [base, dup])

        # Use permissive style for this test (no code block requirement)
        code_config_permissive = PipelineConfig(
            style_validation=False,
            dedup_threshold=0.7,
            ngram_size=3,
            dedup_scope="response",
        )
        stats = run_pipeline(input_path, output_path, code_config_permissive, "code")
        assert stats["after_dedup"] == 1
        output = _read_jsonl(output_path)
        assert len(output) == 1

    def test_pipeline_rejects_wrong_style(self, tmp_path):
        """Code domain sample with no code blocks rejected when require_code_blocks=true."""
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        _write_jsonl(input_path, [_no_code_sample()])

        config = PipelineConfig(
            style_validation=True,
            domains={
                "code": __import__("scripts.pipeline_config", fromlist=["DomainConfig"]).DomainConfig(
                    style=__import__("scripts.pipeline_config", fromlist=["StyleConfig"]).StyleConfig(
                        require_code_blocks=True,
                    ),
                ),
            },
        )
        stats = run_pipeline(input_path, output_path, config, "code")
        assert stats["after_style"] == 0
        assert stats["output_count"] == 0


class TestPipelineOutput:
    """Test pipeline output format and content."""

    def test_pipeline_passes_good_sample(self, tmp_path, code_config):
        """Valid, high-quality, unique, correctly-styled sample passes all 4 stages."""
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        _write_jsonl(input_path, [_valid_code_sample()])

        stats = run_pipeline(input_path, output_path, code_config, "code")
        assert stats["output_count"] == 1
        output = _read_jsonl(output_path)
        assert len(output) == 1

    def test_pipeline_attaches_quality_scores(self, tmp_path, code_config):
        """Output JSONL records contain _quality key with scores."""
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        _write_jsonl(input_path, [_valid_code_sample()])

        run_pipeline(input_path, output_path, code_config, "code")
        output = _read_jsonl(output_path)
        assert len(output) == 1
        assert "_quality" in output[0]
        quality = output[0]["_quality"]
        assert "pass" in quality
        assert "score" in quality
        assert "signals" in quality
        assert quality["pass"] is True

    def test_pipeline_uses_domain_config(self, tmp_path):
        """Code domain uses min_response_chars=20 (not global 10)."""
        from scripts.pipeline_config import DomainConfig

        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"

        # Sample with 15-char assistant response -- passes global (10) but fails code (20)
        short_code = {
            "messages": [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "Help"},
                {"role": "assistant", "content": "Use sorted()."},
            ]
        }
        _write_jsonl(input_path, [short_code])

        config = PipelineConfig(
            defaults=DomainConfig(min_response_chars=10),
            domains={"code": DomainConfig(min_response_chars=20)},
            style_validation=False,
        )
        stats = run_pipeline(input_path, output_path, config, "code")
        # Should fail quality due to min_response_chars=20
        assert stats["quality_pass"] == 0


class TestPipelineCLI:
    """Test CLI entry point."""

    def test_pipeline_cli_runs(self):
        """subprocess call to python -m scripts.curate_pipeline --help exits 0."""
        result = subprocess.run(
            [sys.executable, "-m", "scripts.curate_pipeline", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--input" in result.stdout
        assert "--domain" in result.stdout
        assert "--config" in result.stdout
