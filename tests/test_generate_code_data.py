"""Tests for the code generation batch generation script.

Validates that generate_code_data.py produces valid JSONL batches for all 3
code generation categories: utility functions, file operations, and debugging.
"""
import json
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.validate_format import Conversation


# Load system prompts for assertion comparison
_prompts_data = yaml.safe_load(
    Path("templates/system-prompts.yaml").read_text()
)
EXPECTED_CODE_ASSISTANT = (
    _prompts_data["system_prompts"]["code_assistant"]["content"].strip()
)
EXPECTED_CODE_DEBUGGER = (
    _prompts_data["system_prompts"]["code_debugger"]["content"].strip()
)


class TestUtilityBatch:
    """Tests for generate_utility_batch."""

    def test_returns_correct_count(self):
        from scripts.generate_code_data import generate_utility_batch
        samples = generate_utility_batch(count=5, seed=42)
        assert len(samples) == 5

    def test_all_samples_validate(self):
        from scripts.generate_code_data import generate_utility_batch
        samples = generate_utility_batch(count=5, seed=42)
        for i, sample in enumerate(samples):
            try:
                Conversation.model_validate(sample)
            except Exception as e:
                pytest.fail(f"Sample {i} failed validation: {e}")

    def test_uses_code_assistant_prompt(self):
        from scripts.generate_code_data import generate_utility_batch
        samples = generate_utility_batch(count=5, seed=42)
        for sample in samples:
            assert sample["messages"][0]["role"] == "system"
            assert (
                sample["messages"][0]["content"].strip() == EXPECTED_CODE_ASSISTANT
            )

    def test_language_coverage(self):
        """Over 50 samples with seed=42, all 5 languages appear in code fences."""
        from scripts.generate_code_data import generate_utility_batch
        samples = generate_utility_batch(count=50, seed=42)
        found_languages = set()
        for sample in samples:
            for msg in sample["messages"]:
                if msg["role"] == "assistant" and msg.get("content"):
                    content = msg["content"]
                    for lang in ["python", "javascript", "typescript", "go", "rust"]:
                        if f"```{lang}" in content:
                            found_languages.add(lang)
        expected = {"python", "javascript", "typescript", "go", "rust"}
        assert found_languages == expected, (
            f"Missing languages: {expected - found_languages}"
        )

    def test_responses_have_code_blocks(self):
        """Every assistant response contains triple backtick fences."""
        from scripts.generate_code_data import generate_utility_batch
        samples = generate_utility_batch(count=5, seed=42)
        for i, sample in enumerate(samples):
            for msg in sample["messages"]:
                if msg["role"] == "assistant" and msg.get("content"):
                    assert "```" in msg["content"], (
                        f"Sample {i} assistant response missing code fence"
                    )


class TestFileOpsBatch:
    """Tests for generate_file_ops_batch."""

    def test_returns_correct_count(self):
        from scripts.generate_code_data import generate_file_ops_batch
        samples = generate_file_ops_batch(count=5, seed=42)
        assert len(samples) == 5

    def test_all_samples_validate(self):
        from scripts.generate_code_data import generate_file_ops_batch
        samples = generate_file_ops_batch(count=5, seed=42)
        for i, sample in enumerate(samples):
            try:
                Conversation.model_validate(sample)
            except Exception as e:
                pytest.fail(f"Sample {i} failed validation: {e}")

    def test_uses_code_assistant_prompt(self):
        from scripts.generate_code_data import generate_file_ops_batch
        samples = generate_file_ops_batch(count=5, seed=42)
        for sample in samples:
            assert sample["messages"][0]["role"] == "system"
            assert (
                sample["messages"][0]["content"].strip() == EXPECTED_CODE_ASSISTANT
            )

    def test_language_coverage(self):
        """Over 50 samples, all 3 languages appear (python, javascript, typescript)."""
        from scripts.generate_code_data import generate_file_ops_batch
        samples = generate_file_ops_batch(count=50, seed=42)
        found_languages = set()
        for sample in samples:
            for msg in sample["messages"]:
                if msg["role"] == "assistant" and msg.get("content"):
                    content = msg["content"]
                    for lang in ["python", "javascript", "typescript"]:
                        if f"```{lang}" in content:
                            found_languages.add(lang)
        expected = {"python", "javascript", "typescript"}
        assert found_languages == expected, (
            f"Missing languages: {expected - found_languages}"
        )

    def test_error_handling_present(self):
        """At least 50% of samples contain error handling keywords."""
        from scripts.generate_code_data import generate_file_ops_batch
        samples = generate_file_ops_batch(count=10, seed=42)
        error_keywords = ["try", "except", "catch", "finally", "error", "Error", "with open"]
        count_with_error_handling = 0
        for sample in samples:
            for msg in sample["messages"]:
                if msg["role"] == "assistant" and msg.get("content"):
                    if any(kw in msg["content"] for kw in error_keywords):
                        count_with_error_handling += 1
                        break
        assert count_with_error_handling >= len(samples) // 2, (
            f"Only {count_with_error_handling}/{len(samples)} samples have error handling"
        )

    def test_responses_have_code_blocks(self):
        """Every file-ops assistant response contains code fences."""
        from scripts.generate_code_data import generate_file_ops_batch
        samples = generate_file_ops_batch(count=5, seed=42)
        for i, sample in enumerate(samples):
            for msg in sample["messages"]:
                if msg["role"] == "assistant" and msg.get("content"):
                    assert "```" in msg["content"], (
                        f"Sample {i} assistant response missing code fence"
                    )


class TestDebuggingBatch:
    """Tests for generate_debugging_batch."""

    def test_returns_correct_count(self):
        from scripts.generate_code_data import generate_debugging_batch
        samples = generate_debugging_batch(count=5, seed=42)
        assert len(samples) == 5

    def test_all_samples_validate(self):
        from scripts.generate_code_data import generate_debugging_batch
        samples = generate_debugging_batch(count=5, seed=42)
        for i, sample in enumerate(samples):
            try:
                Conversation.model_validate(sample)
            except Exception as e:
                pytest.fail(f"Sample {i} failed validation: {e}")

    def test_uses_code_debugger_prompt(self):
        from scripts.generate_code_data import generate_debugging_batch
        samples = generate_debugging_batch(count=5, seed=42)
        for sample in samples:
            assert sample["messages"][0]["role"] == "system"
            assert (
                sample["messages"][0]["content"].strip() == EXPECTED_CODE_DEBUGGER
            )

    def test_bug_fix_format(self):
        """Every debugging assistant response contains both 'Bug:' and 'Fix:' per D-06."""
        from scripts.generate_code_data import generate_debugging_batch
        samples = generate_debugging_batch(count=5, seed=42)
        for i, sample in enumerate(samples):
            for msg in sample["messages"]:
                if msg["role"] == "assistant" and msg.get("content"):
                    assert "Bug:" in msg["content"], (
                        f"Sample {i} missing 'Bug:' in debugging response"
                    )
                    assert "Fix:" in msg["content"], (
                        f"Sample {i} missing 'Fix:' in debugging response"
                    )

    def test_responses_have_code_blocks(self):
        """Every debugging response has code fences."""
        from scripts.generate_code_data import generate_debugging_batch
        samples = generate_debugging_batch(count=5, seed=42)
        for i, sample in enumerate(samples):
            for msg in sample["messages"]:
                if msg["role"] == "assistant" and msg.get("content"):
                    assert "```" in msg["content"], (
                        f"Sample {i} debugging response missing code fence"
                    )

    def test_language_coverage(self):
        """Over 50 samples, all 3 languages appear (python, javascript, typescript)."""
        from scripts.generate_code_data import generate_debugging_batch
        samples = generate_debugging_batch(count=50, seed=42)
        found_languages = set()
        for sample in samples:
            for msg in sample["messages"]:
                if msg["role"] == "assistant" and msg.get("content"):
                    content = msg["content"]
                    for lang in ["python", "javascript", "typescript"]:
                        if f"```{lang}" in content:
                            found_languages.add(lang)
        expected = {"python", "javascript", "typescript"}
        assert found_languages == expected, (
            f"Missing languages: {expected - found_languages}"
        )


class TestSystemMessageFirst:
    """All 3 generators produce samples where messages[0].role == 'system'."""

    def test_utility_system_first(self):
        from scripts.generate_code_data import generate_utility_batch
        for sample in generate_utility_batch(count=5, seed=42):
            assert sample["messages"][0]["role"] == "system"

    def test_file_ops_system_first(self):
        from scripts.generate_code_data import generate_file_ops_batch
        for sample in generate_file_ops_batch(count=5, seed=42):
            assert sample["messages"][0]["role"] == "system"

    def test_debugging_system_first(self):
        from scripts.generate_code_data import generate_debugging_batch
        for sample in generate_debugging_batch(count=5, seed=42):
            assert sample["messages"][0]["role"] == "system"


class TestQueryDiversity:
    """Generated conversations have diverse user queries."""

    def test_no_duplicate_queries_utility(self):
        from scripts.generate_code_data import generate_utility_batch
        samples = generate_utility_batch(count=50, seed=42)
        user_messages = []
        for sample in samples:
            for msg in sample["messages"]:
                if msg["role"] == "user":
                    user_messages.append(msg["content"])
        assert len(user_messages) == len(set(user_messages)), (
            "Duplicate user messages found in utility batch of 50"
        )

    def test_no_duplicate_queries_file_ops(self):
        from scripts.generate_code_data import generate_file_ops_batch
        samples = generate_file_ops_batch(count=50, seed=42)
        user_messages = []
        for sample in samples:
            for msg in sample["messages"]:
                if msg["role"] == "user":
                    user_messages.append(msg["content"])
        assert len(user_messages) == len(set(user_messages)), (
            "Duplicate user messages found in file-ops batch of 50"
        )

    def test_no_duplicate_queries_debugging(self):
        from scripts.generate_code_data import generate_debugging_batch
        samples = generate_debugging_batch(count=50, seed=42)
        user_messages = []
        for sample in samples:
            for msg in sample["messages"]:
                if msg["role"] == "user":
                    user_messages.append(msg["content"])
        assert len(user_messages) == len(set(user_messages)), (
            "Duplicate user messages found in debugging batch of 50"
        )


class TestCliEntryPoint:
    """CLI entry point writes JSONL file."""

    def test_writes_jsonl(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, "-m", "scripts.generate_code_data",
                 "--category", "utility", "--count", "5", "--batch", "1",
                 "--seed", "42", "--output-dir", tmpdir],
                capture_output=True, text=True,
                cwd=str(Path(__file__).parent.parent),
            )
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            output_path = Path(tmpdir) / "utility-batch-01.jsonl"
            assert output_path.exists(), f"Output file not created: {output_path}"
            lines = output_path.read_text().strip().split("\n")
            assert len(lines) == 5
            for line in lines:
                data = json.loads(line)
                Conversation.model_validate(data)

    def test_rejects_invalid_category(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "scripts.generate_code_data",
             "--category", "invalid", "--count", "5", "--batch", "1"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode != 0

    def test_rejects_negative_batch(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "scripts.generate_code_data",
             "--category", "utility", "--count", "5", "--batch", "-1"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode != 0


class TestValidateBatch:
    """Test the validate_batch function."""

    def test_validate_batch_all_valid(self):
        from scripts.generate_code_data import generate_utility_batch, validate_batch
        samples = generate_utility_batch(count=5, seed=42)
        results = validate_batch(samples)
        assert results["total"] == 5
        assert results["valid"] == 5
        assert results["invalid"] == 0

    def test_validate_batch_catches_invalid(self):
        from scripts.generate_code_data import validate_batch
        bad_samples = [{"messages": []}]
        results = validate_batch(bad_samples)
        assert results["invalid"] >= 1


class TestWriteBatch:
    """Test the write_batch function."""

    def test_write_batch_creates_file(self):
        from scripts.generate_code_data import generate_utility_batch, write_batch
        samples = generate_utility_batch(count=3, seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test-batch-01.jsonl"
            result = write_batch(samples, output_path)
            assert result.exists()
            lines = result.read_text().strip().split("\n")
            assert len(lines) == 3
