"""Tests for the tool-calling batch generation script.

Validates that generate_tool_data.py produces valid JSONL batches for all 5
tool-calling categories, drawing from the schema pool with 25% edge cases.
"""
import json
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.validate_format import Conversation


class TestSingleCallBatch:
    """Test 1: generate_single_call_batch returns valid conversations."""

    def test_returns_correct_count(self):
        from scripts.generate_tool_data import generate_single_call_batch
        samples = generate_single_call_batch(count=5, seed=42)
        assert len(samples) == 5

    def test_all_samples_validate(self):
        from scripts.generate_tool_data import generate_single_call_batch
        samples = generate_single_call_batch(count=5, seed=42)
        for i, sample in enumerate(samples):
            try:
                Conversation.model_validate(sample)
            except Exception as e:
                pytest.fail(f"Sample {i} failed validation: {e}")

    def test_uses_tool_assistant_prompt(self):
        from scripts.generate_tool_data import generate_single_call_batch
        prompts = yaml.safe_load(Path("templates/system-prompts.yaml").read_text())
        expected_prompt = prompts["system_prompts"]["tool_assistant"]["content"].strip()
        samples = generate_single_call_batch(count=5, seed=42)
        for sample in samples:
            assert sample["messages"][0]["role"] == "system"
            assert sample["messages"][0]["content"].strip() == expected_prompt


class TestCliBatch:
    """Test 2: generate_cli_batch returns valid conversations using run_command."""

    def test_returns_correct_count(self):
        from scripts.generate_tool_data import generate_cli_batch
        samples = generate_cli_batch(count=5, seed=42)
        assert len(samples) == 5

    def test_all_samples_validate(self):
        from scripts.generate_tool_data import generate_cli_batch
        samples = generate_cli_batch(count=5, seed=42)
        for i, sample in enumerate(samples):
            try:
                Conversation.model_validate(sample)
            except Exception as e:
                pytest.fail(f"Sample {i} failed validation: {e}")

    def test_uses_cli_assistant_prompt(self):
        from scripts.generate_tool_data import generate_cli_batch
        prompts = yaml.safe_load(Path("templates/system-prompts.yaml").read_text())
        expected_prompt = prompts["system_prompts"]["cli_assistant"]["content"].strip()
        samples = generate_cli_batch(count=5, seed=42)
        for sample in samples:
            assert sample["messages"][0]["role"] == "system"
            assert sample["messages"][0]["content"].strip() == expected_prompt

    def test_uses_run_command_tool(self):
        from scripts.generate_tool_data import generate_cli_batch
        samples = generate_cli_batch(count=5, seed=42)
        for sample in samples:
            tool_names = [t["function"]["name"] for t in sample.get("tools", [])]
            assert "run_command" in tool_names or "run_command_in_dir" in tool_names


class TestMultiTurnBatch:
    """Test 3: generate_multi_turn_batch returns valid multi-turn conversations."""

    def test_returns_correct_count(self):
        from scripts.generate_tool_data import generate_multi_turn_batch
        samples = generate_multi_turn_batch(count=5, seed=42)
        assert len(samples) == 5

    def test_all_samples_validate(self):
        from scripts.generate_tool_data import generate_multi_turn_batch
        samples = generate_multi_turn_batch(count=5, seed=42)
        for i, sample in enumerate(samples):
            try:
                Conversation.model_validate(sample)
            except Exception as e:
                pytest.fail(f"Sample {i} failed validation: {e}")

    def test_has_multiple_tool_rounds(self):
        """At least some samples should have 2+ tool call rounds."""
        from scripts.generate_tool_data import generate_multi_turn_batch
        samples = generate_multi_turn_batch(count=5, seed=42)
        multi_round_count = 0
        for sample in samples:
            tool_call_rounds = sum(
                1 for msg in sample["messages"]
                if msg["role"] == "assistant" and msg.get("tool_calls")
            )
            if tool_call_rounds >= 2:
                multi_round_count += 1
        # At least 3 of 5 should be multi-round (75% happy path)
        assert multi_round_count >= 3, f"Only {multi_round_count}/5 had 2+ tool rounds"


class TestParallelBatch:
    """Test 4: generate_parallel_batch returns valid parallel call conversations."""

    def test_returns_correct_count(self):
        from scripts.generate_tool_data import generate_parallel_batch
        samples = generate_parallel_batch(count=5, seed=42)
        assert len(samples) == 5

    def test_all_samples_validate(self):
        from scripts.generate_tool_data import generate_parallel_batch
        samples = generate_parallel_batch(count=5, seed=42)
        for i, sample in enumerate(samples):
            try:
                Conversation.model_validate(sample)
            except Exception as e:
                pytest.fail(f"Sample {i} failed validation: {e}")

    def test_has_parallel_tool_calls(self):
        """At least some samples should have 2+ tool_calls in one assistant message."""
        from scripts.generate_tool_data import generate_parallel_batch
        samples = generate_parallel_batch(count=5, seed=42)
        parallel_count = 0
        for sample in samples:
            for msg in sample["messages"]:
                if msg["role"] == "assistant" and msg.get("tool_calls"):
                    if len(msg["tool_calls"]) >= 2:
                        parallel_count += 1
                        break
        # At least 3 of 5 should have parallel calls (75% happy path)
        assert parallel_count >= 3, f"Only {parallel_count}/5 had 2+ parallel tool_calls"


class TestMcpBatch:
    """Test 5: generate_mcp_batch returns valid MCP discovery conversations."""

    def test_returns_correct_count(self):
        from scripts.generate_tool_data import generate_mcp_batch
        samples = generate_mcp_batch(count=5, seed=42)
        assert len(samples) == 5

    def test_all_samples_validate(self):
        from scripts.generate_tool_data import generate_mcp_batch
        samples = generate_mcp_batch(count=5, seed=42)
        for i, sample in enumerate(samples):
            try:
                Conversation.model_validate(sample)
            except Exception as e:
                pytest.fail(f"Sample {i} failed validation: {e}")

    def test_uses_mcp_assistant_prompt(self):
        from scripts.generate_tool_data import generate_mcp_batch
        prompts = yaml.safe_load(Path("templates/system-prompts.yaml").read_text())
        expected_prompt = prompts["system_prompts"]["mcp_assistant"]["content"].strip()
        samples = generate_mcp_batch(count=5, seed=42)
        for sample in samples:
            assert sample["messages"][0]["role"] == "system"
            assert sample["messages"][0]["content"].strip() == expected_prompt

    def test_uses_mcp_tools(self):
        """Samples should use mcp_list_servers or mcp_list_tools."""
        from scripts.generate_tool_data import generate_mcp_batch
        samples = generate_mcp_batch(count=5, seed=42)
        for sample in samples:
            tool_names = [t["function"]["name"] for t in sample.get("tools", [])]
            has_mcp = any(n.startswith("mcp_") for n in tool_names)
            assert has_mcp, f"No MCP tools found in sample tools: {tool_names}"


class TestSystemMessageFirst:
    """Test 6: All generated samples have system message as first message."""

    def test_single_call_system_first(self):
        from scripts.generate_tool_data import generate_single_call_batch
        for sample in generate_single_call_batch(count=5, seed=42):
            assert sample["messages"][0]["role"] == "system"

    def test_cli_system_first(self):
        from scripts.generate_tool_data import generate_cli_batch
        for sample in generate_cli_batch(count=5, seed=42):
            assert sample["messages"][0]["role"] == "system"

    def test_multi_turn_system_first(self):
        from scripts.generate_tool_data import generate_multi_turn_batch
        for sample in generate_multi_turn_batch(count=5, seed=42):
            assert sample["messages"][0]["role"] == "system"

    def test_parallel_system_first(self):
        from scripts.generate_tool_data import generate_parallel_batch
        for sample in generate_parallel_batch(count=5, seed=42):
            assert sample["messages"][0]["role"] == "system"

    def test_mcp_system_first(self):
        from scripts.generate_tool_data import generate_mcp_batch
        for sample in generate_mcp_batch(count=5, seed=42):
            assert sample["messages"][0]["role"] == "system"


class TestEdgeCases:
    """Test 7: ~25% of samples are edge cases (no-tool-needed)."""

    def test_single_call_has_edge_cases(self):
        from scripts.generate_tool_data import generate_single_call_batch
        samples = generate_single_call_batch(count=8, seed=42)
        no_tool_count = sum(
            1 for s in samples
            if not any(
                msg.get("tool_calls")
                for msg in s["messages"]
                if msg["role"] == "assistant"
            )
        )
        # At least 1 in 8 should be a no-tool edge case
        assert no_tool_count >= 1, f"No edge cases found in 8 samples"

    def test_cli_has_edge_cases(self):
        from scripts.generate_tool_data import generate_cli_batch
        samples = generate_cli_batch(count=8, seed=42)
        no_tool_count = sum(
            1 for s in samples
            if not any(
                msg.get("tool_calls")
                for msg in s["messages"]
                if msg["role"] == "assistant"
            )
        )
        assert no_tool_count >= 1, f"No edge cases found in 8 CLI samples"

    def test_multi_turn_has_edge_cases(self):
        from scripts.generate_tool_data import generate_multi_turn_batch
        samples = generate_multi_turn_batch(count=8, seed=42)
        no_tool_count = sum(
            1 for s in samples
            if not any(
                msg.get("tool_calls")
                for msg in s["messages"]
                if msg["role"] == "assistant"
            )
        )
        assert no_tool_count >= 1, f"No edge cases found in 8 multi-turn samples"


class TestCliEntryPoint:
    """Test 8: CLI entry point writes JSONL file to datasets/tool-calling/ directory."""

    def test_cli_writes_jsonl(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "scripts.generate_tool_data",
             "--category", "single-call", "--count", "3", "--batch", "99"],
            capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        output_path = Path("datasets/tool-calling/single-call-batch-99.jsonl")
        assert output_path.exists(), f"Output file not created: {output_path}"
        # Verify it contains valid JSONL
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 3
        for line in lines:
            data = json.loads(line)
            Conversation.model_validate(data)
        # Cleanup
        output_path.unlink()


class TestSchemaPoolUsage:
    """Test 9: Generated conversations use tools from tool_schemas.yaml."""

    def test_tools_from_schema_pool(self):
        from scripts.generate_tool_data import generate_single_call_batch, load_schemas
        schemas = load_schemas()
        # Collect all known tool names from schema pool
        all_names = set()
        for domain_val in schemas.values():
            if isinstance(domain_val, list):
                for s in domain_val:
                    all_names.add(s["name"])
            elif isinstance(domain_val, dict):
                for subcat_val in domain_val.values():
                    for s in subcat_val:
                        all_names.add(s["name"])

        samples = generate_single_call_batch(count=5, seed=42)
        for sample in samples:
            if sample.get("tools"):
                for tool in sample["tools"]:
                    assert tool["function"]["name"] in all_names, (
                        f"Tool '{tool['function']['name']}' not in schema pool"
                    )


class TestQueryDiversity:
    """Test 10: Generated conversations have diverse user queries."""

    def test_no_duplicate_queries_single_call(self):
        from scripts.generate_tool_data import generate_single_call_batch
        samples = generate_single_call_batch(count=5, seed=42)
        user_messages = []
        for sample in samples:
            for msg in sample["messages"]:
                if msg["role"] == "user":
                    user_messages.append(msg["content"])
        # All user messages should be unique
        assert len(user_messages) == len(set(user_messages)), (
            f"Duplicate user messages found in batch of 5"
        )

    def test_no_duplicate_queries_cli(self):
        from scripts.generate_tool_data import generate_cli_batch
        samples = generate_cli_batch(count=5, seed=42)
        user_messages = []
        for sample in samples:
            for msg in sample["messages"]:
                if msg["role"] == "user":
                    user_messages.append(msg["content"])
        assert len(user_messages) == len(set(user_messages)), (
            f"Duplicate user messages found in CLI batch of 5"
        )
