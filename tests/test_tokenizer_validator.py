"""Tests for tokenizer alignment validation against SmolLM2-1.7B-Instruct.

Validates that conversations are correctly processed by the SmolLM2 tokenizer,
checking token counts, EOS markers, chat template markers, and default system
prompt injection detection.
"""
import json

import pytest
from pathlib import Path

from scripts.validate_tokenizer import validate_conversation, validate_file, load_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    """Load SmolLM2-1.7B-Instruct tokenizer once for all tests in this module."""
    return load_tokenizer()


@pytest.mark.slow
class TestTokenizerValidation:
    """Test suite for tokenizer alignment validation."""

    def test_valid_conversation_passes_tokenizer(self, tokenizer, valid_conversation):
        """Valid basic conversation should pass tokenizer validation with reasonable token count."""
        result = validate_conversation(tokenizer, valid_conversation)
        assert result["valid"] is True
        assert result["token_count"] > 0
        assert result["token_count"] <= 2048
        assert len(result["errors"]) == 0

    def test_valid_tool_call_passes_tokenizer(self, tokenizer, valid_tool_call_conversation):
        """Valid tool call conversation should pass tokenizer validation."""
        result = validate_conversation(tokenizer, valid_tool_call_conversation)
        assert result["valid"] is True
        assert result["token_count"] > 0

    def test_rejects_oversized_conversation(self, tokenizer):
        """Conversation exceeding 2048 tokens should be rejected with specific error."""
        oversized = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "word " * 3000},
                {"role": "assistant", "content": "That is a lot of words."},
            ]
        }
        result = validate_conversation(tokenizer, oversized)
        assert result["valid"] is False
        assert any("exceeds 2048 limit" in err for err in result["errors"])

    def test_detects_chat_template_markers(self, tokenizer, valid_conversation):
        """Tokenized output should contain im_start and im_end markers."""
        result = validate_conversation(tokenizer, valid_conversation)
        assert result["valid"] is True
        # The decoded output should contain the chat template markers
        # Check the full decoded output by running tokenizer directly
        messages = valid_conversation["messages"]
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors=None)
        decoded = tokenizer.decode(token_ids)
        assert "<|im_start|>" in decoded
        assert "<|im_end|>" in decoded

    def test_detects_eos_token(self, tokenizer, valid_conversation):
        """Valid conversation should end with EOS token."""
        messages = valid_conversation["messages"]
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors=None)
        assert token_ids[-1] == tokenizer.eos_token_id

    def test_detects_default_system_prompt_injection(self, tokenizer):
        """Conversation without system message should trigger default prompt detection."""
        no_system = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        result = validate_conversation(tokenizer, no_system)
        # If apply_chat_template succeeds, it will inject the default system prompt
        # and our validation should catch it. If it fails, the error should be captured.
        if result["token_count"] > 0:
            # Template succeeded -- should detect default prompt injection
            assert any("Default system prompt" in err for err in result["errors"])
        else:
            # Template failed -- error should be captured
            assert len(result["errors"]) > 0

    def test_validate_file_returns_token_stats(
        self, tokenizer, valid_conversation, valid_tool_call_conversation, tmp_path
    ):
        """validate_file should return per-line results with token statistics."""
        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps(valid_conversation) + "\n")
            f.write(json.dumps(valid_tool_call_conversation) + "\n")

        result = validate_file(tokenizer, jsonl_path)
        assert result["total"] == 2
        assert result["valid"] == 2
        assert "token_stats" in result
        assert result["token_stats"]["min"] > 0
        assert result["token_stats"]["max"] <= 2048

    def test_validate_file_reports_oversized(
        self, tokenizer, valid_conversation, tmp_path
    ):
        """validate_file should report oversized conversations as invalid."""
        oversized = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "x " * 3000},
                {"role": "assistant", "content": "OK."},
            ]
        }
        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps(valid_conversation) + "\n")
            f.write(json.dumps(oversized) + "\n")

        result = validate_file(tokenizer, jsonl_path)
        assert result["invalid"] >= 1
        assert any(
            "exceeds" in str(r["errors"])
            for r in result["results"]
            if not r["valid"]
        )
