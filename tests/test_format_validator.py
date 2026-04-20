"""Unit tests for Lyra format validation covering DATA-01 requirement.

Tests cover:
- Valid conversations (basic, single tool call, parallel tool calls)
- Invalid conversations (missing system, orphan tool, undefined tool, etc.)
- File-level validation via validate_file()
"""
import json
from pathlib import Path

import pytest

from scripts.validate_format import Conversation, validate_file


def test_valid_conversation_passes(valid_conversation):
    """Valid basic conversation passes model_validate without error."""
    result = Conversation.model_validate(valid_conversation)
    assert len(result.messages) == 3
    assert result.messages[0].role == "system"
    assert result.messages[1].role == "user"
    assert result.messages[2].role == "assistant"


def test_valid_tool_call_passes(valid_tool_call_conversation):
    """Valid single tool call conversation passes validation."""
    result = Conversation.model_validate(valid_tool_call_conversation)
    assert len(result.messages) == 5
    assert result.messages[2].tool_calls is not None
    assert len(result.messages[2].tool_calls) == 1
    assert result.messages[2].tool_calls[0].function.name == "get_weather"


def test_valid_parallel_tool_calls_pass(valid_parallel_tool_call_conversation):
    """Valid parallel tool call conversation passes validation."""
    result = Conversation.model_validate(valid_parallel_tool_call_conversation)
    assert len(result.messages[2].tool_calls) == 2
    assert result.messages[3].role == "tool"
    assert result.messages[4].role == "tool"


def test_rejects_missing_system(invalid_no_system):
    """Conversation without system as first message is rejected."""
    with pytest.raises(ValueError, match="First message must be system role"):
        Conversation.model_validate(invalid_no_system)


def test_rejects_orphan_tool(invalid_orphan_tool):
    """Tool message not following assistant with tool_calls is rejected."""
    with pytest.raises(ValueError, match="must follow assistant or tool"):
        Conversation.model_validate(invalid_orphan_tool)


def test_rejects_undefined_tool(invalid_undefined_tool):
    """Tool call referencing undefined tool name is rejected."""
    with pytest.raises(ValueError, match="not in defined tools"):
        Conversation.model_validate(invalid_undefined_tool)


def test_rejects_empty_messages():
    """Empty messages list is rejected."""
    with pytest.raises(ValueError, match="Empty conversation"):
        Conversation.model_validate({"messages": []})


def test_rejects_invalid_role():
    """Invalid role string is rejected with specific error."""
    data = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "function_call", "content": "bad role"},
        ]
    }
    with pytest.raises(ValueError, match="Invalid role"):
        Conversation.model_validate(data)


def test_rejects_tool_without_name():
    """Tool message without name field is rejected."""
    data = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": {}}}
            ]},
            {"role": "tool", "content": '{"temp": 62}'},
            {"role": "assistant", "content": "Done."},
        ],
        "tools": [
            {"type": "function", "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {}
            }}
        ]
    }
    with pytest.raises(ValueError, match="missing 'name' field"):
        Conversation.model_validate(data)


def test_validate_file_valid(valid_conversation, tmp_path):
    """validate_file on valid JSONL returns all valid, no errors."""
    jsonl_path = tmp_path / "test.jsonl"
    jsonl_path.write_text(json.dumps(valid_conversation) + "\n")
    result = validate_file(jsonl_path)
    assert result["total"] == 1
    assert result["valid"] == 1
    assert result["invalid"] == 0
    assert result["errors"] == []


def test_validate_file_mixed(valid_conversation, tmp_path):
    """validate_file on JSONL with one invalid line reports correct error."""
    invalid = {"messages": []}
    jsonl_path = tmp_path / "test.jsonl"
    jsonl_path.write_text(
        json.dumps(valid_conversation) + "\n" + json.dumps(invalid) + "\n"
    )
    result = validate_file(jsonl_path)
    assert result["total"] == 2
    assert result["valid"] == 1
    assert result["invalid"] == 1
    assert result["errors"][0]["line"] == 2
