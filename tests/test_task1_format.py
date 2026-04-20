"""TDD RED tests for Task 1 behavior -- format validation schema."""
import json
import tempfile
from pathlib import Path

import pytest

from scripts.validate_format import Conversation, Message, ToolCall, ToolSchema, validate_file


class TestBasicConversation:
    """Test valid basic conversation passes validation."""

    def test_valid_basic_conversation(self):
        data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        result = Conversation.model_validate(data)
        assert len(result.messages) == 3


class TestToolCallConversation:
    """Test valid tool call conversations pass validation."""

    def test_valid_single_tool_call(self):
        data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant with tools."},
                {"role": "user", "content": "What is the weather?"},
                {"role": "assistant", "tool_calls": [
                    {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "SF"}}}
                ]},
                {"role": "tool", "name": "get_weather", "content": '{"temp": 62}'},
                {"role": "assistant", "content": "It is 62F."},
            ],
            "tools": [
                {"type": "function", "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                }}
            ]
        }
        result = Conversation.model_validate(data)
        assert len(result.messages) == 5

    def test_valid_parallel_tool_calls(self):
        data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Compare weather in NYC and LA."},
                {"role": "assistant", "tool_calls": [
                    {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "New York"}}},
                    {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Los Angeles"}}},
                ]},
                {"role": "tool", "name": "get_weather", "content": '{"temp": 45}'},
                {"role": "tool", "name": "get_weather", "content": '{"temp": 78}'},
                {"role": "assistant", "content": "NYC is 45F, LA is 78F."},
            ],
            "tools": [
                {"type": "function", "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                }}
            ]
        }
        result = Conversation.model_validate(data)
        assert len(result.messages[2].tool_calls) == 2


class TestInvalidConversations:
    """Test invalid conversations are rejected with correct errors."""

    def test_rejects_missing_system(self):
        data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        }
        with pytest.raises(ValueError, match="First message must be system role"):
            Conversation.model_validate(data)

    def test_rejects_orphan_tool(self):
        data = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "tool", "name": "get_weather", "content": '{"temp": 62}'},
            ]
        }
        with pytest.raises(ValueError, match="must follow assistant or tool"):
            Conversation.model_validate(data)

    def test_rejects_undefined_tool(self):
        data = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "tool_calls": [
                    {"type": "function", "function": {"name": "nonexistent_tool", "arguments": {}}}
                ]},
                {"role": "tool", "name": "nonexistent_tool", "content": "result"},
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
        with pytest.raises(ValueError, match="not in defined tools"):
            Conversation.model_validate(data)

    def test_rejects_tool_without_name(self):
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

    def test_rejects_empty_messages(self):
        data = {"messages": []}
        with pytest.raises(ValueError, match="Empty conversation"):
            Conversation.model_validate(data)

    def test_rejects_invalid_role(self):
        data = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "function_call", "content": "bad role"},
            ]
        }
        with pytest.raises(ValueError, match="Invalid role"):
            Conversation.model_validate(data)


class TestValidateFile:
    """Test validate_file on JSONL files."""

    def test_validate_file_valid(self, tmp_path):
        data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        jsonl_path = tmp_path / "test.jsonl"
        jsonl_path.write_text(json.dumps(data) + "\n")
        result = validate_file(jsonl_path)
        assert result["total"] == 1
        assert result["valid"] == 1
        assert result["invalid"] == 0

    def test_validate_file_mixed(self, tmp_path):
        valid = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        invalid = {"messages": []}
        jsonl_path = tmp_path / "test.jsonl"
        jsonl_path.write_text(json.dumps(valid) + "\n" + json.dumps(invalid) + "\n")
        result = validate_file(jsonl_path)
        assert result["invalid"] == 1
        assert result["errors"][0]["line"] == 2
