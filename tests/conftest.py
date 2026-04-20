"""Shared pytest fixtures for Lyra format validation tests.

Provides sample conversations as Python dicts (not Pydantic objects)
so tests validate the dict-to-model path.
"""
import pytest


@pytest.fixture
def valid_conversation():
    """Basic valid conversation: system + user + assistant."""
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Write a Python function to reverse a string."},
            {"role": "assistant", "content": "def reverse_string(s: str) -> str:\n    return s[::-1]"},
        ]
    }


@pytest.fixture
def valid_tool_call_conversation():
    """Valid single tool call conversation with tools column."""
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant with access to tools."},
            {"role": "user", "content": "What is the weather in San Francisco?"},
            {"role": "assistant", "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "San Francisco"}}}
            ]},
            {"role": "tool", "name": "get_weather", "content": '{"temp": 62, "condition": "foggy"}'},
            {"role": "assistant", "content": "It is 62F and foggy in San Francisco."},
        ],
        "tools": [
            {"type": "function", "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
            }}
        ]
    }


@pytest.fixture
def valid_parallel_tool_call_conversation():
    """Valid parallel tool call conversation: 2 tool_calls, 2 tool responses."""
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
            {"role": "user", "content": "Compare the weather in New York and Los Angeles."},
            {"role": "assistant", "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "New York"}}},
                {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Los Angeles"}}},
            ]},
            {"role": "tool", "name": "get_weather", "content": '{"temp": 45, "condition": "rainy"}'},
            {"role": "tool", "name": "get_weather", "content": '{"temp": 78, "condition": "sunny"}'},
            {"role": "assistant", "content": "New York is 45F and rainy, while Los Angeles is 78F and sunny."},
        ],
        "tools": [
            {"type": "function", "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
            }}
        ]
    }


@pytest.fixture
def invalid_no_system():
    """Invalid conversation: first message is user, not system."""
    return {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }


@pytest.fixture
def invalid_orphan_tool():
    """Invalid conversation: tool message follows user, not assistant with tool_calls."""
    return {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "tool", "name": "get_weather", "content": '{"temp": 62}'},
        ]
    }


@pytest.fixture
def invalid_undefined_tool():
    """Invalid conversation: tool_calls reference a tool not defined in tools column."""
    return {
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
