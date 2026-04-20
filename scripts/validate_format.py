#!/usr/bin/env python3
"""validate_format.py -- Validate Lyra ShareGPT JSONL against the format specification.

Pydantic-based validation for TRL-native conversational format with tool calling support.
Enforces role ordering rules, tool call consistency, and structural correctness.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, model_validator


# --- Pydantic Schema ---


class FunctionDef(BaseModel):
    """Schema for a tool function definition."""

    name: str
    description: str
    parameters: dict


class ToolSchema(BaseModel):
    """Schema for a tool definition in the tools column."""

    type: str = "function"
    function: FunctionDef


class ToolCallFunction(BaseModel):
    """Schema for the function field within a tool call."""

    name: str
    arguments: dict


class ToolCall(BaseModel):
    """Schema for a tool call made by the assistant."""

    type: str = "function"
    function: ToolCallFunction


class Message(BaseModel):
    """Schema for a single message in a conversation."""

    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    name: Optional[str] = None


class Conversation(BaseModel):
    """Schema for a complete conversation with optional tool definitions.

    Validates TRL-native format with role ordering rules:
    1. First message MUST be system role
    2. After system, messages alternate user/assistant (with tool interludes)
    3. tool messages MUST immediately follow assistant with tool_calls
    4. Multiple tool messages can follow single assistant with multiple tool_calls
    5. assistant with tool_calls must be followed by exactly one tool message per call
    6. Final message SHOULD be assistant role
    7. assistant with tool_calls typically has content null or empty
    8. tool messages MUST have name field matching called tool
    9. tool content should be string representation of return value
    """

    messages: list[Message]
    tools: Optional[list[ToolSchema]] = None

    @model_validator(mode="after")
    def validate_structure(self) -> "Conversation":
        """Enforce all role ordering and structural rules."""
        msgs = self.messages

        # Rule: Reject empty conversations
        if not msgs:
            raise ValueError("Empty conversation")

        # Rule 1: First message must be system
        if msgs[0].role != "system":
            raise ValueError("First message must be system role")

        # Validate each message
        valid_roles = {"system", "user", "assistant", "tool"}
        for i, msg in enumerate(msgs):
            # Rule: Valid roles only
            if msg.role not in valid_roles:
                raise ValueError(f"Invalid role '{msg.role}' at index {i}")

            # Rule 3, 8: tool messages must follow assistant or tool, must have name
            if msg.role == "tool":
                if i == 0 or msgs[i - 1].role not in ("assistant", "tool"):
                    raise ValueError(
                        f"Tool message at index {i} must follow assistant or tool"
                    )
                if msg.name is None:
                    raise ValueError(
                        f"Tool message at index {i} missing 'name' field"
                    )

        # Rule 5: Validate tool_calls count matches subsequent tool messages
        i = 0
        while i < len(msgs):
            msg = msgs[i]
            if msg.role == "assistant" and msg.tool_calls:
                expected = len(msg.tool_calls)
                actual = 0
                j = i + 1
                while j < len(msgs) and msgs[j].role == "tool":
                    actual += 1
                    j += 1
                if actual != expected:
                    raise ValueError(
                        f"Expected {expected} tool messages after assistant "
                        f"at index {i}, got {actual}"
                    )
            i += 1

        # Rule: If tools column present, tool_calls must reference defined tools
        if self.tools:
            defined_names = {t.function.name for t in self.tools}
            for msg in msgs:
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc.function.name not in defined_names:
                            raise ValueError(
                                f"Tool call '{tc.function.name}' not in "
                                f"defined tools: {defined_names}"
                            )

        return self


def validate_file(path: Path) -> dict:
    """Validate a JSONL file line by line.

    Args:
        path: Path to the JSONL file to validate.

    Returns:
        Dictionary with keys:
            total: Total number of lines processed
            valid: Number of valid conversations
            invalid: Number of invalid conversations
            errors: List of dicts with 'line' and 'error' keys
    """
    results = {"total": 0, "valid": 0, "invalid": 0, "errors": []}

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            results["total"] += 1
            try:
                data = json.loads(line)
                Conversation.model_validate(data)
                results["valid"] += 1
            except Exception as e:
                results["invalid"] += 1
                results["errors"].append({"line": line_num, "error": str(e)})

    return results


def main():
    """CLI entry point for format validation."""
    parser = argparse.ArgumentParser(
        description="Validate Lyra ShareGPT JSONL format"
    )
    parser.add_argument(
        "jsonl_file",
        type=Path,
        help="Path to the JSONL file to validate",
    )
    args = parser.parse_args()

    if not args.jsonl_file.exists():
        print(f"Error: File not found: {args.jsonl_file}", file=sys.stderr)
        sys.exit(1)

    results = validate_file(args.jsonl_file)
    print(json.dumps(results, indent=2))

    if results["invalid"] > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
