#!/usr/bin/env python3
"""generate_sample.py -- Generate sample conversations for testing the validation pipeline.

Produces 10 hardcoded sample conversations across three domains (tool-calling, code,
knowledge) to verify the format spec and validation scripts work correctly end-to-end.
These are NOT training data -- they are test fixtures for pipeline validation.

Per D-05, actual training data generation happens in Claude Code sessions using the
Anthropic API. This script generates minimal examples to exercise every format feature.
"""
import argparse
import json
import sys
from pathlib import Path

from scripts.validate_format import Conversation


# --- Sample Conversations ---

# Domain: Tool Calling (3 samples)

TOOL_CALL_SINGLE = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant with access to tools. Use them when appropriate."},
        {"role": "user", "content": "What is the weather in San Francisco?"},
        {"role": "assistant", "tool_calls": [
            {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "San Francisco"}}}
        ]},
        {"role": "tool", "name": "get_weather", "content": '{"temp": 62, "condition": "foggy"}'},
        {"role": "assistant", "content": "The weather in San Francisco is 62F and foggy."},
    ],
    "tools": [
        {"type": "function", "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        }}
    ]
}

TOOL_CALL_PARALLEL = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant with access to tools. Use them when needed."},
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

TOOL_CALL_CLI = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant that can run shell commands to help users."},
        {"role": "user", "content": "List all Python files in the current directory."},
        {"role": "assistant", "tool_calls": [
            {"type": "function", "function": {"name": "run_command", "arguments": {"command": "find . -name '*.py' -type f"}}}
        ]},
        {"role": "tool", "name": "run_command", "content": "./main.py\n./utils/helpers.py\n./tests/test_main.py"},
        {"role": "assistant", "content": "There are 3 Python files:\n- ./main.py\n- ./utils/helpers.py\n- ./tests/test_main.py"},
    ],
    "tools": [
        {"type": "function", "function": {
            "name": "run_command",
            "description": "Execute a shell command and return its output",
            "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The shell command to execute"}}, "required": ["command"]}
        }}
    ]
}

# Domain: Code (4 samples)

CODE_REVERSE_STRING = {
    "messages": [
        {"role": "system", "content": "You are a concise coding assistant. Provide working code with minimal explanation."},
        {"role": "user", "content": "Write a Python function to reverse a string."},
        {"role": "assistant", "content": "```python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n```"},
    ]
}

CODE_READ_CSV = {
    "messages": [
        {"role": "system", "content": "You are a concise coding assistant. Provide working code with minimal explanation."},
        {"role": "user", "content": "Write a Python function to read a CSV file and return its contents as a list of dictionaries."},
        {"role": "assistant", "content": "```python\nimport csv\nfrom pathlib import Path\n\n\ndef read_csv(path: str) -> list[dict]:\n    with open(path, newline='') as f:\n        reader = csv.DictReader(f)\n        return list(reader)\n```"},
    ]
}

CODE_FIX_OFF_BY_ONE = {
    "messages": [
        {"role": "system", "content": "You are a concise coding assistant. Help debug code issues with clear explanations."},
        {"role": "user", "content": "This function should return the last N elements of a list, but it has a bug:\n\n```python\ndef last_n(lst, n):\n    return lst[len(lst) - n - 1:]\n```"},
        {"role": "assistant", "content": "The bug is an off-by-one error in the slice start index. `len(lst) - n - 1` skips one extra element.\n\nFixed version:\n```python\ndef last_n(lst, n):\n    return lst[len(lst) - n:]\n```\n\nOr more simply:\n```python\ndef last_n(lst, n):\n    return lst[-n:] if n > 0 else []\n```"},
    ]
}

CODE_FIBONACCI = {
    "messages": [
        {"role": "system", "content": "You are a concise coding assistant. Provide working code with minimal explanation."},
        {"role": "user", "content": "Write an efficient Python function to calculate the nth Fibonacci number."},
        {"role": "assistant", "content": "```python\ndef fibonacci(n: int) -> int:\n    if n < 0:\n        raise ValueError(\"n must be non-negative\")\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n```\n\nO(n) time, O(1) space. Avoids recursion overhead."},
    ]
}

# Domain: Knowledge (3 samples)

KNOWLEDGE_PHOTOSYNTHESIS = {
    "messages": [
        {"role": "system", "content": "You are a knowledgeable assistant. Provide accurate, clear explanations."},
        {"role": "user", "content": "Explain photosynthesis in simple terms."},
        {"role": "assistant", "content": "Photosynthesis is how plants make their own food using sunlight.\n\nThe process:\n1. Plants absorb sunlight through chlorophyll (the green pigment in leaves)\n2. They take in carbon dioxide (CO2) from the air through tiny pores called stomata\n3. They absorb water (H2O) from the soil through their roots\n4. Using light energy, they combine CO2 and H2O to produce glucose (C6H12O6) and oxygen (O2)\n\nThe overall equation: 6CO2 + 6H2O + light energy -> C6H12O6 + 6O2\n\nThe glucose provides energy for the plant to grow, and the oxygen is released into the air -- which is what we breathe."},
    ]
}

KNOWLEDGE_MATH_REASONING = {
    "messages": [
        {"role": "system", "content": "You are a knowledgeable assistant. Show your reasoning step by step."},
        {"role": "user", "content": "A train travels 120 km in 2 hours, then 180 km in 3 hours. What is the average speed for the entire journey?"},
        {"role": "assistant", "content": "Step 1: Find the total distance.\nTotal distance = 120 km + 180 km = 300 km\n\nStep 2: Find the total time.\nTotal time = 2 hours + 3 hours = 5 hours\n\nStep 3: Calculate average speed.\nAverage speed = total distance / total time\nAverage speed = 300 km / 5 hours = 60 km/h\n\nThe average speed for the entire journey is 60 km/h.\n\nNote: This is NOT the average of the two speeds (60 and 60 in this case). In general, average speed is always total distance divided by total time, not the arithmetic mean of individual speeds."},
    ]
}

KNOWLEDGE_TCP_IP = {
    "messages": [
        {"role": "system", "content": "You are a knowledgeable assistant. Provide accurate, clear explanations."},
        {"role": "user", "content": "How does TCP/IP work?"},
        {"role": "assistant", "content": "TCP/IP is the fundamental communication protocol suite of the internet. It has four layers:\n\n1. Application Layer: Where your programs live (HTTP, FTP, DNS, SMTP). Your browser sends an HTTP request here.\n\n2. Transport Layer (TCP): Breaks data into packets, numbers them for ordering, and ensures reliable delivery. TCP establishes a connection using a 3-way handshake (SYN, SYN-ACK, ACK), then sends packets with acknowledgments. Lost packets are retransmitted.\n\n3. Internet Layer (IP): Handles addressing and routing. Each packet gets a source and destination IP address. Routers read these addresses to forward packets toward the destination, potentially via different paths.\n\n4. Network Access Layer: The physical transmission -- Ethernet, Wi-Fi, fiber optics. Converts packets to electrical signals or radio waves.\n\nWhen you visit a website:\n- Your browser creates an HTTP request (Application)\n- TCP breaks it into numbered segments (Transport)\n- IP adds addresses and routes each packet (Internet)\n- Your network card transmits the signals (Network Access)\n\nThe receiving server reverses the process, reassembling packets in order using TCP sequence numbers."},
    ]
}

# All samples grouped by domain
SAMPLES_BY_DOMAIN = {
    "tool-calling": [TOOL_CALL_SINGLE, TOOL_CALL_PARALLEL, TOOL_CALL_CLI],
    "code": [CODE_REVERSE_STRING, CODE_READ_CSV, CODE_FIX_OFF_BY_ONE, CODE_FIBONACCI],
    "knowledge": [KNOWLEDGE_PHOTOSYNTHESIS, KNOWLEDGE_MATH_REASONING, KNOWLEDGE_TCP_IP],
}


def generate_samples() -> list[dict]:
    """Return all 10 sample conversations.

    Returns:
        List of 10 conversation dicts across tool-calling (3), code (4),
        and knowledge (3) domains.
    """
    samples = []
    for domain_samples in SAMPLES_BY_DOMAIN.values():
        samples.extend(domain_samples)
    return samples


def write_samples(output_dir: Path, domain: str, samples: list[dict]) -> Path:
    """Write samples as JSONL to output_dir/domain/samples.jsonl.

    Args:
        output_dir: Base output directory (e.g., datasets/).
        domain: Domain subdirectory name (tool-calling, code, knowledge).
        samples: List of conversation dicts to write.

    Returns:
        Path to the written JSONL file.
    """
    domain_dir = output_dir / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = domain_dir / "samples.jsonl"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    return jsonl_path


def validate_samples(samples: list[dict]) -> dict:
    """Validate samples against the Pydantic format schema.

    Args:
        samples: List of conversation dicts to validate.

    Returns:
        Dict with keys: total, valid, invalid, errors.
    """
    total = len(samples)
    valid = 0
    invalid = 0
    errors = []

    for i, sample in enumerate(samples):
        try:
            Conversation.model_validate(sample)
            valid += 1
        except Exception as e:
            invalid += 1
            errors.append({"index": i, "error": str(e)})

    return {"total": total, "valid": valid, "invalid": invalid, "errors": errors}


def main():
    """CLI entry point for sample generation and validation."""
    parser = argparse.ArgumentParser(
        description="Generate and validate sample conversations for Lyra pipeline testing"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets"),
        help="Base output directory for generated samples (default: datasets)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate samples without writing to disk",
    )
    parser.add_argument(
        "--tokenizer-check",
        action="store_true",
        help="Also run tokenizer validation (loads SmolLM2 tokenizer)",
    )
    args = parser.parse_args()

    # Generate all samples
    samples = generate_samples()
    print(f"Generated {len(samples)} samples across {len(SAMPLES_BY_DOMAIN)} domains")

    # Format validation
    format_result = validate_samples(samples)
    print(
        f"Format validation: {format_result['valid']}/{format_result['total']} valid"
    )
    if format_result["errors"]:
        for err in format_result["errors"]:
            print(f"  Sample {err['index']}: {err['error']}")

    # Tokenizer validation (optional)
    tokenizer_valid = None
    if args.tokenizer_check:
        from scripts.validate_tokenizer import load_tokenizer, validate_conversation

        print("Loading SmolLM2-1.7B-Instruct tokenizer...")
        tokenizer = load_tokenizer()
        tokenizer_valid = 0
        tokenizer_invalid = 0
        for i, sample in enumerate(samples):
            result = validate_conversation(tokenizer, sample)
            if result["valid"]:
                tokenizer_valid += 1
            else:
                tokenizer_invalid += 1
                print(
                    f"  Tokenizer failed sample {i}: {result['errors']}"
                )
        print(
            f"Tokenizer validation: {tokenizer_valid}/{len(samples)} valid"
        )

    # Write samples to disk (unless validate-only)
    if not args.validate_only:
        for domain, domain_samples in SAMPLES_BY_DOMAIN.items():
            path = write_samples(args.output_dir, domain, domain_samples)
            print(f"Wrote {len(domain_samples)} samples to {path}")

    # Summary
    print(f"\nSummary: {len(samples)} total, {format_result['valid']} format-valid", end="")
    if tokenizer_valid is not None:
        print(f", {tokenizer_valid} tokenizer-valid", end="")
    print()

    # Exit code
    if format_result["invalid"] > 0:
        sys.exit(1)
    if tokenizer_valid is not None and tokenizer_valid < len(samples):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
