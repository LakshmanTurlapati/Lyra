#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""validate_tokenizer.py -- Validate conversations against SmolLM2-1.7B-Instruct tokenizer.

Catches the most dangerous failure mode in fine-tuning: data that looks structurally
correct but tokenizes incorrectly. Validates token count limits, EOS presence,
chat template markers, and default system prompt injection.
"""
import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


# --- Constants ---

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# Training sequence length limit. Intentionally conservative vs SmolLM2's 8192 context:
# - Shorter samples train faster and with less memory
# - 2048 matches SmolLM2's native training sequence length
# - Can be overridden via --max-tokens CLI flag for longer samples in future phases
MAX_TOKENS = 2048


def load_tokenizer(model_id: str = MODEL_ID) -> AutoTokenizer:
    """Load the SmolLM2-1.7B-Instruct tokenizer.

    Args:
        model_id: HuggingFace model identifier. Defaults to SmolLM2-1.7B-Instruct.

    Returns:
        The loaded tokenizer instance.
    """
    return AutoTokenizer.from_pretrained(model_id)


def _prepare_messages_for_template(messages: list[dict], tools: list[dict] | None) -> list[dict]:
    """Convert TRL-native messages to SmolLM2 chat template compatible format.

    SmolLM2's chat template is simple concatenation: <|im_start|>{role}\\n{content}<|im_end|>
    It does NOT natively handle structured tool_calls or tools columns.
    This function converts:
    - assistant messages with tool_calls -> content with <tool_call> XML delimiters
    - tool messages -> content prefixed with tool name for clarity
    - tools column -> injected into system prompt as JSON tool definitions

    Args:
        messages: List of TRL-native message dicts.
        tools: Optional list of tool schema dicts.

    Returns:
        List of messages compatible with SmolLM2's chat template.
    """
    prepared = []
    for i, msg in enumerate(messages):
        new_msg = {"role": msg["role"]}

        if msg["role"] == "system" and i == 0 and tools:
            # Inject tool definitions into system prompt
            tools_json = json.dumps([t["function"] for t in tools], indent=2)
            base_content = msg.get("content") or ""
            new_msg["content"] = (
                f"{base_content}\n\n"
                f"You have access to the following tools:\n{tools_json}\n\n"
                f"To call a tool, output: <tool_call>[{{\"name\": \"func\", \"arguments\": {{...}}}}]</tool_call>"
            )
        elif msg["role"] == "assistant" and msg.get("tool_calls"):
            # Convert structured tool_calls to <tool_call> XML format
            calls = []
            for tc in msg["tool_calls"]:
                calls.append({
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                })
            new_msg["content"] = f"<tool_call>{json.dumps(calls)}</tool_call>"
        else:
            new_msg["content"] = msg.get("content") or ""

        prepared.append(new_msg)

    return prepared


def validate_conversation(tokenizer, conversation: dict, max_tokens: int = MAX_TOKENS) -> dict:
    """Validate a conversation dict against the SmolLM2 tokenizer.

    Checks that the conversation tokenizes correctly, fits within the token limit,
    ends with EOS, contains chat template markers, and does not trigger default
    system prompt injection.

    Conversations with tool_calls are pre-processed to SmolLM2's native format
    (<tool_call> XML delimiters) before tokenization, since the chat template
    does not handle structured tool calls.

    Args:
        tokenizer: A loaded HuggingFace tokenizer instance.
        conversation: Dict with "messages" key (list of role/content dicts)
            and optional "tools" key (list of tool schema dicts).
        max_tokens: Maximum allowed token count. Defaults to MAX_TOKENS (2048).

    Returns:
        Dict with keys:
            valid (bool): Whether the conversation passes all checks.
            token_count (int): Number of tokens after apply_chat_template.
            errors (list[str]): List of error descriptions.
            decoded_preview (str): First 200 chars of decoded tokenized output.
    """
    errors = []
    messages = conversation.get("messages", [])
    tools = conversation.get("tools")

    # Pre-process messages for SmolLM2's simple chat template
    prepared_messages = _prepare_messages_for_template(messages, tools)

    # Apply chat template -- use return_dict=True to get flat input_ids list
    # (return_tensors=None on transformers 5.x returns BatchEncoding, not list)
    try:
        encoded = tokenizer.apply_chat_template(
            prepared_messages, tokenize=True, return_dict=True
        )
        token_ids = encoded["input_ids"]
    except Exception as e:
        return {
            "valid": False,
            "token_count": 0,
            "errors": [f"Template error: {e}"],
            "decoded_preview": "",
        }

    # Check token count
    token_count = len(token_ids)
    if token_count > max_tokens:
        errors.append(
            f"Token count {token_count} exceeds {max_tokens} limit"
        )

    # Check EOS token presence at end of conversation.
    # SmolLM2 chat template ends with <|im_end|>\n, so the last token is \n.
    # The EOS token (id=2, same as <|im_end|>) appears as second-to-last.
    # Check that EOS is present in the final N tokens (more robust than stripping
    # trailing whitespace tokens, which can incorrectly remove the EOS token itself
    # if eos_id decodes to an empty or whitespace-only string).
    eos_id = tokenizer.eos_token_id
    # EOS should be among the last 3 tokens for SmolLM2's template
    if eos_id not in token_ids[-3:]:
        errors.append(f"Missing EOS token (expected id={eos_id})")

    # Decode for content checks
    decoded = tokenizer.decode(token_ids)

    # Check for default system prompt injection
    default_prompt_marker = "You are a helpful AI assistant named SmolLM"
    messages_str = str(messages)
    if default_prompt_marker in decoded and "SmolLM" not in messages_str:
        errors.append(
            "Default system prompt injected -- explicit system message missing or ignored"
        )

    # Check chat template role markers
    for msg in messages:
        role = msg.get("role", "")
        marker = f"<|im_start|>{role}"
        if marker not in decoded:
            errors.append(f"Missing role marker for '{role}'")

    return {
        "valid": len(errors) == 0,
        "token_count": token_count,
        "errors": errors,
        "decoded_preview": decoded[:200],
    }


def validate_file(tokenizer, path: Path, max_tokens: int = MAX_TOKENS) -> dict:
    """Validate a JSONL file of conversations against the tokenizer.

    Reads the file line by line, validates each conversation, and computes
    aggregate token statistics.

    Args:
        tokenizer: A loaded HuggingFace tokenizer instance.
        path: Path to the JSONL file to validate.
        max_tokens: Maximum allowed token count per conversation. Defaults to MAX_TOKENS.

    Returns:
        Dict with keys:
            total (int): Total number of conversations processed.
            valid (int): Number of valid conversations.
            invalid (int): Number of invalid conversations.
            results (list[dict]): Per-line results with line number, valid, token_count, errors.
            token_stats (dict): Min, max, mean token counts from valid conversations.
    """
    results = []
    total = 0
    valid_count = 0
    invalid_count = 0
    valid_token_counts = []

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                invalid_count += 1
                results.append({
                    "line": line_num,
                    "valid": False,
                    "token_count": 0,
                    "errors": [f"JSON parse error: {e}"],
                })
                continue

            result = validate_conversation(tokenizer, data, max_tokens=max_tokens)
            line_result = {
                "line": line_num,
                "valid": result["valid"],
                "token_count": result["token_count"],
                "errors": result["errors"],
            }
            results.append(line_result)

            if result["valid"]:
                valid_count += 1
                valid_token_counts.append(result["token_count"])
            else:
                invalid_count += 1

    # Compute token statistics from valid conversations
    token_stats = {"min": 0, "max": 0, "mean": 0.0}
    if valid_token_counts:
        token_stats = {
            "min": min(valid_token_counts),
            "max": max(valid_token_counts),
            "mean": round(sum(valid_token_counts) / len(valid_token_counts), 1),
        }

    return {
        "total": total,
        "valid": valid_count,
        "invalid": invalid_count,
        "results": results,
        "token_stats": token_stats,
    }


def main():
    """CLI entry point for tokenizer validation."""
    parser = argparse.ArgumentParser(
        description="Validate conversations against SmolLM2-1.7B-Instruct tokenizer"
    )
    parser.add_argument(
        "jsonl_file",
        type=Path,
        help="Path to the JSONL file to validate",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_ID,
        help=f"HuggingFace model ID for tokenizer (default: {MODEL_ID})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help=f"Maximum token count per conversation (default: {MAX_TOKENS})",
    )
    args = parser.parse_args()

    if not args.jsonl_file.exists():
        print(f"Error: File not found: {args.jsonl_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading tokenizer: {args.model}")
    tokenizer = load_tokenizer(args.model)

    print(f"Validating: {args.jsonl_file} (max_tokens={args.max_tokens})")
    results = validate_file(tokenizer, args.jsonl_file, max_tokens=args.max_tokens)

    # Print summary
    print(f"\nResults: {results['valid']}/{results['total']} valid")
    if results["token_stats"]["min"] > 0:
        stats = results["token_stats"]
        print(
            f"Token stats: min={stats['min']}, max={stats['max']}, "
            f"mean={stats['mean']}"
        )

    # Print errors if any
    if results["invalid"] > 0:
        print(f"\nInvalid conversations ({results['invalid']}):")
        for r in results["results"]:
            if not r["valid"]:
                print(f"  Line {r['line']}: {r['errors']}")

    # Output full JSON to stdout
    print(f"\n{json.dumps(results, indent=2)}")

    if results["invalid"] > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
