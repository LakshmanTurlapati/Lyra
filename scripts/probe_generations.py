#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""probe_generations.py -- One-off diagnostic runner for D-03 + D-05 (Phase 09.2).

Combines the dataset-mode raw-generation probe (D-03) and the single-turn
hand-picked probe (D-05) behind one argparse CLI. Outputs per-sample rows to
JSONL with an auto-assigned `category` tag; a human then finalizes the
categorization table in ``D03-D05-CATEGORIZATION.md`` + ``.json``.

Usage (D-11 — run sequentially, never in parallel shells):

    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m scripts.probe_generations \\
        --model models/lyra-merged \\
        --mode dataset \\
        --limit 20 \\
        --output results/v2_probe_raw.jsonl

    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m scripts.probe_generations \\
        --model models/lyra-merged \\
        --mode single-turn \\
        --output results/v2_probe_single_turn.jsonl

Generation kwargs follow Pattern 3 of 09.2-RESEARCH.md (greedy / fp32 on MPS;
no ``repetition_penalty`` so the echo-tool-result failure mode is not masked).

Threat mitigations:
  T-09.2-01: ``--model`` arg validated against ``MODEL_PATH_PATTERN`` before
             any heavy import or library call (mirrors eval_inference.py pattern).
  T-09.2-05: Model output parsed with ``json.loads`` in try/except inside
             ``categorize()``; output is never ``exec``/``eval``-ed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

# MPS fallback must be set before torch is imported anywhere in the process.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Model path validation pattern per T-09.2-01 / T-03-07 (mirrors eval_runner.py
# and eval_inference.py). Applied before any library call in main().
MODEL_PATH_PATTERN = re.compile(r"^[a-zA-Z0-9._/~\-]+$")

# Hand-picked single-turn probe prompts (D-05). These are fixed per the plan so
# any future re-run is byte-comparable with prior runs. Coverage: weather /
# time / search / explicit-tool-hint / CLI / git / file-read.
PROBES: list[str] = [
    "What's the weather in New York?",
    "What time is it in Tokyo?",
    "Search for the latest news about AI.",
    "Call get_weather for London.",
    "I need the current temperature in Paris. Use the weather tool.",
    "Run `ls -la` in the current directory.",
    "Commit my changes with message 'fix regression'.",
    "Read the contents of /etc/hosts.",
]

# System prompt for the single-turn probe. Mirrors the plan exactly so the probe
# is reproducible: available tool signatures are listed so a well-aligned model
# knows what to emit.
SINGLE_TURN_SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to tools. When a tool is needed, "
    "respond with <tool_call>[{...}]</tool_call>. Available tools: "
    "get_weather(location), get_time(location), search_web(query), "
    "run_command(cmd), git_commit(message), read_file(path)."
)


def categorize(generated: str, expected_content: str) -> str:
    """Assign a coarse failure-mode category to a raw model generation.

    Category ladder (first match wins):

    1. ``valid_tool_call`` -- contains ``<tool_call>...</tool_call>`` whose
       interior parses as JSON.
    2. ``malformed_tool_call_json`` -- has the XML wrapper but the interior is
       not valid JSON.
    3. ``empty`` -- nothing but whitespace.
    4. ``echo_tool_result_prefix`` -- starts with the ``[Tool result from X]:``
       prefix that the eval-time normalization injects into the user side; v2
       frequently echoes this verbatim in its assistant turn.
    5. ``echoed_expected_text`` -- contains the first 30 characters of the
       expected label verbatim (train-set memorization).
    6. ``incomplete_tool_call`` -- ``<tool_call>`` appears but the closing tag
       does not (truncation or partial XML).
    7. ``plain_text`` -- default bucket.

    Args:
        generated: Raw decoded model output (new tokens only).
        expected_content: Canonical label content (may be empty for
            single-turn probes that have no reference).

    Returns:
        One of the seven category strings above.
    """
    m = re.search(r"<tool_call>(.*?)</tool_call>", generated, re.DOTALL)
    if m:
        try:
            # T-09.2-05: parse-only; never exec/eval on model output.
            json.loads(m.group(1))
            return "valid_tool_call"
        except json.JSONDecodeError:
            return "malformed_tool_call_json"
    if not generated.strip():
        return "empty"
    if generated.lstrip().startswith("[Tool result from"):
        return "echo_tool_result_prefix"
    if expected_content and expected_content[:30] and expected_content[:30] in generated:
        return "echoed_expected_text"
    if "<tool_call>" in generated:
        return "incomplete_tool_call"
    return "plain_text"


def normalize_messages(msgs: list[dict]) -> list[dict]:
    """Local fallback mirror of ``_normalize_messages_for_template``.

    Kept byte-identical in behaviour to ``scripts/eval_inference.py`` lines
    147-188 so the probe's prompt-rendering path matches the eval path and
    any divergence here is a bug, not a confound. Used only if the import of
    the eval-side helper fails (circular-import / mis-scoped name); see
    ``_load_normalize_helper``.
    """
    out: list[dict] = []
    for msg in msgs:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "assistant" and not content and msg.get("tool_calls"):
            tool_calls_json = json.dumps(msg["tool_calls"])
            out.append({
                "role": "assistant",
                "content": f"<tool_call>{tool_calls_json}</tool_call>",
            })
        elif role == "tool":
            tool_name = msg.get("name", "tool")
            out.append({
                "role": "user",
                "content": f"[Tool result from {tool_name}]: {content or ''}",
            })
        else:
            out.append({"role": role, "content": content})
    return out


def _load_normalize_helper():
    """Return ``_normalize_messages_for_template`` from the eval module if
    importable, otherwise fall back to the local ``normalize_messages``.

    The local fallback is intentionally byte-identical behaviour to the
    eval helper; we prefer the shared import so any future fix to the eval
    path is picked up here automatically.
    """
    try:
        from scripts.eval_inference import _normalize_messages_for_template
        return _normalize_messages_for_template
    except Exception:  # noqa: BLE001 -- broad on purpose: any import failure falls back
        return normalize_messages


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser (factored out so --help works without torch).

    Lazy-imports (torch, transformers, datasets) happen inside ``main`` only
    AFTER argparse returns, so ``--help`` works on any machine with a stdlib
    Python regardless of ML-lib installation state (Phase 8 pattern).
    """
    parser = argparse.ArgumentParser(
        prog="probe_generations",
        description=(
            "One-off diagnostic probe for Phase 09.2 D-03 (dataset-mode raw "
            "generations) and D-05 (single-turn hand-picked probes). Writes "
            "per-sample JSONL rows with an auto-assigned category for human "
            "review."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Local model path or HuggingFace model ID. Validated against "
            "MODEL_PATH_PATTERN before any library call (T-09.2-01)."
        ),
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["dataset", "single-turn"],
        help=(
            "dataset = iterate datasets/assembled/test tool-calling samples "
            "up to --limit; single-turn = run the 8 hand-picked probes."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max tool-calling samples to probe in dataset mode (default: 20).",
    )
    parser.add_argument(
        "--dataset-dir",
        default="datasets/assembled",
        help="Path to assembled dataset dir (default: datasets/assembled).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file path (parent dirs created as needed).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help=(
            "Greedy generation max new tokens (default: 256, matches "
            "eval_inference.py)."
        ),
    )
    return parser


def main() -> int:
    """CLI entry point. Returns 0 on success, 2 on argument/validation failure."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    # T-09.2-01: validate --model BEFORE any heavy import. Exit 2 = usage error.
    if not MODEL_PATH_PATTERN.match(args.model):
        print(
            f"Error: invalid --model {args.model!r}. Must match "
            f"pattern {MODEL_PATH_PATTERN.pattern}",
            file=sys.stderr,
        )
        return 2

    # Lazy imports (Phase 8 pattern — enables `--help` without torch installed,
    # and keeps argparse usage errors fast).
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[probe] loading tokenizer + model from {args.model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Mirror eval_inference.py: fall back to chat_template.jinja if the
    # tokenizer_config.json does not carry chat_template inline (D-03 fix).
    if tokenizer.chat_template is None:
        jinja_path = Path(args.model) / "chat_template.jinja"
        if jinja_path.exists():
            tokenizer.chat_template = jinja_path.read_text()
            print(
                f"[probe] loaded chat_template from {jinja_path} "
                f"(not in tokenizer_config)",
                file=sys.stderr,
            )

    # Pattern 3 defaults: fp32 on MPS — bf16/fp16 numerics can mask or
    # introduce failure modes. PYTORCH_ENABLE_MPS_FALLBACK=1 already set.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
    ).to("mps")
    model.eval()

    normalize = _load_normalize_helper()
    out_rows: list[dict] = []

    if args.mode == "dataset":
        from datasets import load_from_disk

        print(f"[probe] loading dataset from {args.dataset_dir}", file=sys.stderr)
        ds = load_from_disk(args.dataset_dir)
        test_split = ds["test"]

        count = 0
        for i, sample in enumerate(test_split):
            if sample.get("domain") != "tool-calling":
                continue
            count += 1
            if count > args.limit:
                break

            messages = sample["messages"]
            # Label is the final assistant turn; prompt is everything before it.
            last_asst_idx = max(
                j for j, m in enumerate(messages) if m["role"] == "assistant"
            )
            prompt_msgs = normalize(messages[:last_asst_idx])
            expected = messages[last_asst_idx]

            prompt = tokenizer.apply_chat_template(
                prompt_msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to("mps")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,  # greedy / deterministic, Pattern 3
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            generated = tokenizer.decode(new_tokens, skip_special_tokens=True)

            expected_content = expected.get("content") or ""
            row = {
                "idx": i,
                "expected_content": expected_content[:200],
                "expected_has_tool_calls": bool(expected.get("tool_calls")),
                "generated": generated,
                "category": categorize(generated, expected_content),
            }
            out_rows.append(row)
            print(
                f"[probe] dataset {count}/{args.limit} idx={i} "
                f"category={row['category']}",
                file=sys.stderr,
            )

    else:  # single-turn
        for p_text in PROBES:
            msgs = [
                {"role": "system", "content": SINGLE_TURN_SYSTEM_PROMPT},
                {"role": "user", "content": p_text},
            ]
            prompt = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to("mps")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,  # greedy / deterministic, Pattern 3
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            generated = tokenizer.decode(new_tokens, skip_special_tokens=True)

            row = {
                "user": p_text,
                "generated": generated,
                "category": categorize(generated, ""),
            }
            out_rows.append(row)
            print(
                f"[probe] single-turn {p_text!r} -> {row['category']}",
                file=sys.stderr,
            )

    # Write JSONL artifact. Parent dirs created on demand.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")

    # Category histogram for quick review at end of run.
    cats = Counter(r["category"] for r in out_rows)
    print("=== Category histogram ===")
    for cat, n in cats.most_common():
        print(f"  {cat:30s} {n}")
    print(f"[probe] wrote {len(out_rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
