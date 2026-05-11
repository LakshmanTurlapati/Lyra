#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""gen_tc_09_3_A2.py -- Generate 500 ShareGPT tool-calling samples (TC-A2, Wave 2).

Domain: weather, geolocation, maps, calendar, scheduling.

Seed: 1009308A.

Output: datasets/tool-calling/raw-09.3/batch-08-A.jsonl  (exactly 500 lines)

Reuses tool catalog and helpers from gen_tc_09_3_A.py / gen_tc_09_3_A_data.py.
"""
import hashlib
import json
import random
from pathlib import Path

from gen_tc_09_3_A_data import (
    TOOL_SAMPLES,
    CITIES,
    ADDRESSES,
    TIMEZONES,
    COUNTRIES,
    COORDS,
    DATETIMES,
    EVENT_IDS,
    USER_IDS,
    CALENDAR_IDS,
    EVENT_TITLES,
    FLIGHTS,
)
from gen_tc_09_3_A import (
    SYSTEM_PROMPT,
    SUFFIX_POOL,
    _follow_up,
    _interp,
    build_context,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "datasets" / "tool-calling" / "raw-09.3" / "batch-08-A.jsonl"


def main():
    seed_int = int(hashlib.md5(b"1009308A").hexdigest(), 16) % (2**32)
    rng = random.Random(seed_int)

    tool_names = list(TOOL_SAMPLES.keys())
    n_tools = len(tool_names)
    rng.shuffle(tool_names)  # different ordering than wave 1

    plan = []
    i = 0
    while len(plan) < 500:
        tool = tool_names[i % n_tools]
        variant = (i // n_tools + (i * 7 % 13)) % 13  # mix variants
        plan.append((tool, variant))
        i += 1
    rng.shuffle(plan)

    indices = list(range(500))
    rng.shuffle(indices)
    single_turn_idx = set(indices[:75])

    # Suffix pool: distribute roughly evenly across 30 suffixes for 425 multi-turn.
    suffix_assignments = []
    for k in range(425):
        suffix_assignments.append(SUFFIX_POOL[k % 30])
    rng.shuffle(suffix_assignments)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    suffix_iter = iter(suffix_assignments)

    written = 0
    user_msgs_seen = set()
    with OUT_PATH.open("w") as f:
        for idx, (tool_name, variant) in enumerate(plan):
            ctx = build_context(rng)
            variants = TOOL_SAMPLES[tool_name]
            user_tpl, args_tpl, result_tpl = variants[variant % len(variants)]
            user_msg = _interp(user_tpl, ctx)

            base_user = user_msg
            n_dup = 0
            while user_msg in user_msgs_seen:
                n_dup += 1
                user_msg = base_user + (" " * n_dup)
            user_msgs_seen.add(user_msg)

            args = _interp(args_tpl, ctx)

            assistant_tool_call = {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "type": "function",
                    "function": {"name": tool_name, "arguments": args},
                }],
            }

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                assistant_tool_call,
            ]

            if idx in single_turn_idx:
                pass
            else:
                result_str = _interp(result_tpl, ctx) if result_tpl else _interp(
                    '{"status":"ok"}', ctx)
                messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": result_str,
                })
                opener = next(suffix_iter)
                final_text = _follow_up(opener, tool_name, args, result_str)
                messages.append({"role": "assistant", "content": final_text})

            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written} samples to {OUT_PATH}")


if __name__ == "__main__":
    main()
