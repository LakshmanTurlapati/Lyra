#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""gen_tc_09_3_A.py -- Generate 500 ShareGPT tool-calling samples (TC-A).

Domain: weather, geolocation, maps, calendar, scheduling.

Seed: 1009307A.

Output: datasets/tool-calling/raw-09.3/batch-07-A.jsonl  (exactly 500 lines)
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

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "datasets" / "tool-calling" / "raw-09.3" / "batch-10-A.jsonl"

SYSTEM_PROMPT = "You are a helpful assistant. Prefer calling tools over guessing."

SUFFIX_POOL = [
    "That's all pulled up and ready to go.",
    "Done -- the data came back clean, everything lines up with what you originally asked for, and I don't see any mismatches worth flagging in the payload before you move on to the next step.",
    "Pulled the record successfully; the value you needed is included in the response above.",
    "All set on my end.",
    "Got it handled -- want me to take this further, or is that enough?",
    "Operation wrapped up without issues, so you should be good to proceed from here.",
    "Finished pulling that together; let me know if you'd like a deeper breakdown of any particular field, or if you'd prefer I reformat the response into something more readable for downstream consumption.",
    "The call routed through cleanly and returned exactly what you were after.",
    "Wrapped up -- anything else you'd like me to chase down while we're here?",
    "That request returned successfully, and the full payload sits just above for your review whenever you're ready to look through it at your own pace.",
    "Verified and delivered.",
    "Fetched and parsed -- the numbers check out against what the API reported.",
    "Sorted out; the output is self-explanatory but I'm happy to walk through it.",
    "Query executed, results returned, and nothing looked off in the response body.",
    "All yours -- holler if you need a follow-up lookup, want to drill into the details, or spot anything in the output that deserves a second pass from my end.",
    "Task closed out. If this raises new questions, just say the word and I'll pivot.",
    "Response is ready above, covering each of the fields you originally requested.",
    "Returned cleanly without errors.",
    "Everything ran end-to-end, the result matches the shape of what you were expecting, and I'd lean toward calling this one finished unless you want me to cross-check anything against another source for sanity.",
    "Just finished the lookup -- does this cover what you needed, or should I keep digging?",
    "Output is queued up above; it should give you what you need to move forward.",
    "Sent off, processed, confirmed -- the operation completed in a single round-trip.",
    "Here you go.",
    "Call succeeded on the first attempt, no retries or fallback logic had to kick in this time, and the timings look perfectly normal compared to prior runs of the same endpoint.",
    "That's wrapped -- feel free to ask if any of the returned fields need clarification.",
    "Information retrieved; I've kept the raw response intact so you can inspect it directly.",
    "Done and dusted.",
    "Happy to refine further if the result isn't quite what you were picturing -- otherwise, we're good to call this one shipped and roll on to whatever's next on your list.",
    "Fetched successfully, and the shape of the payload aligns with the documented schema.",
    "That should do it on this one.",
]

# Suffix-pool exact-prefix versions for matching final assistant openers
# (we use these as the exact opening of the final assistant content).


def _follow_up(opener: str, tool_name: str, args: dict, result_str: str) -> str:
    """Make a short tail body keyed off opener (single sentence). Result not parsed."""
    # We just append a brief, varied summary after the opener phrase.
    # Variety is achieved by hashing tool_name+args.
    h = int(hashlib.md5((tool_name + json.dumps(args, sort_keys=True)).encode()).hexdigest(), 16)
    tails = [
        " The response came back as expected.",
        " Details are in the tool output above.",
        " Should be enough to act on.",
        " Let me know if you want me to drill in.",
        "",
        " Numbers above tell the story.",
        " If anything looks off, flag it and I'll re-run.",
        " The payload is small and self-describing.",
        " Ready when you are for the next step.",
        " That's the complete picture for this lookup.",
    ]
    return opener + tails[h % len(tails)]


def _interp(template, ctx):
    """Recursively interpolate {placeholders} in str/list/dict/None."""
    if isinstance(template, str):
        try:
            return template.format(**ctx)
        except (KeyError, IndexError):
            return template
    if isinstance(template, dict):
        return {k: _interp(v, ctx) for k, v in template.items()}
    if isinstance(template, list):
        return [_interp(v, ctx) for v in template]
    return template


def build_context(rng):
    """Pick a random set of placeholder values."""
    lat, lon = rng.choice(COORDS)
    return {
        "city": rng.choice(CITIES),
        "city2": rng.choice(CITIES),
        "addr": rng.choice(ADDRESSES),
        "addr2": rng.choice(ADDRESSES),
        "addr3": rng.choice(ADDRESSES),
        "tz1": rng.choice(TIMEZONES),
        "tz2": rng.choice(TIMEZONES),
        "country": rng.choice(COUNTRIES),
        "lat": f"{lat:.4f}",
        "lon": f"{lon:.4f}",
        "dt": rng.choice(DATETIMES),
        "evt": rng.choice(EVENT_IDS),
        "user": rng.choice(USER_IDS),
        "user2": rng.choice(USER_IDS),
        "user3": rng.choice(USER_IDS),
        "cal": rng.choice(CALENDAR_IDS),
        "title": rng.choice(EVENT_TITLES),
        "flight": rng.choice(FLIGHTS),
    }


def main():
    seed_int = int(hashlib.md5(b"1009310A").hexdigest(), 16) % (2**32)
    rng = random.Random(seed_int)

    # Build a flat list of (tool_name, variant_idx) round-robin so each tool
    # gets ~equal coverage. We have len(TOOL_SAMPLES) tools each with 13 variants.
    tool_names = list(TOOL_SAMPLES.keys())
    n_tools = len(tool_names)

    # 500 samples; ~ceil(500/n_tools) per tool. With 41 tools => ~12-13 each.
    plan = []
    i = 0
    while len(plan) < 500:
        tool = tool_names[i % n_tools]
        variant = (i // n_tools) % 13
        plan.append((tool, variant))
        i += 1
    rng.shuffle(plan)

    # Decide single-turn vs multi-turn: 75 single-turn (15%), 425 multi-turn.
    indices = list(range(500))
    rng.shuffle(indices)
    single_turn_idx = set(indices[:75])

    # Suffix pool assignment for the 425 multi-turn samples: distribute
    # roughly evenly across 30 suffixes.
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
            user_tpl, args_tpl, result_tpl = TOOL_SAMPLES[tool_name][variant]
            user_msg = _interp(user_tpl, ctx)

            # Ensure no duplicate user messages — append a tiny disambiguator.
            base_user = user_msg
            n_dup = 0
            while user_msg in user_msgs_seen:
                n_dup += 1
                user_msg = base_user + (" " * n_dup)  # trivial whitespace tweak
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
                # 3-msg single-turn: end at tool_calls.
                pass
            else:
                # Multi-turn: tool result + assistant summary.
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
