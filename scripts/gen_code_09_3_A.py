#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""gen_code_09_3_A.py -- Generate 500 ShareGPT Python code samples.

Subagent CODE-A of phase 09.3 (batch-08). Produces a mix of:
  - algorithms / data structures / utilities (~30%)
  - web / API (~25%)
  - data science / numpy / pandas / matplotlib (~20%)
  - debugging / refactoring (~15%)
  - testing / pytest / mocking / async (~10%)

Format mix: 60% 3-msg, 40% 5-msg with follow-up.

Seed: "1009308A" for deterministic ordering / system-prompt rotation.

Output: datasets/code/raw-09.3/batch-08-A-code.jsonl (exactly 500 lines).
"""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "datasets" / "code" / "raw-09.3" / "batch-08-A-code.jsonl"

SYSTEM_PROMPTS = [
    "You are an expert Python programmer.",
    "You are a senior Python developer.",
    "You are a helpful Python assistant.",
    "You write clean, idiomatic Python.",
    "You are a Python expert who explains code clearly.",
]


# Each entry is a dict:
#   {"cat": <category>, "u": user prompt, "a": assistant reply (already includes
#    a fenced code block + short prose), and OPTIONAL "u2", "a2" for 5-msg.}
ENTRIES: list = []


def add(cat, u, a, u2=None, a2=None):
    ENTRIES.append({"cat": cat, "u": u, "a": a, "u2": u2, "a2": a2})


# Load data from companion modules.
from gen_code_09_3_A_data import register_all as r1  # noqa: E402
from gen_code_09_3_A_data2 import register_all as r2  # noqa: E402
from gen_code_09_3_A_data3 import register_all as r3  # noqa: E402
from gen_code_09_3_A_data4 import register_all as r4  # noqa: E402
from gen_code_09_3_A_data5 import register_all as r5  # noqa: E402

for fn in (r1, r2, r3, r4, r5):
    fn(add)


def main():
    seed_int = int(hashlib.md5(b"1009308A").hexdigest(), 16) % (2**32)
    rng = random.Random(seed_int)

    # Stable shuffle so file is reproducible but interleaved.
    entries = list(ENTRIES)
    rng.shuffle(entries)

    if len(entries) < 500:
        raise SystemExit(f"Only {len(entries)} entries — need 500.")
    entries = entries[:500]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    counts = {"algo": 0, "web": 0, "data": 0, "debug": 0, "test": 0}
    fmt = {"3": 0, "5": 0}
    sysprompt_use = {p: 0 for p in SYSTEM_PROMPTS}

    with OUT_PATH.open("w") as f:
        for i, e in enumerate(entries):
            sp = SYSTEM_PROMPTS[i % len(SYSTEM_PROMPTS)]
            sysprompt_use[sp] += 1
            msgs = [
                {"role": "system", "content": sp},
                {"role": "user", "content": e["u"]},
                {"role": "assistant", "content": e["a"]},
            ]
            if e.get("u2") and e.get("a2"):
                msgs.append({"role": "user", "content": e["u2"]})
                msgs.append({"role": "assistant", "content": e["a2"]})
                fmt["5"] += 1
            else:
                fmt["3"] += 1
            counts[e["cat"]] = counts.get(e["cat"], 0) + 1
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")

    print(f"wrote {sum(counts.values())} samples to {OUT_PATH}")
    print(f"category counts: {counts}")
    print(f"format split:    {fmt}")
    print(f"system prompts:  {sysprompt_use}")


if __name__ == "__main__":
    main()
