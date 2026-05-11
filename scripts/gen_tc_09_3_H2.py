"""TC-H2 batch 08: 500 tool-calling samples for cloud/devops/iot/media/ecommerce.

Wave 2 — fresh seed, same domain coverage as H but different RNG draw and
a handful of additional tools sourced from gen_tc_09_3_H2_data.py.

Output: /Users/lakshman/Documents/Lyra/datasets/tool-calling/raw-09.3/batch-08-H.jsonl
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

# Import the wave-1 tool catalog and reuse it for the bulk of tools.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_tc_09_3_H import make_tools as make_base_tools  # noqa: E402
from gen_tc_09_3_H2_data import extra_tools  # noqa: E402

SEED = "1009308H"
OUT = Path("/Users/lakshman/Documents/Lyra/datasets/tool-calling/raw-09.3/batch-08-H.jsonl")
TOTAL = 500
SINGLE_TURN_TARGET = 75
MULTI_TURN_TARGET = TOTAL - SINGLE_TURN_TARGET

SYSTEM_PROMPTS = [
    "You are a helpful assistant. Prefer calling tools over guessing.",
    "You are a helpful assistant. Use tools when they would help answer the user.",
    "You are a helpful assistant. Call tools whenever they let you give a precise answer.",
    "You are a helpful assistant. Tools are available; use them rather than fabricating data.",
]

SUFFIX_POOL = [
    "That's all pulled up and ready to go.",
    "Done — the data came back clean, everything lines up with what you originally asked for, and I don't see any mismatches worth flagging in the payload before you move on to the next step.",
    "Pulled the record successfully; the value you needed is included in the response above.",
    "All set on my end.",
    "Got it handled — want me to take this further, or is that enough?",
    "Operation wrapped up without issues, so you should be good to proceed from here.",
    "Finished pulling that together; let me know if you'd like a deeper breakdown of any particular field, or if you'd prefer I reformat the response into something more readable for downstream consumption.",
    "The call routed through cleanly and returned exactly what you were after.",
    "Wrapped up — anything else you'd like me to chase down while we're here?",
    "That request returned successfully, and the full payload sits just above for your review whenever you're ready to look through it at your own pace.",
    "Verified and delivered.",
    "Fetched and parsed — the numbers check out against what the API reported.",
    "Sorted out; the output is self-explanatory but I'm happy to walk through it.",
    "Query executed, results returned, and nothing looked off in the response body.",
    "All yours — holler if you need a follow-up lookup, want to drill into the details, or spot anything in the output that deserves a second pass from my end.",
    "Task closed out. If this raises new questions, just say the word and I'll pivot.",
    "Response is ready above, covering each of the fields you originally requested.",
    "Returned cleanly without errors.",
    "Everything ran end-to-end, the result matches the shape of what you were expecting, and I'd lean toward calling this one finished unless you want me to cross-check anything against another source for sanity.",
    "Just finished the lookup — does this cover what you needed, or should I keep digging?",
    "Output is queued up above; it should give you what you need to move forward.",
    "Sent off, processed, confirmed — the operation completed in a single round-trip.",
    "Here you go.",
    "Call succeeded on the first attempt, no retries or fallback logic had to kick in this time, and the timings look perfectly normal compared to prior runs of the same endpoint.",
    "That's wrapped — feel free to ask if any of the returned fields need clarification.",
    "Information retrieved; I've kept the raw response intact so you can inspect it directly.",
    "Done and dusted.",
    "Happy to refine further if the result isn't quite what you were picturing — otherwise, we're good to call this one shipped and roll on to whatever's next on your list.",
    "Fetched successfully, and the shape of the payload aligns with the documented schema.",
    "That should do it on this one.",
]

BLACKLISTED = (
    "I've gathered all the information",
    "I've completed the task",
    "Here's what I found:",
    "Based on the results,",
    "The results show that",
)


def main():
    rng = random.Random(SEED)
    tools = dict(make_base_tools())
    tools.update(extra_tools())
    tool_names = list(tools.keys())
    assert len(tool_names) >= 40, f"Need >=40 tools, have {len(tool_names)}"

    OUT.parent.mkdir(parents=True, exist_ok=True)

    indices = list(range(TOTAL))
    rng.shuffle(indices)
    single_set = set(indices[:SINGLE_TURN_TARGET])

    cap_per_tool = max(1, int(TOTAL * 0.05))  # 25
    tool_counts = Counter()
    suffix_counts = Counter()
    suffix_target = MULTI_TURN_TARGET // len(SUFFIX_POOL)
    suffix_extra = MULTI_TURN_TARGET - suffix_target * len(SUFFIX_POOL)

    suffix_queue = []
    for s in SUFFIX_POOL:
        suffix_queue.extend([s] * suffix_target)
    extras = rng.sample(SUFFIX_POOL, suffix_extra)
    suffix_queue.extend(extras)
    rng.shuffle(suffix_queue)

    samples = []
    suffix_iter = iter(suffix_queue)

    def pick_tool():
        candidates = [t for t in tool_names if tool_counts[t] < cap_per_tool]
        if not candidates:
            return rng.choice(tool_names)
        return rng.choice(candidates)

    for i in range(TOTAL):
        tname = pick_tool()
        tool_counts[tname] += 1
        desc, builder = tools[tname]
        user_text, args, result_str, summary = builder(rng)
        sysprompt = rng.choice(SYSTEM_PROMPTS)

        if i in single_set:
            messages = [
                {"role": "system", "content": sysprompt},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"type": "function", "function": {"name": tname, "arguments": args}}
                ]},
            ]
        else:
            suffix = next(suffix_iter)
            for bl in BLACKLISTED:
                assert not suffix.startswith(bl)
            final = f"{suffix} {summary}"
            messages = [
                {"role": "system", "content": sysprompt},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"type": "function", "function": {"name": tname, "arguments": args}}
                ]},
                {"role": "tool", "name": tname, "content": result_str},
                {"role": "assistant", "content": final},
            ]
            suffix_counts[suffix] += 1

        sample = {
            "messages": messages,
            "tools": [{
                "type": "function",
                "function": {
                    "name": tname,
                    "description": desc,
                    "parameters": {"type": "object", "properties": {k: {"type": "string"} for k in args}, "required": list(args.keys())},
                },
            }],
            "domain": "tool-calling",
        }
        samples.append(sample)

    with OUT.open("w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    line_count = sum(1 for _ in OUT.open())
    distinct_tools = len({s["messages"][2]["tool_calls"][0]["function"]["name"] for s in samples})
    single_count = sum(1 for s in samples if len(s["messages"]) == 3)
    suffix_used = len(suffix_counts)
    max_tool, max_n = tool_counts.most_common(1)[0]
    pct = max_n / TOTAL * 100

    print(f"Wrote {OUT}")
    print(f"line_count={line_count}")
    print(f"distinct_tools={distinct_tools}")
    print(f"single_turn={single_count}  multi_turn={TOTAL - single_count}")
    print(f"max_tool={max_tool} count={max_n} ({pct:.1f}%)  cap={cap_per_tool}")
    print(f"suffix_pool_coverage={suffix_used}/{len(SUFFIX_POOL)}")
    print("suffix_distribution:", sorted(suffix_counts.values()))


if __name__ == "__main__":
    main()
