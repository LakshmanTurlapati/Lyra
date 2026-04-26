#!/usr/bin/env python3
"""Combine all blocks into final JSONL, validate, and truncate to 160."""
import json
import re
from pathlib import Path

out_path = Path("/Users/lakshman/Documents/Lyra/datasets/code/v2-supplementary/batch-code-04.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)

# Gather all blocks
blocks = [
    "/tmp/gen_batch_code_04_partial.json",
    "/tmp/gen_block_2.json",
    "/tmp/gen_block_3.json",
    "/tmp/gen_block_4.json",
    "/tmp/gen_block_5.json",
    "/tmp/gen_block_6.json",
]

all_samples = []
for p in blocks:
    data = json.loads(Path(p).read_text())
    all_samples.extend(data)

print(f"Total samples before validation: {len(all_samples)}")

# Validate and count follow-ups
valid = []
issues = 0
followup_count = 0

def estimate_tokens(text: str) -> int:
    # rough approximation: 1 token ~ 4 chars
    return max(1, len(text) // 4)

for i, sample in enumerate(all_samples):
    msgs = sample["messages"]
    # Schema checks
    if msgs[0]["role"] != "system":
        print(f"sample {i}: first msg not system"); issues += 1; continue
    if sample.get("domain") != "code":
        print(f"sample {i}: bad domain"); issues += 1; continue
    if sample.get("tools") != []:
        print(f"sample {i}: tools not []"); issues += 1; continue
    # Check no tool_calls
    bad = False
    for m in msgs:
        if "tool_calls" in m:
            print(f"sample {i}: has tool_calls"); issues += 1; bad = True; break
    if bad: continue
    # Check assistant has python block
    assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
    for am in assistant_msgs:
        if "```python" not in am["content"]:
            print(f"sample {i}: assistant missing ```python"); issues += 1; bad = True; break
        if estimate_tokens(am["content"]) < 150:
            print(f"sample {i}: too short ({estimate_tokens(am['content'])} tok)"); issues += 1; bad = True; break
    if bad: continue
    # Count follow-ups
    if len(msgs) == 5:
        followup_count += 1
    valid.append(sample)

print(f"Valid samples: {len(valid)}, issues: {issues}")
print(f"Follow-up samples: {followup_count} ({100*followup_count/len(valid):.1f}%)")

# Truncate to exactly 160
if len(valid) > 160:
    # Keep a balanced selection: preserve first 160 which spans all blocks
    valid = valid[:160]

# Re-count after truncation
followup_count = sum(1 for s in valid if len(s["messages"]) == 5)
print(f"Final: {len(valid)} samples, {followup_count} follow-ups ({100*followup_count/len(valid):.1f}%)")

# Write JSONL
with out_path.open("w") as f:
    for s in valid:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"Written to {out_path}")
