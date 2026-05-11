#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""gen_knowledge_09_3_A.py -- Generate 500 ShareGPT knowledge samples (science+math).

Subagent A of phase 09.3-01. Sibling B and C handle other categories.

Seed: 1009031A (deterministic shuffle of all (cat, topic, sub_angle) tuples
restricted to categories 'science' and 'math', then take first 500).

Each sample is a 3-message ShareGPT dict written to:
  datasets/knowledge/raw-09.3/batch-08-A-knowledge.jsonl
"""
import hashlib
import json
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TOPICS_PATH = ROOT / "templates" / "knowledge-topics-v2.yaml"
OUT_PATH = ROOT / "datasets" / "knowledge" / "raw-09.3" / "batch-08-A-knowledge.jsonl"

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Provide accurate, clear answers.\n"
    "For complex topics, break your explanation into steps.\n"
    "For simple factual questions, be concise."
)

# Per-sub_angle question templates (varied phrasings, picked deterministically by topic hash)
QUESTION_TEMPLATES = {
    "definition":   ["What is {t}?", "Explain what {t} means.", "Define {t}.", "Can you describe what {t} refers to?", "In simple terms, what is {t}?"],
    "mechanism":    ["How does {t} work?", "Explain the mechanism behind {t}.", "What's the underlying process of {t}?", "Walk me through how {t} actually works.", "How does {t} operate at a deeper level?"],
    "significance": ["Why does {t} matter?", "What's the practical importance of {t}?", "Why is {t} significant?", "What makes {t} important in the real world?", "Why should we care about {t}?"],
    "comparison":   ["How does {t} differ from related concepts?", "Compare {t} with similar ideas.", "What distinguishes {t} from neighboring concepts?", "How is {t} different from what people often confuse it with?", "What are the key contrasts that define {t}?"],
    "example":      ["Give a concrete example of {t}.", "Walk me through a real-world case of {t}.", "Can you illustrate {t} with an example?", "Show me {t} in action with a specific case.", "What's a tangible example that demonstrates {t}?"],
    "misconception":["What's a common misunderstanding about {t}?", "What do people often get wrong about {t}?", "What's a popular but incorrect belief about {t}?", "What myth surrounds {t} that's worth correcting?", "Where does intuition tend to mislead people about {t}?"],
}


def make_question(topic: str, sub_angle: str) -> str:
    """Pick a question template deterministically from topic+sub_angle hash."""
    h = int(hashlib.md5(f"{topic}|{sub_angle}".encode()).hexdigest(), 16)
    templates = QUESTION_TEMPLATES[sub_angle]
    return templates[h % len(templates)].format(t=topic)


# ENTRIES maps (topic, sub_angle) -> answer string.
# Built inline below by the subagent's own knowledge — no API calls.
ENTRIES: dict = {}


def _r(topic: str, sub_angle: str, answer: str):
    """Register an answer for a (topic, sub_angle) key."""
    ENTRIES[(topic, sub_angle)] = answer


# ============================================================================
# ANSWERS — inlined by the generating subagent.
# ============================================================================
# Loaded from companion module to keep this file readable.
from gen_knowledge_09_3_A_data import register_all  # noqa: E402
register_all(_r)


def main():
    data = yaml.safe_load(TOPICS_PATH.read_text())
    sub_angles = data["sub_angles"]
    seeds = []
    for cat in ("science", "math"):
        for topic in data["categories"][cat]:
            for sa in sub_angles:
                seeds.append((cat, topic, sa))

    seed_int = int(hashlib.md5(b"1009031A").hexdigest(), 16) % (2**32)
    rng = random.Random(seed_int)
    rng.shuffle(seeds)
    chosen = seeds[:500]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    missing = []
    with OUT_PATH.open("w") as f:
        for cat, topic, sa in chosen:
            ans = ENTRIES.get((topic, sa))
            if not ans:
                missing.append((cat, topic, sa))
                continue
            q = make_question(topic, sa)
            sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": ans},
                ]
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written} samples to {OUT_PATH}")
    if missing:
        print(f"MISSING {len(missing)} entries:")
        for m in missing[:20]:
            print(" -", m)


if __name__ == "__main__":
    main()
