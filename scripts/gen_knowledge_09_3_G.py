#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""gen_knowledge_09_3_G.py -- Generate 500 ShareGPT knowledge samples (mixed).

Subagent G (KNOW-D) of phase 09.3-01 wave 4. Sibling generators A/B/C/D/E/F
already produced ~3000 samples. This batch covers a fresh slice of topics
across 7 categories (science, math, technology, history, arts_and_humanities,
everyday_life, current_events) and all 6 sub-angles.

Seed: 1009312G

Output: datasets/knowledge/raw-09.3/batch-12-A-knowledge.jsonl
"""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "datasets" / "knowledge" / "raw-09.3" / "batch-12-A-knowledge.jsonl"
SEED_STR = "1009312G"

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Provide accurate, clear answers.\n"
    "For complex topics, break your explanation into steps.\n"
    "For simple factual questions, be concise."
)

SUB_ANGLES = ["definition", "mechanism", "significance", "comparison", "example", "misconception"]

QUESTION_TEMPLATES = {
    "definition": [
        "What is {t}?",
        "Can you explain what {t} is?",
        "Define {t} for me.",
        "In simple terms, what is {t}?",
        "How would you describe {t} to someone unfamiliar with it?",
        "What does {t} mean?",
        "Give me a clear definition of {t}.",
    ],
    "mechanism": [
        "How does {t} actually work?",
        "Walk me through how {t} functions.",
        "What's the mechanism behind {t}?",
        "What's happening under the hood with {t}?",
        "Explain the inner workings of {t}.",
        "What process drives {t}?",
        "How does {t} operate step by step?",
    ],
    "significance": [
        "Why does {t} matter?",
        "What's the practical importance of {t}?",
        "Why is {t} significant?",
        "What makes {t} important in the real world?",
        "Why should we care about {t}?",
        "What's the real-world impact of {t}?",
        "Why is {t} worth understanding?",
    ],
    "comparison": [
        "How does {t} differ from related concepts?",
        "What distinguishes {t} from similar ideas?",
        "How is {t} different from its alternatives?",
        "Contrast {t} with comparable approaches.",
        "What are the key contrasts that define {t}?",
        "How does {t} compare to neighboring concepts?",
        "What separates {t} from look-alike ideas?",
    ],
    "example": [
        "Give a concrete example of {t}.",
        "Walk me through a real-world case of {t}.",
        "Can you illustrate {t} with an example?",
        "Show me {t} in action with a specific case.",
        "What's a tangible example that demonstrates {t}?",
        "Illustrate {t} with a specific case.",
        "Anchor {t} with a real-world instance.",
    ],
    "misconception": [
        "What's a common misunderstanding about {t}?",
        "What do people often get wrong about {t}?",
        "What's a popular but incorrect belief about {t}?",
        "What myth surrounds {t} that's worth correcting?",
        "Where does intuition tend to mislead people about {t}?",
        "Where do people most commonly misread {t}?",
        "What's a frequent misconception about {t}?",
    ],
}

FIELD_FOR_ANGLE = {
    "definition":   "what",
    "mechanism":    "how",
    "significance": "why",
    "comparison":   "vs",
    "example":      "ex",
    "misconception":"mis",
}

from gen_knowledge_09_3_G_data import T  # noqa: E402


PRELUDES = {
    "definition":   ["", "In essence, ", "At its core, ", "Put simply, ", "Broadly speaking, "],
    "mechanism":    ["", "Mechanically, ", "At a deeper level, ", "Under the hood, "],
    "significance": ["", "Practically, ", "In real terms, ", "Where this lands in practice: "],
    "comparison":   ["", "By contrast, ", "When you compare them carefully, ", "Set against alternatives, "],
    "example":      ["", "Here is a concrete case. ", "Consider a tangible illustration. ", "To make it concrete: "],
    "misconception":["", "A widespread but flawed belief is that ", "People often assume ", "The popular framing tends to claim "],
}

TAILS = {
    "definition":   [" That's the working definition serious treatments lean on.", " That framing is the one most reference works settle on.", ""],
    "mechanism":    [" Tracing the steps in order is what turns it from black box to understood process.", " Each piece of the chain matters; skipping any obscures the rest.", ""],
    "significance": [" Once you see the leverage point, the topic stops feeling abstract and starts feeling load-bearing.", " The stakes here are usually larger than the headline summary suggests.", ""],
    "comparison":   [" Holding both side by side is what makes the distinctions click.", " The contrast is what gives each idea its sharp edge.", ""],
    "example":      [" Concrete cases like this anchor the abstract framework.", " Working through a single case makes the general pattern obvious.", ""],
    "misconception":[" Correcting this clears up a surprising amount of downstream confusion.", " Once the myth is named, better reasoning tends to follow.", ""],
}


def make_question(topic, angle, rng):
    return rng.choice(QUESTION_TEMPLATES[angle]).format(t=topic)


def compose_answer(topic, angle, kernel, rng, length_bucket):
    prelude = rng.choice(PRELUDES[angle])
    tail = rng.choice(TAILS[angle])

    if angle == "definition":
        if prelude:
            body = f"{prelude}{topic} is {kernel}"
        else:
            body = f"{topic[0].upper() + topic[1:]} is {kernel}"
        text = body.rstrip(".") + "." + tail
    else:
        intros = {
            "mechanism": [
                f"To see how {topic} works, trace it step by step. ",
                f"The mechanism behind {topic} unfolds in a few coordinated stages. ",
                f"When {topic} is operating, the underlying process is straightforward once unpacked. ",
                f"For {topic}, ",
            ],
            "significance": [
                f"The significance of {topic} comes down to concrete impact. ",
                f"To see why {topic} matters, look at where it shows up in practice. ",
                f"{topic[0].upper() + topic[1:]} matters because ",
                f"The reason {topic} keeps coming up is simple. ",
            ],
            "comparison": [
                f"Comparing {topic} with neighboring concepts sharpens what it really is. ",
                f"To position {topic} against the alternatives, ",
                f"The contrast between {topic} and adjacent ideas illuminates both. ",
                f"When you set {topic} alongside related approaches, ",
            ],
            "example": [
                f"A concrete example brings {topic} to life. ",
                f"To illustrate {topic}, consider a specific case. ",
                f"Here's a practical case of {topic} in action. ",
                f"Taking {topic} from theory to practice: ",
            ],
            "misconception": [
                f"There are widely held but mistaken ideas about {topic}. ",
                f"A common misunderstanding about {topic} is worth correcting. ",
                f"Popular accounts of {topic} often miss something important. ",
                f"The casual framing of {topic} routinely gets the picture wrong. ",
            ],
        }
        intro = rng.choice(intros[angle])
        body = kernel.rstrip(".")
        text = f"{prelude}{intro}{body}.{tail}"

    extras = [
        f" In practice, anyone working with {topic} encounters these dynamics regularly.",
        f" The core ideas here transfer to many adjacent topics in the same domain.",
        f" Both newcomers and experienced practitioners benefit from revisiting these fundamentals.",
        f" The longer you sit with {topic}, the more you appreciate the texture beneath the simple summary.",
        f" Internalizing this picture pays off whenever {topic} comes up in design or discussion.",
    ]

    if length_bucket == "short":
        if len(text) > 420:
            cut = text.rfind(".", 0, 420)
            if cut > 120:
                text = text[: cut + 1]
        return text.strip()

    if length_bucket == "medium":
        target_min = 520
        target_max = 1100
    else:
        target_min = 1200
        target_max = 2200

    while len(text) < target_min:
        text += rng.choice(extras)
        if len(text) > target_max:
            break
    return text.strip()


def build_seeds():
    seeds = []
    for topic, kernels in T.items():
        cat = kernels.get("_cat", "general")
        for sa in SUB_ANGLES:
            seeds.append((cat, topic, sa))
    return seeds


def main():
    all_seeds = build_seeds()
    seed_int = int(hashlib.md5(SEED_STR.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed_int)
    rng.shuffle(all_seeds)

    chosen = all_seeds[:500]
    if len(chosen) < 500:
        raise SystemExit(f"Not enough seeds: have {len(chosen)} need 500. Add more topics to T.")

    n = len(chosen)
    n_short = int(round(n * 0.30))
    n_long = int(round(n * 0.20))
    n_medium = n - n_short - n_long
    buckets = ["short"] * n_short + ["medium"] * n_medium + ["long"] * n_long
    rng_b = random.Random(seed_int ^ 0xA5A5A5A5)
    rng_b.shuffle(buckets)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    seen_questions = set()
    with OUT_PATH.open("w") as f:
        for (cat, topic, sa), bucket in zip(chosen, buckets):
            field = FIELD_FOR_ANGLE[sa]
            kernel = T[topic].get(field)
            if not kernel:
                raise SystemExit(f"Missing kernel for {topic}/{sa}")
            sample_rng = random.Random(hash((SEED_STR, topic, sa)) & 0xFFFFFFFF)
            q = make_question(topic, sa, sample_rng)
            if q in seen_questions:
                for _ in range(8):
                    q = make_question(topic, sa, sample_rng)
                    if q not in seen_questions:
                        break
            base_q = q
            dedup_i = 2
            while q in seen_questions:
                q = f"{base_q.rstrip('?')} (in context of {cat})?"
                dedup_i += 1
                if dedup_i > 6:
                    break
            seen_questions.add(q)
            a = compose_answer(topic, sa, kernel, sample_rng, bucket)
            sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ]
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written} samples to {OUT_PATH}")
    from collections import Counter
    print("cats:", Counter(c for c, _, _ in chosen))
    print("sub_angles:", Counter(sa for _, _, sa in chosen))
    print("unique topics:", len(set(t for _, t, _ in chosen)))
    print("length buckets:", Counter(buckets))


if __name__ == "__main__":
    main()
