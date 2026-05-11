#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""gen_code_09_3_C.py -- Generate 500 ShareGPT Python code samples (wave 3).

Subagent CODE-C of phase 09.3. Mix:
  - algorithms / data structures / utilities (~30%)
  - web / API (~25%)
  - data science / numpy / pandas / matplotlib (~20%)
  - debugging / refactoring (~15%)
  - testing / pytest / mocking (~10%)

Format mix: 60% 3-msg, 40% 5-msg with follow-up.
Seed: "1009310C" -- deterministic ordering / system-prompt rotation.
Output: datasets/code/raw-09.3/batch-10-A-code.jsonl (exactly 500 lines).
"""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "datasets" / "code" / "raw-09.3" / "batch-10-A-code.jsonl"

SYSTEM_PROMPTS = [
    "You are an expert Python programmer.",
    "You are a senior Python developer.",
    "You are a helpful Python assistant.",
    "You write clean, idiomatic Python.",
    "You are a Python expert who explains code clearly.",
]


ENTRIES: list = []


def add(cat, u, a, u2=None, a2=None):
    ENTRIES.append({"cat": cat, "u": u, "a": a, "u2": u2, "a2": a2})


from gen_code_09_3_C_data import register_all as r1  # noqa: E402
from gen_code_09_3_C_data2 import register_all as r2  # noqa: E402
from gen_code_09_3_C_data3 import register_all as r3  # noqa: E402
from gen_code_09_3_C_data4 import register_all as r4  # noqa: E402
from gen_code_09_3_C_data5 import register_all as r5  # noqa: E402

for fn in (r1, r2, r3, r4, r5):
    fn(add)


FOLLOWUP_POOL = {
    "algo": [
        ("Can you analyze the time and space complexity?",
         "The main loop processes each element once, so time is O(n). Space is O(1) for the in-place variants and O(n) when we accumulate output. If you need to push it further, profile first with `timeit` -- micro-optimizations rarely matter compared to algorithmic choices."),
        ("How do I make this work on iterators instead of lists?",
         "Take an `Iterable` parameter and avoid indexing or `len()`. Use `itertools.tee` if you need two passes, or buffer the input into a list. For unbounded streams, return a generator so the consumer controls when work happens."),
        ("What edge cases would you add tests for?",
         "Empty input, single element, all-equal elements, sorted vs reverse-sorted, and the largest realistic input. For numeric inputs add negatives, zeros, and floats. A Hypothesis property test on an invariant catches most cases you wouldn't think of."),
        ("Is there a standard-library function that does this?",
         "Check `itertools`, `functools`, and `collections` first -- they cover a surprising amount. `more_itertools` on PyPI fills most of the remaining gaps. Reaching for stdlib first means less code to test and less to maintain."),
    ],
    "web": [
        ("How would I add request validation here?",
         "Use Pydantic models for the request body and query params; FastAPI wires them up automatically. For Flask, use `marshmallow` or `pydantic` directly inside the handler. Validation errors should return 422 with a structured body listing each field's problem."),
        ("What's the right way to handle background work?",
         "Don't block the request. For light work use `BackgroundTasks` (FastAPI) or `threading` for fire-and-forget. For real workloads put a queue (Redis + RQ, Celery, or Arq) between the API and the worker process. Always make tasks idempotent so retries are safe."),
        ("How should I structure config for this app?",
         "Use Pydantic's `BaseSettings` (pydantic-settings) to load from env vars with type coercion. Keep secrets out of the repo; load them from a vault or platform-managed env. Never read `os.environ` directly scattered across the codebase -- centralize it."),
        ("How do I add rate limiting?",
         "For a single instance, `slowapi` (FastAPI) or `Flask-Limiter` work out of the box with in-memory storage. For multi-instance deployments, point those at Redis. Limit by API key for authenticated routes and by IP for public ones, and surface the limit headers (`X-RateLimit-*`) in responses."),
    ],
    "data": [
        ("How would I scale this past memory?",
         "Switch to chunked iteration with `pd.read_csv(chunksize=...)` or use Polars / DuckDB which work on larger-than-memory data natively. For numpy, `np.memmap` keeps arrays on disk. If the workload pushes down to SQL, DuckDB over Parquet is often the simplest big win."),
        ("How do I make the result reproducible?",
         "Pin library versions in a lockfile, set every random seed (`numpy.random.default_rng(seed)`, `random.seed`, framework seeds), and avoid nondeterministic GPU kernels when correctness matters more than speed. Snapshot input data with a hash so you know which inputs produced a result."),
        ("What's the right plot for this kind of data?",
         "Distributions: histogram or KDE. Comparisons across groups: boxplot or violin. Trends over time: line plot. Relationships between two numeric variables: scatter (or hexbin if dense). Avoid pie charts -- bar charts are almost always easier to read."),
        ("How do I check the output is correct?",
         "Pick a small sample you can verify by hand and assert the function reproduces it. For aggregations, check totals match the input total. For transformations, run a round-trip test (encode then decode) and assert equality. These cheap sanity checks catch most regressions."),
    ],
    "debug": [
        ("What was the underlying lesson?",
         "Most Python footguns come from mutable defaults, late binding in closures, or assumptions about iteration order across versions. A strict ruff config catches a lot of these automatically; mypy catches the type-shaped ones. Reading the data model docs once carefully also saves a lot of time later."),
        ("How can I prevent this class of bug going forward?",
         "Add a regression test that fails before your fix and passes after. Turn on the relevant ruff rule (most patterns have one). Code review by someone else catches a lot too -- the author's eyes slide over their own assumptions."),
        ("Is there a linter rule that would flag this?",
         "Often yes. Ruff has hundreds of rules covering common bugs; flake8-bugbear (B) and flake8-simplify (SIM) catch many subtle ones. Mypy with strict_equality catches type-shaped bugs. The cost of strict linting is low and the bug-prevention value is high."),
    ],
    "test": [
        ("How would I scale this test pattern across the suite?",
         "Lift shared setup into `conftest.py` fixtures at the right scope (module/session). Group related tests by feature in their own files. Mark slow tests with `@pytest.mark.slow` so the fast feedback loop stays under a few seconds. Aim for tests that finish in under a second each."),
        ("Should this be a unit or integration test?",
         "If most collaborators are mocked, it's a unit test. If you're hitting a real DB, HTTP, or filesystem, it's integration. Keep most tests as fast unit tests but reserve enough integration coverage to catch wiring bugs that mocks miss."),
        ("Where's the right boundary to mock at?",
         "Mock at your code's edges -- HTTP clients, DB drivers, the clock -- not your own internal classes. If you're tempted to mock your own methods, the design probably needs a seam (an injected dependency or interface) instead."),
        ("How do I test something that depends on the current time?",
         "Inject a `clock` callable that defaults to `time.time`; pass a fake in tests. Or use `freezegun` for legacy code where injection isn't feasible. Same pattern for randomness -- inject an RNG so tests are deterministic."),
    ],
}


def main():
    seed_int = int(hashlib.md5(b"1009310C").hexdigest(), 16) % (2**32)
    rng = random.Random(seed_int)

    entries = list(ENTRIES)
    rng.shuffle(entries)

    if len(entries) < 500:
        raise SystemExit(f"Only {len(entries)} entries -- need 500.")
    entries = entries[:500]

    target_5msg = 200
    have_5msg = sum(1 for e in entries if e.get("u2") and e.get("a2"))
    need = max(0, target_5msg - have_5msg)
    pool_idx = {cat: 0 for cat in FOLLOWUP_POOL}
    for e in entries:
        if need <= 0:
            break
        if e.get("u2") and e.get("a2"):
            continue
        cat = e["cat"]
        pool = FOLLOWUP_POOL.get(cat, [])
        if not pool:
            continue
        u2, a2 = pool[pool_idx[cat] % len(pool)]
        pool_idx[cat] += 1
        e["u2"], e["a2"] = u2, a2
        need -= 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    counts: dict = {}
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
