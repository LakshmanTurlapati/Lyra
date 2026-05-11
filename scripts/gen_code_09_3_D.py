#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""gen_code_09_3_D.py -- Generate 500 ShareGPT Python code samples (wave 4).

Subagent CODE-D of phase 09.3. Mix:
  - algorithms / data structures / utilities (~30%)
  - web / API (~25%)
  - data science / numpy / pandas / matplotlib (~20%)
  - debugging / refactoring (~15%)
  - testing / pytest / mocking (~10%)

Format mix: 60% 3-msg, 40% 5-msg with follow-up.
Seed: "1009311D" -- deterministic ordering / system-prompt rotation.
Output: datasets/code/raw-09.3/batch-11-A-code.jsonl (exactly 500 lines).
"""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "datasets" / "code" / "raw-09.3" / "batch-11-A-code.jsonl"

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


from gen_code_09_3_D_data import register_all as r1  # noqa: E402
from gen_code_09_3_D_data2 import register_all as r2  # noqa: E402
from gen_code_09_3_D_data3 import register_all as r3  # noqa: E402
from gen_code_09_3_D_data4 import register_all as r4  # noqa: E402
from gen_code_09_3_D_data5 import register_all as r5  # noqa: E402

for fn in (r1, r2, r3, r4, r5):
    fn(add)


FOLLOWUP_POOL = {
    "algo": [
        ("How would you write a docstring for this?",
         "Use a one-line summary, a blank line, then args / returns / raises sections. Google or NumPy style both work; pick one and stay consistent. Example: `\"\"\"Return the kth-smallest item.\\n\\nArgs:\\n    xs: Iterable of comparable items.\\n    k: 1-based rank.\\nReturns:\\n    The kth-smallest element.\\n\"\"\"`. Tools like pdoc and Sphinx render either style cleanly."),
        ("How would I parallelize this if the input is huge?",
         "If the work per element is CPU-bound, use `concurrent.futures.ProcessPoolExecutor` with `executor.map(fn, chunks)`. For IO-bound work, `ThreadPoolExecutor` or asyncio. Always benchmark first -- the GIL plus pickling overhead can make threads slower than a tight serial loop for short tasks."),
        ("Can you rewrite this as a generator?",
         "Replace `return [...]` with `yield` inside the loop. The caller can then `list(...)` or stream. Generators win when the consumer may stop early or when the full result wouldn't fit in memory; they lose if you need `len()` or random access."),
        ("How do I make this type-safe for mypy strict mode?",
         "Add explicit annotations on every parameter and return, prefer `Sequence` / `Iterable` over `list` for inputs, and use `TypeVar` when the input and output element types must match. Run `mypy --strict` in CI so regressions surface on the offending PR."),
    ],
    "web": [
        ("How do I add structured logging to this?",
         "Use the stdlib `logging` module with a JSON formatter (`python-json-logger` is the smallest dependency). Log at INFO for request boundaries, DEBUG for internal steps, WARNING for retries, and ERROR for failures. Always include a request ID in the log context so you can correlate lines."),
        ("How do I document this API?",
         "FastAPI generates OpenAPI automatically; just write good Pydantic models and route docstrings. For Flask, use `flask-smorest` or `apispec`. Keep examples in the schema -- consumers copy-paste them and that's a good thing."),
        ("How would I add auth to this endpoint?",
         "Use a dependency that validates a bearer token and returns the user (FastAPI's `Depends`). Reject with 401 for missing/invalid, 403 for valid-but-unauthorized. Keep the auth logic in one place -- spreading token-decoding across handlers is how you get inconsistent behavior."),
        ("How do I make this resilient to downstream failures?",
         "Wrap outbound calls with `tenacity.retry` for transient errors, set explicit timeouts (never the default), and add a circuit breaker (`pybreaker`) for repeated failure. Surface a clear 503 with a `Retry-After` header to your callers when the downstream is unavailable."),
    ],
    "data": [
        ("How do I profile this to find the slow part?",
         "Use `cProfile` for an overall view: `python -m cProfile -o out.prof script.py` then inspect with `snakeviz`. For line-level detail, `line_profiler` (`@profile` decorator + `kernprof -l -v`). Don't optimize before you've measured."),
        ("How would I save and reload this result?",
         "Parquet for tabular data (`df.to_parquet(...)`), `np.save` for arrays, joblib for scikit-learn models. Avoid pickle for cross-language or long-term storage; the format is fragile and a security risk if the file is ever from an untrusted source."),
        ("How do I make this work with categorical data?",
         "Convert string columns with `df[col] = df[col].astype('category')` -- you save memory and groupby gets faster. For modeling, use `pd.get_dummies` or scikit-learn's `OneHotEncoder` depending on whether you need the encoder object later."),
        ("What's the right way to handle missing values here?",
         "There's no universal answer -- it depends on the mechanism. Drop if the column is mostly missing or rows are independent. Impute with median (numeric) or mode (categorical) for quick baselines. Add a `was_missing` indicator column when the missingness itself carries signal."),
    ],
    "debug": [
        ("How do I add a regression test for this?",
         "Reduce the failing input to the smallest example you can, then write a `pytest` test that asserts the fixed behavior. Commit it in the same PR as the fix. Future-you (or a teammate) reading the test learns what the bug was without needing to read the diff."),
        ("What tooling could have caught this earlier?",
         "Strict ruff (especially the B and SIM rules), mypy with `--strict`, and pytest with coverage gates in CI catch most of the common shapes. For runtime issues, structured logging plus an error tracker (Sentry) shortens the debug loop dramatically."),
        ("How do I bisect a regression like this?",
         "`git bisect start; git bisect bad HEAD; git bisect good <known-good>` then either run a script per step (`git bisect run pytest tests/test_regression.py`) or test manually. With a fast test, bisect over hundreds of commits in a few minutes."),
    ],
    "test": [
        ("How do I share setup across these tests?",
         "Move common setup into a `pytest` fixture in `conftest.py`. Pick the right scope (`function` is safest, `module` and `session` for expensive setup). Yield-based fixtures handle teardown cleanly."),
        ("How do I run only the fast tests in development?",
         "Mark slow tests with `@pytest.mark.slow` and add `addopts = -m 'not slow'` to `pyproject.toml`. Run the full suite in CI, the fast subset locally. Keeping the local loop under five seconds keeps you actually running tests."),
        ("How do I assert on log output?",
         "Use the built-in `caplog` fixture: `caplog.set_level(logging.INFO); ...; assert 'expected message' in caplog.text`. For structured logs, iterate `caplog.records` and assert on individual fields rather than the formatted string."),
        ("How do I parametrize this test cleanly?",
         "`@pytest.mark.parametrize('x,expected', [(1, 2), (3, 4)])` -- use `ids=` to give each case a readable name in the report. For larger datasets, load from a fixture file and parametrize over the loaded list."),
    ],
}


def main():
    seed_int = int(hashlib.md5(b"1009311D").hexdigest(), 16) % (2**32)
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
