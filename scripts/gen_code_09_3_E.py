#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""gen_code_09_3_E.py -- Generate 500 ShareGPT Python code samples (wave 5).

Subagent CODE-E of phase 09.3. Mix:
  - algorithms / data structures / utilities (~30%)
  - web / API (~25%)
  - data science / numpy / pandas / matplotlib (~20%)
  - debugging / refactoring (~15%)
  - testing / pytest / mocking (~10%)

Format mix: 60% 3-msg, 40% 5-msg with follow-up.
Seed: "1009312E" -- deterministic ordering / system-prompt rotation.
Output: datasets/code/raw-09.3/batch-12-A-code.jsonl (exactly 500 lines).
"""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "datasets" / "code" / "raw-09.3" / "batch-12-A-code.jsonl"

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


from gen_code_09_3_E_data import register_all as r1  # noqa: E402
from gen_code_09_3_E_data2 import register_all as r2  # noqa: E402
from gen_code_09_3_E_data3 import register_all as r3  # noqa: E402
from gen_code_09_3_E_data4 import register_all as r4  # noqa: E402
from gen_code_09_3_E_data5 import register_all as r5  # noqa: E402

for fn in (r1, r2, r3, r4, r5):
    fn(add)


FOLLOWUP_POOL = {
    "algo": [
        ("How would you write a docstring for this?",
         "Use a one-line summary, a blank line, then args / returns / raises sections. Google or NumPy style both work; pick one and stay consistent. Example: `\"\"\"Return the kth-smallest item.\\n\\nArgs:\\n    xs: Iterable of comparable items.\\n    k: 1-based rank.\\nReturns:\\n    The kth-smallest element.\\n\"\"\"`. Tools like pdoc and Sphinx render either style cleanly."),
        ("What's the time complexity of this approach?",
         "Walk through it: each input element is visited a constant number of times, dictionary insert/lookup is amortized O(1), so the whole function is O(n) time and O(n) extra space. For sorting-based variants you'd be at O(n log n) instead -- always cheaper to hash if order doesn't matter."),
        ("Can you rewrite this using a generator?",
         "Replace the accumulating list with `yield` inside the loop. Callers can `list(...)` for the eager version or stream lazily. Generators win when the consumer may stop early or the full result wouldn't fit in memory; they lose if you need `len()` or random access."),
        ("How do I make this type-safe under mypy strict?",
         "Annotate every parameter and return, prefer `Sequence` / `Iterable` over `list` for inputs, and use a `TypeVar` if input and output element types must match. Wire `mypy --strict` into CI so regressions fail the offending PR rather than landing silently."),
        ("How would you handle empty input?",
         "Decide explicitly: raise `ValueError` if empty input is a programmer error, return a sensible default (`0`, `None`, `[]`) if it's a normal case. Document the choice in the docstring -- silent fallthrough is the worst option because it hides bugs."),
    ],
    "web": [
        ("How would I add request validation to this?",
         "With FastAPI, declare a Pydantic model on the route signature -- 422 errors are returned automatically for bad input. With Flask, use `flask-pydantic` or `marshmallow`. Validate at the edge so handlers always work with clean, typed data."),
        ("How do I rate-limit this endpoint?",
         "For FastAPI, `slowapi` decorates routes with `@limiter.limit('10/minute')`. For Flask, `flask-limiter`. Use Redis as the storage backend so limits work across multiple workers; in-process counters silently fall apart behind a load balancer."),
        ("How do I add a health-check endpoint?",
         "Expose `/healthz` that returns 200 with a small JSON body, and `/readyz` that also pings critical dependencies (DB, cache). Keep `/healthz` cheap -- it gets hammered by orchestrators -- and never put auth in front of it."),
        ("How do I handle errors consistently across routes?",
         "Register a global exception handler that maps your app's exception types to HTTP responses with a stable JSON shape (`{\"error\": {\"code\": ..., \"message\": ...}}`). Clients depend on the shape; surprise variations break their parsing."),
        ("How would I add CORS to this app?",
         "FastAPI: `app.add_middleware(CORSMiddleware, allow_origins=[...], allow_methods=['*'])`. Flask: `flask-cors`. Always use an explicit origin allowlist in production -- `'*'` plus credentialed requests is rejected by browsers and is a security smell."),
    ],
    "data": [
        ("How would I save and reload this DataFrame?",
         "Parquet (`df.to_parquet('out.parquet')`) is the right default: typed, compressed, and cross-tool. CSV loses dtypes and is 10x larger. Avoid pickle for anything you'll read later -- it's fragile across pandas versions."),
        ("How do I make this work with a much larger dataset?",
         "Switch to chunked reads (`pd.read_csv(..., chunksize=...)`) and aggregate per chunk, or move to Polars / DuckDB for out-of-core query execution. Profile first -- 'larger' often still fits in RAM if you choose dtypes carefully."),
        ("How do I plot this result with matplotlib?",
         "`fig, ax = plt.subplots(); ax.plot(df['x'], df['y']); ax.set_xlabel(...); ax.set_ylabel(...); fig.tight_layout(); fig.savefig('out.png', dpi=150)`. Always create explicit `fig, ax` -- the `plt.plot(...)` global state pattern bites you in notebooks."),
        ("How do I handle missing values here?",
         "Decide based on mechanism. `df.dropna()` if rows are independent and missingness is rare. `df.fillna(df.median(numeric_only=True))` for quick numeric imputation. Add a `was_missing` indicator column when missingness itself carries signal for downstream models."),
        ("How do I profile this pipeline?",
         "`%%time` in a notebook for a quick read; `cProfile` + `snakeviz` for an overview; `line_profiler` (`@profile` decorator + `kernprof -l -v`) for line-level detail. Optimize after you've measured -- intuition about pandas hotspots is usually wrong."),
    ],
    "debug": [
        ("How do I add a regression test for this fix?",
         "Reduce the failing input to the smallest example you can, then write a `pytest` test that asserts the corrected behavior. Commit it in the same PR as the fix so future readers learn what the bug was without rereading the diff."),
        ("What tooling could have caught this earlier?",
         "Strict ruff (especially the B and SIM rule sets), mypy `--strict`, and pytest with coverage gates in CI catch most common shapes. For runtime issues, structured logging plus an error tracker (Sentry, Honeybadger) shortens the loop dramatically."),
        ("How do I bisect a regression like this?",
         "`git bisect start; git bisect bad HEAD; git bisect good <tag>` then either run the test interactively per step or `git bisect run pytest tests/test_regression.py`. With a fast test, bisect over hundreds of commits in minutes."),
        ("How do I reproduce this reliably in a test?",
         "Pin the inputs that triggered the bug as fixtures (use `pytest`'s `tmp_path` for file IO). If the bug depends on time, freeze it with `freezegun`; if randomness, seed the RNG. A flaky repro is barely better than no repro."),
    ],
    "test": [
        ("How do I share setup across these tests?",
         "Move common setup into a `pytest` fixture in `conftest.py`. Pick the scope deliberately (`function` is safest, `module` and `session` for expensive setup). Yield-based fixtures handle teardown cleanly even when the test raises."),
        ("How do I run only the fast tests locally?",
         "Mark slow tests with `@pytest.mark.slow` and add `addopts = -m 'not slow'` to `pyproject.toml`. Run the full suite in CI, the fast subset locally. Keeping the local loop under five seconds keeps you actually running tests."),
        ("How do I assert on log output?",
         "Use the built-in `caplog` fixture: `caplog.set_level(logging.INFO); ...; assert 'expected' in caplog.text`. For structured logs, iterate `caplog.records` and assert on individual fields rather than the rendered string."),
        ("How do I parametrize this test cleanly?",
         "`@pytest.mark.parametrize('x,expected', [(1, 2), (3, 4)])`; pass `ids=` for readable case names. For larger inputs, load from a fixture file and parametrize over the loaded list -- keeps the test body small."),
    ],
}


def main():
    seed_int = int(hashlib.md5(b"1009312E").hexdigest(), 16) % (2**32)
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
