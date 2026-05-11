#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""gen_code_09_3_F.py -- Generate 500 ShareGPT Python code samples (wave 6).

Subagent CODE-F of phase 09.3. Mix:
  - algorithms / data structures / utilities (~30%)
  - web / API (~25%)
  - data science / numpy / pandas / matplotlib (~20%)
  - debugging / refactoring (~15%)
  - testing / pytest / mocking (~10%)

Format mix: 60% 3-msg, 40% 5-msg with follow-up.
Seed: "1009313F" -- deterministic ordering / system-prompt rotation.
Output: datasets/code/raw-09.3/batch-13-A-code.jsonl (exactly 500 lines).
"""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "datasets" / "code" / "raw-09.3" / "batch-13-A-code.jsonl"

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


from gen_code_09_3_F_data import register_all as r1  # noqa: E402
from gen_code_09_3_F_data2 import register_all as r2  # noqa: E402
from gen_code_09_3_F_data3 import register_all as r3  # noqa: E402
from gen_code_09_3_F_data4 import register_all as r4  # noqa: E402
from gen_code_09_3_F_data5 import register_all as r5  # noqa: E402

for fn in (r1, r2, r3, r4, r5):
    fn(add)


FOLLOWUP_POOL = {
    "algo": [
        ("Can you add type hints to this?",
         "Annotate parameters and return values; prefer `Sequence`/`Iterable` for inputs and concrete `list`/`dict` for outputs. Run `mypy --strict` to catch implicit `Any`s. Hints are documentation that the type checker enforces."),
        ("How would you test this function?",
         "Cover the happy path, the empty input, a single-element input, and one boundary case. Use `pytest.mark.parametrize` so each case is its own test ID. If the function has invariants (sorted output, length preserved), encode them as `hypothesis` properties for free fuzz coverage."),
        ("What's the time complexity?",
         "Each element is processed a constant number of times and dict ops are amortized O(1), so it's O(n) time and O(n) space. Sorting variants land at O(n log n). Always state both time and space -- reviewers notice when one is missing."),
        ("How would I make this lazy with a generator?",
         "Replace the result list with `yield` inside the loop. Callers can still `list(...)` when they want eager evaluation. Generators shine for early termination and unbounded streams; they trade away `len()` and indexing."),
        ("Can you make this more Pythonic?",
         "Look for explicit index loops to replace with `enumerate`, manual accumulators to replace with comprehensions or `sum`/`any`/`all`, and nested conditionals to flatten with early returns. The standard library (`itertools`, `collections`) usually has the primitive you're hand-rolling."),
    ],
    "web": [
        ("How do I add authentication to this endpoint?",
         "FastAPI: declare a `Depends(get_current_user)` that decodes a JWT or queries a session store, and raise `HTTPException(401)` on failure. Flask: a `@login_required` decorator wrapping the view. Never hand-roll crypto -- use `passlib` for password hashing and `pyjwt`/`authlib` for tokens."),
        ("How do I log requests cleanly?",
         "Add middleware that logs method, path, status, and duration as structured JSON. Include a request ID so multi-line traces correlate. Put logging behind a feature flag at DEBUG level for body capture -- you don't want PII in production logs."),
        ("How would I add pagination?",
         "Cursor-based is the right default: `?cursor=<opaque>&limit=50` returns `{items, next_cursor}`. Offset pagination is fine for small admin lists but breaks under concurrent inserts. Document the max limit so clients can't ask for a million rows."),
        ("How do I version this API?",
         "Mount the router under `/v1/...` and never break v1 once it's published. Additive changes (new fields, new endpoints) don't need a new version; removing or renaming does. Keep `/v1` available for at least one major release after `/v2` ships."),
        ("How would I deploy this?",
         "Containerize with a slim Python base image, run `uvicorn` or `gunicorn` with worker count = `2*CPU+1`, put nginx or a managed load balancer in front for TLS. Healthcheck endpoints, structured logs to stdout, secrets via environment -- the standard 12-factor playbook."),
    ],
    "data": [
        ("How do I write this back to a database?",
         "`df.to_sql('table', engine, if_exists='append', index=False, chunksize=10_000, method='multi')`. Use a SQLAlchemy engine, not a raw connection, so reflection works. For Postgres bulk loads, `COPY` via `psycopg.copy` is 10x faster than `to_sql`."),
        ("How would I parallelize this computation?",
         "If it's CPU-bound and per-row, `df.swifter.apply` or split into chunks and use `concurrent.futures.ProcessPoolExecutor`. For real scale, switch to Dask or Polars -- they parallelize natively without you having to manage workers."),
        ("How do I save this plot in higher quality?",
         "`fig.savefig('out.png', dpi=200, bbox_inches='tight')` for raster, or `'out.pdf'` / `'out.svg'` for vector. Vector beats raster for line charts in publications. Set `plt.rcParams['savefig.dpi']` once if you want a global default."),
        ("How would I make this faster with numpy?",
         "Replace Python-level loops with vectorized array ops -- `np.where`, broadcasting, `np.add.reduceat` for grouped sums. The rule of thumb: if you wrote a `for` loop over array elements, there's almost always a 10-100x faster vector form."),
        ("How do I handle outliers here?",
         "First decide whether they're errors (drop or cap) or signal (keep). For Gaussian-ish data, winsorize at the 1st/99th percentile. For skewed data, use IQR fences (`q1 - 1.5*IQR`, `q3 + 1.5*IQR`). Always log how many rows were affected -- silent filtering is a bug magnet."),
    ],
    "debug": [
        ("How do I prevent this kind of bug in the future?",
         "Add a regression test, then look for the broader pattern: was this a missing type hint, an unchecked None, an off-by-one? Add a ruff or mypy rule (or a custom AST check) that flags the shape so you catch siblings of this bug at lint time."),
        ("How do I trace this in production?",
         "Structured logging at INFO with request IDs, plus an APM (OpenTelemetry, Datadog, Sentry performance) so you can follow a request across services. For one-off forensics, `py-spy dump --pid <pid>` gives you a stack trace of every thread without restarting the process."),
        ("What's the smallest reproduction I can make?",
         "Strip the input until the bug disappears, then add the last thing back. Inline external calls into stubs that return the recorded data. The goal is a test that fits in 20 lines and runs in 100ms -- big repros stay un-debugged."),
        ("How do I confirm the fix works under load?",
         "Write a stress test that hammers the previously-broken path with concurrent workers (`asyncio.gather` or `threading`). Run it long enough to get past first-failure -- some bugs only show after the GC kicks in or a connection pool churns."),
    ],
    "test": [
        ("How do I mock an external HTTP call here?",
         "`pytest`'s `monkeypatch` plus `responses` (for `requests`) or `respx` (for `httpx`). Match on method + URL + body so the test fails loudly if the call signature drifts. Avoid `unittest.mock.patch` of the client object -- it doesn't validate the URL."),
        ("How do I test code that uses datetime.now?",
         "Inject a clock: take `now: Callable[[], datetime] = datetime.utcnow` as a parameter, default to the real one, pass a stub in tests. If you can't refactor, `freezegun`'s `@freeze_time('2024-01-01')` patches `datetime` globally for the test."),
        ("How do I get coverage on this branch?",
         "`pytest --cov=mypkg --cov-report=term-missing` shows uncovered lines. Add a parametrized case for each branch. Don't chase 100% -- some branches (defensive `else` raises) are fine to leave uncovered if covered by mypy."),
        ("How do I test async code?",
         "Install `pytest-asyncio`, mark tests `@pytest.mark.asyncio`, and `await` the function under test directly. For fixtures that yield resources, use `async def` fixtures. Set `asyncio_mode = auto` in `pyproject.toml` so you don't have to mark every test."),
    ],
}


def main():
    seed_int = int(hashlib.md5(b"1009313F").hexdigest(), 16) % (2**32)
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
