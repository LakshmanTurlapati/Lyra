#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""gen_code_09_3_G.py -- Generate 500 ShareGPT Python code samples (wave 7, final).

Subagent CODE-G of phase 09.3. Mix:
  - algorithms / data structures / utilities (~30%)
  - web / API (~25%)
  - data science / numpy / pandas / matplotlib (~20%)
  - debugging / refactoring (~15%)
  - testing / pytest / mocking (~10%)

Format mix: 60% 3-msg, 40% 5-msg with follow-up.
Seed: "1009314G" -- deterministic ordering / system-prompt rotation.
Output: datasets/code/raw-09.3/batch-14-A-code.jsonl (exactly 500 lines).
"""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "datasets" / "code" / "raw-09.3" / "batch-14-A-code.jsonl"

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


from gen_code_09_3_G_data import register_all as r1  # noqa: E402
from gen_code_09_3_G_data2 import register_all as r2  # noqa: E402
from gen_code_09_3_G_data3 import register_all as r3  # noqa: E402
from gen_code_09_3_G_data4 import register_all as r4  # noqa: E402
from gen_code_09_3_G_data5 import register_all as r5  # noqa: E402

for fn in (r1, r2, r3, r4, r5):
    fn(add)


FOLLOWUP_POOL = {
    "algo": [
        ("Can you add type hints to this?",
         "Annotate parameters and return types; prefer abstract `Iterable`/`Sequence` for inputs and concrete `list`/`dict` for outputs. Run `mypy --strict` to catch implicit `Any`. Treat hints as enforceable documentation."),
        ("How would you test this function?",
         "Cover the happy path, an empty input, a single-element input, and at least one boundary case. Use `pytest.mark.parametrize` for distinct test IDs, and `hypothesis` properties for invariants like length preservation or sortedness."),
        ("What is the time complexity?",
         "Each element is touched a constant number of times and dict ops are amortized O(1), so it's O(n) time and O(n) space. State both bounds; reviewers notice when one is missing."),
        ("Can you make this lazy with a generator?",
         "Replace the result list with `yield` inside the loop. Callers can wrap with `list(...)` if they need eager evaluation. Generators win on early termination and unbounded streams but lose `len()` and indexing."),
        ("Can you make this more Pythonic?",
         "Replace explicit index loops with `enumerate`, manual accumulators with `sum`/`any`/`all` or comprehensions, and nested conditionals with early returns. `itertools` and `collections` usually have the primitive you're hand-rolling."),
    ],
    "web": [
        ("How do I add authentication to this endpoint?",
         "FastAPI: a `Depends(get_current_user)` that decodes a JWT or queries a session and raises `HTTPException(401)` on failure. Use `passlib` for password hashing and `pyjwt` for tokens; never roll your own crypto."),
        ("How do I log requests cleanly?",
         "Add middleware that emits structured JSON with method, path, status, and duration. Include a request ID so multi-line traces correlate. Gate body capture behind DEBUG-only flags so PII never lands in production logs."),
        ("How would I add pagination?",
         "Cursor-based by default: `?cursor=<opaque>&limit=50` returns `{items, next_cursor}`. Offset pagination is fine for small admin lists but skips/duplicates rows under concurrent inserts. Document and enforce a max limit."),
        ("How do I version this API?",
         "Mount the router under `/v1/...` and never break v1 once published. Additive changes don't need a new version; renaming or removing fields does. Keep `/v1` available for one major release after `/v2` ships."),
        ("How would I deploy this?",
         "Slim Python base image, `uvicorn` or `gunicorn` with `2*CPU+1` workers, TLS terminated by nginx or a managed LB. Healthcheck endpoint, structured logs to stdout, secrets via environment -- standard 12-factor."),
    ],
    "data": [
        ("How do I write this back to a database?",
         "`df.to_sql('table', engine, if_exists='append', index=False, chunksize=10_000, method='multi')`. Use a SQLAlchemy engine so reflection works. For Postgres bulk loads, `COPY` via `psycopg.copy` is ~10x faster than `to_sql`."),
        ("How would I parallelize this computation?",
         "If it's CPU-bound and per-row, chunk the frame and use `concurrent.futures.ProcessPoolExecutor`. For real scale, switch to Polars or Dask -- they parallelize natively without you babysitting workers."),
        ("How do I save this plot in higher quality?",
         "`fig.savefig('out.png', dpi=200, bbox_inches='tight')` for raster, or `'out.pdf'`/`'out.svg'` for vector. Vector beats raster for line charts in print. Set `plt.rcParams['savefig.dpi']` once for a global default."),
        ("How would I make this faster with numpy?",
         "Replace Python-level loops with vectorized ops -- `np.where`, broadcasting, `np.add.reduceat`. If you wrote a `for` over array elements, there's almost always a 10-100x faster vectorized form."),
        ("How do I handle outliers here?",
         "Decide first whether they're errors (drop/cap) or signal (keep). For Gaussian-ish data, winsorize at 1st/99th percentile. For skewed data, IQR fences (`q1 - 1.5*IQR`, `q3 + 1.5*IQR`). Always log the affected row count."),
    ],
    "debug": [
        ("How do I prevent this kind of bug in the future?",
         "Add a regression test, then look for the broader pattern: missing type hint, unchecked None, off-by-one. Add a ruff or mypy rule (or a custom AST check) so siblings of this bug get caught at lint time."),
        ("How do I trace this in production?",
         "Structured logging at INFO with request IDs, plus an APM (OpenTelemetry, Datadog, Sentry). For one-off forensics, `py-spy dump --pid <pid>` gives a stack of every thread without restarting the process."),
        ("What is the smallest reproduction I can make?",
         "Strip input until the bug disappears, then add the last thing back. Inline external calls into stubs that replay recorded data. Aim for under 20 lines and 100ms runtime; big repros stay un-debugged."),
        ("How do I confirm the fix works under load?",
         "Stress test the previously-broken path with concurrent workers via `asyncio.gather` or `threading`. Run long enough to past first-failure -- some bugs only appear after GC kicks in or a connection pool churns."),
    ],
    "test": [
        ("How do I mock an external HTTP call here?",
         "`pytest`'s `monkeypatch` plus `responses` (for `requests`) or `respx` (for `httpx`). Match on method + URL + body so the test fails loudly when the call signature drifts. Avoid raw `unittest.mock.patch` of clients."),
        ("How do I test code that uses datetime.now?",
         "Inject a clock: take `now: Callable[[], datetime] = datetime.utcnow` as a parameter and pass a stub in tests. If you can't refactor, `freezegun`'s `@freeze_time('2024-01-01')` patches `datetime` globally."),
        ("How do I get coverage on this branch?",
         "`pytest --cov=mypkg --cov-report=term-missing` shows uncovered lines. Add a parametrized case per branch. Don't chase 100% -- defensive `else: raise` paths are fine to skip if mypy already proves them unreachable."),
        ("How do I test async code?",
         "Install `pytest-asyncio`, mark tests `@pytest.mark.asyncio`, and `await` the function directly. Use `async def` fixtures for resources. Set `asyncio_mode = auto` in `pyproject.toml` to skip the marker boilerplate."),
    ],
}


def main():
    seed_int = int(hashlib.md5(b"1009314G").hexdigest(), 16) % (2**32)
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
