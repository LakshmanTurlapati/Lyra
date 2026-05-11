#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""gen_code_09_3_B.py -- Generate 500 ShareGPT Python code samples (wave 2).

Subagent CODE-B of phase 09.3. Mix:
  - algorithms / data structures / utilities (~30%)
  - web / API (~25%)
  - data science / numpy / pandas / sklearn (~20%)
  - debugging / refactoring (~15%)
  - testing / pytest / mocking (~10%)

Format mix: 60% 3-msg, 40% 5-msg with follow-up.
Seed: "1009309B" -- deterministic ordering / system-prompt rotation.
Output: datasets/code/raw-09.3/batch-09-A-code.jsonl (exactly 500 lines).
"""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "datasets" / "code" / "raw-09.3" / "batch-09-A-code.jsonl"

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


from gen_code_09_3_B_data import register_all as r1  # noqa: E402
from gen_code_09_3_B_data2 import register_all as r2  # noqa: E402
from gen_code_09_3_B_data3 import register_all as r3  # noqa: E402
from gen_code_09_3_B_data4 import register_all as r4  # noqa: E402
from gen_code_09_3_B_data5 import register_all as r5  # noqa: E402

for fn in (r1, r2, r3, r4, r5):
    fn(add)


FOLLOWUP_POOL = {
    "algo": [
        ("Walk me through the time and space complexity.",
         "Time is dominated by the main loop's work per element; space is whatever auxiliary structures grow with input size. For most of these single-pass implementations that's O(n) time and O(1) extra space, or O(n) extra when we accumulate results. Profile with `timeit` on representative inputs before optimizing further."),
        ("Are there edge cases I should add tests for?",
         "Empty input, single element, all-equal elements, and the maximum reasonable size for your domain. Also boundary values (negatives, zeros, floats with subnormals). Hypothesis-style property tests catch most of these without you having to enumerate them."),
        ("How would you adapt this to a streaming input?",
         "Replace the materialized list with an iterable parameter, accumulate state across `for` instead of relying on indexing, and return a generator if the output is also unbounded. The trickier part is anything that needs a second pass -- you may need to either buffer or reformulate the algorithm to be single-pass."),
        ("What's the idiomatic way to test this?",
         "A handful of parametrized pytest cases covering the obvious shapes (empty, single, typical, boundary) plus a Hypothesis test asserting an invariant (e.g. round-trip, monotonicity, length preservation). That combination usually catches both the cases you thought of and the ones you didn't."),
    ],
    "web": [
        ("How would I add structured logging to this?",
         "Wrap the route in a small helper that logs request id, method, path, status, and duration as JSON. Use `logging.getLogger(__name__)` and a JSON formatter; configure once at application startup. Avoid `print` -- log aggregators expect proper log records with levels."),
        ("What's the right error-handling boundary?",
         "Catch domain exceptions inside handlers and translate them to HTTP responses (422 for validation, 404 for missing, 409 for conflicts). Let unexpected exceptions bubble to a global handler that returns 500 with a correlation id but hides internals. Never leak stack traces to clients in production."),
        ("How should I add authentication here?",
         "For session-based apps, use cookie sessions with HTTPS-only and SameSite. For APIs, prefer Bearer tokens via Authorization header. Wrap auth in a dependency or before-request hook so the route bodies stay focused on domain logic. Never roll your own crypto -- use a vetted library."),
        ("What's a reasonable timeout to set?",
         "A few seconds for typical user-facing APIs, longer (30s-60s) for known-slow operations. Always set both connect and read timeouts. For dependent services, use timeouts shorter than your own SLA so you have a chance to retry or return a degraded response."),
    ],
    "data": [
        ("How does this scale to a dataset that doesn't fit in memory?",
         "Switch to chunked processing with `pd.read_csv(..., chunksize=N)` or use Polars / DuckDB / Dask, which all support out-of-core operations. For numpy, use memmapped arrays or HDF5 via h5py. If you can push the work to a database (DuckDB over Parquet is great for this), the runtime almost always wins."),
        ("How do I make this reproducible across machines?",
         "Pin numpy/sklearn/torch versions, set seeds for every random source (`numpy.random.default_rng(seed)`, `torch.manual_seed`, `random.seed`), and avoid algorithms with nondeterministic GPU kernels. Save preprocessor state (scaler stats, encoder vocab) alongside the model so test-time transforms match training."),
        ("What's the best way to deploy this model?",
         "For low-latency online use, serialize via ONNX or TorchScript and serve with FastAPI. For batch scoring, schedule a job that reads input from object storage, applies the model, and writes results back. Keep model loading outside the request path; cache the model in memory at process start."),
        ("How do I evaluate whether the result is good?",
         "Hold out a clean test set you don't touch during training. Compute the metric the business cares about (often not accuracy -- think AUC for ranking, F1 for imbalance, MAE for regressions). Compare to a sensible baseline (most-common class, last value, simple linear model) so you know whether the complex model earns its keep."),
    ],
    "debug": [
        ("What's the underlying lesson here?",
         "Most Python gotchas come from the language's mutability defaults, late binding in closures, or assumptions about iteration order. Reading the data model docs (https://docs.python.org/3/reference/datamodel.html) once carefully prevents a lot of these surprises. Linters like pyflakes and ruff catch many automatically."),
        ("How can I prevent regressions like this?",
         "Add a test that captures the failing scenario before you fix it, then watch it go green. Configure ruff or mypy in CI -- many of these patterns trigger linter warnings if you turn the right rules on. Code review by someone unfamiliar with the change often catches gotchas the author missed."),
        ("Is there a linter rule for this?",
         "Often yes -- ruff has hundreds of rules covering common bugs. For mutable-default-arg specifically, B006 from flake8-bugbear catches it. For any-comparison errors, mypy with strict_equality flags many. Adopt a strict ruff config early in projects; the cost is low and it eliminates entire classes of bugs."),
    ],
    "test": [
        ("How would I structure this for a larger test suite?",
         "Group related tests in a class or file by feature. Move shared fixtures to conftest.py at the appropriate level (project root or feature directory). Mark slow tests with `@pytest.mark.slow` and run only fast tests in pre-commit; full suite in CI. Aim for tests that finish under a second each so the feedback loop stays tight."),
        ("Should this be a unit test or an integration test?",
         "If the function under test has many collaborators stubbed, it's a unit test. If you're exercising real subsystems (DB, HTTP, filesystem) it's integration. Both are valuable: unit tests are fast and isolate failures; integration tests catch wiring bugs that unit tests miss. Keep the ratio roughly 70/30 and you'll be in good shape."),
        ("What's the right level to mock at?",
         "Mock at the boundary of your code, not inside it. Mock external systems (HTTP, DB) via injection or library-level stubs. Avoid mocking your own classes' methods -- if you find yourself doing that, the design probably needs a seam (interface, dependency-inject the dependency)."),
        ("How do I test code that depends on the current time?",
         "Inject a `clock` callable (defaulting to `time.time`) instead of calling time functions directly. In tests, pass a fake clock you control. Alternatively use `freezegun` to globally freeze time. The injection approach is cleaner long-term; freezegun is a quick fix for legacy code."),
    ],
}


def main():
    seed_int = int(hashlib.md5(b"1009309B").hexdigest(), 16) % (2**32)
    rng = random.Random(seed_int)

    entries = list(ENTRIES)
    rng.shuffle(entries)

    if len(entries) < 500:
        raise SystemExit(f"Only {len(entries)} entries -- need 500.")
    entries = entries[:500]

    # Boost 5-msg ratio to ~40% by attaching pool follow-ups to some 3-msg
    # entries that don't already have one.
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
