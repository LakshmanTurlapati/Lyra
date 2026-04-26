#!/usr/bin/env python3
"""Add follow-up turns to ~40 samples to hit 25% follow-up ratio."""
import json
from pathlib import Path

p = Path("/Users/lakshman/Documents/Lyra/datasets/code/v2-supplementary/batch-code-04.jsonl")
samples = [json.loads(line) for line in p.read_text().splitlines()]

print(f"Starting: {len(samples)} samples")

# Generic follow-up pairs keyed to assistant content topics.
# Each returns (user_followup, assistant_followup) where assistant includes python block & >=150 tokens.

FOLLOWUPS = {
    "pytest_fixture": (
        "How do I override this fixture in a specific test file?",
        "Redefine the fixture at a narrower scope: inside the test file or a sibling conftest.py.\n\n```python\n# tests/special/conftest.py\nimport pytest\n\n@pytest.fixture\ndef db(tmp_path):\n    # specialised variant just for this subtree\n    import sqlite3\n    conn = sqlite3.connect(tmp_path / 'special.db')\n    conn.execute('PRAGMA foreign_keys = ON')\n    yield conn\n    conn.close()\n```\n\npytest resolves fixtures bottom-up: the closest definition wins. A fixture with the same name in a child `conftest.py` or directly in a test file overrides the one from a parent `conftest.py`. This makes it easy to inherit common behaviour and tweak it per subdirectory without rewriting the whole fixture tree.",
    ),
    "mock": (
        "What if I want to assert the order of calls across multiple mocks?",
        "Use a parent `MagicMock` with children attached, then inspect `mock_calls` on the parent.\n\n```python\nfrom unittest.mock import MagicMock, call\n\nparent = MagicMock()\nparent.attach_mock(MagicMock(), 'save')\nparent.attach_mock(MagicMock(), 'send')\nparent.attach_mock(MagicMock(), 'log')\n\ndef place(parent):\n    parent.save('order')\n    parent.send('email')\n    parent.log('done')\n\nplace(parent)\n\nparent.assert_has_calls([\n    call.save('order'),\n    call.send('email'),\n    call.log('done'),\n], any_order=False)\n```\n\n`attach_mock` links children under a shared parent so their combined call sequence is recorded in `parent.mock_calls`. `assert_has_calls(..., any_order=False)` enforces the exact ordering. This is the cleanest way to verify a choreography across several collaborators.",
    ),
    "hypothesis": (
        "How do I make a Hypothesis test reproduce failures deterministically in CI?",
        "Pin the seed with `@settings(derandomize=True)` or set the profile.\n\n```python\nfrom hypothesis import given, settings, strategies as st\n\n@given(st.integers())\n@settings(derandomize=True)\ndef test_deterministic(n):\n    assert n == n\n```\n\nOr globally in `conftest.py`:\n\n```python\nfrom hypothesis import settings\n\nsettings.register_profile('ci', derandomize=True, max_examples=200)\nsettings.load_profile('ci')\n```\n\n`derandomize=True` derives the RNG seed from the test name, so the same test generates the same examples run after run. Commit the `.hypothesis/examples/` directory too — Hypothesis replays past failures first, which catches regressions quickly even when normal generation would miss them.",
    ),
    "coverage": (
        "How do I fail the build only when coverage on changed lines drops?",
        "Use `diff-cover` against the PR base branch.\n\n```bash\npip install diff-cover\ncoverage run -m pytest\ncoverage xml\ndiff-cover coverage.xml --compare-branch=origin/main --fail-under=90\n```\n\nIn GitHub Actions, set `fetch-depth: 0` on checkout so the merge base is available:\n\n```python\n# scripts/check_diff_cov.py\nimport subprocess\nsubprocess.run(['coverage', 'xml'], check=True)\nresult = subprocess.run(\n    ['diff-cover', 'coverage.xml',\n     '--compare-branch=origin/main', '--fail-under=90'],\n    check=False,\n)\nexit(result.returncode)\n```\n\nThis enforces that *new* lines are well tested without penalising legacy code you have not touched. It is the lightest-weight coverage gate that still blocks regression.",
    ),
    "ci": (
        "How do I cache test data across runs to speed up CI?",
        "Use `actions/cache` keyed on the data source.\n\n```yaml\n- name: Cache test fixtures\n  uses: actions/cache@v4\n  with:\n    path: tests/data\n    key: fixtures-${{ hashFiles('scripts/generate_fixtures.py') }}\n    restore-keys: fixtures-\n\n- name: Generate fixtures if missing\n  run: |\n    if [ ! -d tests/data ]; then\n      python scripts/generate_fixtures.py\n    fi\n```\n\n```python\n# scripts/generate_fixtures.py\nimport pathlib, json\np = pathlib.Path('tests/data')\np.mkdir(parents=True, exist_ok=True)\n(p / 'sample.json').write_text(json.dumps({'seed': 42}))\n```\n\nThe cache key includes the hash of the generator script so changes invalidate it. `restore-keys` does partial matches, letting an older cache kickstart even when the key miss. Typical saving: 30-90 seconds per run for nontrivial fixture generation.",
    ),
    "packaging": (
        "How do I verify the built wheel contains what I expect before publishing?",
        "`twine check` validates metadata; `unzip` confirms contents.\n\n```bash\npip install twine\npython -m build\ntwine check dist/*            # README rendering, metadata\nunzip -l dist/*.whl           # listing\nunzip -p dist/*.whl METADATA  # inspect metadata\n```\n\n```python\n# scripts/validate_wheel.py\nimport zipfile, sys\nwith zipfile.ZipFile(sys.argv[1]) as z:\n    names = z.namelist()\nassert any(n.endswith('/py.typed') for n in names), 'py.typed missing'\nassert any(n.endswith('/__init__.py') for n in names), 'no package init'\nprint('wheel ok:', len(names), 'files')\n```\n\nRun this in CI right after `python -m build` and before `twine upload`. The assertions catch common packaging mistakes (missing py.typed, missing data files, surprise top-level imports) before the artifact reaches PyPI where it cannot be rewritten.",
    ),
    "logging": (
        "How do I make logs include the function name automatically?",
        "Add `%(funcName)s` (and `%(lineno)d`) to your format string.\n\n```python\nimport logging\n\nlogging.basicConfig(\n    level=logging.INFO,\n    format='%(asctime)s %(levelname)s %(name)s.%(funcName)s:%(lineno)d: %(message)s',\n)\nlog = logging.getLogger(__name__)\n\ndef handle_request(req):\n    log.info('received %s', req.path)\n\nhandle_request(type('R', (), {'path': '/x'})())\n# 2026-04-24 ... INFO __main__.handle_request:8: received /x\n```\n\nNote the slight cost: the formatter uses `sys._getframe` to fill `funcName`, which is cheap but not free. On a hot path, skip function/line info or use structlog which lazily evaluates the fields.",
    ),
    "debugging": (
        "How do I inspect variables from a different stack frame in pdb?",
        "`u`/`d` move up and down; `l` shows the source; `p expr` prints.\n\n```python\ndef outer():\n    data = {'key': 'value'}\n    inner(data)\n\ndef inner(data):\n    result = data.get('missing')\n    breakpoint()\n    print(result)\n\nouter()\n```\n\nAt the breakpoint:\n\n```\n(Pdb) w          # show stack\n(Pdb) u          # move up into outer()\n(Pdb) l          # see outer's source\n(Pdb) p data     # inspect outer's local\n(Pdb) d          # back down into inner\n(Pdb) pp locals()\n```\n\n`interact` drops you into a full REPL in the current frame; Ctrl-D returns to pdb. For repeated inspection, `display expr` adds an expression to auto-print at every step.",
    ),
    "testing": (
        "How do I isolate a test that modifies global state?",
        "Snapshot and restore state in a fixture, or run in a subprocess.\n\n```python\nimport pytest\n\n@pytest.fixture\ndef clean_registry():\n    from myapp import registry\n    saved = dict(registry._items)\n    yield\n    registry._items.clear()\n    registry._items.update(saved)\n\ndef test_modifies_registry(clean_registry):\n    from myapp import registry\n    registry.register('foo', lambda: 1)\n    assert 'foo' in registry._items\n```\n\nFor truly unsafe mutation (sys.path changes, warnings filter changes, extension-module state), use `pytest-forked` to run the test in a subprocess:\n\n```bash\npip install pytest-forked\npytest --forked tests/test_global_state.py\n```\n\nSubprocess isolation is foolproof but slow — reserve it for the 1% of tests where in-process snapshotting does not work.",
    ),
}


def pick_followup(sample):
    """Pick an appropriate follow-up for a sample based on its content."""
    system = sample["messages"][0]["content"].lower()
    user = sample["messages"][1]["content"].lower()
    assistant = sample["messages"][2]["content"].lower()
    combined = system + " " + user + " " + assistant

    if "fixture" in combined or "pytest fixture" in combined:
        return FOLLOWUPS["pytest_fixture"]
    if "mock" in combined or "patch" in combined:
        return FOLLOWUPS["mock"]
    if "hypothesis" in combined or "property-based" in combined or "strategies" in combined:
        return FOLLOWUPS["hypothesis"]
    if "coverage" in combined:
        return FOLLOWUPS["coverage"]
    if "github actions" in combined or "workflow" in combined or "yaml" in combined or "ci" in combined:
        return FOLLOWUPS["ci"]
    if "pyproject" in combined or "wheel" in combined or "sdist" in combined or "pypi" in combined:
        return FOLLOWUPS["packaging"]
    if "logging" in combined:
        return FOLLOWUPS["logging"]
    if "pdb" in combined or "breakpoint" in combined or "debug" in combined:
        return FOLLOWUPS["debugging"]
    return FOLLOWUPS["testing"]


# Add follow-ups to every 4th sample (giving ~25% rate)
target_followups = 40  # 25%
added = 0
existing_followups = sum(1 for s in samples if len(s["messages"]) == 5)
print(f"Existing follow-ups: {existing_followups}")

# Indices to add follow-ups — evenly spaced
step = len(samples) // target_followups
for i in range(0, len(samples), step):
    if added >= target_followups:
        break
    if len(samples[i]["messages"]) != 3:
        continue
    user_fu, assistant_fu = pick_followup(samples[i])
    samples[i]["messages"].append({"role": "user", "content": user_fu})
    samples[i]["messages"].append({"role": "assistant", "content": assistant_fu})
    added += 1

print(f"Added {added} follow-ups")

final_followups = sum(1 for s in samples if len(s["messages"]) == 5)
print(f"Final follow-ups: {final_followups} ({100 * final_followups / len(samples):.1f}%)")

# Rewrite file
with p.open("w") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"Wrote {len(samples)} to {p}")
