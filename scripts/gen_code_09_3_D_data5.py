# SPDX-License-Identifier: MIT
"""Debugging (47) + testing (50). Wave 4 part 5."""


def register_all(add):
    # ---- debugging (47) ----
    add("debug",
        "Refactor this nested if-else into something flatter.\n```python\ndef classify(x):\n    if x > 0:\n        if x < 10:\n            return 'small'\n        else:\n            if x < 100:\n                return 'medium'\n            else:\n                return 'large'\n    else:\n        return 'nonpositive'\n```",
        "Use early returns to flatten:\n```python\ndef classify(x):\n    if x <= 0:\n        return 'nonpositive'\n    if x < 10:\n        return 'small'\n    if x < 100:\n        return 'medium'\n    return 'large'\n```\nGuard clauses + cascading `if/return` reads top-to-bottom. Or for ranges, `bisect` on a list of thresholds: `('small', 'medium', 'large')[bisect_left([10, 100], x)]` after handling the nonpositive edge.")

    add("debug",
        "This loop double-counts on the boundary. What's wrong?\n```python\ndef count_in_range(xs, lo, hi):\n    return sum(1 for x in xs if lo <= x and x <= hi)\n```",
        "It's not double-counting; the bug is conceptual -- both bounds are inclusive, which is rarely what callers want. Document or change to half-open: `lo <= x < hi`. Always pick a convention (half-open is the Pythonic default, matching `range`) and stick to it across your API.")

    add("debug",
        "Why is `df.iterrows()` so slow?",
        "`iterrows` returns Python objects (Series) per row, doing per-row Python overhead and dtype conversion. Vectorize where possible (`df['c'] = df['a'] + df['b']`). When you must iterate, `df.itertuples(index=False)` is much faster. For the tiny minority of cases that genuinely need per-row Python logic, build a list and assign back.")

    add("debug",
        "My function is failing because of operator precedence. Spot the bug.\n```python\ndef ok(x):\n    return 0 < x < 10 or x == 100\n```",
        "It's correct -- chained comparisons bind tighter than `or`. The likely confusion is that `0 < x < 10` looks like `(0 < x) < 10` which would be wrong, but Python actually treats it as `0 < x and x < 10`. If readability is suffering, parenthesize: `return (0 < x < 10) or x == 100`.")

    add("debug",
        "Refactor this duplicated try/except scaffolding.\n```python\ntry:\n    a = parse_int(x)\nexcept ValueError:\n    a = 0\ntry:\n    b = parse_int(y)\nexcept ValueError:\n    b = 0\n```",
        "Extract the pattern:\n```python\ndef safe_int(s, default=0):\n    try:\n        return parse_int(s)\n    except ValueError:\n        return default\n\na = safe_int(x); b = safe_int(y)\n```\nAny time you've copy-pasted try/except, ask whether the wrapper itself is the function you actually want.")

    add("debug",
        "This regex is catastrophically slow on long strings. Why?\n```python\nimport re\nre.match(r'(a+)+b', 'a' * 30)\n```",
        "Catastrophic backtracking from nested quantifiers. The regex engine tries exponentially many ways to split the run of `a`s. Rewrite without the nested `+`: `r'a+b'`. In general, audit regexes for `(...+)+`, `(...*)*`, or alternations that overlap. Or switch to the `regex` module on PyPI which has a linear-time alternative.")

    add("debug",
        "Refactor this magic-number-laden function.\n```python\ndef discount(price):\n    if price > 100:\n        return price * 0.9\n    if price > 50:\n        return price * 0.95\n    return price\n```",
        "Pull thresholds and rates into named constants:\n```python\nTIERS = [(100, 0.10), (50, 0.05)]\n\ndef discount(price):\n    for threshold, rate in TIERS:\n        if price > threshold:\n            return price * (1 - rate)\n    return price\n```\nNow the policy lives in data, easy to extend or override in tests. The discount-rate is also more honest as the multiplier you subtract.")

    add("debug",
        "Why does `is` sometimes work for string equality and sometimes not?",
        "`is` tests identity, not equality. Python interns short string literals as an optimization, so `'foo' is 'foo'` may be True -- but it's not guaranteed and isn't true for runtime-built strings. Always use `==` for value equality. Linters flag `is` against literals (PLR0124 in ruff).")

    add("debug",
        "I'm hitting a deadlock in this code.\n```python\nimport threading\nlock_a, lock_b = threading.Lock(), threading.Lock()\n# thread 1\nwith lock_a:\n    with lock_b:\n        ...\n# thread 2\nwith lock_b:\n    with lock_a:\n        ...\n```",
        "Classic deadlock: threads acquire locks in opposite orders. Establish a lock-ordering rule (always acquire `lock_a` first) and never violate it. For more than two locks, sort by `id(lock)` and acquire in that order. Or use a single higher-level lock guarding the critical section -- coarser but simpler.")

    add("debug",
        "Refactor this code that opens and closes the file manually.\n```python\nf = open('data.txt')\ntry:\n    data = f.read()\nfinally:\n    f.close()\n```",
        "Use a context manager:\n```python\nwith open('data.txt', encoding='utf-8') as f:\n    data = f.read()\n```\nWhile we're here, set the encoding explicitly -- the platform default has caused real bugs across OSes. Python 3.10+ even surfaces a warning for missing `encoding=` under `-X warn_default_encoding`.")

    add("debug",
        "This sorted output isn't actually sorted. What's wrong?\n```python\nxs = ['10', '2', '1']\nxs.sort()\n```",
        "`xs.sort()` returns `None` (it sorts in place). The list itself is now `['1', '10', '2']` -- sorted lexicographically because they're strings. Two fixes: convert first (`xs.sort(key=int)`) and remember sort is in-place. Use `sorted(xs, key=int)` if you want a new list.")

    add("debug",
        "Refactor this function that builds a string with `+=`.\n```python\ndef build(items):\n    s = ''\n    for x in items:\n        s += str(x) + ','\n    return s\n```",
        "String concatenation with `+=` is O(n\u00b2) -- each += copies the whole string. Use `str.join`:\n```python\ndef build(items):\n    return ','.join(str(x) for x in items)\n```\nClearer and faster. CPython has a special case that makes `+=` linear in some versions, but you shouldn't rely on it.")

    add("debug",
        "My code crashes with `KeyError` only on Python 3.6 but not 3.7+. Why?",
        "Dict iteration order became guaranteed in 3.7. Code relying on insertion-order behavior may have worked accidentally before. Check whether something assumes ordered keys (e.g. compares hashes of dict-derived strings). Pin Python 3.10+ for new code; 3.6 is past EOL.")

    add("debug",
        "Refactor: convert this function-with-flag into two functions.\n```python\ndef fetch(url, async_mode=False):\n    if async_mode:\n        return async_fetch(url)\n    return sync_fetch(url)\n```",
        "Boolean flags that switch between fundamentally different code paths are a smell. Just expose both:\n```python\ndef fetch(url):\n    return sync_fetch(url)\n\nasync def fetch_async(url):\n    return await async_fetch(url)\n```\nThe caller knows whether they're in an async context; let them pick.")

    add("debug",
        "I'm getting `_csv.Error: line contains NUL` when reading a CSV.",
        "The file has embedded NUL bytes -- often because it's actually UTF-16 or has a BOM that wasn't decoded. Open with the right encoding: `open(path, encoding='utf-16')`. Or sniff with `chardet`/`charset-normalizer`. As a quick test, run `file path.csv` in the shell to see what the OS thinks it is.")

    add("debug",
        "Refactor this code with too many parameters.\n```python\ndef create_user(first_name, last_name, email, age, address, city, state, zip_code):\n    ...\n```",
        "Group related fields into a dataclass:\n```python\nfrom dataclasses import dataclass\n\n@dataclass\nclass Address:\n    street: str\n    city: str\n    state: str\n    zip_code: str\n\n@dataclass\nclass User:\n    first_name: str\n    last_name: str\n    email: str\n    age: int\n    address: Address\n```\nTakes a `User` (or builds one). Six unrelated parameters become two structured ones.")

    add("debug",
        "Why does my `requests.Session` leak file descriptors?",
        "You're creating sessions in a loop and never closing them. Either reuse one session across calls, or use `with requests.Session() as s:`. The same applies to `httpx.Client` and database connections -- always use context managers or explicitly close.")

    add("debug",
        "Refactor: this nested for-loop is hard to read.\n```python\nresult = []\nfor row in matrix:\n    for x in row:\n        if x > 0:\n            result.append(x * 2)\n```",
        "Flatten and filter in a comprehension:\n```python\nresult = [x * 2 for row in matrix for x in row if x > 0]\n```\nReads left-to-right like the original loop. For large arrays, a numpy version (`arr[arr > 0] * 2`) is dramatically faster.")

    add("debug",
        "I'm getting `RecursionError` in a tree traversal that should be shallow.",
        "Likely a cycle that you didn't mean to create. Add a `visited` set and skip nodes you've seen. If the structure is intentionally cyclic, switch to iterative DFS with an explicit stack -- the recursion limit in Python isn't a great place to test the depth of your data.")

    add("debug",
        "Refactor this if-elif chain that maps strings to functions.\n```python\ndef dispatch(cmd, *args):\n    if cmd == 'add':\n        return add(*args)\n    if cmd == 'sub':\n        return sub(*args)\n    if cmd == 'mul':\n        return mul(*args)\n    raise ValueError(cmd)\n```",
        "Use a dispatch dict:\n```python\nDISPATCH = {'add': add, 'sub': sub, 'mul': mul}\n\ndef dispatch(cmd, *args):\n    fn = DISPATCH.get(cmd)\n    if fn is None:\n        raise ValueError(cmd)\n    return fn(*args)\n```\nAdding new commands is a one-liner. For Python 3.10+, `match` is also clean -- pick whichever your team finds more readable.")

    add("debug",
        "My `asyncio` task vanishes silently. Why?",
        "Bare `asyncio.create_task(coro)` returns a Task that, if not stored, can be garbage-collected mid-flight, swallowing exceptions. Always store the reference (`tasks.append(task)`) and `await` it later. Or use `asyncio.gather` / `TaskGroup` (3.11+) which manage lifetime for you.")

    add("debug",
        "Refactor: this function returns mixed types making callers branch.\n```python\ndef get_user(uid):\n    user = db.get(uid)\n    if user is None:\n        return None\n    return user\n```",
        "If `None` is a meaningful 'not found', keep `Optional[User]` but be explicit at type-check time. If callers shouldn't have to handle missing, raise:\n```python\ndef get_user(uid: int) -> User:\n    user = db.get(uid)\n    if user is None:\n        raise UserNotFound(uid)\n    return user\n```\nPick one based on whether 'not found' is exceptional. Mixing styles across the codebase is what makes callers branch.")

    add("debug",
        "I get `FileNotFoundError` when running my test from a different directory.",
        "Hard-coded relative paths break when CWD changes. Use `Path(__file__).parent / 'fixtures' / 'data.csv'` to resolve relative to the test file. Or set test fixtures via pytest's `tmp_path` and copy/load with absolute paths.")

    add("debug",
        "Refactor: this code allocates a list just to check length.\n```python\nif len(list(filter(lambda x: x > 0, items))) > 5:\n    ...\n```",
        "Don't materialize the whole list; short-circuit with `any` and a slice... actually use `sum(1 for ...)` plus a generator, or smarter: `sum(1 for x in items if x > 0) > 5` still walks the whole list. The minimum-work form is:\n```python\nfrom itertools import islice\nif sum(1 for _ in islice((x for x in items if x > 0), 6)) > 5:\n    ...\n```\nFor this kind of threshold check, count up to threshold+1 and stop.")

    add("debug",
        "My `pytest` fixture isn't being applied. Why?",
        "Either the test isn't requesting it (parameter name must match the fixture name) or the fixture is in the wrong file. Fixtures in `conftest.py` apply to tests in that directory tree; module-local fixtures only apply in that module. Check the scope (`function`, `module`, `session`) too -- session-scoped fixtures running once may surprise you.")

    add("debug",
        "Refactor: this code compares floats for equality.\n```python\nif x == 0.1 + 0.2:\n    ...\n```",
        "Floating-point equality is fragile: `0.1 + 0.2 != 0.3`. Use `math.isclose` with explicit tolerances:\n```python\nimport math\nif math.isclose(x, 0.3, rel_tol=1e-9):\n    ...\n```\nFor monetary values, `decimal.Decimal` is the right type from the start.")

    add("debug",
        "`shutil.rmtree` fails on Windows with permission errors. How do I fix it?",
        "Windows holds locks on files until handles close, and read-only files can't be deleted. Pass an `onerror` callback that chmods to writable and retries: `shutil.rmtree(path, onerror=lambda fn, p, _: (os.chmod(p, 0o777), fn(p)))`. The `send2trash` package is more robust if you don't actually need permanent deletion.")

    add("debug",
        "Refactor: this `if x: doA(); doB()` is hiding a subtle bug.\n```python\nif x: do_a(); do_b()\n```",
        "On one line, only `do_a()` is conditional -- `do_b()` always runs. The semicolon syntax doesn't bind to the `if`. Always use a block:\n```python\nif x:\n    do_a()\n    do_b()\n```\nRuff/ruff-format prevents this style entirely.")

    add("debug",
        "My `dataclass` doesn't compare equal even with same fields. Why?",
        "You're comparing across class versions or one instance has fields the other doesn't (e.g. you added a field with a default). Or you've set `eq=False`. Default dataclass `__eq__` compares the tuple of all fields. For partial comparison, use a custom method or compare specific tuples.")

    add("debug",
        "Refactor: this `time.sleep(60)` busy-loop in a test is slowing CI.",
        "Don't sleep in tests. Inject the clock or use `freezegun`:\n```python\nimport freezegun\n\n@freezegun.freeze_time('2024-01-01')\ndef test_x():\n    ...\n```\nFor async timeouts, use `asyncio.sleep` with `asyncio.run` and a fake event loop, or refactor the code to accept a `clock` callable.")

    add("debug",
        "I'm getting `TypeError: cannot pickle 'generator' object`.",
        "Generators are not picklable -- they hold a frame, locals, and a Python instruction pointer. Materialize to a list before pickling, or redesign so the work is dispatched as data (function name + args) and reconstructed in the worker. For multiprocessing specifically, `imap` or `imap_unordered` lets you stream from the parent without pickling the generator.")

    add("debug",
        "Refactor: this code mutates the input list.\n```python\ndef sort_evens(xs):\n    xs.sort()\n    return [x for x in xs if x % 2 == 0]\n```",
        "Mutating callers' inputs surprises them. Sort a copy:\n```python\ndef sort_evens(xs):\n    return [x for x in sorted(xs) if x % 2 == 0]\n```\nWhen mutation is intended, name the function accordingly (`sort_in_place`) and return `None`.")

    add("debug",
        "My logging shows the wrong log level. Why?",
        "Likely two `basicConfig` calls -- the second is a no-op because the root logger already has handlers. Either configure once at startup, or use `logging.dictConfig` for explicit, declarative setup. In libraries, never call `basicConfig` -- only the application should configure.")

    add("debug",
        "Refactor: this code uses `eval` to dispatch.\n```python\nfn_name = 'compute_' + kind\nresult = eval(f'{fn_name}(data)')\n```",
        "`eval` is a security and maintainability disaster. Use `getattr` (if methods on a class) or a dispatch dict:\n```python\nDISPATCH = {'sum': compute_sum, 'avg': compute_avg}\nresult = DISPATCH[kind](data)\n```\nA `KeyError` is also more debuggable than a `NameError` from a typo'd function name.")

    add("debug",
        "I get `ValueError: I/O operation on closed file`.",
        "You're returning a file object or a generator that reads from one, then closing the file. Either read everything before returning, or use `pathlib.Path.read_text()` for the simple case. For streaming reads, the file must stay open while the consumer iterates -- pass the open file and let the caller close.")

    add("debug",
        "Refactor: this function takes a dict where named arguments would be clearer.\n```python\ndef render(opts):\n    width = opts.get('width', 80)\n    height = opts.get('height', 24)\n    color = opts.get('color', 'white')\n    ...\n```",
        "Use explicit keyword arguments:\n```python\ndef render(*, width: int = 80, height: int = 24, color: str = 'white'):\n    ...\n```\nThe `*,` makes them keyword-only; mypy and IDEs will catch typos. Dict-of-options is appropriate only when keys are dynamic.")

    add("debug",
        "Why does my `pytest` discover no tests?",
        "Test files must match `test_*.py` or `*_test.py`, classes must be `Test*` (without `__init__`), and functions `test_*`. Your `pyproject.toml` may also have `testpaths` restricting where pytest looks. Run `pytest --collect-only` to see what it found and why.")

    add("debug",
        "Refactor: this code tests too much in one function.\n```python\ndef test_user_full():\n    user = create_user('a@b.c')\n    assert user.email == 'a@b.c'\n    user.update_name('Alice')\n    assert user.name == 'Alice'\n    user.delete()\n    assert User.get(user.id) is None\n```",
        "Split into focused tests; each verifies one behavior:\n```python\ndef test_create_sets_email(): ...\ndef test_update_changes_name(): ...\ndef test_delete_removes_record(): ...\n```\nFailures point directly at what broke. A failing 'full lifecycle' test tells you nothing about which step is wrong.")

    add("debug",
        "I'm seeing memory grow unbounded in a long-running script. What now?",
        "Common causes: caches without size limits (`functools.lru_cache(maxsize=None)`), accumulating logger handlers, growing module-level lists, or holding references to closed objects. Use `tracemalloc` or `memory-profiler` to find allocations; `objgraph` to find unexpected references.")

    add("debug",
        "Refactor: this code parses dates by string-slicing.\n```python\ny = s[:4]; m = s[5:7]; d = s[8:10]\n```",
        "Use `datetime.fromisoformat` (3.7+) or `dateutil.parser.parse` for messy formats:\n```python\nimport datetime as dt\nday = dt.date.fromisoformat(s[:10])\n```\nString-slicing dates breaks the moment a Z, timezone, or alternate format appears.")

    add("debug",
        "Why does my unicode string look corrupted in the terminal?",
        "Likely the terminal's encoding doesn't match Python's stdout encoding. Set `PYTHONIOENCODING=utf-8` (or run with `python -X utf8 ...`). On Windows, `chcp 65001` switches the console to UTF-8. The `colorama` package fixes ANSI handling on Windows too.")

    add("debug",
        "Refactor: this code uses bare `except:`.\n```python\ntry:\n    do_thing()\nexcept:\n    log.error('failed')\n```",
        "Bare except catches `KeyboardInterrupt` and `SystemExit`, which usually you don't want. Be specific:\n```python\ntry:\n    do_thing()\nexcept Exception as e:\n    log.exception('failed')\n```\n`log.exception` automatically captures the traceback. Catching specific exception types is even better when you can.")

    add("debug",
        "Why does my coroutine never run?",
        "You probably called it without awaiting, getting just the coroutine object. `asyncio.iscoroutine(result)` will be True. Either `await` it, schedule it with `asyncio.create_task`, or `asyncio.run` at the top-level entry point.")

    add("debug",
        "Refactor: this code copies a list explicitly.\n```python\nresult = []\nfor x in xs:\n    result.append(x)\n```",
        "Just use the constructor:\n```python\nresult = list(xs)\n```\nOr `xs.copy()` for the same effect. The loop is also slower because it goes through Python opcodes per element.")

    add("debug",
        "I'm getting `RuntimeError: Working outside of application context` in Flask.",
        "You're using `current_app`, `g`, or `session` outside a request handler -- in module-level code or a thread. Wrap with `with app.app_context():` for app-level access, or `with app.test_request_context():` for request-level. For background jobs, pass needed values explicitly rather than relying on context.")

    add("debug",
        "Refactor: this comprehension does too much in one expression.\n```python\nresult = [(k, sum(v for v in d[k] if v > 0) / max(1, len([v for v in d[k] if v > 0]))) for k in d]\n```",
        "Pull the inner work into a helper:\n```python\ndef avg_positive(xs):\n    pos = [x for x in xs if x > 0]\n    return sum(pos) / len(pos) if pos else 0.0\n\nresult = [(k, avg_positive(d[k])) for k in d]\n```\nReadable, testable, and faster (avoids walking the inner list twice).")

    # ---- testing (50) ----
    add("test",
        "Write a pytest test for a function that adds two numbers.",
        "```python\nfrom myapp import add\n\ndef test_add():\n    assert add(2, 3) == 5\n    assert add(-1, 1) == 0\n    assert add(0, 0) == 0\n```\nKeep cases small and focused; one logical idea per test function.")

    add("test",
        "Implement a parametrized pytest test for a string utility.",
        "```python\nimport pytest\nfrom myapp import slugify\n\n@pytest.mark.parametrize('raw, expected', [\n    ('Hello World', 'hello-world'),\n    ('  spaces  ', 'spaces'),\n    ('Camel/Case', 'camel-case'),\n], ids=['basic', 'trim', 'punct'])\ndef test_slugify(raw, expected):\n    assert slugify(raw) == expected\n```\nIDs make failure output readable when one case fails.")

    add("test",
        "Write a pytest fixture that creates a temporary database.",
        "```python\nimport pytest\nimport sqlite3\n\n@pytest.fixture\ndef db(tmp_path):\n    path = tmp_path / 'test.db'\n    conn = sqlite3.connect(path)\n    conn.execute('CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)')\n    yield conn\n    conn.close()\n```\n`tmp_path` cleans up automatically; `yield` separates setup from teardown.")

    add("test",
        "Write a test that mocks an HTTP call.",
        "```python\nfrom unittest.mock import patch\nfrom myapp import fetch\n\ndef test_fetch():\n    with patch('myapp.requests.get') as mock_get:\n        mock_get.return_value.json.return_value = {'ok': True}\n        mock_get.return_value.status_code = 200\n        assert fetch('https://x') == {'ok': True}\n```\nFor httpx, `httpx.MockTransport` is more idiomatic than patching.")

    add("test",
        "Implement a test using pytest's `monkeypatch` for environment variables.",
        "```python\ndef test_reads_env(monkeypatch):\n    monkeypatch.setenv('API_KEY', 'test-key')\n    from myapp import config\n    assert config.api_key == 'test-key'\n```\n`monkeypatch` undoes changes after the test, so other tests aren't affected.")

    add("test",
        "Write a test that checks an exception is raised.",
        "```python\nimport pytest\nfrom myapp import divide\n\ndef test_divide_by_zero():\n    with pytest.raises(ZeroDivisionError, match='division by zero'):\n        divide(1, 0)\n```\nAssert on `match=` so the test fails if a different exception with similar message slips through.")

    add("test",
        "Write a fixture that provides a mock httpx client.",
        "```python\nimport httpx\nimport pytest\n\n@pytest.fixture\ndef http_client():\n    def handler(request: httpx.Request) -> httpx.Response:\n        return httpx.Response(200, json={'ok': True})\n    transport = httpx.MockTransport(handler)\n    with httpx.Client(transport=transport) as client:\n        yield client\n```\n`MockTransport` is the supported way to stub httpx in tests.")

    add("test",
        "Implement a parametrized test using `pytest.param` with marks.",
        "```python\nimport pytest\n\n@pytest.mark.parametrize('x, expected', [\n    (1, 1),\n    pytest.param(2, 4, marks=pytest.mark.slow),\n    pytest.param(0, 0, marks=pytest.mark.xfail(reason='known')),\n])\ndef test_square(x, expected):\n    assert x * x == expected\n```\n`pytest.param` lets per-case marks ride along.")

    add("test",
        "Write a test that captures log output.",
        "```python\nimport logging\nfrom myapp import do_thing\n\ndef test_logs(caplog):\n    caplog.set_level(logging.INFO)\n    do_thing()\n    assert 'started' in caplog.text\n    assert any('finished' in r.message for r in caplog.records)\n```\nUse `caplog.records` for structured assertions; `caplog.text` is the rendered text.")

    add("test",
        "Implement a test using pytest's `tmp_path`.",
        "```python\nfrom myapp import write_data\n\ndef test_writes_file(tmp_path):\n    path = tmp_path / 'out.txt'\n    write_data(path, 'hello')\n    assert path.read_text() == 'hello'\n```\n`tmp_path` is per-test; cleanup is automatic.")

    add("test",
        "Write a session-scoped fixture for a shared resource.",
        "```python\nimport pytest\n\n@pytest.fixture(scope='session')\ndef expensive_setup():\n    resource = build_thing()\n    yield resource\n    resource.close()\n```\nSession scope means once per test run -- great for spinning up a Docker container or a shared DB.")

    add("test",
        "Implement a test that uses `freezegun` for time-dependent code.",
        "```python\nimport datetime as dt\nfrom freezegun import freeze_time\nfrom myapp import current_year\n\n@freeze_time('2030-06-15')\ndef test_current_year():\n    assert current_year() == 2030\n```\n`freeze_time` patches `datetime` and `time` globally for the test scope.")

    add("test",
        "Write a test for an async function.",
        "```python\nimport pytest\nfrom myapp import fetch_data\n\n@pytest.mark.asyncio\nasync def test_fetch_data():\n    result = await fetch_data('id-1')\n    assert result['id'] == 'id-1'\n```\nRequires `pytest-asyncio`; configure mode in pyproject (`asyncio_mode = 'auto'`) to drop the marker.")

    add("test",
        "Implement a fixture that auto-uses for setup.",
        "```python\nimport pytest\n\n@pytest.fixture(autouse=True)\ndef silence_logs(caplog):\n    caplog.set_level('WARNING')\n```\n`autouse=True` applies to every test in scope -- use sparingly, but handy for global setup.")

    add("test",
        "Write a test that uses `assert_called_with` to verify a call.",
        "```python\nfrom unittest.mock import Mock\n\ndef test_passes_args():\n    cb = Mock()\n    do_work(cb, x=5)\n    cb.assert_called_once_with(x=5)\n```\n`assert_called_once_with` catches both 'never called' and 'called wrong'.")

    add("test",
        "Implement a test using `pytest-mock`'s `mocker` fixture.",
        "```python\ndef test_with_mocker(mocker):\n    fake_get = mocker.patch('myapp.requests.get')\n    fake_get.return_value.json.return_value = {'ok': True}\n    assert load() == {'ok': True}\n```\n`mocker` undoes patches automatically -- no `with` statement nesting.")

    add("test",
        "Write a test verifying a class method is called once.",
        "```python\nfrom unittest.mock import patch\n\ndef test_calls_save():\n    with patch.object(MyClass, 'save') as mock_save:\n        do_create()\n        mock_save.assert_called_once()\n```\n`patch.object` patches a single attribute -- safer than module-level patching when the import path is uncertain.")

    add("test",
        "Implement a test that reads from a fixture file.",
        "```python\nfrom pathlib import Path\n\nFIXTURES = Path(__file__).parent / 'fixtures'\n\ndef test_parse_sample():\n    raw = (FIXTURES / 'sample.csv').read_text()\n    assert parse(raw)[0]['id'] == 1\n```\nResolving relative to `__file__` makes the test independent of CWD.")

    add("test",
        "Write a test for a Flask endpoint.",
        "```python\nfrom myapp import create_app\n\ndef test_hello():\n    client = create_app().test_client()\n    r = client.get('/hello')\n    assert r.status_code == 200\n    assert r.json == {'message': 'hello'}\n```\nUse the app factory pattern so each test gets a fresh app.")

    add("test",
        "Implement a test for a FastAPI endpoint with TestClient.",
        "```python\nfrom fastapi.testclient import TestClient\nfrom myapp.main import app\n\nclient = TestClient(app)\n\ndef test_create_item():\n    r = client.post('/items', json={'name': 'x', 'price': 1.0})\n    assert r.status_code == 200\n    assert r.json()['name'] == 'x'\n```\n`TestClient` is sync and uses httpx internally.")

    add("test",
        "Write a fixture that mocks the system clock.",
        "```python\nimport pytest\n\n@pytest.fixture\ndef clock(monkeypatch):\n    state = {'now': 1_700_000_000.0}\n    monkeypatch.setattr('time.time', lambda: state['now'])\n    return state\n```\nReturning the state dict lets tests advance time: `clock['now'] += 60`.")

    add("test",
        "Implement a test that asserts approximate equality.",
        "```python\nimport pytest\nfrom myapp import compute\n\ndef test_compute():\n    assert compute(0.1, 0.2) == pytest.approx(0.3)\n```\n`pytest.approx` handles floating-point tolerance with sane defaults.")

    add("test",
        "Write a test using `xfail` for a known bug.",
        "```python\nimport pytest\n\n@pytest.mark.xfail(reason='issue #123: edge case fails')\ndef test_edge_case():\n    assert handle('') == 'empty'\n```\n`xfail` records the failure without failing the suite. Pair with a referenced issue so it doesn't linger.")

    add("test",
        "Implement a test that skips on missing dependency.",
        "```python\nimport pytest\n\nnumpy = pytest.importorskip('numpy')\n\ndef test_numpy_thing():\n    assert numpy.array([1, 2]).sum() == 3\n```\n`importorskip` skips the whole module if the import fails.")

    add("test",
        "Write a test that uses `pytest.mark.slow` to mark slow tests.",
        "```python\nimport pytest\n\n@pytest.mark.slow\ndef test_big_load():\n    process_million_rows()\n```\nIn `pyproject.toml`: `addopts = '-m \"not slow\"'`. Run slow tests in CI with `pytest -m slow`.")

    add("test",
        "Implement a test that asserts a list is sorted.",
        "```python\nfrom myapp import sort_users\n\ndef test_sort_users():\n    users = sort_users([{'age': 30}, {'age': 20}, {'age': 25}], by='age')\n    ages = [u['age'] for u in users]\n    assert ages == sorted(ages)\n```\nAsserting via `sorted` is robust to changes in the underlying ordering algorithm.")

    add("test",
        "Write a test using `unittest.TestCase` with setUp.",
        "```python\nimport unittest\nfrom myapp import Calc\n\nclass CalcTest(unittest.TestCase):\n    def setUp(self):\n        self.c = Calc()\n    def test_add(self):\n        self.assertEqual(self.c.add(2, 3), 5)\n```\nPytest discovers `unittest.TestCase` subclasses too; convert as you have time, no big bang required.")

    add("test",
        "Implement a fixture that yields a temporary working directory.",
        "```python\nimport os, pytest\n\n@pytest.fixture\ndef cwd(tmp_path, monkeypatch):\n    monkeypatch.chdir(tmp_path)\n    return tmp_path\n```\n`monkeypatch.chdir` restores the original CWD after the test.")

    add("test",
        "Write a test that asserts a function is idempotent.",
        "```python\nfrom myapp import normalize\n\ndef test_idempotent():\n    once = normalize('Hello World')\n    twice = normalize(once)\n    assert once == twice\n```\nIdempotence tests catch bugs where the function double-applies a transformation.")

    add("test",
        "Implement a test that checks a generator yields the right sequence.",
        "```python\nfrom myapp import countdown\n\ndef test_countdown():\n    assert list(countdown(3)) == [3, 2, 1]\n```\nMaterialize with `list` to compare; iterators are otherwise hard to introspect.")

    add("test",
        "Write a Hypothesis property test for a serializer.",
        "```python\nfrom hypothesis import given\nfrom hypothesis import strategies as st\nfrom myapp import dumps, loads\n\n@given(st.dictionaries(st.text(), st.integers()))\ndef test_roundtrip(d):\n    assert loads(dumps(d)) == d\n```\nProperty tests find edge cases you wouldn't think of by example.")

    add("test",
        "Implement a test using `caplog` to verify a specific record.",
        "```python\nimport logging\nfrom myapp import warn_on_low\n\ndef test_warns(caplog):\n    caplog.set_level(logging.WARNING)\n    warn_on_low(5)\n    record = next(r for r in caplog.records if r.levelno == logging.WARNING)\n    assert 'low' in record.message\n```\nTraversing `caplog.records` lets you assert structured log fields too.")

    add("test",
        "Write a fixture that builds a sample DataFrame.",
        "```python\nimport pandas as pd\nimport pytest\n\n@pytest.fixture\ndef sample_df():\n    return pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})\n```\nKeep test fixtures small -- big DataFrames make tests slow and brittle.")

    add("test",
        "Implement a parametrized test with `indirect=True` for fixture parameters.",
        "```python\nimport pytest\n\n@pytest.fixture\ndef user(request):\n    return {'role': request.param}\n\n@pytest.mark.parametrize('user', ['admin', 'guest'], indirect=True)\ndef test_role(user):\n    assert user['role'] in {'admin', 'guest'}\n```\n`indirect=True` passes the parameter to the fixture.")

    add("test",
        "Write a test that asserts an asyncio task was scheduled.",
        "```python\nimport asyncio, pytest\n\n@pytest.mark.asyncio\nasync def test_schedules_task():\n    task = asyncio.create_task(do_async())\n    await task\n    assert task.done()\n    assert task.exception() is None\n```\nCheck both completion and exception state -- tasks can finish via failure too.")

    add("test",
        "Implement a fixture that mocks a Redis client.",
        "```python\nimport pytest\nfrom fakeredis import FakeRedis\n\n@pytest.fixture\ndef redis():\n    return FakeRedis(decode_responses=True)\n```\n`fakeredis` is API-compatible enough for most tests; use real Redis for integration tests.")

    add("test",
        "Write a test that uses `tmp_path_factory` for shared temp dirs.",
        "```python\nimport pytest\n\n@pytest.fixture(scope='session')\ndef shared_dir(tmp_path_factory):\n    return tmp_path_factory.mktemp('shared')\n```\nSession-scoped temp directory -- handy for caching expensive setup.")

    add("test",
        "Implement a test that verifies dict equality ignoring extra keys.",
        "```python\ndef test_subset():\n    actual = {'a': 1, 'b': 2, 'extra': 3}\n    expected = {'a': 1, 'b': 2}\n    assert expected.items() <= actual.items()\n```\nUse the items-view subset to assert 'contains at least these'.")

    add("test",
        "Write a test that asserts on call ordering across mocks.",
        "```python\nfrom unittest.mock import Mock, call\n\ndef test_ordering():\n    m = Mock()\n    pipeline(m)\n    m.assert_has_calls([call.start(), call.process(), call.finish()])\n```\n`assert_has_calls` checks the sequence appears in order.")

    add("test",
        "Implement a test that checks all expected exceptions chain correctly.",
        "```python\nimport pytest\nfrom myapp import wrap_error\n\ndef test_chains():\n    with pytest.raises(MyError) as info:\n        wrap_error()\n    assert isinstance(info.value.__cause__, ValueError)\n```\nAccess the underlying exception via `__cause__` (set by `raise X from y`).")

    add("test",
        "Write a fixture that yields a context-managed websocket connection.",
        "```python\nimport pytest, websockets\n\n@pytest.fixture\nasync def ws():\n    async with websockets.connect('ws://localhost:8000') as conn:\n        yield conn\n```\nAsync fixtures need pytest-asyncio's auto mode or explicit marker.")

    add("test",
        "Implement a test that verifies a function does not call a side effect.",
        "```python\nfrom unittest.mock import patch\n\ndef test_no_save():\n    with patch('myapp.save_to_db') as mock_save:\n        compute_only()\n    mock_save.assert_not_called()\n```\nNegative assertions are as important as positive ones for testing pure functions.")

    add("test",
        "Write a test using `pytest-benchmark` for performance.",
        "```python\ndef test_speed(benchmark):\n    result = benchmark(compute_heavy, 1000)\n    assert result > 0\n```\nThe `benchmark` fixture runs the function many times and reports stats; great for catching perf regressions.")

    add("test",
        "Implement a test that asserts a Pydantic model rejects bad input.",
        "```python\nimport pytest\nfrom pydantic import ValidationError\nfrom myapp.models import User\n\ndef test_invalid_email():\n    with pytest.raises(ValidationError):\n        User(name='x', email='not-an-email')\n```\nAssert on ValidationError; pydantic's error has rich detail you can inspect.")

    add("test",
        "Write a test that uses `pytest.fixture(params=...)` for variations.",
        "```python\nimport pytest\n\n@pytest.fixture(params=['sqlite', 'memory'])\ndef storage(request):\n    return Storage(backend=request.param)\n\ndef test_storage(storage):\n    storage.put('k', 'v')\n    assert storage.get('k') == 'v'\n```\nThe test runs once per param, automatically.")

    add("test",
        "Implement a fixture that records calls across a test.",
        "```python\nimport pytest\n\n@pytest.fixture\ndef recorder():\n    events: list = []\n    return events\n\ndef test_records(recorder):\n    recorder.append('a')\n    recorder.append('b')\n    assert recorder == ['a', 'b']\n```\nLightweight alternative to mocks when you just want to observe calls.")

    add("test",
        "Write a test that asserts a coroutine raises after a timeout.",
        "```python\nimport asyncio\nimport pytest\n\n@pytest.mark.asyncio\nasync def test_times_out():\n    with pytest.raises(asyncio.TimeoutError):\n        await asyncio.wait_for(slow_thing(), timeout=0.01)\n```\nForce the timeout in the test rather than waiting for production timeouts to fire.")

    add("test",
        "Implement a regression test for a specific past bug.",
        "```python\nfrom myapp import handle\n\ndef test_issue_456_empty_list_returns_zero():\n    # Regression: handle([]) used to raise IndexError.\n    assert handle([]) == 0\n```\nName the test after the bug; future readers will thank you.")

    add("test",
        "Write a test that verifies a CLI runs successfully.",
        "```python\nimport subprocess\nimport sys\n\ndef test_cli_runs():\n    result = subprocess.run([sys.executable, '-m', 'myapp', '--help'], capture_output=True, text=True, check=True)\n    assert 'usage' in result.stdout.lower()\n```\nUse `sys.executable` so the test uses the same Python as the test runner.")
