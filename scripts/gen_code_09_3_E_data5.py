# SPDX-License-Identifier: MIT
"""Debugging (50) + testing (50). Wave 5 part 5."""


def register_all(add):
    # ----- debugging / refactoring (50) -----
    add("debug",
        "How do I find which line in a long script is the slowest?",
        "Use `line_profiler`:\n```bash\npip install line_profiler\nkernprof -l -v script.py\n```\nDecorate the suspect function with `@profile` (no import needed -- the decorator is injected by `kernprof`). It prints time per line; the hotspot is usually obvious from there.")

    add("debug",
        "I'm getting `JSONDecodeError` on what looks like valid JSON. What now?",
        "Print the raw bytes you're trying to parse:\n```python\nprint(repr(payload[:200]))\n```\nUsual causes: a UTF-8 BOM (`\\ufeff`), trailing whitespace, single quotes (Python-style, not JSON), or trailing commas. JSON is strict; non-strict JSON-ish inputs need `json5` or a manual cleanup pass.")

    add("debug",
        "Why is my multiprocessing pool slower than the serial loop?",
        "Likely causes: (1) the per-task work is too small relative to pickling cost, (2) the worker function imports heavy modules at the top of the file, (3) you're running on Windows where forking is unavailable so each worker re-imports everything. Benchmark with `chunksize=` larger than 1; if speedup never appears, threads or batched serial code may be the right answer.")

    add("debug",
        "How do I refactor a function with mixed return types?",
        "Decide on one shape and stick to it. If the function sometimes returns a value and sometimes None, decide whether None is success or failure. If failure, raise. If absent, use `Optional[T]` and force callers to unwrap. If success/failure with detail, return a small dataclass `Result`. Mixed returns make every caller's life harder.")

    add("debug",
        "My SQLAlchemy query returns stale data. What's wrong?",
        "Common causes: (1) you're caching the result; (2) another process committed but your session is reading from its own snapshot (PostgreSQL's repeatable read), so call `session.expire_all()` or open a fresh session; (3) you're reading from a read replica with replication lag. Fix: make the read path use a fresh session per request.")

    add("debug",
        "Why am I getting `EOFError` when reading from stdin in tests?",
        "Tests usually have no real stdin. Mock it:\n```python\nfrom unittest.mock import patch\nimport io\n\n@patch('sys.stdin', new=io.StringIO('input data\\n'))\ndef test_reads_stdin():\n    ...\n```\nOr inject the input source as a parameter so tests can pass a `StringIO` directly without patching.")

    add("debug",
        "How do I refactor `try: ... except: pass`?",
        "Bare except hides everything including KeyboardInterrupt and bugs. Always:\n1. Catch the specific exception type you expect.\n2. Log it (don't silently swallow).\n3. Decide intentionally to continue or re-raise.\n```python\ntry:\n    risky()\nexcept ConnectionError:\n    log.warning('downstream unavailable, falling back')\n    use_fallback()\n```")

    add("debug",
        "I'm getting `ValueError: I/O operation on closed file`.",
        "You're using a file outside its `with` block:\n```python\nwith open('a.txt') as f:\n    data = f.readlines()\nfor line in f:   # BUG: f is closed here\n    print(line)\n```\nMove the work inside the `with`, or read into a list first if you need it after. The same pitfall hits `csv.reader(f)` when `f` is the closed handle.")

    add("debug",
        "Why is my regex slow on long inputs?",
        "Likely catastrophic backtracking from nested quantifiers like `(a+)+`. Rewrite to be non-ambiguous, use possessive quantifiers via `regex` (PyPI), or anchor with `\\A` / `\\Z`. Profile by feeding small inputs of increasing size -- if runtime explodes, it's backtracking. Sometimes the right answer is to split the problem and not use regex at all.")

    add("debug",
        "How do I refactor logging-heavy code into something cleaner?",
        "Pull formatting into structured logging:\n```python\nlog.info('order processed', extra={'order_id': oid, 'status': status, 'ms': dur})\n```\nWith a JSON formatter (`python-json-logger`), each call becomes a single structured event your aggregator can index on. Long log strings are a code smell -- `extra` keeps the variables visible.")

    add("debug",
        "I get different float results on different machines. Why?",
        "Floating-point isn't perfectly reproducible across CPUs, BLAS versions, or even across cores when summation order differs. Mitigations: (1) seed RNGs explicitly, (2) sum in a deterministic order (`sum(sorted(xs))`), (3) for testing use `pytest.approx` with explicit `abs=` and `rel=` tolerances. If exact reproducibility matters, switch to integer math or `Decimal`.")

    add("debug",
        "How do I refactor a long parametrize list in pytest?",
        "Pull the data into a fixture file (JSON/YAML/CSV) and load it in `conftest.py`:\n```python\nimport json, pytest\n\nwith open('test_cases.json') as f:\n    CASES = json.load(f)\n\n@pytest.fixture(params=CASES, ids=lambda c: c['name'])\ndef case(request):\n    return request.param\n```\nKeeps the test source short and lets non-coders contribute cases.")

    add("debug",
        "Why does `print(x)` show different output than `print(repr(x))`?",
        "`str(x)` is for end users -- it's pretty-printed and may lose information. `repr(x)` is for developers -- it should round-trip via `eval` when possible. For debugging always use `repr` (or `f'{x!r}'`); for user-facing output use `str`. The classic gotcha: a string that contains a newline prints as multiple lines via `str`, single line via `repr`.")

    add("debug",
        "I'm getting `BrokenPipeError` writing to stdout. How do I handle it?",
        "Standard fix when piped through `head` or similar:\n```python\nimport sys, signal\ntry:\n    sys.stdout.write(big_output)\nexcept BrokenPipeError:\n    sys.stdout = None\n```\nOr ignore SIGPIPE at startup: `signal.signal(signal.SIGPIPE, signal.SIG_DFL)`. Either way, the program should exit gracefully when its consumer goes away.")

    add("debug",
        "How do I refactor when adding a feature requires changes in 10 files?",
        "Look for the abstraction those 10 files are trying to share. Common shapes:\n- They all subclass the same base -- add the feature to the base.\n- They all dispatch on a string/enum -- replace with a registry.\n- They all wrap the same underlying object -- add a method on the wrapper.\nIf no abstraction exists, that's a sign you should create one. If there are exactly 10 places, write the helper, update all callers in one PR, and add a lint rule that enforces using it.")

    add("debug",
        "Why is my context manager not running its `__exit__` on KeyboardInterrupt?",
        "It does run `__exit__` -- but if the exception escapes, there's no time to handle it. To finalize work cleanly:\n```python\ntry:\n    with cm() as x:\n        long_running()\nexcept KeyboardInterrupt:\n    print('user cancelled')\n```\nFor truly critical cleanup register an `atexit` handler or use a process-level signal handler that flips a flag the work loop checks.")

    add("debug",
        "How do I refactor a global variable that's shared across modules?",
        "Move it into a small config or registry object:\n```python\n# state.py\nfrom dataclasses import dataclass, field\n@dataclass\nclass AppState:\n    cache: dict = field(default_factory=dict)\n\nstate = AppState()\n```\nImport `state` everywhere instead of mutating a free-standing global. The module is still effectively a singleton, but tests can replace `state.cache` cleanly.")

    add("debug",
        "I'm seeing inconsistent test results when tests are run in parallel.",
        "Shared state between tests. Common offenders: a module-level cache, the system clock, the filesystem, env vars, the global random state. Fix by making each test create its own resources (use `tmp_path`, `monkeypatch.setenv`, fresh random seed). Until you find it, run `pytest -p no:cacheprovider --randomly-seed=last` repeatedly to surface the flake.")

    add("debug",
        "Why does my function silently corrupt data when called with unicode?",
        "You're probably encoding/decoding with the wrong codec somewhere -- look for any `.encode()` or `.decode()` without an explicit encoding (the default of 'utf-8' is what you want, but the OS default for `open()` is locale-dependent on Windows). Always pass `encoding='utf-8'` to `open` and to `bytes`/`str` conversions.")

    add("debug",
        "How do I refactor a big switch on type into polymorphism?",
        "Replace the switch with a method on the type itself. Before:\n```python\ndef area(shape):\n    if shape.kind == 'circle':\n        return 3.14 * shape.r ** 2\n    elif shape.kind == 'square':\n        return shape.side ** 2\n```\nAfter, each shape class has its own `area(self)` method. Adding a new shape no longer touches existing code -- the open/closed principle in action.")

    add("debug",
        "Why is `os.path.exists` returning False for a file I can see?",
        "Three usual reasons: (1) the working directory differs from where you ran the script; print `os.getcwd()`. (2) the path contains tilde (`~`) or env vars that aren't expanded -- use `os.path.expanduser` and `os.path.expandvars`. (3) the file is on a different filesystem your process doesn't have permission to traverse.")

    add("debug",
        "I'm getting `RuntimeError: Event loop is closed` in async code.",
        "Usually you scheduled work on a loop after it finished. Common patterns: a destructor calling an async cleanup, or a thread/queue draining after `asyncio.run` returned. Fix by tying the lifetime of the async resource to the loop -- close it inside the same `async with` or `try/finally` that contains your loop.")

    add("debug",
        "How do I refactor code that has copy-pasted try/except blocks?",
        "Extract the shared error handling into a context manager or decorator:\n```python\nfrom contextlib import contextmanager\n\n@contextmanager\ndef tolerant(name: str):\n    try:\n        yield\n    except SomeError as exc:\n        log.warning('%s failed: %s', name, exc)\n```\nUsage: `with tolerant('upload'): do_upload()`. The error handling is in one place; new sites just need a `with` block.")

    add("debug",
        "Why does `subprocess.run` hang on a process that produces lots of output?",
        "You're using `shell=True` or capturing without reading. The OS pipe buffer fills up and the child blocks writing. Fix: use `capture_output=True` (which uses `communicate()` internally and reads concurrently), or stream output:\n```python\nproc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)\nfor line in proc.stdout:\n    log.info(line.rstrip())\n```")

    add("debug",
        "I want to find which import is taking 2 seconds at startup.",
        "Run with import-time profiling:\n```bash\npython -X importtime -c 'import myapp' 2> import.log\n```\nThe log shows cumulative time per module; sort by self-time to find the culprit. Common offenders: `pandas`, `tensorflow`, `boto3`, anything that pulls in `sklearn` indirectly. Lazy-import heavy deps inside functions to keep CLI startup fast.")

    add("debug",
        "How do I refactor SQL strings scattered across the codebase?",
        "Move them to a single `queries.py` (or split per domain) with named constants:\n```python\nGET_USER = 'SELECT id, email FROM users WHERE id = %s'\nLIST_ACTIVE = 'SELECT * FROM users WHERE is_active = TRUE'\n```\nFor anything dynamic use SQLAlchemy Core or query parameters -- never f-string user input into SQL.")

    add("debug",
        "Why does my pandas operation say `SettingWithCopyWarning`?",
        "You're modifying a slice that may or may not be a view. Either be explicit:\n```python\ndf.loc[df.amount > 100, 'flag'] = True\n```\nOr take an explicit copy first:\n```python\nbig = df[df.amount > 100].copy()\nbig['flag'] = True\n```\nThe warning is correct: relying on view-vs-copy semantics is fragile and pandas may change it.")

    add("debug",
        "I'm getting flaky test failures on macOS but not Linux.",
        "Likely culprits: case-insensitive filesystem (a test creates `Foo.txt`, another creates `foo.txt`), `fork` vs `spawn` for multiprocessing (macOS defaults to spawn since 3.8), or different default timezone. Make tests independent of all three: use `tmp_path` per test, set `multiprocessing.set_start_method('spawn')` explicitly, and freeze the timezone with `freezegun`.")

    add("debug",
        "How do I find which test left a file behind?",
        "Run with `--basetemp` and inspect what's left:\n```bash\npytest --basetemp=/tmp/pytest_run\nls -la /tmp/pytest_run\n```\nUse `tmp_path` fixture instead of hardcoded paths -- pytest cleans those automatically. Files in the project root are usually the smoking gun for missing cleanup.")

    add("debug",
        "Why does `dataclass(frozen=True)` not actually prevent mutation?",
        "It does prevent rebinding fields (`obj.x = 1` raises) but it cannot freeze nested mutable values: `obj.items.append(...)` still works because the *list* isn't frozen, only the binding to it. For deep immutability use `tuple` for sequences and `frozenset`/`MappingProxyType` for collections, or use `@frozen` from `attrs` which integrates better with type checkers.")

    add("debug",
        "I'm getting `OperationalError: database is locked` with SQLite.",
        "SQLite's default journal allows only one writer. Causes: (1) a connection holding an open transaction blocks others; commit or close it. (2) Two processes trying to write simultaneously. Fix: enable WAL mode (`PRAGMA journal_mode=WAL`) which allows concurrent reads with one writer. For high write concurrency, SQLite is the wrong tool.")

    add("debug",
        "How do I refactor a callback hell pattern into linear code?",
        "Use `async`/`await`:\n```python\n# Before: callbacks\nfetch(url, lambda data: process(data, lambda result: save(result, lambda: done())))\n# After:\nasync def pipeline(url):\n    data = await fetch(url)\n    result = await process(data)\n    await save(result)\n```\nReads top-to-bottom and errors propagate as normal exceptions. The asyncio runtime handles the scheduling.")

    add("debug",
        "Why does `argparse` accept unknown args silently?",
        "It doesn't by default -- `parse_args()` errors on unknowns. You're probably using `parse_known_args()`. Switch back:\n```python\nargs = parser.parse_args()\n```\nOr if you genuinely need pass-through, document the rest:\n```python\nargs, rest = parser.parse_known_args()\nlog.info('passing through: %s', rest)\n```")

    add("debug",
        "How do I refactor when the same logic exists with subtle differences in three places?",
        "Find the parameter that differs and pull it out as an argument. If the differences feel structural (different control flow), use the strategy pattern -- each variant becomes a small class implementing a common interface, and the shared code lives once with `strategy.do(...)` calls.")

    add("debug",
        "I'm getting `MemoryError` reading a large CSV.",
        "Three options ordered by effort: (1) `pd.read_csv(..., usecols=[...], dtype={...})` -- read only what you need, often shrinks 10x. (2) Process in chunks (`chunksize=`). (3) Switch to a tool designed for this -- DuckDB or Polars -- and run a SQL/lazy query that doesn't materialize the full table.")

    add("debug",
        "How do I refactor when an API caller has to know too much about internal details?",
        "Hide the details behind a method or function. If callers do `obj._private.extra.get(...)`, that's a leak; expose `obj.extra(key)` instead. Each leak is a future maintenance burden. The principle: clients should depend on what your code does, not how it does it.")

    add("debug",
        "My function works on lists but errors on numpy arrays. Why?",
        "Likely you're using `if x:` which raises `ValueError: ambiguous truth value` for arrays. Use `len(x) > 0` for emptiness, `np.any(x)` for any-truthy, `(x == y).all()` for equality. For polymorphic functions, document accepted types or coerce at the boundary: `x = np.asarray(x)`.")

    add("debug",
        "How do I find which test changed a global module-level variable?",
        "Add a session-scope autouse fixture that snapshots and restores:\n```python\nimport copy\n\n@pytest.fixture(autouse=True)\ndef restore_globals():\n    snapshot = copy.deepcopy(some_module._GLOBAL)\n    yield\n    if some_module._GLOBAL != snapshot:\n        pytest.fail(f'test mutated global: {some_module._GLOBAL}')\n```\nThe failure message names the test that did it.")

    add("debug",
        "Why is `pip install` picking the wrong package version?",
        "Usually a constraint elsewhere. Check `pip install -v package==1.2.3` for the resolver's chain of reasoning, or `pip install pipdeptree && pipdeptree -p package` to see who's pinning it. Pin transitively in `pyproject.toml`'s `dependencies` rather than relying on the resolver.")

    add("debug",
        "I'm getting `FileNotFoundError` only in production. What changed?",
        "Likely the working directory or path resolution. Always anchor to `__file__`:\n```python\nfrom pathlib import Path\nROOT = Path(__file__).resolve().parent\ndata = (ROOT / 'config' / 'app.yaml').read_text()\n```\nNever rely on `os.getcwd()` -- it depends on how the process was launched.")

    add("debug",
        "How do I refactor a class that needs three different constructors?",
        "Use `@classmethod` factory methods:\n```python\nclass User:\n    def __init__(self, id: int, email: str):\n        self.id = id; self.email = email\n    @classmethod\n    def from_row(cls, row): return cls(row['id'], row['email'])\n    @classmethod\n    def from_token(cls, token): payload = decode(token); return cls(payload['sub'], payload['email'])\n```\nKeeps `__init__` simple and gives readable factory names at call sites.")

    add("debug",
        "Why does `len(generator)` fail?",
        "Generators don't expose a length -- they're potentially infinite and consuming them is destructive. If you need the count, materialize: `len(list(gen))`. If you want the count without losing the items, use `more_itertools.ilen` or `sum(1 for _ in gen)` -- but that consumes the generator.")

    add("debug",
        "How do I refactor when functions silently return None for invalid input?",
        "Switch to raising or returning a Result/Optional type. Silent None gets shrugged off until something downstream crashes with `'NoneType' has no attribute X`. Better:\n```python\ndef parse_user(payload: dict) -> User:\n    if 'id' not in payload:\n        raise ValueError('missing id')\n    return User(...)\n```\nFail loudly at the boundary; succeed quietly in the middle.")

    add("debug",
        "I'm getting `TypeError: object of type X is not JSON serializable`.",
        "Pass a `default` callable to `json.dumps`:\n```python\nimport json\nfrom datetime import datetime\nfrom decimal import Decimal\n\ndef default(o):\n    if isinstance(o, (datetime,)):\n        return o.isoformat()\n    if isinstance(o, Decimal):\n        return str(o)\n    raise TypeError(f'unserializable: {type(o).__name__}')\n\njson.dumps(payload, default=default)\n```\nKeep the converters narrow and explicit so you don't accidentally lose precision.")

    add("debug",
        "Why does `for x in obj:` raise `TypeError: 'X' object is not iterable`?",
        "Missing `__iter__` (or `__getitem__` for the legacy protocol). Implement one:\n```python\nclass Box:\n    def __init__(self, items): self._items = items\n    def __iter__(self): return iter(self._items)\n```\nFor lazy iteration return a generator instead. The error message is honest -- the class doesn't claim to be iterable.")

    add("debug",
        "How do I refactor a test suite that takes 20 minutes to run?",
        "Profile with `pytest --durations=20`. Common wins:\n- Move expensive setup into `scope='session'` fixtures.\n- Parallelize with `pytest-xdist` (`-n auto`).\n- Replace network/DB calls with in-memory fakes for unit tests.\n- Mark slow integration tests with `@pytest.mark.slow` and skip them in the developer loop.\nGoal: under five seconds for the inner loop, under two minutes for full local run.")

    add("debug",
        "I'm getting `KeyError` deep inside a third-party library.",
        "Wrap the call at your boundary with a clean error:\n```python\ntry:\n    result = thirdparty.api.do(payload)\nexcept KeyError as exc:\n    raise InvalidPayloadError(f'missing field: {exc}') from exc\n```\nThe `from exc` preserves the chain for debugging while giving callers a sensible exception type to catch.")

    add("debug",
        "Why is my Docker container OOM-killed but the same code runs fine locally?",
        "Containers have memory limits; locally Python can grow indefinitely. Causes: (1) a runaway pandas/numpy operation, (2) accidental list materialization (`list(generator)` over a huge stream), (3) reference cycles in long-running processes. Profile with `tracemalloc.start()` and `tracemalloc.take_snapshot()` to find the growing allocations.")

    add("debug",
        "How do I refactor when a test file is 2000 lines long?",
        "Split by behaviour, not by structure. Each test file should focus on one feature or module. Move shared fixtures into `conftest.py` (closest scope wins). Tests are documentation -- if you can't find the test for a behaviour in 30 seconds, the file is too big.")

    add("debug",
        "Why does my CLI accept `--flag yes` instead of erroring?",
        "argparse is permissive when you used `nargs='?'` or `type=str`. For boolean flags use `action='store_true'`:\n```python\nparser.add_argument('--flag', action='store_true')\n```\nThis ignores any following arg and only sets the flag when present. For tri-state flags, use `--no-flag` via `BooleanOptionalAction` (Python 3.9+).")

    # ----- testing / pytest / mocking (50) -----
    add("test",
        "Write a pytest test for a simple add function.",
        "```python\nimport pytest\n\nfrom mymath import add\n\ndef test_add_positive():\n    assert add(2, 3) == 5\n\ndef test_add_zero():\n    assert add(0, 5) == 5\n\ndef test_add_negative():\n    assert add(-2, -3) == -5\n```\nKeep each test focused on one behaviour; the test name is your documentation.")

    add("test",
        "How do I parametrize a pytest test over multiple inputs?",
        "```python\nimport pytest\n\n@pytest.mark.parametrize(\n    'value, expected',\n    [(0, 0), (1, 1), (4, 2), (9, 3), (16, 4)],\n    ids=['zero', 'one', 'four', 'nine', 'sixteen'],\n)\ndef test_isqrt(value, expected):\n    assert int(value ** 0.5) == expected\n```\n`ids=` produces readable case names in the test report. Without it pytest auto-generates terse ids that are harder to scan.")

    add("test",
        "Write a pytest fixture that creates a temporary file.",
        "```python\nimport pytest\n\n@pytest.fixture\ndef sample_file(tmp_path):\n    path = tmp_path / 'sample.txt'\n    path.write_text('hello\\nworld\\n')\n    return path\n\ndef test_count_lines(sample_file):\n    assert sample_file.read_text().count('\\n') == 2\n```\n`tmp_path` is the built-in fixture for temporary directories; pytest cleans it up automatically.")

    add("test",
        "How do I mock a function with unittest.mock?",
        "```python\nfrom unittest.mock import patch\n\n@patch('mymodule.fetch_user')\ndef test_with_mocked_fetch(mock_fetch):\n    mock_fetch.return_value = {'id': 1, 'name': 'alice'}\n    result = mymodule.greet(1)\n    assert 'alice' in result\n    mock_fetch.assert_called_once_with(1)\n```\nPatch where it's *used*, not where it's defined -- this is the most common gotcha.")

    add("test",
        "Write a pytest test that expects an exception.",
        "```python\nimport pytest\n\ndef test_divide_by_zero():\n    with pytest.raises(ZeroDivisionError, match='division by zero'):\n        1 / 0\n```\n`match=` checks the exception message via regex. Always pin the type and ideally the message -- `pytest.raises(Exception)` is too forgiving.")

    add("test",
        "How do I share fixtures across multiple test files?",
        "Put them in `conftest.py` at the appropriate directory:\n```python\n# tests/conftest.py\nimport pytest\n\n@pytest.fixture\ndef alice():\n    return {'id': 1, 'name': 'alice'}\n```\nAny test under `tests/` can request `alice` without importing. Closer-to-leaf conftests override outer ones.")

    add("test",
        "Write a test that uses pytest's monkeypatch fixture.",
        "```python\nimport os\n\ndef get_db_url() -> str:\n    return os.environ['DATABASE_URL']\n\ndef test_get_db_url(monkeypatch):\n    monkeypatch.setenv('DATABASE_URL', 'postgresql://test')\n    assert get_db_url() == 'postgresql://test'\n```\n`monkeypatch` undoes its changes at the end of the test automatically.")

    add("test",
        "How do I assert that two floats are approximately equal?",
        "```python\nimport pytest\n\ndef test_close_enough():\n    assert 0.1 + 0.2 == pytest.approx(0.3)\n```\nFor tighter control:\n```python\nassert result == pytest.approx(expected, rel=1e-6, abs=1e-9)\n```\nNever compare floats with `==`. Use `approx` for both scalars and sequences.")

    add("test",
        "Write a test that mocks an HTTP request with responses.",
        "```python\nimport responses, requests\n\n@responses.activate\ndef test_fetch_user():\n    responses.get('https://api.example.com/users/1', json={'id': 1, 'name': 'alice'})\n    r = requests.get('https://api.example.com/users/1', timeout=2)\n    assert r.json()['name'] == 'alice'\n```\n`responses` is the cleanest way to mock `requests`-based code; use `respx` for `httpx`.")

    add("test",
        "How do I test code that reads from stdin?",
        "Inject the input source as a parameter so the test can pass a `StringIO`:\n```python\nimport io\n\ndef sum_numbers(stream):\n    return sum(int(line) for line in stream)\n\ndef test_sum_numbers():\n    assert sum_numbers(io.StringIO('1\\n2\\n3\\n')) == 6\n```\nDependency injection beats `monkeypatch.setattr('sys.stdin', ...)` -- the production code is simpler too.")

    add("test",
        "Write a test that uses a context manager mock.",
        "```python\nfrom unittest.mock import MagicMock, patch\n\ndef test_with_db(monkeypatch):\n    fake_conn = MagicMock()\n    fake_conn.__enter__.return_value = fake_conn\n    fake_conn.execute.return_value.fetchone.return_value = (1, 'alice')\n    with patch('myapp.db.connect', return_value=fake_conn):\n        assert myapp.get_user(1) == {'id': 1, 'name': 'alice'}\n```\n`MagicMock` auto-implements the context manager protocol, but you usually want to chain return values explicitly.")

    add("test",
        "How do I run only one test from the command line?",
        "```bash\npytest tests/test_users.py::test_create_user\n# or by keyword:\npytest -k 'create_user'\n# or by mark:\npytest -m smoke\n```\nDuring debugging add `-x` to stop on first failure and `-s` to show prints.")

    add("test",
        "Write a fixture that tears down after the test.",
        "```python\nimport pytest\n\n@pytest.fixture\ndef tempdb():\n    db = create_temp_db()\n    yield db\n    db.drop()   # teardown after the yield\n```\nThe `yield` form is cleaner than the older finalizer pattern; teardown runs even if the test raises.")

    add("test",
        "How do I test asynchronous code with pytest?",
        "```python\nimport pytest\n\n@pytest.mark.asyncio\nasync def test_fetch():\n    result = await fetch('https://example.com')\n    assert result.status == 200\n```\nInstall `pytest-asyncio` and configure `asyncio_mode = 'auto'` in `pyproject.toml` so you can drop the marker.")

    add("test",
        "Write a test that captures log output.",
        "```python\nimport logging\n\ndef test_logs_warning(caplog):\n    caplog.set_level(logging.WARNING)\n    do_risky_thing()\n    assert any('quota near limit' in r.message for r in caplog.records)\n```\nFor structured logs assert on `record.args` or `record.extra` instead of the formatted string.")

    add("test",
        "How do I mock a class method?",
        "```python\nfrom unittest.mock import patch\n\nclass Service:\n    def fetch(self, n): return n\n\n@patch.object(Service, 'fetch', return_value=42)\ndef test_service(mock_fetch):\n    assert Service().fetch(1) == 42\n    mock_fetch.assert_called_once_with(1)\n```\n`patch.object` targets the class directly and is less brittle than patching by string when the class moves.")

    add("test",
        "Write a parametrized test for two related functions.",
        "```python\nimport pytest\n\nfrom mymath import sin_taylor, sin_lookup\n\n@pytest.mark.parametrize('impl', [sin_taylor, sin_lookup])\n@pytest.mark.parametrize('x', [0.0, 0.5, 1.0, 3.14])\ndef test_sin_implementations(impl, x):\n    assert impl(x) == pytest.approx(math.sin(x), abs=1e-3)\n```\nStacking `parametrize` decorators creates the cross-product of inputs.")

    add("test",
        "How do I make a fixture session-scoped?",
        "```python\nimport pytest\n\n@pytest.fixture(scope='session')\ndef big_dataset():\n    return load_expensive_data()\n```\nValid scopes: `function` (default), `class`, `module`, `package`, `session`. Higher scope means fewer setups but shared state -- be careful with mutation.")

    add("test",
        "Write a test that uses freezegun to fix the clock.",
        "```python\nfrom datetime import datetime\nfrom freezegun import freeze_time\n\ndef expires_at(now: datetime, seconds: int) -> datetime:\n    from datetime import timedelta\n    return now + timedelta(seconds=seconds)\n\n@freeze_time('2026-05-08 12:00:00')\ndef test_expires_at():\n    now = datetime.now()\n    assert expires_at(now, 60).hour == 12\n```\nDeterministic time keeps tests reproducible. Inject `now` as a parameter for testability without needing freezegun.")

    add("test",
        "How do I check that a function was called with specific args?",
        "```python\nfrom unittest.mock import Mock\n\nm = Mock()\nm(1, 2, name='alice')\nm.assert_called_once_with(1, 2, name='alice')\n# or check at call site:\nassert m.call_args == ((1, 2), {'name': 'alice'})\n```\n`assert_called_once_with` raises `AssertionError` with a useful diff if the args don't match.")

    add("test",
        "Write a test that runs a Flask app via test_client.",
        "```python\ndef test_get_user(client):\n    response = client.get('/users/1')\n    assert response.status_code == 200\n    assert response.json == {'id': 1, 'name': 'alice'}\n```\nThe `client` fixture is provided by the app factory pattern; each test gets a fresh client.")

    add("test",
        "How do I test a FastAPI endpoint that requires authentication?",
        "Override the auth dependency in tests:\n```python\nfrom fastapi.testclient import TestClient\nfrom myapp import app, get_current_user\n\ndef fake_user():\n    return {'id': 1, 'role': 'admin'}\n\napp.dependency_overrides[get_current_user] = fake_user\nclient = TestClient(app)\n\ndef test_admin():\n    r = client.get('/admin/dashboard')\n    assert r.status_code == 200\n```\nClear overrides between tests so suites stay isolated.")

    add("test",
        "Write a test that uses pytest's tmp_path_factory.",
        "```python\nimport pytest\n\n@pytest.fixture(scope='session')\ndef shared_dir(tmp_path_factory):\n    d = tmp_path_factory.mktemp('shared')\n    (d / 'config.yaml').write_text('debug: true')\n    return d\n```\n`tmp_path_factory` works for non-function-scope fixtures where `tmp_path` would error.")

    add("test",
        "How do I assert that a list of mock calls is in a specific order?",
        "```python\nfrom unittest.mock import Mock, call\n\nm = Mock()\nm(1); m(2); m(3)\nm.assert_has_calls([call(1), call(2), call(3)])\n```\nFor strict ordering pass `any_order=False` (the default). For unordered checks use `any_order=True`.")

    add("test",
        "Write a test that mocks multiple imports in a module.",
        "```python\nfrom unittest.mock import patch\n\n@patch('myapp.service.fetch_user')\n@patch('myapp.service.send_email')\ndef test_signup(mock_email, mock_fetch):\n    mock_fetch.return_value = {'id': 1}\n    myapp.service.signup(1)\n    mock_email.assert_called_once()\n```\nDecorators apply bottom-up -- the closest decorator is the leftmost argument. This trips everyone the first time.")

    add("test",
        "How do I skip a test conditionally?",
        "```python\nimport pytest, sys\n\n@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX only')\ndef test_unix_socket():\n    ...\n```\nFor known-broken tests use `@pytest.mark.xfail(reason='bug-1234', strict=True)` -- the test is expected to fail and the suite errors if it suddenly passes.")

    add("test",
        "Write a test that uses faker to generate test data.",
        "```python\nfrom faker import Faker\n\ndef test_signup():\n    fake = Faker()\n    fake.seed_instance(42)\n    email = fake.email()\n    user = signup(email)\n    assert user.email == email\n```\nSeed Faker for reproducibility; otherwise tests will pass differently each run.")

    add("test",
        "How do I run pytest with code coverage?",
        "```bash\npip install pytest-cov\npytest --cov=mypkg --cov-report=term-missing --cov-fail-under=80\n```\n`--cov-fail-under` makes CI fail when coverage drops. Don't fetishize 100% -- aim for the right tests in the right places.")

    add("test",
        "Write a test that uses hypothesis property-based testing.",
        "```python\nfrom hypothesis import given\nimport hypothesis.strategies as st\n\n@given(st.lists(st.integers()))\ndef test_sort_idempotent(xs):\n    assert sorted(sorted(xs)) == sorted(xs)\n```\nHypothesis generates many random inputs and shrinks failures to a minimal example. Great for invariants.")

    add("test",
        "How do I test a function that uses random numbers?",
        "Inject the RNG as a parameter:\n```python\nimport random\n\ndef pick(items, rng=None):\n    rng = rng or random\n    return rng.choice(items)\n\ndef test_pick():\n    rng = random.Random(42)\n    assert pick(['a', 'b', 'c'], rng) == 'b'\n```\nMakes randomness deterministic in tests without `random.seed(...)` polluting global state.")

    add("test",
        "Write a test that uses respx to mock httpx requests.",
        "```python\nimport httpx, pytest, respx\n\n@respx.mock\nasync def test_fetch():\n    route = respx.get('https://api.example.com/users/1').mock(\n        return_value=httpx.Response(200, json={'id': 1, 'name': 'alice'})\n    )\n    async with httpx.AsyncClient() as client:\n        r = await client.get('https://api.example.com/users/1')\n    assert route.called\n    assert r.json()['name'] == 'alice'\n```\n`respx` is the httpx-equivalent of `responses`.")

    add("test",
        "How do I test database code with a transaction rollback per test?",
        "```python\nimport pytest\nfrom sqlalchemy.orm import Session\n\n@pytest.fixture\ndef db_session(engine):\n    connection = engine.connect()\n    trans = connection.begin()\n    session = Session(bind=connection)\n    yield session\n    session.close()\n    trans.rollback()\n    connection.close()\n```\nEach test gets a fresh transaction that's rolled back -- fast and isolated.")

    add("test",
        "Write a test for a CLI built with Click.",
        "```python\nfrom click.testing import CliRunner\n\nfrom mycli import cli\n\ndef test_version():\n    runner = CliRunner()\n    result = runner.invoke(cli, ['--version'])\n    assert result.exit_code == 0\n    assert 'mycli, version' in result.output\n```\n`CliRunner` captures stdout/stderr and exit codes without spawning subprocesses.")

    add("test",
        "How do I test code that writes to the console?",
        "Use the `capsys` fixture:\n```python\ndef hello():\n    print('hello, world')\n\ndef test_hello(capsys):\n    hello()\n    captured = capsys.readouterr()\n    assert captured.out == 'hello, world\\n'\n```\n`capsys` works for `print`; use `capfd` for code that writes to fd 1/2 directly (subprocess output, C extensions).")

    add("test",
        "Write a test that uses a class-scoped fixture.",
        "```python\nimport pytest\n\n@pytest.fixture(scope='class')\ndef shared_state():\n    return {'counter': 0}\n\nclass TestSharedState:\n    def test_first(self, shared_state):\n        shared_state['counter'] += 1\n    def test_second(self, shared_state):\n        assert shared_state['counter'] == 1\n```\nClass-scoped fixtures share state across tests in the class; useful but easy to abuse -- prefer function scope when you can.")

    add("test",
        "How do I assert that a Pandas DataFrame matches an expected shape?",
        "```python\nimport pandas as pd\nfrom pandas.testing import assert_frame_equal\n\ndef test_pivot():\n    actual = pivot_orders(...)\n    expected = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})\n    assert_frame_equal(actual, expected, check_dtype=False)\n```\n`assert_frame_equal` gives diff-style output. `check_dtype=False` is forgiving for int8 vs int64 mismatches.")

    add("test",
        "Write a test that uses pytest-mock.",
        "```python\ndef greet(fetch):\n    return f'hello, {fetch()[\"name\"]}'\n\ndef test_greet(mocker):\n    fake = mocker.Mock(return_value={'name': 'alice'})\n    assert greet(fake) == 'hello, alice'\n    fake.assert_called_once_with()\n```\n`pytest-mock`'s `mocker` fixture is just `unittest.mock` with automatic teardown -- nicer than `with patch(...)` blocks.")

    add("test",
        "How do I test code that writes to a file?",
        "Use `tmp_path`:\n```python\ndef save_report(path, data):\n    path.write_text('\\n'.join(data))\n\ndef test_save_report(tmp_path):\n    target = tmp_path / 'out.txt'\n    save_report(target, ['a', 'b'])\n    assert target.read_text() == 'a\\nb'\n```\nNever write to the project root in tests -- pytest cleans `tmp_path` for you.")

    add("test",
        "Write a test that runs an integration test only when an env var is set.",
        "```python\nimport os, pytest\n\n@pytest.mark.skipif(\n    not os.environ.get('RUN_INTEGRATION'),\n    reason='set RUN_INTEGRATION=1 to enable',\n)\ndef test_real_database():\n    ...\n```\nKeep slow / external-dependency tests opt-in so the developer loop stays fast.")

    add("test",
        "How do I unit-test code that uses `time.time()`?",
        "Inject the clock or monkeypatch `time.time`:\n```python\ndef test_with_clock(monkeypatch):\n    monkeypatch.setattr('time.time', lambda: 1_700_000_000.0)\n    assert get_now_iso() == '2023-11-14T22:13:20+00:00'\n```\nFor more elaborate scenarios use `freezegun`. Designs that take a `clock` callable are easier to test than ones that call `time.time` directly.")

    add("test",
        "Write a test that uses pytest's `request` fixture for indirect parametrization.",
        "```python\nimport pytest\n\n@pytest.fixture\ndef user(request):\n    return {'name': request.param, 'role': 'admin' if request.param == 'alice' else 'user'}\n\n@pytest.mark.parametrize('user', ['alice', 'bob'], indirect=True)\ndef test_user(user):\n    assert 'name' in user\n```\nIndirect parametrization runs the fixture once per parameter value -- useful when fixture setup depends on the parameter.")

    add("test",
        "How do I test code that uses environment variables without leaking?",
        "Use `monkeypatch.setenv` -- it auto-undoes:\n```python\ndef test_with_env(monkeypatch):\n    monkeypatch.setenv('FEATURE_FLAG', 'on')\n    assert is_feature_on()\n# outside the test, FEATURE_FLAG is unset again\n```\nSetting `os.environ[...]` directly leaks into other tests and is a classic flake source.")

    add("test",
        "Write a test that uses pytest fixtures for setup AND teardown.",
        "```python\nimport pytest\n\n@pytest.fixture\ndef temp_user(db):\n    user = db.create_user(email='temp@example.com')\n    yield user\n    db.delete_user(user.id)\n\ndef test_login(temp_user):\n    assert login(temp_user.email)\n```\nThe `yield` separates setup from teardown; teardown runs even if the test fails.")

    add("test",
        "How do I run my tests in a randomized order to find ordering bugs?",
        "Install `pytest-randomly`:\n```bash\npip install pytest-randomly\npytest -p randomly\n```\nThe seed is printed at the top of the report; reproduce with `pytest -p randomly --randomly-seed=12345`.")

    add("test",
        "Write a test using approxequal-style assertion for nested structures.",
        "```python\nimport pytest\n\ndef test_nested():\n    actual = {'a': 1.0001, 'b': [0.1 + 0.2, 0.3]}\n    expected = {'a': 1.0, 'b': [0.3, 0.3]}\n    assert actual == pytest.approx(expected, rel=1e-2)\n```\n`pytest.approx` recurses into dicts, lists, tuples, and numpy arrays.")

    add("test",
        "How do I test a generator function?",
        "Materialize and assert:\n```python\ndef countdown(n):\n    while n > 0:\n        yield n\n        n -= 1\n\ndef test_countdown():\n    assert list(countdown(3)) == [3, 2, 1]\n```\nFor lazy properties (no infinite loop / no eager work) call `next()` step by step.")

    add("test",
        "Write a test that uses a mock to simulate an exception.",
        "```python\nfrom unittest.mock import patch\nimport requests\n\n@patch('requests.get', side_effect=requests.ConnectionError)\ndef test_handles_network_error(mock_get):\n    result = fetch_user_robustly(1)\n    assert result is None\n```\n`side_effect` can be an exception class, an exception instance, or a callable for richer behaviour.")

    add("test",
        "How do I share complex test data via JSON fixtures?",
        "```python\nimport json, pytest\nfrom pathlib import Path\n\n@pytest.fixture(scope='session')\ndef payload(request):\n    path = Path(request.config.rootdir) / 'tests' / 'fixtures' / 'payload.json'\n    return json.loads(path.read_text())\n\ndef test_uses_payload(payload):\n    assert payload['user_id'] == 1\n```\nKeeps tests short; lets non-coders author cases by editing JSON.")

    add("test",
        "Write a test that uses pytest-xdist for parallel execution.",
        "```bash\npip install pytest-xdist\npytest -n auto\n```\nTests must be independent. Anything that writes to a shared resource (the same DB, the same temp file path, the same env var) becomes flaky under parallelism. `tmp_path` and `monkeypatch` keep tests isolated by construction.")

    add("test",
        "How do I clean up resources after every test in a module?",
        "```python\nimport pytest\n\n@pytest.fixture(autouse=True)\ndef cleanup():\n    yield\n    reset_global_state()\n```\n`autouse=True` applies the fixture to every test in the file (or directory if placed in `conftest.py`). The teardown runs even on test failure.")
