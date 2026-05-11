# SPDX-License-Identifier: MIT
"""Debug part 2 (25) + Testing entries (50). Wave 6 part 5."""


def register_all(add):
    # ---- 25 more debug entries ----
    add("debug",
        "Why does my multiprocessing.Pool ignore exceptions in workers?",
        "`pool.map` propagates exceptions when you iterate the result, but `pool.apply_async` requires you to call `.get()` to surface them. Always retrieve results -- silent failures in workers are how processes complete with wrong data and pass CI.")

    add("debug",
        "Help me debug 'list index out of range' in this loop.",
        "Print `len(xs)` and the index inside the loop. The bug is almost always (1) a hardcoded index `xs[3]` when the list shrank, or (2) a comprehension assumed n elements but produced fewer. Replace explicit indexing with `for x in xs:` whenever possible.")

    add("debug",
        "Why does my sqlalchemy connection pool exhaust?",
        "Sessions aren't being closed -- pool grows until it hits the limit, then blocks. Use `with Session(engine) as s:` so the session always closes. In FastAPI, dependency-yield style guarantees teardown.")

    add("debug",
        "My environment variable isn't visible to my subprocess.",
        "Subprocesses inherit env vars only if you don't pass `env=` explicitly. If you set `env={'FOO': 'bar'}`, that REPLACES the parent's env -- you'd lose PATH and everything else. Either omit `env=` or merge: `env={**os.environ, 'FOO': 'bar'}`.")

    add("debug",
        "Why does my pandas to_sql call fail with foreign key error?",
        "`if_exists='replace'` drops and recreates the table -- foreign keys disappear. Use `if_exists='append'` and create the schema separately. For one-off bulk loads with FKs, disable the FK constraint, load, then re-enable.")

    add("debug",
        "Help me debug a flaky test.",
        "Most flaky tests are time, network, or order dependent. Strategies: (1) freeze time with `freezegun`; (2) mock all I/O at the boundary, never let tests touch the network; (3) randomize test order via `pytest-randomly` to expose order coupling. Run flaky tests 100x in a row to catch them.")

    add("debug",
        "Why does my json.dumps produce '<class' for an enum value?",
        "Enums aren't natively serializable. `json.dumps(MyEnum.X)` raises; if you saw '<class' it's because something stringified the enum object. Pass `default=lambda x: x.value if isinstance(x, Enum) else str(x)` or use `MyEnum.X.value` explicitly.")

    add("debug",
        "Help me figure out why my asyncio.gather swallows exceptions.",
        "Default behavior raises the first exception and cancels the rest. If you want all results (success or failure), pass `return_exceptions=True` -- exceptions are returned in the result list. Then iterate and re-raise or report as needed.")

    add("debug",
        "Why is my sklearn cross_val_score score wildly different across runs?",
        "No `random_state` set on the splitter or the model. Pass `cv=KFold(5, shuffle=True, random_state=42)` and instantiate models with `random_state=42`. Without seeds, you get a noisy upper-bound on noise -- not a model evaluation.")

    add("debug",
        "My click command doesn't show the help text I wrote.",
        "Either the docstring is below the decorator (it should be the first line of the function body), or `--help` is being captured by another tool. `python -m mycli --help` should always work; if it doesn't, check sys.argv handling.")

    add("debug",
        "Why does pip install put the package in the wrong location?",
        "Wrong Python interpreter. Run `python -m pip install pkg` rather than `pip install pkg` -- ensures pip and python are paired. In venvs, activate first or use the absolute path: `/path/to/venv/bin/pip`.")

    add("debug",
        "Help me debug this AttributeError on a Pydantic model.",
        "Pydantic v2 renamed several methods: `.dict()` is now `.model_dump()`, `.json()` is `.model_dump_json()`, `Config` class became `model_config = ConfigDict(...)`. The migration guide in pydantic docs lists every rename.")

    add("debug",
        "Why does my Flask test client return 308 instead of the expected 200?",
        "You're calling `/path` but the route is registered as `/path/`. Flask redirects to the trailing-slash version. Either fix the test URL, set `strict_slashes=False` on the route, or follow redirects: `client.get('/path', follow_redirects=True)`.")

    add("debug",
        "My fixtures load before patches apply. How do I order them correctly?",
        "Pytest fixtures resolve in dependency order based on the parameters they accept. If a fixture needs the patch, take the patch fixture as a parameter: `def my_fixture(monkeypatch): ...`. Don't rely on fixture file ordering -- it's not guaranteed.")

    add("debug",
        "Why does my numpy random sample look biased?",
        "Likely you're using a global RNG that was seeded earlier and exhausted by other code. Use `np.random.default_rng(seed)` with an explicit seed per call site so results are reproducible and isolated.")

    add("debug",
        "Help me debug this 'Unable to allocate' MemoryError on a numpy array.",
        "You asked for more memory than available. Check the requested size: `np.zeros((10000, 10000), dtype=np.float64)` is 800MB. Drop to `np.float32` (halves), use sparse arrays if mostly zero, or process in chunks. `np.lib.format.open_memmap` for arrays larger than RAM.")

    add("debug",
        "Why does my pytest-asyncio test hang?",
        "Likely it awaits something that never resolves: a queue, a lock, an unfulfilled future. Set a timeout: `@pytest.mark.timeout(5)` from `pytest-timeout`. Then check what's pending: `asyncio.all_tasks()` lists every task in the loop.")

    add("debug",
        "Help me trace a slow SQL query in SQLAlchemy.",
        "Enable echo: `create_engine(url, echo=True)` logs every query. For per-query timing, add an event listener on `before_cursor_execute` and `after_cursor_execute`. Or just use the DB's slow query log -- it sees what the engine sends, including bind params.")

    add("debug",
        "Why does my CI fail with 'No matching distribution found'?",
        "pip can't find a package version compatible with the CI's Python. Common: M1 wheel-only deps on x86 CI, or Python 3.12-only packages on 3.10. Pin the package's expected version, ensure CI uses a matching Python, or build wheels for the missing platforms.")

    add("debug",
        "My `if x` check fails for valid inputs like 0 or empty list.",
        "Falsy values: 0, 0.0, '', [], {}, None, False all evaluate to False. If you mean 'is the variable set', use `x is not None`. Truthiness is concise but error-prone -- always be explicit when the type permits zero/empty as valid.")

    add("debug",
        "Why does my Django migration apply but the schema doesn't change?",
        "Django records migration as applied in `django_migrations` table even if the SQL fails (in some DBs without transactional DDL like MySQL). Check the actual schema with `python manage.py dbshell` then `\\d table`. Roll back with `migrate app <previous>` and reapply once you fix the migration.")

    add("debug",
        "Help me debug this missing log line in production.",
        "Three usual causes: (1) logger level too high (your INFO call but root level is WARNING); (2) propagation off on the logger or its parent; (3) the logged code path isn't actually hit. Add a temporary `logging.warning` (which always shows) at the same site to confirm.")

    add("debug",
        "Why does my pytest test pass but coverage shows the line uncovered?",
        "Likely the test imports the module but doesn't exercise that branch. Coverage tracks executed lines, not just import-time code. Add a parametrized case that hits the branch. `--cov-report=html` produces a clickable view of every line's status.")

    add("debug",
        "My Pillow image save produces a corrupted file.",
        "Two common causes: (1) wrong file extension for the save format (`save('out.png')` works because PIL infers, but `save('out')` doesn't); (2) the image's mode (`RGBA`) doesn't match the format (JPEG can't save RGBA -- convert with `.convert('RGB')` first). PIL's error message is usually enlightening.")

    add("debug",
        "Why does my requests-mock not intercept the call?",
        "You're calling a different client than the one mocked. `requests_mock` patches `requests`, not `httpx`. For httpx use `respx`. Verify the URL pattern matches exactly -- trailing slashes and query params count.")

    # ---- 50 testing entries ----
    add("test",
        "Write a basic pytest test.",
        "```python\ndef add(a, b):\n    return a + b\n\ndef test_add():\n    assert add(2, 3) == 5\n```\nNo class needed. Pytest discovers any function named `test_*` in a file named `test_*.py`. Run with `pytest`.")

    add("test",
        "Parametrize a pytest test.",
        "```python\nimport pytest\n\n@pytest.mark.parametrize('a,b,expected', [\n    (2, 3, 5),\n    (-1, 1, 0),\n    (0, 0, 0),\n])\ndef test_add(a, b, expected):\n    assert a + b == expected\n```\nEach tuple becomes a separate test ID. Pass `ids=['positive', 'opposite', 'zeros']` for readable case names.")

    add("test",
        "Use a pytest fixture.",
        "```python\nimport pytest\n\n@pytest.fixture\ndef sample_user():\n    return {'name': 'alice', 'age': 30}\n\ndef test_user_name(sample_user):\n    assert sample_user['name'] == 'alice'\n```\nFixtures injected by parameter name. Default `function` scope -- one per test.")

    add("test",
        "Use a yield-based fixture for setup and teardown.",
        "```python\nimport pytest\nfrom pathlib import Path\n\n@pytest.fixture\ndef tmp_log(tmp_path):\n    log = tmp_path / 'test.log'\n    log.write_text('')\n    yield log\n    # teardown happens automatically since tmp_path is removed\n\ndef test_writes(tmp_log):\n    tmp_log.write_text('hello')\n    assert tmp_log.read_text() == 'hello'\n```\n`tmp_path` is a built-in fixture that auto-cleans. Yield style ensures teardown even on test failure.")

    add("test",
        "Mock a function with monkeypatch.",
        "```python\nimport time\n\ndef get_now():\n    return time.time()\n\ndef test_time(monkeypatch):\n    monkeypatch.setattr(time, 'time', lambda: 1234567890.0)\n    assert get_now() == 1234567890.0\n```\nMonkeypatch is auto-undone after the test. No need to manually restore.")

    add("test",
        "Mock an HTTP call with respx.",
        "```python\nimport httpx, respx\n\n@respx.mock\ndef test_fetch():\n    respx.get('https://api.example.com/users').mock(return_value=httpx.Response(200, json={'name': 'a'}))\n    r = httpx.get('https://api.example.com/users')\n    assert r.json() == {'name': 'a'}\n```\nrespx integrates with httpx properly -- `unittest.mock.patch` of the client misses URL validation.")

    add("test",
        "Test that a function raises an exception.",
        "```python\nimport pytest\n\ndef divide(a, b):\n    if b == 0:\n        raise ValueError('cannot divide by zero')\n    return a / b\n\ndef test_divide_by_zero():\n    with pytest.raises(ValueError, match='cannot divide'):\n        divide(1, 0)\n```\n`match=` checks the message via regex. Stronger assertion than just the type.")

    add("test",
        "Use a session-scoped fixture.",
        "```python\nimport pytest\n\n@pytest.fixture(scope='session')\ndef expensive_resource():\n    print('setting up')\n    return {'data': 'shared'}\n\ndef test_a(expensive_resource):\n    assert expensive_resource['data'] == 'shared'\n```\nCreated once per pytest session. Use only for read-only resources -- mutations leak across tests.")

    add("test",
        "Mark a test as slow.",
        "```python\nimport pytest\n\n@pytest.mark.slow\ndef test_full_pipeline():\n    pass  # imagine a 30s test\n```\nRegister in `pyproject.toml`:\n```toml\n[tool.pytest.ini_options]\nmarkers = ['slow: marks tests as slow']\n```\nRun fast tests only with `pytest -m 'not slow'`.")

    add("test",
        "Skip a test conditionally.",
        "```python\nimport sys\nimport pytest\n\n@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX only')\ndef test_unix_socket():\n    pass\n```\n`skipif` evaluates at collection time. For runtime decisions, use `pytest.skip('reason')` inside the test body.")

    add("test",
        "Assert log output with caplog.",
        "```python\nimport logging\n\ndef worker():\n    logging.warning('something bad happened')\n\ndef test_warning(caplog):\n    caplog.set_level(logging.WARNING)\n    worker()\n    assert 'something bad' in caplog.text\n    assert caplog.records[0].levelname == 'WARNING'\n```\n`caplog.records` gives structured access; `caplog.text` is the rendered output.")

    add("test",
        "Capture stdout with capsys.",
        "```python\ndef hello(name: str):\n    print(f'hello, {name}')\n\ndef test_hello(capsys):\n    hello('alice')\n    captured = capsys.readouterr()\n    assert captured.out == 'hello, alice\\n'\n```\nWorks for `print`, anything writing to sys.stdout/stderr. Use `capfd` for OS-level captures (subprocess output).")

    add("test",
        "Use a fixture from conftest.py.",
        "```python\n# conftest.py\nimport pytest\n\n@pytest.fixture\ndef db():\n    return {'users': []}\n\n# test_users.py\ndef test_db(db):\n    db['users'].append('alice')\n    assert len(db['users']) == 1\n```\nFixtures in `conftest.py` are auto-discovered by every test in the same directory and below. The right place for project-wide fixtures.")

    add("test",
        "Test code with frozen time using freezegun.",
        "```python\nfrom datetime import datetime\nfrom freezegun import freeze_time\n\ndef now() -> datetime:\n    return datetime.utcnow()\n\n@freeze_time('2024-01-15 12:00:00')\ndef test_now():\n    assert now() == datetime(2024, 1, 15, 12, 0, 0)\n```\nFreezegun patches `datetime.now`/`utcnow`/`time.time` globally for the test scope.")

    add("test",
        "Test async code with pytest-asyncio.",
        "```python\nimport asyncio, pytest\n\nasync def fetch() -> str:\n    await asyncio.sleep(0)\n    return 'data'\n\n@pytest.mark.asyncio\nasync def test_fetch():\n    assert await fetch() == 'data'\n```\nSet `asyncio_mode = 'auto'` in pyproject so you don't need the marker on every test.")

    add("test",
        "Mock environment variables.",
        "```python\nimport os\n\ndef test_env(monkeypatch):\n    monkeypatch.setenv('API_KEY', 'fake')\n    assert os.environ['API_KEY'] == 'fake'\n```\nAuto-undone after the test. For `unittest`, use `mock.patch.dict(os.environ, ...)`.")

    add("test",
        "Test a Flask app with the test client.",
        "```python\nfrom myapp import app\n\ndef test_index():\n    client = app.test_client()\n    r = client.get('/')\n    assert r.status_code == 200\n    assert b'hello' in r.data\n```\nFlask's test_client is in-process; no real port binding. Use `r.get_json()` for JSON responses.")

    add("test",
        "Test a FastAPI app with TestClient.",
        "```python\nfrom fastapi.testclient import TestClient\nfrom myapp import app\n\ndef test_root():\n    client = TestClient(app)\n    r = client.get('/')\n    assert r.status_code == 200\n    assert r.json()['ok']\n```\nTestClient builds on httpx and supports the full client API including websockets.")

    add("test",
        "Override a FastAPI dependency in tests.",
        "```python\nfrom fastapi.testclient import TestClient\nfrom myapp import app, get_db\n\ndef fake_db():\n    return {'fake': True}\n\ndef test_with_fake_db():\n    app.dependency_overrides[get_db] = fake_db\n    try:\n        client = TestClient(app)\n        r = client.get('/items')\n        assert r.status_code == 200\n    finally:\n        app.dependency_overrides.clear()\n```\nAlways clear overrides in `finally` -- otherwise test ordering matters.")

    add("test",
        "Use unittest.mock.patch on a class method.",
        "```python\nfrom unittest.mock import patch\n\nclass Service:\n    def fetch(self):\n        return 'real'\n\ndef test_fetch():\n    with patch.object(Service, 'fetch', return_value='mocked') as m:\n        s = Service()\n        assert s.fetch() == 'mocked'\n        m.assert_called_once()\n```\n`patch.object` is type-safe vs `patch('module.Service.fetch')` (string mistakes raise AttributeError instead of silently doing nothing).")

    add("test",
        "Use a hypothesis property test.",
        "```python\nfrom hypothesis import given\nfrom hypothesis import strategies as st\n\ndef sort_then_reverse(xs):\n    return sorted(xs)[::-1]\n\n@given(st.lists(st.integers()))\ndef test_reverse_idempotent(xs):\n    assert sort_then_reverse(sort_then_reverse(xs)) == sorted(xs)\n```\nHypothesis generates many random inputs and shrinks failures to minimal counterexamples. Catches edges you didn't think to write.")

    add("test",
        "Test a generator function.",
        "```python\ndef counter(n):\n    for i in range(n):\n        yield i\n\ndef test_counter():\n    assert list(counter(3)) == [0, 1, 2]\n```\nMaterialize with `list()` for assertion. For infinite generators, take a finite prefix with `itertools.islice`.")

    add("test",
        "Use parametrize with multiple decorators.",
        "```python\nimport pytest\n\n@pytest.mark.parametrize('x', [1, 2])\n@pytest.mark.parametrize('y', ['a', 'b'])\ndef test_pair(x, y):\n    assert x in (1, 2)\n    assert y in ('a', 'b')\n```\nProduces the Cartesian product: 4 tests total. Useful when both axes vary independently.")

    add("test",
        "Mock a context manager with MagicMock.",
        "```python\nfrom unittest.mock import MagicMock, patch\n\ndef read_first_line(path):\n    with open(path) as f:\n        return f.readline()\n\ndef test_read_first(monkeypatch):\n    m = MagicMock()\n    m.return_value.__enter__.return_value.readline.return_value = 'first\\n'\n    monkeypatch.setattr('builtins.open', m)\n    assert read_first_line('x') == 'first\\n'\n```\nFor file IO, `tmp_path` + writing a real fixture file is cleaner; use mocks only when the dependency is expensive or external.")

    add("test",
        "Use pytest-cov for coverage reporting.",
        "```python\n# pyproject.toml\n[tool.pytest.ini_options]\naddopts = '--cov=mypkg --cov-report=term-missing --cov-report=html'\n\n[tool.coverage.run]\nbranch = true\nsource = ['mypkg']\n```\nRun `pytest` and inspect `htmlcov/index.html`. Branch coverage tells you about untested if/else paths, not just unreached lines.")

    add("test",
        "Use a tmp_path fixture for file IO.",
        "```python\ndef test_write(tmp_path):\n    p = tmp_path / 'test.txt'\n    p.write_text('hello')\n    assert p.read_text() == 'hello'\n```\n`tmp_path` is per-test and auto-cleaned. `tmp_path_factory` provides a session-scoped equivalent.")

    add("test",
        "Test a CLI built with Click.",
        "```python\nfrom click.testing import CliRunner\nfrom mycli import cli\n\ndef test_cli_help():\n    runner = CliRunner()\n    r = runner.invoke(cli, ['--help'])\n    assert r.exit_code == 0\n    assert 'Usage' in r.output\n```\nCliRunner runs the command in-process; no subprocess overhead and full assertion access.")

    add("test",
        "Use approx for floating-point comparison.",
        "```python\nimport pytest\n\ndef test_pi():\n    assert 3.14159 == pytest.approx(3.14, abs=0.01)\n```\nDefault tolerance is 1e-6 relative. Use `abs=` for absolute tolerance, `rel=` for relative -- never compare floats with `==`.")

    add("test",
        "Group related tests in a class.",
        "```python\nclass TestUser:\n    def test_create(self):\n        assert True\n    def test_delete(self):\n        assert True\n```\nNo `unittest.TestCase` inheritance needed for pytest. Class fixtures (`setup_method`/`teardown_method`) work; prefer pytest fixtures for new code.")

    add("test",
        "Run only a specific test by name.",
        "```\npytest tests/test_users.py::test_create_user\npytest -k 'create and not slow'\n```\n`-k` does substring matching across test names. Powerful for focused debugging.")

    add("test",
        "Test exception traceback context.",
        "```python\nimport pytest\n\ndef inner():\n    raise ValueError('bad')\n\ndef outer():\n    try:\n        inner()\n    except ValueError as e:\n        raise RuntimeError('wrapper') from e\n\ndef test_chain():\n    with pytest.raises(RuntimeError) as exc_info:\n        outer()\n    assert isinstance(exc_info.value.__cause__, ValueError)\n```\nThe `__cause__` chain confirms the original error was preserved via `raise from`.")

    add("test",
        "Use pytest plugins listed in pyproject.toml.",
        "```toml\n[tool.pytest.ini_options]\nminversion = '7.0'\naddopts = '-ra --strict-markers --strict-config'\ntestpaths = ['tests']\n```\n`--strict-markers` errors on unknown marks. `-ra` shows a short summary of skips/xfails/errors at the end.")

    add("test",
        "Test SQLAlchemy code with an in-memory SQLite.",
        "```python\nimport pytest\nfrom sqlalchemy import create_engine\nfrom sqlalchemy.orm import Session\nfrom mymodels import Base\n\n@pytest.fixture\ndef db_session():\n    engine = create_engine('sqlite:///:memory:')\n    Base.metadata.create_all(engine)\n    with Session(engine) as s:\n        yield s\n```\nIn-memory SQLite is fast and isolated per test. Beware: SQLite has subtle SQL dialect differences from Postgres.")

    add("test",
        "Mock time.sleep to speed up tests.",
        "```python\nimport time\n\ndef test_no_sleep(monkeypatch):\n    monkeypatch.setattr(time, 'sleep', lambda *a, **kw: None)\n    # code under test that calls time.sleep runs instantly\n    time.sleep(60)  # would sleep 60s, now returns immediately\n```\nGreat for testing retry/backoff logic without slowing the suite.")

    add("test",
        "Use a fixture that returns a callable factory.",
        "```python\nimport pytest\n\n@pytest.fixture\ndef make_user():\n    def _make(name='alice', age=30):\n        return {'name': name, 'age': age}\n    return _make\n\ndef test_default_user(make_user):\n    assert make_user()['name'] == 'alice'\n\ndef test_custom_user(make_user):\n    assert make_user(name='bob')['name'] == 'bob'\n```\nFactory fixtures parameterize fixture creation per test -- more flexible than plain value fixtures.")

    add("test",
        "Verify a function was called with specific arguments.",
        "```python\nfrom unittest.mock import MagicMock\n\ndef test_call_args():\n    m = MagicMock()\n    m(1, 'two', kw='three')\n    m.assert_called_with(1, 'two', kw='three')\n    assert m.call_count == 1\n```\nUse `assert_called_once_with` to assert both arguments and call count in one line.")

    add("test",
        "Use xfail for tests expected to fail.",
        "```python\nimport pytest\n\n@pytest.mark.xfail(reason='known bug #123')\ndef test_known_bug():\n    assert 1 / 0  # raises\n```\nxfail tests don't fail the suite. If they pass unexpectedly, pytest reports XPASS which prompts you to remove the marker.")

    add("test",
        "Run tests in parallel with pytest-xdist.",
        "```\npytest -n auto\n```\n`-n auto` uses one worker per CPU. Tests must be isolated -- shared state (DB, files, sockets) breaks parallelism. Use unique tmp_path per test and unique DB per worker (`pytest-xdist`'s `worker_id` fixture).")

    add("test",
        "Test that includes a cleanup step.",
        "```python\nimport pytest\n\n@pytest.fixture\ndef temp_resource(request):\n    resource = {'open': True}\n    def cleanup():\n        resource['open'] = False\n    request.addfinalizer(cleanup)\n    return resource\n\ndef test_uses(temp_resource):\n    assert temp_resource['open']\n```\n`request.addfinalizer` runs even if the test errors. Equivalent to yield-style fixtures with teardown after the yield.")

    add("test",
        "Test database transactions are rolled back.",
        "```python\nimport pytest\nfrom sqlalchemy.orm import Session\n\n@pytest.fixture\ndef db_session(engine):\n    conn = engine.connect()\n    trans = conn.begin()\n    session = Session(bind=conn)\n    yield session\n    session.close()\n    trans.rollback()\n    conn.close()\n```\nRunning every test in a savepoint that's rolled back keeps the DB pristine and tests fast (no truncate-and-reseed).")

    add("test",
        "Use side_effect to make a mock raise an exception.",
        "```python\nfrom unittest.mock import MagicMock\nimport pytest\n\ndef test_raises():\n    m = MagicMock(side_effect=ValueError('bad'))\n    with pytest.raises(ValueError):\n        m()\n```\n`side_effect` can be an exception (raised), a callable (called), or an iterable (consumed sequentially).")

    add("test",
        "Use a pytest fixture for app config.",
        "```python\nimport pytest\n\n@pytest.fixture\ndef app_config(monkeypatch):\n    monkeypatch.setenv('DATABASE_URL', 'sqlite:///:memory:')\n    monkeypatch.setenv('DEBUG', 'true')\n    from myapp import create_app\n    return create_app()\n```\nSet env vars before importing the app so config-on-import respects them.")

    add("test",
        "Check the exit code of a CLI subprocess.",
        "```python\nimport subprocess\n\ndef test_cli_exit_code():\n    r = subprocess.run(['python', '-c', 'import sys; sys.exit(0)'], capture_output=True)\n    assert r.returncode == 0\n```\nFor in-process CLIs (Click, Typer) prefer their test runner -- subprocess is for true integration tests.")

    add("test",
        "Use a fixture to seed a random generator.",
        "```python\nimport random, pytest\n\n@pytest.fixture(autouse=True)\ndef seed_random():\n    random.seed(42)\n```\n`autouse=True` applies to every test in scope. Reproducible randomness is critical for non-flaky tests.")

    add("test",
        "Test that a deprecated function emits a warning.",
        "```python\nimport pytest, warnings\n\ndef old_func():\n    warnings.warn('use new_func', DeprecationWarning, stacklevel=2)\n\ndef test_deprecation():\n    with pytest.warns(DeprecationWarning, match='new_func'):\n        old_func()\n```\nMatches the warning message via regex. `stacklevel=2` makes the warning point at the caller, not the deprecated function.")

    add("test",
        "Use a fixture with parametrize.",
        "```python\nimport pytest\n\n@pytest.fixture(params=['sqlite', 'postgres'])\ndef db_kind(request):\n    return request.param\n\ndef test_db(db_kind):\n    assert db_kind in ('sqlite', 'postgres')\n```\nThe test runs once per param value. Param-fixture combos are how you test the same logic across multiple backends.")

    add("test",
        "Test that a thread completes within a timeout.",
        "```python\nimport threading, pytest\n\ndef test_thread_completes():\n    done = threading.Event()\n    t = threading.Thread(target=done.set)\n    t.start()\n    assert done.wait(timeout=1.0)\n    t.join()\n```\nNever rely on `time.sleep` to wait for threads -- use `Event.wait` or `Thread.join(timeout=)` so a hung thread fails fast.")

    add("test",
        "Test JSON output structure with jsonschema.",
        "```python\nimport jsonschema\n\ndef test_user_shape():\n    schema = {\n        'type': 'object',\n        'required': ['id', 'email'],\n        'properties': {\n            'id': {'type': 'integer'},\n            'email': {'type': 'string', 'format': 'email'},\n        },\n    }\n    user = {'id': 1, 'email': 'a@example.com'}\n    jsonschema.validate(user, schema)\n```\nGreat for API contract tests. Pair with Pydantic models for the actual production validation.")

    add("test",
        "Use a fixture that depends on another fixture.",
        "```python\nimport pytest\n\n@pytest.fixture\ndef config():\n    return {'debug': True}\n\n@pytest.fixture\ndef app(config):\n    return {'config': config}\n\ndef test_app(app):\n    assert app['config']['debug']\n```\nFixture composition is the pytest superpower; chain freely. Pytest builds the dependency graph automatically.")

    # ---- 6 balancing entries: 3 algo, 1 web, 2 data ----
    add("algo",
        "Implement a function that returns the longest substring containing at most k distinct characters.",
        "```python\nfrom collections import Counter\n\ndef longest_k_distinct(s: str, k: int) -> int:\n    counts: Counter = Counter()\n    start = best = 0\n    for i, c in enumerate(s):\n        counts[c] += 1\n        while len(counts) > k:\n            counts[s[start]] -= 1\n            if counts[s[start]] == 0:\n                del counts[s[start]]\n            start += 1\n        best = max(best, i - start + 1)\n    return best\n```\nSliding window with character counts; O(n) time, O(k) space.")

    add("algo",
        "Write a function that flattens a list of lists with itertools.",
        "```python\nfrom itertools import chain\n\ndef flatten(xss: list[list]) -> list:\n    return list(chain.from_iterable(xss))\n```\n`chain.from_iterable` is the canonical idiom; cleaner than `sum(xss, [])` which is O(n^2).")

    add("algo",
        "Implement a function that computes the running minimum of a list.",
        "```python\nfrom itertools import accumulate\n\ndef running_min(xs: list) -> list:\n    return list(accumulate(xs, min))\n```\n`accumulate` accepts a custom binary op via the second arg. One line for what would otherwise be a manual loop.")

    add("web",
        "Write a FastAPI route that streams JSON line-by-line.",
        "```python\nimport json\nfrom fastapi import FastAPI\nfrom fastapi.responses import StreamingResponse\n\napp = FastAPI()\n\nasync def stream():\n    for i in range(100):\n        yield json.dumps({'i': i}) + '\\n'\n\n@app.get('/stream')\ndef stream_endpoint():\n    return StreamingResponse(stream(), media_type='application/x-ndjson')\n```\nNDJSON is parser-friendly for streamed records. Consumers can parse one line at a time without loading the whole response.")

    add("data",
        "Compute weighted average of a column grouped by another column.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({\n    'cat': ['a','a','b','b'],\n    'val': [10, 20, 30, 40],\n    'weight': [1, 3, 2, 2],\n})\nout = df.groupby('cat').apply(\n    lambda g: (g['val'] * g['weight']).sum() / g['weight'].sum(),\n    include_groups=False,\n)\nprint(out)\n```\n`include_groups=False` (pandas 2.2+) silences the deprecation warning about including the group column.")

    add("data",
        "Convert a numpy array to a torch tensor and back.",
        "```python\nimport numpy as np\nimport torch\n\na = np.arange(12).reshape(3, 4).astype(np.float32)\nt = torch.from_numpy(a)\nb = t.numpy()\nprint(np.array_equal(a, b))\n```\n`from_numpy` shares memory -- modifying one mutates the other. Use `.clone()` if you need an independent copy.")

    add("test",
        "Verify a file's contents in a test.",
        "```python\nfrom pathlib import Path\n\ndef test_readme_has_title(tmp_path):\n    readme = tmp_path / 'README.md'\n    readme.write_text('# Title\\n\\nbody')\n    content = readme.read_text()\n    assert content.startswith('# Title')\n    assert 'body' in content\n```\nFor large files, snapshot testing (`syrupy`) is more maintainable than inline string assertions.")
