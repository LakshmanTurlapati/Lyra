# SPDX-License-Identifier: MIT
"""Testing / pytest / mocking / async entries (53 total)."""


def register_all(add):
    # ---- 3-msg (30) ----
    add("test",
        "Write a pytest test for a function `add(a, b)` that returns a + b.",
        "```python\nfrom mymodule import add\n\ndef test_add():\n    assert add(2, 3) == 5\n    assert add(-1, 1) == 0\n    assert add(0, 0) == 0\n```\nA test name starting with `test_` and clear assertions is all pytest needs.")

    add("test",
        "Write a parametrized pytest test for a function `is_even(n)`.",
        "```python\nimport pytest\nfrom mymodule import is_even\n\n@pytest.mark.parametrize('n,expected', [\n    (0, True), (1, False), (2, True), (-3, False), (-4, True),\n])\ndef test_is_even(n, expected):\n    assert is_even(n) == expected\n```\n`@parametrize` runs the body once per case with each set of args.")

    add("test",
        "Write a pytest fixture that creates a temporary directory.",
        "```python\nimport pytest\nfrom pathlib import Path\n\n@pytest.fixture\ndef workspace(tmp_path: Path):\n    (tmp_path / 'data').mkdir()\n    return tmp_path\n\ndef test_uses_workspace(workspace):\n    assert (workspace / 'data').is_dir()\n```\nThe `tmp_path` builtin gives you a pristine `Path` per test; cleanup is automatic.")

    add("test",
        "Write a pytest test that asserts an exception is raised.",
        "```python\nimport pytest\n\ndef divide(a, b):\n    return a / b\n\ndef test_divide_by_zero():\n    with pytest.raises(ZeroDivisionError):\n        divide(1, 0)\n```\nUse `pytest.raises(...) as exc:` to inspect the exception's message.")

    add("test",
        "Write a pytest test using monkeypatch to override an env var.",
        "```python\nimport os\n\ndef test_env(monkeypatch):\n    monkeypatch.setenv('API_KEY', 'test-key')\n    assert os.environ['API_KEY'] == 'test-key'\n```\n`monkeypatch` automatically reverts the change at test teardown.")

    add("test",
        "Write a pytest test that mocks `requests.get` using unittest.mock.",
        "```python\nfrom unittest.mock import patch, MagicMock\n\n@patch('mymodule.requests.get')\ndef test_fetch(mock_get):\n    mock_get.return_value = MagicMock(status_code=200, json=lambda: {'ok': True})\n    from mymodule import fetch\n    assert fetch('http://x')['ok'] is True\n```\nPatch where the name is *used*, not where it's defined.")

    add("test",
        "Write a pytest test that uses tmp_path to test a file-writing function.",
        "```python\nfrom pathlib import Path\n\ndef write_hello(path: Path) -> None:\n    path.write_text('hello')\n\ndef test_write_hello(tmp_path):\n    target = tmp_path / 'out.txt'\n    write_hello(target)\n    assert target.read_text() == 'hello'\n```\n`tmp_path` is a per-test temporary `Path` cleaned up automatically.")

    add("test",
        "Write a pytest test that asserts a warning is raised.",
        "```python\nimport pytest\nimport warnings\n\ndef deprecated_thing():\n    warnings.warn('deprecated', DeprecationWarning)\n\ndef test_deprecation():\n    with pytest.warns(DeprecationWarning, match='deprecated'):\n        deprecated_thing()\n```\nThe `match=` arg uses regex on the warning message.")

    add("test",
        "Write a pytest fixture that yields and cleans up a database connection.",
        "```python\nimport sqlite3\nimport pytest\n\n@pytest.fixture\ndef db():\n    conn = sqlite3.connect(':memory:')\n    yield conn\n    conn.close()\n```\nCode after `yield` runs at teardown — even if the test fails.")

    add("test",
        "Write a pytest test for an async function.",
        "```python\nimport asyncio\nimport pytest\n\nasync def add(a, b):\n    await asyncio.sleep(0)\n    return a + b\n\n@pytest.mark.asyncio\nasync def test_add():\n    assert await add(1, 2) == 3\n```\nRequires `pytest-asyncio`. Mark each async test with `@pytest.mark.asyncio` (or set `asyncio_mode = 'auto'` in config).")

    add("test",
        "Write a pytest test that measures execution time.",
        "```python\nimport time\n\ndef test_under_one_second():\n    start = time.time()\n    do_work()\n    assert time.time() - start < 1.0\n```\nFor benchmarking, prefer `pytest-benchmark`. Wall-clock asserts can be flaky on slow CI.")

    add("test",
        "Write a pytest test that verifies stdout output.",
        "```python\ndef shout(msg):\n    print(msg.upper())\n\ndef test_shout(capsys):\n    shout('hello')\n    captured = capsys.readouterr()\n    assert captured.out.strip() == 'HELLO'\n```\n`capsys` captures stdout/stderr per test. `capfd` captures at the file-descriptor level (also catches subprocess output).")

    add("test",
        "Write a unittest TestCase with setUp and tearDown.",
        "```python\nimport unittest\n\nclass MyTest(unittest.TestCase):\n    def setUp(self):\n        self.values = [1, 2, 3]\n\n    def tearDown(self):\n        self.values.clear()\n\n    def test_sum(self):\n        self.assertEqual(sum(self.values), 6)\n```\n`setUp` runs before each test method; `tearDown` after — even on failure.")

    add("test",
        "Write a pytest fixture with module scope.",
        "```python\nimport pytest\n\n@pytest.fixture(scope='module')\ndef heavy_resource():\n    obj = expensive_setup()\n    yield obj\n    obj.close()\n```\nModule-scoped fixtures are created once per module — handy for expensive setup.")

    add("test",
        "Write a pytest test that uses freezegun to fix the clock.",
        "```python\nfrom datetime import datetime\nfrom freezegun import freeze_time\n\n@freeze_time('2024-01-01 12:00:00')\ndef test_now_is_fixed():\n    assert datetime.now().year == 2024\n```\n`freezegun` patches `datetime.now`, `time.time`, etc. globally for the test's duration.")

    add("test",
        "Write a pytest test that asserts a dict matches a partial structure.",
        "```python\ndef test_partial():\n    actual = {'name': 'a', 'meta': {'id': 1, 'tags': ['x']}, 'extra': 5}\n    assert actual.items() >= {'name': 'a'}.items()\n```\nDict-views support set semantics — `actual.items() >= expected.items()` checks subset of pairs.")

    add("test",
        "Write a pytest test that uses requests_mock to mock an HTTP call.",
        "```python\nimport requests\n\ndef test_get(requests_mock):\n    requests_mock.get('https://api.example.com/data', json={'value': 42})\n    r = requests.get('https://api.example.com/data')\n    assert r.json()['value'] == 42\n```\n`requests_mock` is a pytest plugin that intercepts `requests` calls without monkeypatching by hand.")

    add("test",
        "Write a pytest fixture that mocks the system clock for the duration of a test.",
        "```python\nimport time\nimport pytest\n\n@pytest.fixture\ndef fixed_time(monkeypatch):\n    monkeypatch.setattr(time, 'time', lambda: 1_700_000_000)\n```\nNow any code calling `time.time()` gets the fixed value.")

    add("test",
        "Write a pytest test that uses assert_called_with on a mock.",
        "```python\nfrom unittest.mock import MagicMock\n\ndef test_calls():\n    callback = MagicMock()\n    do_work(callback)\n    callback.assert_called_with('done', code=0)\n```\nUse `assert_called_once_with` if exactly one call is required.")

    add("test",
        "Write an async pytest test that asserts on a value from an async generator.",
        "```python\nimport pytest\n\nasync def numbers():\n    for i in range(3):\n        yield i\n\n@pytest.mark.asyncio\nasync def test_async_gen():\n    result = [x async for x in numbers()]\n    assert result == [0, 1, 2]\n```\n`async for` and async comprehensions are the standard tools here.")

    add("test",
        "Write a pytest test that mocks a database query using AsyncMock.",
        "```python\nimport pytest\nfrom unittest.mock import AsyncMock\n\n@pytest.mark.asyncio\nasync def test_async_mock():\n    db = AsyncMock()\n    db.fetch.return_value = [{'id': 1}]\n    rows = await db.fetch('SELECT 1')\n    assert rows == [{'id': 1}]\n```\n`AsyncMock` (Python 3.8+) is the right choice for awaitable methods.")

    add("test",
        "Write a pytest test that captures log output.",
        "```python\nimport logging\n\ndef test_logs(caplog):\n    with caplog.at_level(logging.INFO):\n        logging.getLogger('app').info('hello')\n    assert 'hello' in caplog.text\n```\n`caplog` is the built-in fixture for asserting on log output.")

    add("test",
        "Write a pytest test that skips on a missing dependency.",
        "```python\nimport importlib\nimport pytest\n\nrequests = pytest.importorskip('requests')\n\ndef test_uses_requests():\n    assert requests.utils.urlparse('http://x').netloc == 'x'\n```\n`importorskip` skips the whole test module if the import fails — useful for optional dependencies.")

    add("test",
        "Write a pytest test that runs only on Linux.",
        "```python\nimport platform\nimport pytest\n\n@pytest.mark.skipif(platform.system() != 'Linux', reason='Linux only')\ndef test_linux_only():\n    pass\n```\nThe reason text shows up in the test report so you remember why.")

    add("test",
        "Write a pytest test using xfail for an expected failure.",
        "```python\nimport pytest\n\n@pytest.mark.xfail(reason='not yet implemented')\ndef test_future():\n    assert future_thing() == 42\n```\n`xfail` runs the test but doesn't error if it fails. Pass `strict=True` to fail when it unexpectedly passes.")

    add("test",
        "Write a hypothesis test that asserts a function is its own inverse.",
        "```python\nfrom hypothesis import given\nfrom hypothesis import strategies as st\n\nfrom mymodule import encode, decode\n\n@given(st.text())\ndef test_round_trip(s):\n    assert decode(encode(s)) == s\n```\nHypothesis generates many inputs automatically and shrinks failing cases to a minimal example.")

    add("test",
        "Write a pytest test that verifies a function is idempotent.",
        "```python\ndef test_idempotent():\n    state = {'count': 0}\n    apply_change(state)\n    after_first = dict(state)\n    apply_change(state)\n    assert state == after_first\n```\nApply the operation twice and check it doesn't change after the first.")

    add("test",
        "Write a pytest plugin (conftest.py) that adds a custom CLI option.",
        "```python\n# conftest.py\nimport pytest\n\ndef pytest_addoption(parser):\n    parser.addoption('--env', default='local', help='target environment')\n\n@pytest.fixture\ndef env(request):\n    return request.config.getoption('--env')\n```\nUse via `pytest --env=staging`. Inject into tests with the `env` fixture.")

    add("test",
        "Write a pytest test that checks a function under concurrent access.",
        "```python\nfrom concurrent.futures import ThreadPoolExecutor\n\ndef test_thread_safety():\n    counter = Counter()\n    def task():\n        for _ in range(100):\n            counter.inc()\n    with ThreadPoolExecutor(max_workers=10) as ex:\n        list(ex.map(lambda _: task(), range(10)))\n    assert counter.value == 10 * 100\n```\nCatches missing locks. For race-condition-heavy code, also try `pytest-repeat`.")

    add("test",
        "Write a pytest test that uses a custom assertion helper with proper introspection.",
        "```python\ndef _assert_equal_records(a, b):\n    __tracebackhide__ = True\n    assert a == b, f'{a!r} != {b!r}'\n\ndef test_records():\n    _assert_equal_records({'id': 1}, {'id': 1})\n```\n`__tracebackhide__ = True` hides the helper frame from pytest's traceback so the failure points at the test's call site.")

    # ---- 5-msg with follow-ups (20) ----
    add("test",
        "Write a pytest test for a function that reads a JSON file.",
        "```python\nimport json\nfrom pathlib import Path\n\ndef load(path: Path) -> dict:\n    return json.loads(path.read_text())\n\ndef test_load(tmp_path):\n    p = tmp_path / 'a.json'\n    p.write_text('{\"x\": 1}')\n    assert load(p) == {'x': 1}\n```\n`tmp_path` keeps the test self-contained.",
        "Add a parametrized version that covers an empty file and a malformed file.",
        "```python\nimport json\nimport pytest\nfrom pathlib import Path\n\ndef load(path: Path) -> dict:\n    return json.loads(path.read_text())\n\n@pytest.mark.parametrize('content,expected', [\n    ('{\"x\": 1}', {'x': 1}),\n    ('{}', {}),\n])\ndef test_load_ok(tmp_path, content, expected):\n    p = tmp_path / 'a.json'\n    p.write_text(content)\n    assert load(p) == expected\n\ndef test_load_malformed(tmp_path):\n    p = tmp_path / 'bad.json'\n    p.write_text('not json')\n    with pytest.raises(json.JSONDecodeError):\n        load(p)\n```\nSplitting the malformed case lets us assert on the specific exception.")

    add("test",
        "Write a pytest test that mocks `time.sleep` so the test runs instantly.",
        "```python\nimport time\nfrom unittest.mock import patch\n\ndef do_work():\n    time.sleep(5)\n    return 42\n\n@patch('time.sleep')\ndef test_no_sleep(mock_sleep):\n    assert do_work() == 42\n    mock_sleep.assert_called_once_with(5)\n```\nThe `@patch` replaces `time.sleep` only inside the test.",
        "Use monkeypatch instead of @patch.",
        "```python\nimport time\n\ndef do_work():\n    time.sleep(5)\n    return 42\n\ndef test_no_sleep(monkeypatch):\n    calls = []\n    monkeypatch.setattr(time, 'sleep', lambda s: calls.append(s))\n    assert do_work() == 42\n    assert calls == [5]\n```\nThe monkeypatch fixture is more pytest-idiomatic; the assertion on `calls` mirrors `assert_called_once_with`.")

    add("test",
        "Write a pytest test for a Flask route.",
        "```python\nimport pytest\nfrom myapp import app\n\n@pytest.fixture\ndef client():\n    return app.test_client()\n\ndef test_root(client):\n    r = client.get('/')\n    assert r.status_code == 200\n```\nFlask's test client doesn't need a running server.",
        "Now test a POST that requires JSON.",
        "```python\nimport pytest\nfrom myapp import app\n\n@pytest.fixture\ndef client():\n    return app.test_client()\n\ndef test_create(client):\n    r = client.post('/items', json={'name': 'widget'})\n    assert r.status_code == 201\n    assert r.get_json()['name'] == 'widget'\n```\nPassing `json=` sets the Content-Type and serializes the body in one shot.")

    add("test",
        "Write a pytest test for a FastAPI endpoint.",
        "```python\nfrom fastapi.testclient import TestClient\nfrom myapp import app\n\nclient = TestClient(app)\n\ndef test_root():\n    r = client.get('/')\n    assert r.status_code == 200\n```\n`TestClient` is built on httpx and runs the app in-process.",
        "Test an endpoint that requires a Bearer token.",
        "```python\nfrom fastapi.testclient import TestClient\nfrom myapp import app\n\nclient = TestClient(app)\n\ndef test_protected():\n    r = client.get('/private', headers={'Authorization': 'Bearer testtoken'})\n    assert r.status_code == 200\n```\nFor real auth flows, use FastAPI's `app.dependency_overrides` to swap in a fake auth function during tests.")

    add("test",
        "Write an async test using pytest-asyncio for an async DB query.",
        "```python\nimport pytest\nfrom myapp import fetch_user\n\n@pytest.mark.asyncio\nasync def test_fetch():\n    user = await fetch_user(1)\n    assert user.name == 'alice'\n```\nRequires real DB — replace with mocks in unit tests.",
        "Mock the DB call instead of hitting it.",
        "```python\nimport pytest\nfrom unittest.mock import AsyncMock, patch\nfrom myapp import fetch_user, User\n\n@pytest.mark.asyncio\nasync def test_fetch_mocked():\n    with patch('myapp.db.fetchrow', new=AsyncMock(return_value={'id': 1, 'name': 'alice'})):\n        user = await fetch_user(1)\n        assert user.name == 'alice'\n```\n`new=AsyncMock(...)` creates an awaitable mock, which is required for awaited calls.")

    add("test",
        "Write a pytest test that uses parametrize for boundary values.",
        "```python\nimport pytest\n\ndef clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n\n@pytest.mark.parametrize('x,expected', [\n    (-1, 0), (0, 0), (5, 5), (10, 10), (11, 10),\n])\ndef test_clamp(x, expected):\n    assert clamp(x, 0, 10) == expected\n```\nBoundary tests catch off-by-one bugs at the edges.",
        "Add IDs so failing cases are easier to identify.",
        "```python\nimport pytest\n\ndef clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n\n@pytest.mark.parametrize(\n    'x,expected',\n    [(-1, 0), (0, 0), (5, 5), (10, 10), (11, 10)],\n    ids=['below', 'lower-edge', 'middle', 'upper-edge', 'above'],\n)\ndef test_clamp(x, expected):\n    assert clamp(x, 0, 10) == expected\n```\nNow `pytest -v` shows `test_clamp[lower-edge]` etc. — much friendlier when one fails.")

    add("test",
        "Write a pytest test that mocks an HTTP API.",
        "```python\nfrom unittest.mock import patch, MagicMock\n\n@patch('requests.get')\ndef test_api(mock_get):\n    mock_get.return_value = MagicMock(\n        status_code=200,\n        json=lambda: {'count': 5},\n    )\n    from myapp import get_count\n    assert get_count() == 5\n```\nMocking at the boundary keeps tests fast and deterministic.",
        "Assert the URL was called with the right query params.",
        "```python\nfrom unittest.mock import patch, MagicMock\n\n@patch('requests.get')\ndef test_api_params(mock_get):\n    mock_get.return_value = MagicMock(status_code=200, json=lambda: {})\n    from myapp import get_count\n    get_count(team='alpha')\n    mock_get.assert_called_once_with(\n        'https://api.example.com/count',\n        params={'team': 'alpha'},\n        timeout=10,\n    )\n```\n`assert_called_once_with` is the safest assertion because it pins both args *and* the call count.")

    add("test",
        "Write a pytest fixture that creates a fresh sqlite database for each test.",
        "```python\nimport sqlite3\nimport pytest\n\n@pytest.fixture\ndef db():\n    conn = sqlite3.connect(':memory:')\n    conn.execute('CREATE TABLE users (id INTEGER, name TEXT)')\n    yield conn\n    conn.close()\n\ndef test_insert(db):\n    db.execute('INSERT INTO users VALUES (1, \"alice\")')\n    assert db.execute('SELECT name FROM users WHERE id=1').fetchone()[0] == 'alice'\n```\nIn-memory SQLite makes tests fast and isolated.",
        "Now share the schema across tests but reset data each time.",
        "```python\nimport sqlite3\nimport pytest\n\n@pytest.fixture(scope='session')\ndef schema_sql():\n    return 'CREATE TABLE users (id INTEGER, name TEXT)'\n\n@pytest.fixture\ndef db(schema_sql):\n    conn = sqlite3.connect(':memory:')\n    conn.executescript(schema_sql)\n    yield conn\n    conn.close()\n```\nSession-scoped schema, function-scoped DB. Each test still starts empty but doesn't repeat the schema definition.")

    add("test",
        "Write a pytest test that reads a fixture file from disk.",
        "```python\nimport json\nfrom pathlib import Path\n\nFIXTURES = Path(__file__).parent / 'fixtures'\n\ndef test_loads_user():\n    data = json.loads((FIXTURES / 'user.json').read_text())\n    assert data['id'] == 1\n```\nKeep fixture files alongside test files for proximity.",
        "Make the path resolvable as a fixture so tests can declare it.",
        "```python\nimport json\nfrom pathlib import Path\nimport pytest\n\n@pytest.fixture(scope='session')\ndef fixtures_dir():\n    return Path(__file__).parent / 'fixtures'\n\ndef test_loads_user(fixtures_dir):\n    data = json.loads((fixtures_dir / 'user.json').read_text())\n    assert data['id'] == 1\n```\nThe fixture pattern centralizes the path so renames only happen in one place.")

    add("test",
        "Write a hypothesis test for a sort function.",
        "```python\nfrom hypothesis import given, strategies as st\n\ndef test_sort_idempotent():\n    @given(st.lists(st.integers()))\n    def inner(xs):\n        assert sorted(sorted(xs)) == sorted(xs)\n    inner()\n```\nIdempotence is one of the easier invariants to property-test.",
        "Test more invariants of sorted: output length, output is a permutation, output is non-decreasing.",
        "```python\nfrom hypothesis import given, strategies as st\nfrom collections import Counter\n\n@given(st.lists(st.integers()))\ndef test_sort_invariants(xs):\n    s = sorted(xs)\n    assert len(s) == len(xs)\n    assert Counter(s) == Counter(xs)\n    assert all(a <= b for a, b in zip(s, s[1:]))\n```\nThree invariants: length preserved, multiset preserved, non-decreasing. Together they uniquely characterize 'sort'.")

    add("test",
        "Write a pytest test that times out after 5 seconds using pytest-timeout.",
        "```python\nimport time\nimport pytest\n\n@pytest.mark.timeout(5)\ndef test_slow():\n    time.sleep(0.1)\n    assert True\n```\nRequires the `pytest-timeout` plugin. Useful as a guard against deadlocks in CI.",
        "Set a global timeout for all tests.",
        "```ini\n# pyproject.toml\n[tool.pytest.ini_options]\ntimeout = 5\n```\nYou can override per-test with `@pytest.mark.timeout(N)` (or disable with `0`).")

    add("test",
        "Write a pytest test that uses dependency injection.",
        "```python\nimport pytest\n\nclass UserService:\n    def __init__(self, repo):\n        self.repo = repo\n    def names(self):\n        return [u['name'] for u in self.repo.list()]\n\nclass FakeRepo:\n    def list(self):\n        return [{'name': 'alice'}, {'name': 'bob'}]\n\n@pytest.fixture\ndef svc():\n    return UserService(FakeRepo())\n\ndef test_names(svc):\n    assert svc.names() == ['alice', 'bob']\n```\nInjecting a fake repo decouples the service from the database.",
        "Use Mock instead of a hand-written FakeRepo.",
        "```python\nimport pytest\nfrom unittest.mock import MagicMock\n\nclass UserService:\n    def __init__(self, repo):\n        self.repo = repo\n    def names(self):\n        return [u['name'] for u in self.repo.list()]\n\n@pytest.fixture\ndef svc():\n    repo = MagicMock()\n    repo.list.return_value = [{'name': 'alice'}, {'name': 'bob'}]\n    return UserService(repo), repo\n\ndef test_names(svc):\n    service, repo = svc\n    assert service.names() == ['alice', 'bob']\n    repo.list.assert_called_once()\n```\nMagicMock saves you from writing the fake class and lets you assert on call patterns.")

    add("test",
        "Write a pytest test for a CLI command using click.testing.",
        "```python\nimport click\nfrom click.testing import CliRunner\n\n@click.command()\n@click.argument('name')\ndef hello(name):\n    click.echo(f'hello {name}')\n\ndef test_hello():\n    runner = CliRunner()\n    result = runner.invoke(hello, ['alice'])\n    assert result.exit_code == 0\n    assert result.output.strip() == 'hello alice'\n```\n`CliRunner` runs click commands in-process and captures output.",
        "Test that a missing argument exits with the usage error.",
        "```python\nfrom click.testing import CliRunner\n\ndef test_missing_arg():\n    runner = CliRunner()\n    result = runner.invoke(hello, [])\n    assert result.exit_code != 0\n    assert 'Missing argument' in result.output\n```\nClick exits with code 2 on usage errors and prints a helpful message — both are stable enough to assert on.")

    add("test",
        "Write a pytest fixture that uses freezegun for the duration of a test.",
        "```python\nimport pytest\nfrom freezegun import freeze_time\n\n@pytest.fixture\ndef frozen_clock():\n    with freeze_time('2024-01-01 12:00:00') as ft:\n        yield ft\n\ndef test_uses_frozen_clock(frozen_clock):\n    from datetime import datetime\n    assert datetime.now().year == 2024\n```\nThe `with` block scopes the freeze to the test duration.",
        "Let the test advance the clock manually.",
        "```python\nfrom datetime import datetime, timedelta\nfrom freezegun import freeze_time\n\ndef test_advance():\n    with freeze_time('2024-01-01 12:00:00') as ft:\n        t0 = datetime.now()\n        ft.tick(delta=timedelta(seconds=30))\n        assert (datetime.now() - t0).total_seconds() == 30\n```\n`ft.tick` moves the frozen clock forward without unfreezing it.")

    add("test",
        "Write a pytest test that asserts an event was published to a queue.",
        "```python\nfrom unittest.mock import MagicMock\n\ndef do_thing(queue):\n    queue.publish({'type': 'created', 'id': 1})\n\ndef test_publishes():\n    queue = MagicMock()\n    do_thing(queue)\n    queue.publish.assert_called_once_with({'type': 'created', 'id': 1})\n```\nBoundary mocks are the right place for unit tests of side-effecting code.",
        "Use a more flexible matcher in case the payload also includes a timestamp.",
        "```python\nfrom unittest.mock import MagicMock, ANY\n\ndef test_publishes_flexible():\n    queue = MagicMock()\n    do_thing(queue)\n    queue.publish.assert_called_once()\n    args, _ = queue.publish.call_args\n    assert args[0]['type'] == 'created'\n    assert args[0]['id'] == 1\n```\nInstead of using `ANY`, picking out the fields you care about gives much better failure messages.")

    add("test",
        "Write a pytest test that runs against multiple Python versions using tox.",
        "```ini\n# tox.ini\n[tox]\nenvlist = py310, py311, py312\n\n[testenv]\ndeps = pytest\ncommands = pytest\n```\nRun `tox` to test against each interpreter listed.",
        "Convert to nox (Python config) for the same effect.",
        "```python\n# noxfile.py\nimport nox\n\n@nox.session(python=['3.10', '3.11', '3.12'])\ndef tests(session):\n    session.install('pytest', '.')\n    session.run('pytest')\n```\n`nox` lets you write the matrix in Python and is more flexible for non-trivial setup.")

    add("test",
        "Write a pytest test that ensures a logger emits a message at a specific level.",
        "```python\nimport logging\nimport pytest\n\nlog = logging.getLogger('app')\n\ndef alert(msg):\n    log.warning(msg)\n\ndef test_warns(caplog):\n    with caplog.at_level(logging.WARNING, logger='app'):\n        alert('uh oh')\n    assert any('uh oh' in r.message and r.levelno == logging.WARNING\n               for r in caplog.records)\n```\nChecking `levelno` ensures the message went out at the expected severity, not just that the text appeared anywhere.",
        "Switch to assertLogs-style context manager.",
        "```python\nimport logging\nimport unittest\n\nlog = logging.getLogger('app')\n\nclass MyTest(unittest.TestCase):\n    def test_warns(self):\n        with self.assertLogs('app', level='WARNING') as cm:\n            log.warning('uh oh')\n        self.assertIn('uh oh', cm.output[0])\n```\n`assertLogs` is the unittest way; it raises if no log of the specified level was emitted.")

    add("test",
        "Write a pytest test that reads test data from a CSV.",
        "```python\nimport csv\nimport pytest\nfrom pathlib import Path\n\nCASES_FILE = Path(__file__).parent / 'cases.csv'\n\ndef _load_cases():\n    with CASES_FILE.open() as f:\n        return [tuple(row) for row in csv.reader(f)]\n\n@pytest.mark.parametrize('a,b,expected', _load_cases())\ndef test_add(a, b, expected):\n    assert int(a) + int(b) == int(expected)\n```\nLoading at collection time means `pytest --collect-only` already shows each row as a separate test.",
        "Add a header row and use DictReader.",
        "```python\nimport csv\nimport pytest\nfrom pathlib import Path\n\nCASES_FILE = Path(__file__).parent / 'cases.csv'\n\ndef _load_cases():\n    with CASES_FILE.open() as f:\n        reader = csv.DictReader(f)\n        return [(row['a'], row['b'], row['expected']) for row in reader]\n\n@pytest.mark.parametrize('a,b,expected', _load_cases())\ndef test_add(a, b, expected):\n    assert int(a) + int(b) == int(expected)\n```\nWith a CSV header, the test reads as documentation: the column names tell you what each value is.")

    add("test",
        "Write an async pytest test that runs two coroutines concurrently.",
        "```python\nimport asyncio\nimport pytest\n\nasync def fetch(x):\n    await asyncio.sleep(0)\n    return x * 2\n\n@pytest.mark.asyncio\nasync def test_concurrent():\n    results = await asyncio.gather(fetch(1), fetch(2), fetch(3))\n    assert results == [2, 4, 6]\n```\n`asyncio.gather` runs them concurrently and collects results in order.",
        "What if I want to ensure they actually ran concurrently and not sequentially?",
        "```python\nimport asyncio\nimport time\nimport pytest\n\nasync def slow(x):\n    await asyncio.sleep(0.1)\n    return x\n\n@pytest.mark.asyncio\nasync def test_truly_concurrent():\n    start = time.monotonic()\n    results = await asyncio.gather(slow(1), slow(2), slow(3))\n    elapsed = time.monotonic() - start\n    assert results == [1, 2, 3]\n    assert elapsed < 0.25  # should be ~0.1s, not 0.3s\n```\nTiming the wall-clock duration is the standard way to assert concurrency. Sequential execution would take ~0.3s; concurrent ~0.1s with margin.")

    add("test",
        "Write a pytest test that confirms a generator stops after a given count.",
        "```python\nfrom itertools import islice\n\ndef counting():\n    i = 0\n    while True:\n        yield i\n        i += 1\n\ndef test_first_5():\n    assert list(islice(counting(), 5)) == [0, 1, 2, 3, 4]\n```\n`islice` is the safe way to test infinite generators — it bounds the consumption.")

    add("test",
        "Write a pytest test that asserts a function is called with kwargs.",
        "```python\nfrom unittest.mock import MagicMock\n\ndef test_kwargs():\n    fn = MagicMock()\n    fn(name='alice', role='admin')\n    fn.assert_called_once_with(name='alice', role='admin')\n```\n`assert_called_once_with` matches positional and keyword args exactly.")

    add("test",
        "Write a pytest test that runs against multiple inputs using a fixture-parametrize combo.",
        "```python\nimport pytest\n\n@pytest.fixture(params=[1, 2, 3])\ndef num(request):\n    return request.param\n\ndef test_double(num):\n    assert num * 2 in (2, 4, 6)\n```\nA parametrized fixture runs the test once per param. Combine with `@parametrize` for matrix coverage.")

    add("test",
        "Write a pytest fixture that mocks the entire requests library.",
        "```python\nimport pytest\nfrom unittest.mock import MagicMock\n\n@pytest.fixture\ndef mock_requests(monkeypatch):\n    m = MagicMock()\n    monkeypatch.setattr('requests.get', m.get)\n    monkeypatch.setattr('requests.post', m.post)\n    return m\n\ndef test_api(mock_requests):\n    mock_requests.get.return_value.json.return_value = {'ok': True}\n    from myapp import call_api\n    assert call_api()['ok'] is True\n```\nReturning the parent `m` lets the test arrange both sides at once.",
        "Switch to responses or respx for cleaner request matching.",
        "```python\nimport responses\nimport requests\n\n@responses.activate\ndef test_api():\n    responses.add(responses.GET, 'https://api.example.com/data', json={'ok': True}, status=200)\n    r = requests.get('https://api.example.com/data')\n    assert r.json() == {'ok': True}\n    assert len(responses.calls) == 1\n```\n`responses` matches by URL/method, asserts on calls, and reads more like an integration test. For httpx use `respx` instead.")
