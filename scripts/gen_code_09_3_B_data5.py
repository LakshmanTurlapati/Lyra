# SPDX-License-Identifier: MIT
"""Testing entries: pytest, mocks, fixtures (~85 total)."""


def register_all(add):
    add("test",
        "Write a pytest test for an `add` function.",
        "```python\nfrom myapp import add\n\ndef test_add():\n    assert add(2, 3) == 5\n    assert add(-1, 1) == 0\n```\nKeep tests focused; one logical assertion per behavior is the ideal.")

    add("test",
        "Use pytest.parametrize for multiple inputs.",
        "```python\nimport pytest\nfrom myapp import is_even\n\n@pytest.mark.parametrize('n,expected', [(2, True), (3, False), (0, True)])\ndef test_is_even(n, expected):\n    assert is_even(n) == expected\n```\nParametrize avoids duplicating test bodies and gives each case its own pass/fail line.")

    add("test",
        "Write a fixture that creates a temp file.",
        "```python\nimport pytest\nfrom pathlib import Path\n\n@pytest.fixture\ndef tmpfile(tmp_path):\n    p = tmp_path / 'data.txt'\n    p.write_text('hello')\n    return p\n\ndef test_read(tmpfile: Path):\n    assert tmpfile.read_text() == 'hello'\n```\nUse the built-in `tmp_path` fixture for safe per-test cleanup.")

    add("test",
        "Use unittest.mock.patch to replace a function.",
        "```python\nfrom unittest.mock import patch\nfrom myapp import call_external\n\n@patch('myapp.requests.get')\ndef test_call(mock_get):\n    mock_get.return_value.json.return_value = {'ok': True}\n    assert call_external() == {'ok': True}\n```\nPatch where the name is *looked up*, not where it's defined.")

    add("test",
        "Test that a function raises a specific exception.",
        "```python\nimport pytest\nfrom myapp import divide\n\ndef test_divide_zero():\n    with pytest.raises(ZeroDivisionError):\n        divide(1, 0)\n```\n`pytest.raises` doubles as a check that the exception happened *and* a context manager.")

    add("test",
        "Use pytest.fixture with yield for setup/teardown.",
        "```python\nimport pytest\n\n@pytest.fixture\ndef db():\n    conn = open_db()\n    yield conn\n    conn.close()\n\ndef test_query(db):\n    assert db.execute('SELECT 1') == [(1,)]\n```\nThe code after `yield` runs even if the test fails -- guaranteed cleanup.")

    add("test",
        "Mock an HTTP call using responses.",
        "```python\nimport responses\nimport requests\n\n@responses.activate\ndef test_fetch():\n    responses.get('https://api.example.com/x', json={'ok': True})\n    r = requests.get('https://api.example.com/x')\n    assert r.json() == {'ok': True}\n```\n`responses` intercepts at the urllib3 layer so no real network happens.")

    add("test",
        "Test an async function with pytest-asyncio.",
        "```python\nimport pytest\nfrom myapp import fetch_async\n\n@pytest.mark.asyncio\nasync def test_fetch_async():\n    result = await fetch_async('http://x.test')\n    assert result['ok']\n```\nRequires `pytest-asyncio` installed and `asyncio_mode = auto` in pyproject for the marker to be implicit.")

    add("test",
        "Use pytest-mock instead of unittest.mock.",
        "```python\ndef test_call(mocker):\n    mock_get = mocker.patch('myapp.requests.get')\n    mock_get.return_value.json.return_value = {'k': 1}\n    from myapp import fetch\n    assert fetch() == {'k': 1}\n```\nThe `mocker` fixture handles cleanup automatically -- no decorators needed.")

    add("test",
        "Write a fixture with a class scope.",
        "```python\nimport pytest\n\n@pytest.fixture(scope='class')\ndef shared_resource():\n    print('expensive setup')\n    yield 42\n    print('teardown')\n```\nExpensive setup runs once per test class instead of per test.")

    add("test",
        "Use freezegun to freeze time during a test.",
        "```python\nfrom freezegun import freeze_time\nimport datetime as dt\n\n@freeze_time('2026-01-01')\ndef test_today():\n    assert dt.date.today() == dt.date(2026, 1, 1)\n```\nMakes time-dependent code deterministic.")

    add("test",
        "Test a Flask endpoint with the test_client.",
        "```python\nfrom myapp import app\n\ndef test_root():\n    client = app.test_client()\n    resp = client.get('/')\n    assert resp.status_code == 200\n```\nNo network involved -- the test_client dispatches WSGI calls directly.")

    add("test",
        "Test a FastAPI endpoint with TestClient.",
        "```python\nfrom fastapi.testclient import TestClient\nfrom myapp import app\n\ndef test_health():\n    client = TestClient(app)\n    r = client.get('/health')\n    assert r.status_code == 200\n    assert r.json() == {'ok': True}\n```\nStarlette's TestClient is sync and built on httpx.")

    add("test",
        "Use pytest.fixture(autouse=True) to apply to every test.",
        "```python\nimport pytest, logging\n\n@pytest.fixture(autouse=True)\ndef silence_logs():\n    logging.disable(logging.CRITICAL)\n    yield\n    logging.disable(logging.NOTSET)\n```\nPlace in conftest.py so it applies to every test in the directory.")

    add("test",
        "Use `caplog` to assert a log message was emitted.",
        "```python\nimport logging\nimport pytest\n\ndef test_warn(caplog):\n    with caplog.at_level(logging.WARNING):\n        logging.warning('something off')\n    assert 'something off' in caplog.text\n```\nThe built-in fixture captures log records without monkeypatching.")

    add("test",
        "Test an exception's message.",
        "```python\nimport pytest\n\ndef test_msg():\n    with pytest.raises(ValueError, match='must be positive'):\n        validate(-1)\n```\n`match=` is a regex, so escape special chars if you need a literal match.")

    add("test",
        "Use pytest.fixture(params=[...]) to parameterize a fixture.",
        "```python\nimport pytest\n\n@pytest.fixture(params=['sqlite', 'postgres'])\ndef db(request):\n    return open_db(request.param)\n\ndef test_query(db):\n    assert db.execute('SELECT 1')\n```\nEach test using `db` runs once per parameter -- a quick way to test backend agnosticism.")

    add("test",
        "Use Hypothesis for property-based testing.",
        "```python\nfrom hypothesis import given, strategies as st\n\n@given(st.lists(st.integers()))\ndef test_sort_idempotent(xs):\n    assert sorted(sorted(xs)) == sorted(xs)\n```\nHypothesis generates many random inputs and shrinks failures to minimal counterexamples.")

    add("test",
        "Mock a class instance method with autospec.",
        "```python\nfrom unittest.mock import patch\nfrom myapp import Client\n\n@patch('myapp.Client', autospec=True)\ndef test_use_client(MockClient):\n    instance = MockClient.return_value\n    instance.fetch.return_value = 'x'\n    assert use_client() == 'x'\n```\n`autospec=True` ensures the mock has the same signature as the real class -- catches misuse.")

    add("test",
        "Test that two floats are approximately equal.",
        "```python\nimport pytest\n\ndef test_pi():\n    assert compute_pi() == pytest.approx(3.14159, rel=1e-4)\n```\n`pytest.approx` handles relative and absolute tolerances cleanly.")

    add("test",
        "Use a fixture to seed a deterministic random generator.",
        "```python\nimport random\nimport pytest\n\n@pytest.fixture\ndef rng():\n    random.seed(42)\n    yield random\n```\nFor numpy use `np.random.default_rng(42)` to get a thread-safe generator.")

    add("test",
        "Skip a test conditionally.",
        "```python\nimport sys\nimport pytest\n\n@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX only')\ndef test_signal():\n    ...\n```\nGood for platform-specific behavior or feature-flagged code.")

    add("test",
        "Mark a test as expected failure.",
        "```python\nimport pytest\n\n@pytest.mark.xfail(reason='bug #123 not fixed yet')\ndef test_known_bug():\n    assert wrong_thing()\n```\nIf it unexpectedly passes, pytest reports XPASS which is loud -- great for catching when bugs are fixed silently.")

    add("test",
        "Use `monkeypatch` to set an environment variable for one test.",
        "```python\ndef test_env(monkeypatch):\n    monkeypatch.setenv('API_KEY', 'test-key')\n    import os\n    assert os.environ['API_KEY'] == 'test-key'\n```\nThe original env is restored automatically after the test.")

    add("test",
        "Test a generator function.",
        "```python\nfrom myapp import counter\n\ndef test_counter():\n    g = counter()\n    assert next(g) == 1\n    assert next(g) == 2\n```\nFor full sequences, materialize with `list(...)` and compare.")

    add("test",
        "Use pytest.fixture to share data across tests in a module.",
        "```python\nimport pytest\n\n@pytest.fixture(scope='module')\ndef expensive_data():\n    return load_dataset()\n```\nLoaded once per file, then reused -- great for read-only test data.")

    add("test",
        "Mock a context manager.",
        "```python\nfrom unittest.mock import MagicMock, patch\n\n@patch('myapp.open', new_callable=MagicMock)\ndef test_open(mock_open):\n    mock_open.return_value.__enter__.return_value.read.return_value = 'hi'\n    from myapp import read_file\n    assert read_file('x.txt') == 'hi'\n```\n`unittest.mock.mock_open` is a more direct helper for this case.")

    add("test",
        "Test a timeout with pytest-timeout.",
        "```python\nimport pytest\n\n@pytest.mark.timeout(5)\ndef test_slow_query():\n    run_query()\n```\nFails if the test takes longer than 5 seconds. Useful for catching infinite loops in CI.")

    add("test",
        "Use pytest's tmp_path for filesystem tests.",
        "```python\nfrom pathlib import Path\nfrom myapp import write_log\n\ndef test_write_log(tmp_path: Path):\n    log_file = tmp_path / 'app.log'\n    write_log(log_file, 'hello')\n    assert log_file.read_text() == 'hello\\n'\n```\nEach test gets a fresh isolated directory.")

    add("test",
        "Use pytest's capsys to capture stdout.",
        "```python\nfrom myapp import greet\n\ndef test_greet(capsys):\n    greet('alice')\n    captured = capsys.readouterr()\n    assert 'hello, alice' in captured.out\n```\nUse `capsys.disabled()` inside a `with` block if you need to print for debugging.")

    add("test",
        "Skip a test class entirely.",
        "```python\nimport pytest\n\n@pytest.mark.skip(reason='deprecated module')\nclass TestOldThing:\n    def test_a(self): ...\n    def test_b(self): ...\n```\nFor entire files, put the marker at module level: `pytestmark = pytest.mark.skip(...)`.")

    add("test",
        "Use Hypothesis stateful testing.",
        "```python\nfrom hypothesis.stateful import RuleBasedStateMachine, rule\n\nclass MyStack(RuleBasedStateMachine):\n    def __init__(self):\n        super().__init__()\n        self.stack = []\n    @rule(x=st.integers())\n    def push(self, x):\n        self.stack.append(x)\n    @rule()\n    def pop(self):\n        if self.stack:\n            self.stack.pop()\n\nTestStack = MyStack.TestCase\n```\nState machines find sequences of operations that violate invariants.")

    add("test",
        "Test that a function calls another with specific args.",
        "```python\nfrom unittest.mock import patch\nfrom myapp import wrapper, inner\n\ndef test_wrapper_calls_inner():\n    with patch('myapp.inner') as m:\n        wrapper(5)\n        m.assert_called_once_with(5)\n```\n`assert_called_once_with` is stricter and clearer than `called` + manual arg checks.")

    add("test",
        "Use pytest fixtures with conftest.py for shared setup.",
        "```python\n# tests/conftest.py\nimport pytest\nfrom myapp import create_app\n\n@pytest.fixture\ndef app():\n    return create_app(testing=True)\n```\nFixtures in conftest.py are auto-discovered for tests in the same directory tree.")

    add("test",
        "Test exception chaining (raise from).",
        "```python\nimport pytest\n\ndef test_chain():\n    with pytest.raises(RuntimeError) as exc_info:\n        outer()\n    assert isinstance(exc_info.value.__cause__, ValueError)\n```\n`__cause__` is set when you `raise X from Y`.")

    add("test",
        "Use `@pytest.mark.slow` to organize slow tests.",
        "```python\nimport pytest\n\n@pytest.mark.slow\ndef test_slow_thing():\n    ...\n# pytest -m 'not slow' to skip\n```\nRegister the marker in pyproject to silence warnings: `[tool.pytest.ini_options]\\nmarkers = ['slow: slow tests']`.")

    add("test",
        "Mock a property.",
        "```python\nfrom unittest.mock import patch, PropertyMock\nfrom myapp import Service\n\ndef test_status():\n    with patch.object(Service, 'status', new_callable=PropertyMock) as m:\n        m.return_value = 'ok'\n        s = Service()\n        assert s.status == 'ok'\n```\n`PropertyMock` is required because properties are class-level descriptors.")

    add("test",
        "Test that a function returns within a tolerance over many trials.",
        "```python\nimport statistics\n\ndef test_average():\n    samples = [random_sample() for _ in range(1000)]\n    assert abs(statistics.mean(samples)) < 0.05\n```\nStochastic tests should set tolerances generously enough to avoid flakes.")

    add("test",
        "Test SQLAlchemy code with an in-memory SQLite.",
        "```python\nimport pytest\nfrom sqlalchemy import create_engine\nfrom myapp.models import Base\n\n@pytest.fixture\ndef engine():\n    e = create_engine('sqlite:///:memory:')\n    Base.metadata.create_all(e)\n    yield e\n    e.dispose()\n```\nIn-memory SQLite is fast and isolated; only a problem if you rely on PostgreSQL-specific features.")

    add("test",
        "Use pytest's `request` fixture to access the test name.",
        "```python\nimport pytest\n\n@pytest.fixture\ndef name(request):\n    return request.node.name\n\ndef test_a(name):\n    assert name == 'test_a'\n```\nUseful for fixtures that produce test-specific resources (e.g. log files named after the test).")

    add("test",
        "Test a CLI built with click using CliRunner.",
        "```python\nfrom click.testing import CliRunner\nfrom myapp.cli import main\n\ndef test_help():\n    result = CliRunner().invoke(main, ['--help'])\n    assert result.exit_code == 0\n    assert 'Usage:' in result.output\n```\nCliRunner runs the command in-process and captures stdout/stderr cleanly.")

    add("test",
        "Use a fixture to provide a fresh logger per test.",
        "```python\nimport logging\nimport pytest\n\n@pytest.fixture\ndef logger():\n    log = logging.getLogger('test')\n    log.handlers.clear()\n    yield log\n```\nClearing handlers prevents pollution from other tests' setups.")

    add("test",
        "Use pytest's `tmp_path_factory` for session-scoped temp dirs.",
        "```python\nimport pytest\n\n@pytest.fixture(scope='session')\ndef big_dir(tmp_path_factory):\n    d = tmp_path_factory.mktemp('shared')\n    return d\n```\nUseful when test fixtures share heavy disk state across many tests.")

    add("test",
        "Mock time.time directly.",
        "```python\nfrom unittest.mock import patch\nfrom myapp import timestamp\n\n@patch('myapp.time.time', return_value=1700000000.0)\ndef test_ts(mock_time):\n    assert timestamp() == 1700000000\n```\nPatch where the function is *imported*, not the time module itself.")

    add("test",
        "Use Hypothesis to test that a function is its own inverse.",
        "```python\nfrom hypothesis import given, strategies as st\nfrom myapp import encode, decode\n\n@given(st.binary())\ndef test_roundtrip(data):\n    assert decode(encode(data)) == data\n```\nRound-trip properties are extremely valuable; one line of test catches dozens of edge cases.")

    add("test",
        "Test a websocket handler with FastAPI's TestClient.",
        "```python\nfrom fastapi.testclient import TestClient\nfrom myapp import app\n\ndef test_ws():\n    client = TestClient(app)\n    with client.websocket_connect('/ws') as ws:\n        ws.send_text('hi')\n        assert ws.receive_text() == 'hi'\n```\nThe context manager handles connect/disconnect cleanly.")

    add("test",
        "Group tests with classes.",
        "```python\nimport pytest\n\nclass TestUser:\n    def test_create(self):\n        ...\n    def test_delete(self):\n        ...\n```\nClasses are cosmetic in pytest -- no need to inherit from anything. Use them when grouping makes the test list more readable.")

    add("test",
        "Use pytest plugins via `-p`.",
        "```bash\npytest -p no:cacheprovider tests/\n```\nDisable specific plugins on the command line. For per-project config, use `pyproject.toml`'s `[tool.pytest.ini_options]`.")

    add("test",
        "Test that an iterable is empty.",
        "```python\ndef test_empty():\n    items = list(filter_invalid([]))\n    assert items == []\n```\n`assert not items` works too but has worse failure messages.")

    add("test",
        "Use parametrize with ids for nicer test names.",
        "```python\nimport pytest\n\n@pytest.mark.parametrize('s,expected', [\n    ('hello', 5),\n    ('', 0),\n    ('python', 6),\n], ids=['short', 'empty', 'medium'])\ndef test_len(s, expected):\n    assert len(s) == expected\n```\n`ids=` makes failing tests easy to spot in the output.")

    add("test",
        "Mock a chained method call.",
        "```python\nfrom unittest.mock import MagicMock\n\ndef test_chain():\n    m = MagicMock()\n    m.client.users.get().json.return_value = {'id': 1}\n    assert m.client.users.get().json() == {'id': 1}\n```\nMagicMock auto-creates attributes; useful for fluent APIs.")

    add("test",
        "Use pytest-cov to measure coverage.",
        "```bash\npytest --cov=myapp --cov-report=term-missing tests/\n```\nCoverage of branches not just lines: add `--cov-branch`.")

    add("test",
        "Test a class invariant with Hypothesis.",
        "```python\nfrom hypothesis import given, strategies as st\nfrom myapp import Counter\n\n@given(st.lists(st.integers(min_value=0, max_value=10)))\ndef test_counter(ops):\n    c = Counter()\n    for x in ops:\n        c.add(x)\n    assert c.value >= 0\n```\nHypothesis is great at finding invariant violations.")

    add("test",
        "Use a session-scoped fixture for an HTTP server.",
        "```python\nimport pytest, threading\nfrom http.server import HTTPServer, BaseHTTPRequestHandler\n\n@pytest.fixture(scope='session')\ndef server():\n    srv = HTTPServer(('localhost', 0), BaseHTTPRequestHandler)\n    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()\n    yield f'http://localhost:{srv.server_port}'\n    srv.shutdown()\n```\nSession scope amortizes the startup cost across all tests.")

    add("test",
        "Test that a deprecation warning is raised.",
        "```python\nimport pytest\nimport warnings\nfrom myapp import old_api\n\ndef test_warn():\n    with pytest.warns(DeprecationWarning, match='use new_api'):\n        old_api()\n```\n`pytest.warns` parallels `pytest.raises` but for warnings.")

    add("test",
        "Use a `conftest.py` plugin hook to add a CLI option.",
        "```python\n# conftest.py\ndef pytest_addoption(parser):\n    parser.addoption('--runslow', action='store_true', default=False)\n\ndef pytest_collection_modifyitems(config, items):\n    if not config.getoption('--runslow'):\n        skip_slow = pytest.mark.skip(reason='need --runslow')\n        for item in items:\n            if 'slow' in item.keywords:\n                item.add_marker(skip_slow)\n```\nNow `pytest --runslow` enables the slow tests on demand.")

    add("test",
        "Mock a Redis client.",
        "```python\nfrom unittest.mock import MagicMock\nfrom myapp import cache_get\n\ndef test_cache_hit():\n    redis = MagicMock()\n    redis.get.return_value = b'cached'\n    assert cache_get(redis, 'k') == 'cached'\n```\nThe `fakeredis` library is also a great option for higher-fidelity tests.")

    add("test",
        "Test JSON output structure with `==`.",
        "```python\ndef test_payload():\n    out = build_payload(user_id=1)\n    assert out == {'user': 1, 'role': 'guest'}\n```\nDict equality is order-independent in Python -- safer than string comparison.")

    add("test",
        "Use freezegun to advance time in a test.",
        "```python\nfrom freezegun import freeze_time\nimport time\n\ndef test_advance():\n    with freeze_time('2026-01-01') as frozen:\n        start = time.time()\n        frozen.tick(60)\n        assert time.time() - start == 60\n```\n`tick(seconds)` advances the frozen clock without sleeping.")

    add("test",
        "Use pytest's `--lf` to rerun only last-failed tests.",
        "```bash\npytest --lf\n```\nGreat for tight TDD loops; rerun just the failures until they pass, then `pytest` again for full suite.")

    add("test",
        "Test that a JSON file matches a schema.",
        "```python\nimport json\nfrom jsonschema import validate\n\ndef test_schema(tmp_path):\n    data = json.loads(tmp_path.joinpath('out.json').read_text())\n    schema = {'type': 'object', 'required': ['id'], 'properties': {'id': {'type': 'integer'}}}\n    validate(data, schema)\n```\n`validate` raises ValidationError on mismatch.")

    add("test",
        "Snapshot test using syrupy.",
        "```python\ndef test_render(snapshot):\n    output = render_user({'name': 'alice'})\n    assert output == snapshot\n```\nFirst run records the snapshot; subsequent runs compare. Update with `pytest --snapshot-update`.")

    add("test",
        "Mock a database query result.",
        "```python\nfrom unittest.mock import MagicMock\n\ndef test_query():\n    db = MagicMock()\n    db.fetch_all.return_value = [{'id': 1}]\n    from myapp import list_items\n    assert list_items(db) == [{'id': 1}]\n```\nDependency-injecting the db keeps the test fast and deterministic.")

    add("test",
        "Test logging level configuration.",
        "```python\nimport logging\n\ndef test_level(caplog):\n    with caplog.at_level(logging.DEBUG, logger='myapp'):\n        from myapp import work\n        work()\n    assert any(r.levelno == logging.DEBUG for r in caplog.records)\n```\n`at_level` accepts a logger name to scope the capture.")

    add("test",
        "Use parametrize with `pytest.param` and a marker.",
        "```python\nimport pytest\n\n@pytest.mark.parametrize('x', [\n    1,\n    2,\n    pytest.param(3, marks=pytest.mark.xfail),\n])\ndef test_x(x):\n    assert x in (1, 2)\n```\nMix per-case markers with parametrize when one input has special handling.")

    add("test",
        "Use a fixture to roll back a DB transaction after each test.",
        "```python\nimport pytest\n\n@pytest.fixture\ndef session(engine):\n    with engine.connect() as conn:\n        with conn.begin() as txn:\n            yield Session(bind=conn)\n            txn.rollback()\n```\nRollback gives you isolated, fast tests without recreating the schema each time.")

    add("test",
        "Test that two CSV files have equal contents.",
        "```python\nimport pandas as pd\nfrom pandas.testing import assert_frame_equal\n\ndef test_csv(tmp_path):\n    a = pd.read_csv('expected.csv')\n    b = pd.read_csv(tmp_path / 'output.csv')\n    assert_frame_equal(a, b)\n```\n`assert_frame_equal` shows clear diffs on failure; better than `df1.equals(df2)`.")

    add("test",
        "Use pytest fixtures with finalizers (alternative to yield).",
        "```python\nimport pytest\n\n@pytest.fixture\ndef temp(request):\n    obj = create()\n    request.addfinalizer(obj.close)\n    return obj\n```\nThe yield-based pattern is simpler and more common; finalizers help when you need conditional cleanup.")

    add("test",
        "Use parametrize indirectly to drive a fixture.",
        "```python\nimport pytest\n\n@pytest.fixture\ndef db(request):\n    return open_db(request.param)\n\n@pytest.mark.parametrize('db', ['sqlite', 'postgres'], indirect=True)\ndef test_query(db):\n    assert db.execute('SELECT 1')\n```\n`indirect=True` makes parametrize values flow into fixtures rather than test args.")

    add("test",
        "Test exception groups (Python 3.11+).",
        "```python\nimport pytest\n\ndef test_group():\n    with pytest.raises(ExceptionGroup) as eg:\n        raise ExceptionGroup('boom', [ValueError('a'), KeyError('b')])\n    assert len(eg.value.exceptions) == 2\n```\nExceptionGroup is the new mechanism for raising multiple exceptions at once (used by asyncio.TaskGroup).")

    add("test",
        "Use approx for sequence comparison with tolerance.",
        "```python\nimport pytest\n\ndef test_seq():\n    actual = [0.1 + 0.2, 0.3 + 0.4]\n    assert actual == pytest.approx([0.3, 0.7])\n```\n`pytest.approx` works element-wise for sequences and even nested mappings.")

    add("test",
        "Test that an HTTP retry actually retried.",
        "```python\nimport responses\nimport requests\n\n@responses.activate\ndef test_retry():\n    responses.get('https://api.example.com', status=500)\n    responses.get('https://api.example.com', status=500)\n    responses.get('https://api.example.com', json={'ok': True})\n    from myapp import fetch_with_retry\n    r = fetch_with_retry()\n    assert r == {'ok': True}\n    assert len(responses.calls) == 3\n```\nQueue multiple responses; `responses.calls` records every request made.")

    add("test",
        "Skip a test if a dependency is missing.",
        "```python\nimport pytest\n\nbson = pytest.importorskip('bson')\n\ndef test_bson_roundtrip():\n    payload = bson.dumps({'x': 1})\n    assert bson.loads(payload) == {'x': 1}\n```\n`pytest.importorskip` skips the test gracefully when the import fails.")

    add("test",
        "Use a fixture to create a unique namespace per test.",
        "```python\nimport pytest, uuid\n\n@pytest.fixture\ndef ns():\n    return f'test_{uuid.uuid4().hex[:8]}'\n```\nGreat for parallel test runs that touch shared external state (Redis keys, S3 prefixes, etc.).")

    add("test",
        "Test stdout output of a script.",
        "```python\nimport subprocess\n\ndef test_script():\n    result = subprocess.run(['python', '-m', 'myapp', '--name', 'alice'], capture_output=True, text=True)\n    assert result.returncode == 0\n    assert 'hello, alice' in result.stdout\n```\nFull-process tests are slow but catch issues import-level tests can miss.")

    add("test",
        "Test using a fake clock with FreezeGun and tick().",
        "```python\nfrom freezegun import freeze_time\nimport time\n\ndef test_tick():\n    with freeze_time('2026-01-01 00:00:00') as frozen:\n        a = time.time()\n        frozen.tick()  # default 1 second\n        b = time.time()\n        assert b - a == 1\n```\n`tick()` defaults to 1 second; pass `delta=` for custom amounts.")

    add("test",
        "Test that two pandas Series are equal up to dtype.",
        "```python\nimport pandas as pd\nfrom pandas.testing import assert_series_equal\n\ndef test_series():\n    a = pd.Series([1, 2, 3])\n    b = pd.Series([1.0, 2.0, 3.0])\n    assert_series_equal(a, b, check_dtype=False)\n```\n`check_dtype=False` ignores int vs float; full equality is the default.")

    add("test",
        "Test custom JSON encoder.",
        "```python\nimport json\nfrom myapp import MyEncoder\n\ndef test_encoder():\n    out = json.dumps({'set': {1, 2}}, cls=MyEncoder)\n    assert json.loads(out) == {'set': [1, 2]}\n```\nThe encoder turns sets into lists; verify the round-trip semantics.")

    add("test",
        "Use a fixture for a temporary SQLite DB on disk.",
        "```python\nimport pytest\nimport sqlite3\n\n@pytest.fixture\ndef db(tmp_path):\n    path = tmp_path / 'test.db'\n    conn = sqlite3.connect(path)\n    conn.execute('CREATE TABLE users(id INTEGER, name TEXT)')\n    yield conn\n    conn.close()\n```\nDisk-backed SQLite is helpful when you want to inspect the file after a failure.")

    # ---- 5-msg follow-ups ----
    add("test",
        "Use pytest-mock to spy on a function call.",
        "```python\ndef test_spy(mocker):\n    spy = mocker.spy(myapp, 'inner')\n    myapp.outer(5)\n    spy.assert_called_once_with(5)\n```\n`spy` lets the real function run while still recording calls.",
        "When would I prefer a spy over a mock?",
        "Use a spy when you want the real implementation to run (so downstream behavior is correct) but also want to assert how it was invoked. Mocks replace the function entirely -- great for isolating units. Spies are perfect for integration tests where you want both real behavior and call inspection.")

    add("test",
        "Test that an async function times out.",
        "```python\nimport asyncio\nimport pytest\n\n@pytest.mark.asyncio\nasync def test_timeout():\n    with pytest.raises(asyncio.TimeoutError):\n        async with asyncio.timeout(0.1):\n            await asyncio.sleep(1)\n```\n`asyncio.timeout` (3.11+) is the preferred way to bound async work.",
        "What about pre-3.11 codebases?",
        "Use `asyncio.wait_for(coro, timeout=0.1)` instead. The semantics are the same: raises `asyncio.TimeoutError` if the coroutine doesn't finish in time. The 3.11+ context-manager form is preferred because you can wrap multiple awaits and it composes cleanly with `TaskGroup`. For older code, `wait_for` does the job.")

    add("test",
        "Mock a class method on an existing instance.",
        "```python\nfrom unittest.mock import patch\n\ndef test_instance_method():\n    obj = MyClass()\n    with patch.object(obj, 'compute', return_value=42):\n        assert obj.compute() == 42\n```\n`patch.object` on the instance leaves other instances unchanged.",
        "Why patch the instance instead of the class?",
        "Patching the class affects every instance, which is usually too broad and can leak between tests. Patching a single instance scopes the mock precisely. If you do want to mock for the lifetime of a test (and you're sure no other code creates the class), patching the class is fine -- patch+context manager handles cleanup.")

    add("test",
        "Test a Flask route that requires authentication.",
        "```python\ndef test_auth(client):\n    headers = {'Authorization': 'Bearer test-token'}\n    r = client.get('/secure', headers=headers)\n    assert r.status_code == 200\n```\nFor Flask, the `client` fixture comes from `app.test_client()`.",
        "Should I have a fixture that logs in once and reuses the session?",
        "Yes, for higher-level tests. Define a `logged_client` fixture that performs the login flow and returns an authenticated client. Tests then run against the authenticated state without each one repeating the login. For lower-level routes you're testing in isolation, mocking the auth dependency is cleaner.")

    add("test",
        "Use pytest's `monkeypatch.setattr` to patch a class attribute.",
        "```python\ndef test_patch_class(monkeypatch):\n    from myapp import Settings\n    monkeypatch.setattr(Settings, 'API_KEY', 'test-key')\n    assert Settings.API_KEY == 'test-key'\n```\n`monkeypatch` undoes the patch after the test.",
        "How does monkeypatch differ from unittest.mock.patch?",
        "`monkeypatch` is a pytest fixture for simple attribute replacement -- it doesn't create a Mock object, just swaps the value. `patch` from unittest.mock creates a MagicMock by default, with call tracking, return_value, side_effect, etc. Use monkeypatch for plain value swaps (env vars, config); use patch when you need to assert on calls.")

    add("test",
        "Use Hypothesis with assume() to filter inputs.",
        "```python\nfrom hypothesis import given, assume, strategies as st\n\n@given(st.integers())\ndef test_div(n):\n    assume(n != 0)\n    assert (10 / n) * n == 10  # may fail due to floats but illustrative\n```\n`assume` discards the example and tries again -- but excessive filtering slows tests, so prefer constrained strategies when possible.",
        "What's the difference between assume and a constrained strategy?",
        "`st.integers().filter(lambda n: n != 0)` and `assume(n != 0)` look similar but the strategy version is more efficient because Hypothesis can shape generation. `assume` is useful when the constraint is hard to express directly in a strategy. Hypothesis emits a `HealthCheck.filter_too_much` warning if assume rejects too many examples -- a signal to refactor into a constrained strategy.")

    add("test",
        "Mock a function with side effects per call.",
        "```python\nfrom unittest.mock import MagicMock\n\ndef test_seq():\n    m = MagicMock(side_effect=[1, 2, 3, ValueError('done')])\n    assert m() == 1\n    assert m() == 2\n    assert m() == 3\n    import pytest\n    with pytest.raises(ValueError):\n        m()\n```\n`side_effect` as a list returns sequential values, then raises if an exception type appears.",
        "What if I want different responses based on input?",
        "Pass a callable as `side_effect`: `m.side_effect = lambda x: x.upper() if isinstance(x, str) else 0`. The callable receives the same args as the call. This lets you simulate complex behavior without writing a full stub class. Combine with `spec=` or `autospec=True` to get signature checking on the mock.")

    add("test",
        "Test an Anthropic-style streaming generator.",
        "```python\nimport pytest\nfrom myapp import stream_response\n\ndef test_stream():\n    chunks = list(stream_response('hi'))\n    assert ''.join(chunks).startswith('hello')\n```\nMaterializing the generator into a list is fine for short streams.",
        "What if the stream is infinite or very long?",
        "Use `itertools.islice` to take the first N items, or use a timeout-based assertion. For streams that should terminate, assert termination explicitly: `next(g, None)` should be None after consuming all expected output. Use `pytest-timeout` to fail tests that hang on broken streaming logic.")

    add("test",
        "Use pytest-xdist to run tests in parallel.",
        "```bash\npytest -n auto tests/\n```\n`auto` uses CPU count. Tests must be independent -- shared mutable state between tests will produce flakes.",
        "How do I make a fixture safe under xdist?",
        "Session-scoped fixtures need to be safe across worker processes. For DB fixtures, give each worker its own database (use the `worker_id` fixture from xdist as a suffix). For files, use the per-worker `tmp_path_factory.getbasetemp()` instead of a global path. The key principle: no shared mutable state across workers.")

    add("test",
        "Test that a context manager properly cleans up on exception.",
        "```python\nimport pytest\nfrom myapp import resource\n\ndef test_cleanup():\n    with pytest.raises(ValueError):\n        with resource() as r:\n            r.use()\n            raise ValueError('boom')\n    assert not resource_is_open()\n```\nThe outer with-pytest.raises lets the exception propagate so __exit__ runs.",
        "What's a common mistake with context-manager testing?",
        "Forgetting that `__exit__` returning True suppresses the exception. If your context manager catches and silently absorbs errors, the test won't see the raise. Always assert the cleanup state happened, not just that the exception was raised. Also: if the context manager allocates external resources (locks, sockets), check that they're released, not just that internal flags reset.")

    add("test",
        "Test a CLI argument parser with subcommands.",
        "```python\nfrom click.testing import CliRunner\nfrom myapp.cli import main\n\ndef test_subcommand():\n    result = CliRunner().invoke(main, ['users', 'list', '--format=json'])\n    assert result.exit_code == 0\n    assert 'alice' in result.output\n```\nTest each subcommand independently so failures point at the right place.",
        "How do I test commands that prompt for input?",
        "CliRunner.invoke accepts an `input=` argument: `runner.invoke(main, ['login'], input='username\\npassword\\n')`. Each `\\n` simulates pressing Enter. For more complex flows, use `runner.invoke(main, args, input='...', catch_exceptions=False)` so the test fails loudly on unexpected exceptions instead of swallowing them.")
