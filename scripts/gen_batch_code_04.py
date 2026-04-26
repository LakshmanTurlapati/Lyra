#!/usr/bin/env python3
"""Generate 160 ShareGPT code conversations for batch-code-04.
Focus: Python testing and tooling."""
import json
from pathlib import Path

OUT = Path("/Users/lakshman/Documents/Lyra/datasets/code/v2-supplementary/batch-code-04.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

samples = []

def add(system, user, assistant, followup=None):
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    if followup:
        msgs.append({"role": "user", "content": followup[0]})
        msgs.append({"role": "assistant", "content": followup[1]})
    samples.append({"messages": msgs, "tools": [], "domain": "code"})


# ========== Block 1: pytest fixtures (items 1-20) ==========

add(
    "You are a Python testing expert.",
    "How do I create a pytest fixture that provides a temporary SQLite database for tests?",
    "Use `tmp_path` combined with a fixture that creates and yields a connection.\n\n```python\nimport sqlite3\nimport pytest\n\n@pytest.fixture\ndef db(tmp_path):\n    path = tmp_path / \"test.db\"\n    conn = sqlite3.connect(path)\n    conn.execute(\"CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)\")\n    conn.commit()\n    yield conn\n    conn.close()\n\ndef test_insert_user(db):\n    db.execute(\"INSERT INTO users (name) VALUES (?)\", (\"alice\",))\n    db.commit()\n    row = db.execute(\"SELECT name FROM users\").fetchone()\n    assert row == (\"alice\",)\n```\n\n`tmp_path` gives you a unique `pathlib.Path` per test function, so the DB is fully isolated. The fixture yields the live connection; the code after `yield` runs during teardown even if the test fails, guaranteeing the handle is closed. For a session-scoped DB that you only build once, add `scope=\"session\"` and use `tmp_path_factory` instead, since `tmp_path` is function-scoped and cannot be used across sessions.",
)

add(
    "You are a pytest coach.",
    "Show me how to parametrize a test across multiple inputs including an expected exception case.",
    "Use `pytest.param` with `marks=pytest.mark.xfail` or combine `pytest.raises` inside the test body.\n\n```python\nimport pytest\n\ndef divide(a, b):\n    return a / b\n\n@pytest.mark.parametrize(\"a, b, expected\", [\n    (10, 2, 5),\n    (9, 3, 3),\n    pytest.param(1, 0, None, marks=pytest.mark.xfail(raises=ZeroDivisionError)),\n    (-6, 3, -2),\n])\ndef test_divide(a, b, expected):\n    if expected is None:\n        divide(a, b)  # xfail will catch the exception\n    else:\n        assert divide(a, b) == expected\n```\n\nThe cleaner pattern uses a separate `raises` column:\n\n```python\n@pytest.mark.parametrize(\"a, b, expected, raises\", [\n    (10, 2, 5, None),\n    (1, 0, None, ZeroDivisionError),\n])\ndef test_divide(a, b, expected, raises):\n    if raises:\n        with pytest.raises(raises):\n            divide(a, b)\n    else:\n        assert divide(a, b) == expected\n```\n\nExplicit is better than xfail here because xfail hides the error type check.",
    followup=(
        "How do I add readable IDs to each parameter set?",
        "Pass the `ids` argument as a list or a callable.\n\n```python\n@pytest.mark.parametrize(\n    \"a, b, expected\",\n    [(10, 2, 5), (9, 3, 3), (-6, 3, -2)],\n    ids=[\"ten-by-two\", \"nine-by-three\", \"neg-six-by-three\"],\n)\ndef test_divide(a, b, expected):\n    assert divide(a, b) == expected\n```\n\nWith a callable: `ids=lambda v: f\"val={v}\"`. Readable IDs make `pytest -k` filters much easier and diagnostics like `test_divide[ten-by-two]` actually readable in CI logs.",
    ),
)

add(
    "You are a pytest expert.",
    "What is the difference between function, class, module, and session scoped fixtures?",
    "Scope controls how often the fixture is constructed and torn down across the test run.\n\n```python\nimport pytest\n\n@pytest.fixture(scope=\"function\")  # default\ndef per_test():\n    print(\"new per test\")\n    yield object()\n\n@pytest.fixture(scope=\"class\")\ndef per_class():\n    yield object()\n\n@pytest.fixture(scope=\"module\")\ndef per_module():\n    yield object()\n\n@pytest.fixture(scope=\"session\")\ndef per_session():\n    yield object()\n```\n\nRules of thumb: `function` for anything mutable (the default, safest). `class` when grouping related tests around shared state. `module` for expensive per-file setup like loading a fixture dataset. `session` for truly global, read-only resources (a compiled wheel, a Docker container, a warm HTTP client). Higher-scoped fixtures cannot depend on lower-scoped ones. If you need a session DB with per-test isolation, wrap each test in a transaction that rolls back on teardown.",
)

add(
    "You are a pytest expert.",
    "How do I share fixtures across multiple test files without importing them?",
    "Put them in a `conftest.py` at the appropriate directory level. pytest autoloads it.\n\n```python\n# tests/conftest.py\nimport pytest\nfrom myapp import create_app\n\n@pytest.fixture\ndef app():\n    app = create_app(testing=True)\n    yield app\n    app.cleanup()\n\n@pytest.fixture\ndef client(app):\n    return app.test_client()\n```\n\nNow any test under `tests/` can request `client` without importing anything:\n\n```python\n# tests/test_routes.py\ndef test_index(client):\n    response = client.get(\"/\")\n    assert response.status_code == 200\n```\n\n`conftest.py` can be nested: a `tests/api/conftest.py` adds fixtures only for the api subtree, overriding ones from the parent if names collide. This keeps test directories self-contained and removes the need for pytest-plugin-style imports.",
)

add(
    "You are a pytest testing expert.",
    "Show me a factory fixture pattern for creating model instances on demand.",
    "Return a function from the fixture so tests can call it with overrides.\n\n```python\nimport pytest\nfrom dataclasses import dataclass, field\n\n@dataclass\nclass User:\n    id: int\n    email: str\n    roles: list = field(default_factory=list)\n\n@pytest.fixture\ndef user_factory():\n    counter = {\"n\": 0}\n    def _make(**overrides):\n        counter[\"n\"] += 1\n        defaults = {\"id\": counter[\"n\"], \"email\": f\"u{counter['n']}@ex.com\", \"roles\": []}\n        defaults.update(overrides)\n        return User(**defaults)\n    return _make\n\ndef test_admin(user_factory):\n    admin = user_factory(roles=[\"admin\"])\n    regular = user_factory()\n    assert admin.roles == [\"admin\"]\n    assert regular.id != admin.id\n```\n\nThe factory fixture pattern avoids the combinatorial explosion of having one fixture per flavour of object. A test can create as many users as needed with just the fields that matter, and defaults stay in one place. For more structured needs, consider `factory-boy` or `polyfactory`, but a plain factory fixture covers 80% of cases.",
)

add(
    "You are a pytest testing specialist.",
    "How do I use pytest markers to skip tests conditionally based on Python version?",
    "Use `pytest.mark.skipif` with a runtime condition.\n\n```python\nimport sys\nimport pytest\n\n@pytest.mark.skipif(sys.version_info < (3, 11), reason=\"requires ExceptionGroup\")\ndef test_exception_group():\n    with pytest.raises(ExceptionGroup):\n        raise ExceptionGroup(\"grp\", [ValueError(\"x\")])\n\n@pytest.mark.skipif(sys.platform != \"linux\", reason=\"linux-only syscall\")\ndef test_fcntl_flock():\n    import fcntl\n    # ...\n```\n\nFor conditional xfail (test runs but failure is expected):\n\n```python\n@pytest.mark.xfail(sys.platform == \"win32\", reason=\"flaky on windows\", strict=False)\ndef test_timing_sensitive():\n    ...\n```\n\nRegister custom markers in `pyproject.toml` to avoid warnings:\n\n```toml\n[tool.pytest.ini_options]\nmarkers = [\n    \"slow: slow tests, deselect with '-m \\\"not slow\\\"'\",\n    \"integration: needs external services\",\n]\n```\n\nThen `pytest -m \"not slow\"` to skip, `pytest -m integration` to run only integration tests.",
)

add(
    "You are a pytest expert.",
    "Write a fixture that captures stdout and stderr for testing CLI output.",
    "Use the built-in `capsys` fixture. It intercepts file descriptors 1 and 2.\n\n```python\nimport sys\n\ndef greet(name):\n    print(f\"hello {name}\")\n    print(f\"warning: unknown user\", file=sys.stderr)\n\ndef test_greet_output(capsys):\n    greet(\"world\")\n    captured = capsys.readouterr()\n    assert captured.out == \"hello world\\n\"\n    assert \"warning\" in captured.err\n```\n\nFor code that writes directly to fd 1/2 (subprocess, C extensions), use `capfd` instead:\n\n```python\ndef test_subprocess_output(capfd):\n    import subprocess\n    subprocess.run([\"echo\", \"hello\"])\n    out, err = capfd.readouterr()\n    assert out == \"hello\\n\"\n```\n\nThere are also binary variants: `capsysbinary` and `capfdbinary` return bytes. Note that `readouterr()` consumes the buffer, so a second call returns empty strings unless new output has been written.",
)

add(
    "You are a pytest expert.",
    "How do I properly test code that reads environment variables?",
    "Use `monkeypatch.setenv` / `delenv` so the change auto-reverts at test end.\n\n```python\nimport os\n\ndef get_timeout():\n    return int(os.environ.get(\"APP_TIMEOUT\", \"30\"))\n\ndef test_default_timeout(monkeypatch):\n    monkeypatch.delenv(\"APP_TIMEOUT\", raising=False)\n    assert get_timeout() == 30\n\ndef test_custom_timeout(monkeypatch):\n    monkeypatch.setenv(\"APP_TIMEOUT\", \"120\")\n    assert get_timeout() == 120\n\ndef test_invalid_timeout(monkeypatch):\n    monkeypatch.setenv(\"APP_TIMEOUT\", \"not-a-number\")\n    import pytest\n    with pytest.raises(ValueError):\n        get_timeout()\n```\n\nThe `monkeypatch` fixture is function-scoped and restores all patches automatically. Never mutate `os.environ` directly in a test; leaked env state causes unpredictable failures in later tests. `monkeypatch` also has `setattr`, `delattr`, `setitem`, and `chdir` for the same safe-teardown pattern on attributes, dicts, and cwd.",
)

add(
    "You are a pytest fixtures expert.",
    "Demonstrate fixture composition where one fixture depends on another.",
    "Just request the dependency by name. pytest resolves the graph.\n\n```python\nimport pytest\n\n@pytest.fixture\ndef config():\n    return {\"host\": \"localhost\", \"port\": 5432}\n\n@pytest.fixture\ndef db_url(config):\n    return f\"postgres://{config['host']}:{config['port']}/test\"\n\n@pytest.fixture\ndef engine(db_url):\n    # imagine sqlalchemy.create_engine(db_url)\n    return {\"url\": db_url, \"connected\": True}\n\n@pytest.fixture\ndef session(engine):\n    sess = {\"engine\": engine, \"tx\": []}\n    yield sess\n    sess[\"tx\"].clear()\n\ndef test_session_wired(session):\n    assert session[\"engine\"][\"url\"] == \"postgres://localhost:5432/test\"\n```\n\npytest builds a DAG of fixtures and creates each one at most once per its scope during a test. A test that asks for `session` gets `config` and `engine` constructed transitively. This composition lets you keep each fixture single-purpose and remix them per test, which is far cleaner than a single monolithic `setup` helper.",
)

add(
    "You are a pytest expert.",
    "How do I run only the tests that failed last time?",
    "Use `--lf` (last-failed) or `--ff` (failed-first).\n\n```bash\n# run only the tests that failed in the previous run\npytest --lf\n\n# run failed tests first, then the rest\npytest --ff\n\n# if no test failed last time, --lf falls back to all; change that:\npytest --lf --last-failed-no-failures=none  # run nothing if none failed\n```\n\nCombine with other filters for a tight iteration loop:\n\n```bash\npytest --lf -x  # stop at first failure again\npytest --lf --pdb  # drop into debugger on failure\n```\n\nRelated cache-based flags: `--nf` runs new tests first (useful for TDD), `--sw` (stepwise) stops at first failure and resumes from there next run, automatically ignoring passed tests until you reach the failure again. Configure cache location with `cache_dir` in `pyproject.toml`:\n\n```toml\n[tool.pytest.ini_options]\ncache_dir = \".pytest_cache\"\n```",
)

add(
    "You are a pytest expert.",
    "Show me how to write a fixture that sets up and tears down a Flask test client.",
    "Create the app, push an application context, yield the client, then pop the context.\n\n```python\nimport pytest\nfrom myapp import create_app, db\n\n@pytest.fixture\ndef app():\n    app = create_app({\"TESTING\": True, \"SQLALCHEMY_DATABASE_URI\": \"sqlite:///:memory:\"})\n    with app.app_context():\n        db.create_all()\n        yield app\n        db.session.remove()\n        db.drop_all()\n\n@pytest.fixture\ndef client(app):\n    return app.test_client()\n\n@pytest.fixture\ndef runner(app):\n    return app.test_cli_runner()\n\ndef test_homepage(client):\n    rv = client.get(\"/\")\n    assert rv.status_code == 200\n    assert b\"Welcome\" in rv.data\n\ndef test_create_cli(runner):\n    result = runner.invoke(args=[\"init-db\"])\n    assert \"Initialized\" in result.output\n```\n\nThe `app_context()` block ensures Flask's `current_app` proxy and SQLAlchemy session work. `test_client()` gives an HTTP-level interface that does not require a running server. `test_cli_runner()` exercises registered Click commands.",
)

add(
    "You are a pytest expert.",
    "What is the autouse fixture and when should I use it?",
    "`autouse=True` makes a fixture run for every test in its scope without being requested explicitly.\n\n```python\nimport pytest\nimport logging\n\n@pytest.fixture(autouse=True)\ndef reset_logging():\n    # runs before every test\n    for h in logging.root.handlers[:]:\n        logging.root.removeHandler(h)\n    logging.basicConfig(level=logging.WARNING)\n    yield\n    # cleanup runs after every test\n\n@pytest.fixture(autouse=True, scope=\"session\")\ndef set_timezone(monkeypatch_session):\n    import os\n    os.environ[\"TZ\"] = \"UTC\"\n```\n\nUse it sparingly. Good cases: resetting global singletons, forcing a known timezone, clearing module-level caches. Bad cases: anything that should be opt-in, because autouse makes the test's dependencies invisible. If every test in a file needs a setup step, prefer autouse in a conftest at that level. If only some tests need it, keep it explicit, because implicit setup is a readability trap.",
)

add(
    "You are a pytest expert.",
    "Show me how to use indirect parametrization to parametrize a fixture.",
    "Mark the parameter as `indirect` so pytest passes values through the fixture instead of directly.\n\n```python\nimport pytest\n\n@pytest.fixture\ndef client(request):\n    backend = request.param\n    if backend == \"sqlite\":\n        return {\"engine\": \"sqlite\", \"url\": \":memory:\"}\n    elif backend == \"postgres\":\n        return {\"engine\": \"postgres\", \"url\": \"postgres://localhost/test\"}\n    raise ValueError(backend)\n\n@pytest.mark.parametrize(\"client\", [\"sqlite\", \"postgres\"], indirect=True)\ndef test_connect(client):\n    assert \"url\" in client\n```\n\nThe fixture sees each parameter in `request.param`. The test receives the fully constructed object. This is the clean way to run the same test suite against multiple real implementations. You can combine direct and indirect:\n\n```python\n@pytest.mark.parametrize(\"client, input, expected\", [\n    (\"sqlite\", 1, 1), (\"postgres\", 1, 1),\n], indirect=[\"client\"])\ndef test_echo(client, input, expected):\n    assert input == expected\n```\n\nOnly `client` is indirect; `input` and `expected` are passed directly.",
)

add(
    "You are a pytest expert.",
    "How do I write a fixture that rolls back database changes after each test?",
    "Wrap the test in a SAVEPOINT and roll back on teardown.\n\n```python\nimport pytest\nfrom sqlalchemy import create_engine\nfrom sqlalchemy.orm import sessionmaker\n\n@pytest.fixture(scope=\"session\")\ndef engine():\n    return create_engine(\"postgresql://localhost/test\")\n\n@pytest.fixture\ndef db_session(engine):\n    connection = engine.connect()\n    transaction = connection.begin()\n    Session = sessionmaker(bind=connection)\n    session = Session()\n    # nested savepoint lets the test code COMMIT without actually committing\n    session.begin_nested()\n    yield session\n    session.close()\n    transaction.rollback()\n    connection.close()\n\ndef test_user_create(db_session):\n    db_session.execute(\"INSERT INTO users (name) VALUES ('alice')\")\n    db_session.commit()  # savepoint release, outer tx still rollbackable\n    rows = db_session.execute(\"SELECT COUNT(*) FROM users\").scalar()\n    assert rows >= 1\n```\n\nThe outer transaction is never committed, so at test end everything the test did vanishes. The nested `begin_nested()` emits a SAVEPOINT so code under test can call `commit()` without ending the outer transaction. This pattern gives you session-scoped schema setup with function-scoped data isolation.",
)

add(
    "You are a pytest expert.",
    "How do I add a pytest fixture that generates random but reproducible test data?",
    "Seed with `random.Random` locally; accept an override to trace flakiness.\n\n```python\nimport pytest\nimport random\n\n@pytest.fixture\ndef rng(request):\n    seed = request.config.getoption(\"--seed\", default=42)\n    r = random.Random(seed)\n    yield r\n\ndef pytest_addoption(parser):\n    parser.addoption(\"--seed\", type=int, default=42)\n\ndef test_shuffle(rng):\n    xs = [1, 2, 3, 4, 5]\n    rng.shuffle(xs)\n    assert sorted(xs) == [1, 2, 3, 4, 5]\n```\n\nPut `pytest_addoption` in `conftest.py`. Now every run uses seed 42; when a test fails on CI, rerun with the logged seed to reproduce. Crucially, `random.Random` is a *local* instance, not the global `random` module, so a test cannot accidentally leak state into another. For numpy, use `numpy.random.default_rng(seed)`. Never use `random.seed()` globally in tests because one test can corrupt another's data.",
)

add(
    "You are a pytest expert.",
    "How do I mark a test as slow and exclude it by default?",
    "Register the marker, add a CLI flag, and skip unless enabled.\n\n```python\n# conftest.py\nimport pytest\n\ndef pytest_addoption(parser):\n    parser.addoption(\"--runslow\", action=\"store_true\", default=False,\n                     help=\"run tests marked @pytest.mark.slow\")\n\ndef pytest_configure(config):\n    config.addinivalue_line(\"markers\", \"slow: mark test as slow to run\")\n\ndef pytest_collection_modifyitems(config, items):\n    if config.getoption(\"--runslow\"):\n        return\n    skip_slow = pytest.mark.skip(reason=\"need --runslow to run\")\n    for item in items:\n        if \"slow\" in item.keywords:\n            item.add_marker(skip_slow)\n```\n\nUsage:\n\n```python\n@pytest.mark.slow\ndef test_expensive_integration():\n    ...\n```\n\nDefault `pytest` skips it. `pytest --runslow` runs it. Alternative one-liner without custom flag: use the marker system directly with `pytest -m \"not slow\"` as your default CI invocation. Both patterns are valid; the opt-in flag is more explicit for newcomers on the team.",
)

add(
    "You are a pytest expert.",
    "Show me the pytest tmp_path_factory fixture for session-scoped temp directories.",
    "`tmp_path_factory` is the session-scoped counterpart to `tmp_path`.\n\n```python\nimport pytest\nimport json\n\n@pytest.fixture(scope=\"session\")\ndef shared_fixture_dir(tmp_path_factory):\n    d = tmp_path_factory.mktemp(\"data\")\n    (d / \"config.json\").write_text(json.dumps({\"env\": \"test\"}))\n    (d / \"payload.bin\").write_bytes(b\"\\x00\" * 1024)\n    return d\n\ndef test_reads_config(shared_fixture_dir):\n    cfg = json.loads((shared_fixture_dir / \"config.json\").read_text())\n    assert cfg[\"env\"] == \"test\"\n\ndef test_reads_payload(shared_fixture_dir):\n    data = (shared_fixture_dir / \"payload.bin\").read_bytes()\n    assert len(data) == 1024\n```\n\nBoth tests reuse the same directory created once per session. pytest manages the base dir under `/tmp/pytest-of-USER/` and keeps the last 3 sessions for debugging. Use this for expensive setup like downloading a fixture tarball, unpacking a big archive, or compiling a binary once. For per-test isolation, stick with `tmp_path`.",
)

add(
    "You are a pytest expert.",
    "How do I test code that uses datetime.now() in a deterministic way?",
    "Use `freezegun` or inject a clock dependency.\n\n```python\nfrom datetime import datetime, timezone\nfrom freezegun import freeze_time\n\ndef timestamp_log(message):\n    return f\"[{datetime.now(timezone.utc).isoformat()}] {message}\"\n\n@freeze_time(\"2026-01-15T10:00:00Z\")\ndef test_timestamp_log():\n    assert timestamp_log(\"boot\") == \"[2026-01-15T10:00:00+00:00] boot\"\n\n@freeze_time(\"2026-01-15\")\ndef test_time_travel():\n    from freezegun import freeze_time as ft\n    with ft(\"2027-06-01\"):\n        assert datetime.now().year == 2027\n    assert datetime.now().year == 2026\n```\n\nInjected-clock alternative, no dependency:\n\n```python\nclass Service:\n    def __init__(self, clock=datetime.utcnow):\n        self._clock = clock\n    def now(self):\n        return self._clock()\n\ndef test_service_clock():\n    svc = Service(clock=lambda: datetime(2026, 1, 15))\n    assert svc.now().year == 2026\n```\n\nDependency injection is cleaner long-term, but `freezegun` is great for existing code you cannot refactor.",
)

add(
    "You are a pytest expert.",
    "How do I organize a large test suite with subdirectories and shared fixtures?",
    "Use nested conftest files and a clear directory layout.\n\n```\nproject/\n  src/\n    myapp/\n      __init__.py\n      users.py\n      orders.py\n  tests/\n    conftest.py          # global fixtures (db, app, http client)\n    unit/\n      conftest.py        # unit-specific (fake clock, in-memory stores)\n      test_users.py\n      test_orders.py\n    integration/\n      conftest.py        # real db, docker compose setup\n      test_user_flow.py\n    e2e/\n      conftest.py        # browser, API keys\n      test_checkout.py\n  pyproject.toml\n```\n\n```toml\n# pyproject.toml\n[tool.pytest.ini_options]\ntestpaths = [\"tests\"]\naddopts = \"-ra --strict-markers --strict-config\"\nmarkers = [\n    \"unit: fast in-process tests\",\n    \"integration: needs a database\",\n    \"e2e: full browser/api end-to-end\",\n]\n```\n\nRun slices independently: `pytest tests/unit`, `pytest tests/integration`, or by marker `pytest -m unit`. Each conftest only applies to its subtree, so unit tests do not pay the cost of integration-level fixtures. Keep a one-to-one structure with your source tree when possible.",
)

add(
    "You are a pytest fixture expert.",
    "Write a fixture that spins up and tears down a real Redis via docker for integration tests.",
    "Use `docker` library or subprocess; expose host/port as a fixture.\n\n```python\nimport subprocess\nimport time\nimport socket\nimport pytest\nimport redis\n\ndef _wait_port(host, port, timeout=20):\n    start = time.time()\n    while time.time() - start < timeout:\n        try:\n            with socket.create_connection((host, port), timeout=1):\n                return\n        except OSError:\n            time.sleep(0.25)\n    raise TimeoutError(f\"{host}:{port} never came up\")\n\n@pytest.fixture(scope=\"session\")\ndef redis_server():\n    name = \"pytest-redis-\" + str(int(time.time()))\n    subprocess.check_call([\n        \"docker\", \"run\", \"-d\", \"--rm\", \"--name\", name,\n        \"-p\", \"6390:6379\", \"redis:7-alpine\"\n    ])\n    try:\n        _wait_port(\"127.0.0.1\", 6390)\n        yield (\"127.0.0.1\", 6390)\n    finally:\n        subprocess.call([\"docker\", \"rm\", \"-f\", name])\n\n@pytest.fixture\ndef redis_client(redis_server):\n    host, port = redis_server\n    client = redis.Redis(host=host, port=port, decode_responses=True)\n    client.flushall()\n    return client\n\ndef test_set_get(redis_client):\n    redis_client.set(\"k\", \"v\")\n    assert redis_client.get(\"k\") == \"v\"\n```\n\nSession-scope the container; function-scope the client with flushall for isolation. Prefer the `testcontainers` library in real projects, it handles random ports and cleanup for you.",
)


# Write progress checkpoint and continue
print(f"Block 1 complete: {len(samples)} samples")
Path("/tmp/gen_batch_code_04_partial.json").write_text(json.dumps(samples))
