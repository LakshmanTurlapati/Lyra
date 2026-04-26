#!/usr/bin/env python3
"""Block 2: Mocking (items 21-40)."""
import json
from pathlib import Path

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


add(
    "You are a Python mocking expert.",
    "How do I mock a function in a module I do not own?",
    "Use `unittest.mock.patch` with the path where the function is *looked up*, not where it is defined.\n\n```python\n# myapp/weather.py\nimport requests\n\ndef current_temp(city):\n    r = requests.get(f\"https://api.weather.com/{city}\")\n    return r.json()[\"temp_c\"]\n```\n\n```python\n# test_weather.py\nfrom unittest.mock import patch, MagicMock\nfrom myapp.weather import current_temp\n\n@patch(\"myapp.weather.requests.get\")\ndef test_current_temp(mock_get):\n    mock_get.return_value = MagicMock(json=lambda: {\"temp_c\": 22.5})\n    assert current_temp(\"sf\") == 22.5\n    mock_get.assert_called_once_with(\"https://api.weather.com/sf\")\n```\n\nThe key rule: patch `myapp.weather.requests.get`, not `requests.get`. `myapp.weather` imported `requests` and now holds its own reference. Patching the original `requests` module leaves that reference untouched, and the test will silently hit the real API. This is the most common `patch` mistake.",
)

add(
    "You are a mocking expert.",
    "Show me MagicMock vs Mock vs AsyncMock differences.",
    "Three variants in `unittest.mock` for three call patterns.\n\n```python\nfrom unittest.mock import Mock, MagicMock, AsyncMock\nimport asyncio\n\n# Mock: bare, raises AttributeError for magic methods like __len__\nm = Mock()\nm.foo.return_value = 1\nassert m.foo() == 1\n# len(m)  # would fail: Mock has no __len__\n\n# MagicMock: Mock + magic method support\nmm = MagicMock()\nmm.__len__.return_value = 5\nassert len(mm) == 5\nfor _ in mm: break  # __iter__ works\n\n# AsyncMock: for coroutines; __call__ returns a coroutine\nasync_m = AsyncMock(return_value=42)\nasync def use():\n    return await async_m()\nassert asyncio.run(use()) == 42\n```\n\nRule of thumb: `MagicMock` is the default choice (and is what `@patch` creates unless you specify otherwise). Use `AsyncMock` when patching an `async def` or any awaitable. Use plain `Mock` when you want strict behavior that fails loudly on unexpected magic-method access.",
)

add(
    "You are a mocking expert.",
    "How do I mock requests.get to return different responses for different URLs?",
    "Assign a `side_effect` callable instead of a fixed `return_value`.\n\n```python\nfrom unittest.mock import patch, MagicMock\n\ndef fake_get(url, *args, **kwargs):\n    if url.endswith(\"/users\"):\n        return MagicMock(status_code=200, json=lambda: [{\"id\": 1}])\n    if url.endswith(\"/health\"):\n        return MagicMock(status_code=200, json=lambda: {\"ok\": True})\n    return MagicMock(status_code=404, json=lambda: {\"error\": \"not found\"})\n\n@patch(\"myapp.client.requests.get\", side_effect=fake_get)\ndef test_multi(mock_get):\n    from myapp.client import fetch\n    assert fetch(\"users\")[0][\"id\"] == 1\n    assert fetch(\"health\")[\"ok\"] is True\n    assert fetch(\"unknown\") is None\n    assert mock_get.call_count == 3\n```\n\nFor simpler cases with a fixed sequence, `side_effect` also accepts an iterable:\n\n```python\nmock_get.side_effect = [resp_a, resp_b, ConnectionError(\"boom\")]\n```\n\nEach call pops the next value; if it is an exception class or instance, it is raised. In real projects, prefer the `responses` or `respx` libraries, which hook at the transport layer and let you register URL patterns declaratively.",
)

add(
    "You are a mocking expert.",
    "Mock an environment variable for a single test.",
    "Use `unittest.mock.patch.dict` or pytest's `monkeypatch`.\n\n```python\nimport os\nfrom unittest.mock import patch\n\ndef get_api_key():\n    return os.environ[\"API_KEY\"]\n\ndef test_key_from_env():\n    with patch.dict(os.environ, {\"API_KEY\": \"secret\"}, clear=False):\n        assert get_api_key() == \"secret\"\n    # os.environ is restored here\n\n# pytest style\ndef test_key_from_env_pytest(monkeypatch):\n    monkeypatch.setenv(\"API_KEY\", \"secret\")\n    assert get_api_key() == \"secret\"\n\n# clear existing env for isolation\ndef test_no_key(monkeypatch):\n    monkeypatch.delenv(\"API_KEY\", raising=False)\n    import pytest\n    with pytest.raises(KeyError):\n        get_api_key()\n```\n\n`clear=True` on `patch.dict` wipes the dict before applying the patch, then restores it, which is the safest choice when you want a fully controlled env. Both approaches restore state on exit, so they are safe with parallel test runners like `pytest-xdist`.",
)

add(
    "You are a mocking expert.",
    "How do I spy on a method call without replacing its behavior?",
    "Use `patch.object(..., wraps=real_object.method)`.\n\n```python\nfrom unittest.mock import patch\n\nclass Cache:\n    def get(self, key):\n        return key.upper()\n\ndef test_spy_on_get():\n    c = Cache()\n    with patch.object(c, \"get\", wraps=c.get) as spy:\n        assert c.get(\"hi\") == \"HI\"\n        assert c.get(\"world\") == \"WORLD\"\n        assert spy.call_count == 2\n        spy.assert_any_call(\"hi\")\n        spy.assert_any_call(\"world\")\n```\n\n`wraps=` causes the mock to delegate to the real callable while still recording calls, arguments, and return values. This is the right tool when you want to observe an interaction without changing behavior, for example to verify a caching layer actually calls through only on cache miss. If you use `return_value` or `side_effect`, they take precedence over `wraps`.",
)

add(
    "You are a mocking expert.",
    "Show me how to mock a context manager like a database connection.",
    "Set `return_value.__enter__.return_value` on a MagicMock.\n\n```python\nfrom unittest.mock import patch, MagicMock\n\ndef read_rows():\n    import sqlite3\n    with sqlite3.connect(\"data.db\") as conn:\n        cur = conn.execute(\"SELECT id, name FROM users\")\n        return cur.fetchall()\n\n@patch(\"myapp.db.sqlite3.connect\")\ndef test_read_rows(mock_connect):\n    mock_conn = MagicMock()\n    mock_cur = MagicMock()\n    mock_cur.fetchall.return_value = [(1, \"alice\"), (2, \"bob\")]\n    mock_conn.execute.return_value = mock_cur\n    # the crucial line for `with` support:\n    mock_connect.return_value.__enter__.return_value = mock_conn\n    mock_connect.return_value.__exit__.return_value = False\n\n    assert read_rows() == [(1, \"alice\"), (2, \"bob\")]\n    mock_conn.execute.assert_called_once_with(\"SELECT id, name FROM users\")\n```\n\nBecause we use `MagicMock`, `__enter__` and `__exit__` exist out of the box; you only need to wire `__enter__`'s return value. If you used plain `Mock`, you would get an `AttributeError` on `__enter__` at runtime.",
)

add(
    "You are a mocking expert.",
    "How do I assert that a mock was NOT called?",
    "Use `assert_not_called` or inspect `call_count`.\n\n```python\nfrom unittest.mock import MagicMock\n\ndef ship(order, notify):\n    if order.is_gift:\n        notify(order.recipient)\n    return \"shipped\"\n\ndef test_no_notify_when_not_gift():\n    class O: is_gift = False; recipient = None\n    notify = MagicMock()\n    ship(O, notify)\n    notify.assert_not_called()\n\ndef test_notify_only_once():\n    class O: is_gift = True; recipient = \"alice\"\n    notify = MagicMock()\n    ship(O, notify)\n    notify.assert_called_once_with(\"alice\")\n    assert notify.call_count == 1\n```\n\nRelated assertions:\n- `assert_called()` — at least once\n- `assert_called_once()` — exactly one call, args ignored\n- `assert_called_with(...)` — last call matched these args\n- `assert_called_once_with(...)` — exactly one call with these args\n- `assert_any_call(...)` — at some point called with these args\n- `assert_has_calls([...], any_order=False)` — a subsequence of calls\n\nPrefer the most specific assertion so a regression that adds extra calls or changes ordering fails loudly.",
)

add(
    "You are a mocking expert.",
    "Mock time.sleep so tests run instantly.",
    "Patch the reference used by the code under test.\n\n```python\n# myapp/retry.py\nimport time\n\ndef retry(fn, attempts=3, backoff=1.0):\n    for i in range(attempts):\n        try:\n            return fn()\n        except Exception:\n            if i == attempts - 1:\n                raise\n            time.sleep(backoff * (2 ** i))\n```\n\n```python\n# test_retry.py\nfrom unittest.mock import patch, MagicMock\nimport pytest\nfrom myapp.retry import retry\n\n@patch(\"myapp.retry.time.sleep\")\ndef test_retry_exponential(mock_sleep):\n    fn = MagicMock(side_effect=[RuntimeError, RuntimeError, \"ok\"])\n    assert retry(fn, attempts=3, backoff=0.5) == \"ok\"\n    waits = [c.args[0] for c in mock_sleep.call_args_list]\n    assert waits == [0.5, 1.0]\n    assert fn.call_count == 3\n```\n\nWe not only assert the function eventually returned, but also that the backoff schedule was correct. Patching `myapp.retry.time.sleep` (not `time.sleep`) ensures we only affect this module, keeping the test targeted.",
)

add(
    "You are a mocking expert.",
    "How do I use autospec to catch signature mismatches?",
    "Pass `autospec=True` to `patch` so the mock enforces the real object's signature.\n\n```python\n# myapp/email.py\ndef send(to, subject, body, cc=None):\n    ...\n```\n\n```python\nfrom unittest.mock import patch\nfrom myapp import email\n\n@patch(\"myapp.email.send\", autospec=True)\ndef test_send_called_correctly(mock_send):\n    email.send(\"a@x\", \"hi\", \"body\")\n    mock_send.assert_called_once_with(\"a@x\", \"hi\", \"body\")\n\n@patch(\"myapp.email.send\", autospec=True)\ndef test_detects_wrong_signature(mock_send):\n    import pytest\n    with pytest.raises(TypeError):\n        email.send(\"only-one-arg\")  # autospec raises, no autospec would pass silently\n```\n\nWithout `autospec`, a `MagicMock` accepts any arguments. When the real function's signature changes and the test still calls the old one, the test passes but production breaks. With `autospec`, the test fails immediately. Use `create_autospec(real_obj)` to build a standalone autospecced mock. The small price: spec-backed mocks are slightly slower to create.",
)

add(
    "You are a mocking expert.",
    "Show the responses library for mocking HTTP requests.",
    "`responses` registers URL patterns and asserts call order.\n\n```python\nimport requests\nimport responses\n\n@responses.activate\ndef test_fetch_user():\n    responses.add(\n        responses.GET,\n        \"https://api.example.com/users/42\",\n        json={\"id\": 42, \"name\": \"alice\"},\n        status=200,\n    )\n    responses.add(\n        responses.POST,\n        \"https://api.example.com/users\",\n        json={\"id\": 43},\n        status=201,\n    )\n\n    r = requests.get(\"https://api.example.com/users/42\")\n    assert r.json()[\"name\"] == \"alice\"\n\n    r = requests.post(\"https://api.example.com/users\", json={\"name\": \"bob\"})\n    assert r.status_code == 201\n\n    assert len(responses.calls) == 2\n    assert responses.calls[0].request.url.endswith(\"/users/42\")\n```\n\nDynamic responses via callbacks:\n\n```python\ndef cb(request):\n    body = json.loads(request.body)\n    return (201, {}, json.dumps({\"echo\": body}))\n\nresponses.add_callback(responses.POST, \"https://api.example.com/echo\", callback=cb)\n```\n\nFor httpx use `respx`; for aiohttp use `aioresponses`. All three beat hand-rolled `patch(requests.get)` for real-world API tests.",
)

add(
    "You are a mocking expert.",
    "How do I patch a classmethod or staticmethod?",
    "`patch.object` handles both; target the class and the method name.\n\n```python\nclass Clock:\n    @classmethod\n    def now(cls):\n        import datetime\n        return datetime.datetime.utcnow()\n\n    @staticmethod\n    def timezone():\n        return \"UTC\"\n\nfrom unittest.mock import patch\nimport datetime\n\ndef test_clock_now():\n    with patch.object(Clock, \"now\", return_value=datetime.datetime(2026, 1, 1)):\n        assert Clock.now().year == 2026\n\ndef test_clock_tz():\n    with patch.object(Clock, \"timezone\", return_value=\"America/Los_Angeles\"):\n        assert Clock.timezone() == \"America/Los_Angeles\"\n```\n\nFor instance methods, patching the class replaces the method on all instances for the duration of the patch. For an already-instantiated object's method, `patch.object(instance, \"method\")` swaps it just on that instance. Remember `autospec=True` if you want signature checking; classmethods are tricky because `cls` is injected, so autospec is worth it to avoid shape mismatches.",
)

add(
    "You are a mocking expert.",
    "How do I use pytest-mock's mocker fixture?",
    "`pytest-mock` exposes `mocker`, a thin wrapper over `unittest.mock` with automatic teardown.\n\n```python\n# install: pip install pytest-mock\n\ndef test_fetch(mocker):\n    mock_get = mocker.patch(\"myapp.weather.requests.get\")\n    mock_get.return_value.json.return_value = {\"temp_c\": 12}\n\n    from myapp.weather import current_temp\n    assert current_temp(\"paris\") == 12\n    mock_get.assert_called_once()\n\ndef test_patch_object(mocker):\n    import myapp.weather\n    mocker.patch.object(myapp.weather, \"current_temp\", return_value=99)\n    assert myapp.weather.current_temp(\"anywhere\") == 99\n\ndef test_spy(mocker):\n    from myapp import weather\n    spy = mocker.spy(weather, \"current_temp\")\n    # code under test calls current_temp...\n```\n\nThe benefit over raw `unittest.mock.patch`: no context manager or decorator, patches unwind at end of test automatically, and you get `mocker.spy`, `mocker.stub`, and `mocker.patch.multiple` without extra imports. It is the most common mocking style in pytest codebases today.",
)

add(
    "You are a mocking expert.",
    "Patch multiple things at once in a single test.",
    "Stack decorators bottom-up, use `patch.multiple`, or use `with` blocks.\n\n```python\nfrom unittest.mock import patch, MagicMock\n\n# Stacked decorators: arguments inject in reverse order\n@patch(\"myapp.svc.db.save\")\n@patch(\"myapp.svc.email.send\")\n@patch(\"myapp.svc.metrics.inc\")\ndef test_order_flow(mock_inc, mock_send, mock_save):\n    from myapp.svc import place_order\n    place_order(user_id=1, item=\"book\")\n    mock_save.assert_called_once()\n    mock_send.assert_called_once()\n    mock_inc.assert_called_with(\"orders.placed\")\n\n# patch.multiple: keyword style\n@patch.multiple(\"myapp.svc\",\n                save=MagicMock(),\n                send=MagicMock(),\n                inc=MagicMock())\ndef test_keyword(save, send, inc):\n    from myapp import svc\n    svc.place_order(user_id=1, item=\"book\")\n    save.assert_called_once()\n\n# Context-manager form, clearer in long tests\ndef test_with(mocker):\n    save = mocker.patch(\"myapp.svc.db.save\")\n    send = mocker.patch(\"myapp.svc.email.send\")\n    # ...\n```\n\nWith decorators, the innermost decorator is the first argument. That reversal trips people up; `mocker` avoids the confusion entirely.",
)

add(
    "You are a mocking expert.",
    "How do I verify the exact arguments of a mock call, ignoring ones I do not care about?",
    "Use `unittest.mock.ANY` or `call` with partial matching.\n\n```python\nfrom unittest.mock import MagicMock, ANY, call\n\nlog = MagicMock()\nlog(\"user.login\", user_id=42, ip=\"10.0.0.1\", ts=1700000000)\n\nlog.assert_called_with(\"user.login\", user_id=42, ip=ANY, ts=ANY)\n\n# With multiple calls, build the expected sequence\nlog(\"user.logout\", user_id=42, ts=1700000050)\nlog.assert_has_calls([\n    call(\"user.login\", user_id=42, ip=ANY, ts=ANY),\n    call(\"user.logout\", user_id=42, ts=ANY),\n])\n```\n\n`ANY` matches any value and is the simplest way to ignore volatile args like timestamps or generated IDs. For more structural checks, inspect `call_args`:\n\n```python\nargs, kwargs = log.call_args\nassert kwargs[\"user_id\"] == 42\nassert isinstance(kwargs[\"ts\"], int)\n```\n\nThis is more flexible when you need type assertions or numeric ranges instead of equality.",
)

add(
    "You are a mocking expert.",
    "Mock a class so every instance is the same mock.",
    "`patch` replaces the class reference; its `return_value` is the instance mock.\n\n```python\nfrom unittest.mock import patch, MagicMock\n\n# myapp/email.py\n# class SMTPClient:\n#     def __init__(self, host): ...\n#     def send(self, to, body): ...\n\n@patch(\"myapp.email.SMTPClient\")\ndef test_uses_smtp(mock_smtp_cls):\n    instance = mock_smtp_cls.return_value  # every SMTPClient(...) returns this\n    instance.send.return_value = True\n\n    from myapp.email import notify\n    assert notify(\"a@x\", \"hi\") is True\n\n    mock_smtp_cls.assert_called_once_with(\"smtp.example.com\")\n    instance.send.assert_called_once_with(\"a@x\", \"hi\")\n```\n\nIf the code constructs multiple different instances and you need them to be distinct mocks, set `mock_smtp_cls.side_effect = [MagicMock(), MagicMock()]` so each call to `SMTPClient(...)` returns a fresh mock. `side_effect` overrides `return_value` for each call.",
)

add(
    "You are a mocking expert.",
    "How do I mock open() when testing code that reads a file?",
    "Use `unittest.mock.mock_open` which provides a ready-made fake for `open`.\n\n```python\nfrom unittest.mock import mock_open, patch\n\ndef load_config(path):\n    with open(path) as f:\n        return f.read().strip()\n\ndef test_load_config():\n    fake = mock_open(read_data=\"host=localhost\\nport=5432\")\n    with patch(\"builtins.open\", fake):\n        result = load_config(\"/etc/app.conf\")\n    assert result == \"host=localhost\\nport=5432\"\n    fake.assert_called_once_with(\"/etc/app.conf\")\n\ndef test_iterate_lines():\n    fake = mock_open(read_data=\"a\\nb\\nc\\n\")\n    with patch(\"builtins.open\", fake):\n        with open(\"f\") as f:\n            lines = list(f)  # iteration supported in py3.8+\n    assert lines == [\"a\\n\", \"b\\n\", \"c\\n\"]\n```\n\nFor anything non-trivial (multiple files, binary reads, seek), prefer `tmp_path` plus real files. `mock_open` shines for unit tests where you want to assert `open` was called with the right path and mode without touching disk.",
)

add(
    "You are a mocking expert.",
    "Mock an async function that is awaited.",
    "Use `AsyncMock`, either directly or via `new_callable=AsyncMock` in patch.\n\n```python\nimport asyncio\nfrom unittest.mock import AsyncMock, patch\n\n# myapp/client.py\n# async def fetch_user(session, id):\n#     async with session.get(f\"/users/{id}\") as r:\n#         return await r.json()\n\nasync def run():\n    from myapp import client\n    from aiohttp import ClientSession\n    async with ClientSession() as s:\n        return await client.fetch_user(s, 1)\n\n@patch(\"myapp.client.fetch_user\", new_callable=AsyncMock)\ndef test_fetch(mock_fetch):\n    mock_fetch.return_value = {\"id\": 1, \"name\": \"alice\"}\n    import asyncio\n    result = asyncio.run(run_wrapper())  # wrapper awaits fetch_user\n    assert result[\"name\"] == \"alice\"\n    mock_fetch.assert_awaited_once()\n```\n\nAsyncMock distinguishes `assert_called`/`assert_awaited`: the former checks the coroutine was *returned*, the latter checks it was *awaited*. Use the awaited variants (`assert_awaited`, `assert_awaited_once_with`, `await_count`) because a forgotten `await` is a common async bug you want your tests to catch.",
)

add(
    "You are a mocking expert.",
    "How do I use side_effect to simulate a flaky service that succeeds on retry?",
    "Provide a list of values/exceptions; each call consumes the next.\n\n```python\nfrom unittest.mock import MagicMock, patch\nimport requests\n\n# myapp/fetcher.py\n# def fetch_with_retry(url, attempts=3):\n#     for i in range(attempts):\n#         try:\n#             r = requests.get(url, timeout=5)\n#             r.raise_for_status()\n#             return r.json()\n#         except requests.RequestException:\n#             if i == attempts - 1:\n#                 raise\n\ndef test_flaky_recovers():\n    good = MagicMock(status_code=200)\n    good.json.return_value = {\"ok\": True}\n    good.raise_for_status = MagicMock()\n\n    with patch(\"myapp.fetcher.requests.get\",\n               side_effect=[requests.ConnectionError(\"net\"),\n                            requests.Timeout(\"slow\"),\n                            good]):\n        from myapp.fetcher import fetch_with_retry\n        assert fetch_with_retry(\"http://x\", attempts=3) == {\"ok\": True}\n```\n\nThe test verifies that two transient failures followed by a success still returns the payload. Each exception instance in the list is raised when its turn comes. Passing exception *classes* instead of instances works too. If the list is exhausted before calls stop, `StopIteration` is raised.",
)

add(
    "You are a mocking expert.",
    "Explain spec= vs autospec= vs spec_set=.",
    "They all constrain a mock's interface; they differ in strictness.\n\n```python\nfrom unittest.mock import Mock, create_autospec\n\nclass Repo:\n    def save(self, obj): ...\n    def find(self, id): ...\n\n# spec= accepts only attributes that exist on the real class, no signature check\nm1 = Mock(spec=Repo)\nm1.save(\"x\")             # ok, any args allowed\nm1.nonexistent           # AttributeError\n\n# spec_set= same, but ALSO blocks setting new attributes\nm2 = Mock(spec_set=Repo)\nm2.extra = 1             # AttributeError\n\n# autospec (via create_autospec or patch(autospec=True)) also validates signatures\nm3 = create_autospec(Repo)\nm3.save(\"x\")             # ok\n# m3.save()              # TypeError: missing 1 required positional arg\n```\n\nUse `spec_set` when you want to prevent tests from sprouting new mock attributes by accident. Use `autospec` when signature drift between tests and production is the bug you want to catch. Plain `spec` is rarely the best choice; either `spec_set` or `autospec` gives a stricter safety net for almost no extra cost.",
)

add(
    "You are a mocking expert.",
    "How do I mock sys.argv for CLI tests?",
    "Patch `sys.argv` for the duration of the test.\n\n```python\nimport sys\nfrom unittest.mock import patch\n\n# myapp/cli.py\n# def main():\n#     if \"--verbose\" in sys.argv:\n#         return \"verbose\"\n#     return \"quiet\"\n\ndef test_cli_verbose():\n    with patch.object(sys, \"argv\", [\"prog\", \"--verbose\"]):\n        from myapp.cli import main\n        assert main() == \"verbose\"\n\ndef test_cli_quiet():\n    with patch.object(sys, \"argv\", [\"prog\"]):\n        from myapp.cli import main\n        assert main() == \"quiet\"\n\n# monkeypatch equivalent\ndef test_cli_monkey(monkeypatch):\n    monkeypatch.setattr(sys, \"argv\", [\"prog\", \"--json\"])\n    from myapp.cli import main\n    assert main() == \"quiet\"  # assuming main ignores unknown flags\n```\n\nFor click/typer CLIs, prefer the framework's test runner:\n\n```python\nfrom click.testing import CliRunner\nfrom myapp.cli import cli\n\ndef test_click():\n    runner = CliRunner()\n    result = runner.invoke(cli, [\"--verbose\"])\n    assert result.exit_code == 0\n    assert \"verbose\" in result.output\n```\n\nFramework runners capture stdout, exit codes, and exceptions in one call, which is much cleaner than patching `sys.argv` and `sys.exit`.",
)


Path("/tmp/gen_block_2.json").write_text(json.dumps(samples))
print(f"Block 2: {len(samples)}")
