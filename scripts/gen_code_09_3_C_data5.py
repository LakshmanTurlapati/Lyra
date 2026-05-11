# SPDX-License-Identifier: MIT
"""Debugging / refactoring (~85) and testing (~55) entries. Wave 3 part 5."""


def register_all(add):
    # ---- debug / refactor ----
    add("debug",
        "Why does this `def f(x, items=[])` accumulate values across calls?",
        "Default mutable arguments are shared between calls. Use `None` as the sentinel:\n```python\ndef f(x, items=None):\n    if items is None:\n        items = []\n    items.append(x)\n    return items\n```\nThis is the most common Python footgun; flake8-bugbear's `B006` rule flags it.")

    add("debug",
        "Why does `for i in range(10): callbacks.append(lambda: i)` give 9 from every callback?",
        "All lambdas close over the same `i`, which is 9 by the time you call them. Bind early:\n```python\ncallbacks = [lambda i=i: i for i in range(10)]\n```\nThe default-argument trick freezes the value at definition time. `functools.partial(lambda i: i, i)` works too.")

    add("debug",
        "Why is `{}.get('missing', expensive_call())` evaluating `expensive_call` even on a hit?",
        "Python evaluates arguments eagerly. Use a `try/except KeyError` or check membership first:\n```python\nif 'missing' in d:\n    return d['missing']\nreturn expensive_call()\n```\nOr use `dict.setdefault(key, lambda: ...)` patterns... actually no, even setdefault evaluates eagerly. The `if` is the cleanest fix.")

    add("debug",
        "My `concurrent.futures.ProcessPoolExecutor` is hanging. Why?",
        "Most likely the worker function references a closure or unpicklable object. Process pools must pickle everything; closures and lambdas usually aren't picklable. Move the worker to module scope and pass plain data."  )

    add("debug",
        "My pytest fixtures are running once per test even with `scope='session'`. Why?",
        "Most likely the fixture lives in a conftest at a deeper directory than the test, or a parametrized test forces re-evaluation. Run pytest with `--setup-show` to see exactly when fixtures run; that usually reveals the cause within minutes.")

    add("debug",
        "Why does `json.dumps(my_dict)` raise `TypeError: Object of type datetime is not JSON serializable`?",
        "`json` only knows about strings, numbers, lists, dicts, bools, and None. Pass a `default` function:\n```python\nimport json\nfrom datetime import datetime\n\ndef default(o):\n    if isinstance(o, datetime):\n        return o.isoformat()\n    raise TypeError(f'unserializable: {type(o).__name__}')\n\njson.dumps(my_dict, default=default)\n```")

    add("debug",
        "I'm getting `RecursionError` from a small input. What's going on?",
        "Either the recursion isn't actually shrinking the input (bug), or Python's default 1000-frame limit is too low for your tree depth. For tree-shaped recursion, prefer iteration with an explicit stack. Don't bump `sys.setrecursionlimit` lightly -- it can crash the interpreter.")

    add("debug",
        "Refactor this nested if into early returns: `if x: if y: if z: return True; else: return False`.",
        "```python\ndef ok(x, y, z) -> bool:\n    if not x:\n        return False\n    if not y:\n        return False\n    if not z:\n        return False\n    return True\n```\nGuard clauses keep the happy path flat and easy to scan.")

    add("debug",
        "Why is `df.iterrows()` slow? How do I speed it up?",
        "`iterrows` boxes each row into a Series and copies values per iteration. Vectorize whenever possible (`df['c'] = df['a'] + df['b']`). When you must loop, `itertuples()` is dramatically faster, and `df.values` (or `df.to_numpy()`) is faster still.")

    add("debug",
        "My `requests.get(url)` hangs forever. Why?",
        "You didn't pass a timeout. The default is *no timeout*, so a slow server can hang your process indefinitely. Always set both connect and read timeouts: `requests.get(url, timeout=(3, 30))`.")

    add("debug",
        "Why does `set([{'a': 1}])` raise `TypeError: unhashable type: 'dict'`?",
        "Sets need hashable elements. Dicts are mutable and therefore unhashable. Convert to `frozenset` of items if you need a hashable representation: `frozenset(d.items())`.")

    add("debug",
        "I changed `True` to `False` in code and tests still pass. Why?",
        "Tests probably don't exercise that branch. Add a test that fails first, then your fix turns it green. If you're unsure where coverage lies, run `coverage run -m pytest && coverage report` to see what's actually executed.")

    add("debug",
        "Refactor: `if user is not None and user.active is True: send(user)`.",
        "```python\nif user and user.active:\n    send(user)\n```\n`is not None` plus `is True` are usually verbose; truthiness covers both. Be careful when 'falsy but not None' values (0, '', []) shouldn't trigger the same path -- in that case keep the `is not None`.")

    add("debug",
        "My pickle file from Python 3.8 won't load in 3.12. Why?",
        "Class definitions changed (or vanished) between versions. Pickle stores qualified names; if the class moved, unpickle fails. Use `dill` for more permissive serialization, or move to a forward-compatible format like JSON or protobuf for long-lived data.")

    add("debug",
        "Why is `os.path.join('/tmp', '/etc/passwd')` returning '/etc/passwd'?",
        "`os.path.join` resets when it sees an absolute path mid-arguments. To safely join user-controlled paths, validate them first and use `pathlib.Path('/tmp') / Path(name).name`. Treating user input as a relative path is critical for avoiding directory traversal bugs.")

    add("debug",
        "Refactor: rewrite `dict((k, v.upper()) for k, v in d.items())` more idiomatically.",
        "```python\n{k: v.upper() for k, v in d.items()}\n```\nDict comprehensions are faster and more readable than passing a generator into `dict()`. Same for set comprehensions vs `set(...)` of a generator.")

    add("debug",
        "Why does `pd.read_csv('big.csv')` blow up memory?",
        "By default, columns are loaded as `object` (Python strings) which is huge for high-cardinality data. Use `dtype` arguments to specify smaller types, `usecols` to load only what you need, and `chunksize` to process in pieces. For analytical work, switch to Parquet -- it's compressed and typed.")

    add("debug",
        "I'm seeing `AttributeError: module 'X' has no attribute 'Y'` after upgrading. Why?",
        "The library renamed or removed `Y`. Check the changelog, then update your import to the new name. If the new version doesn't expose the symbol any longer, you may need to pin to the older version while you migrate.")

    add("debug",
        "Refactor: replace `try: x = d[key] except KeyError: x = default`.",
        "```python\nx = d.get(key, default)\n```\n`dict.get` is the canonical idiom. Use the try/except form only when computing the default is expensive and you want lazy evaluation -- but in that case `if key in d` is clearer.")

    add("debug",
        "Why does my function return `None` instead of the value I expected?",
        "Most likely you forgot `return` on a branch. Run with `-W error` and add type hints + mypy strict; both surface this class of mistake. Or you may have a method that mutates in place but you assumed it returned -- `list.sort()` is a famous example.")

    add("debug",
        "Why does my datetime arithmetic across DST give a one-hour off result?",
        "Naive datetimes don't know about DST. Use `pytz` or, better, `zoneinfo` (stdlib in 3.9+):\n```python\nfrom datetime import datetime, timedelta\nfrom zoneinfo import ZoneInfo\n\nstart = datetime(2025, 3, 9, 0, tzinfo=ZoneInfo('America/Los_Angeles'))\nend = start + timedelta(days=1)\n```\nUse aware datetimes throughout, store/transmit UTC, and convert at the edges.")

    add("debug",
        "Why does `for line in file:` skip the first line sometimes?",
        "If you `next(file)` first (e.g., to skip a header), iteration resumes after that line. To unify, use `csv.reader` with `dialect`, or `for i, line in enumerate(file): if i == 0: continue`. For headers specifically, `csv.DictReader` handles this for you.")

    add("debug",
        "Refactor: simplify `result = []; for x in xs: if cond(x): result.append(transform(x))`.",
        "```python\nresult = [transform(x) for x in xs if cond(x)]\n```\nList comprehension is the canonical 'filter+map' in Python. Generator expressions (`(...)`) work the same way for streaming.")

    add("debug",
        "Why is `assert` not catching errors in production?",
        "Python's `-O` flag strips assertions. Don't rely on `assert` for runtime checks of untrusted input -- raise explicit exceptions instead. Asserts are for documenting invariants you believe are always true.")

    add("debug",
        "My `multiprocessing` code prints duplicates of imports running. Why?",
        "On platforms using `spawn` (Windows, recent macOS), child processes re-execute the module. Always wrap entry points with `if __name__ == '__main__':` or move them out of module top-level.")

    add("debug",
        "Why does `re.match(r'\\d+', 'abc123')` return None?",
        "`re.match` anchors at the start. Use `re.search` for 'anywhere in the string' or `re.fullmatch` to require the whole string to match.")

    add("debug",
        "Refactor: turn this 30-line function with three responsibilities into smaller pieces.",
        "Extract each responsibility into a helper named for what it does, then have the original function call them:\n```python\ndef process(items):\n    cleaned = _clean(items)\n    enriched = _enrich(cleaned)\n    return _summarize(enriched)\n```\nUnit tests get easier and the original function's intent becomes obvious. The 'one screen of code' rule is a good ceiling.")

    add("debug",
        "Why does my logging not appear when I run the script?",
        "Without explicit configuration, the root logger's level is WARNING. Add `logging.basicConfig(level='INFO', format='%(asctime)s %(levelname)s %(name)s %(message)s')` at startup. Configure once, not per-module.")

    add("debug",
        "Why does `decimal.Decimal('0.1') + 0.1` give a weird value?",
        "Mixing Decimal and float is forbidden in some operations and lossy in others. Convert the float first: `Decimal('0.1') + Decimal(str(0.1))`. Decimal is for *exact* arithmetic; mixing breaks the contract.")

    add("debug",
        "Why does `import * from module` not import everything I expect?",
        "Wildcard imports respect `__all__` if defined, otherwise import names not starting with `_`. Wildcard imports are also linter-flagged because they obscure where names come from. Prefer explicit imports.")

    add("debug",
        "I'm getting `BlockingIOError` from a non-blocking socket. What's wrong?",
        "Non-blocking IO returns immediately whether or not data is ready. Wrap reads/writes with `try/except BlockingIOError` and retry, or use `select`/`selectors` to wait for readiness. Better, use `asyncio` and let it manage the readiness loop.")

    add("debug",
        "Refactor: replace this manual JSON build with dataclasses.",
        "```python\nfrom dataclasses import dataclass, asdict\n\n@dataclass\nclass User:\n    id: int\n    name: str\n    active: bool = True\n\nuser = User(1, 'Lakshman')\npayload = asdict(user)\n```\nDataclasses give you `__init__`, `__repr__`, `__eq__`, and structured serialization without boilerplate.")

    add("debug",
        "Why does my decorator change the function's signature in `help()`?",
        "Decorators replace the function with the wrapper. Use `functools.wraps` to copy name, docstring, and signature:\n```python\nfrom functools import wraps\n\ndef trace(fn):\n    @wraps(fn)\n    def wrapper(*args, **kwargs):\n        return fn(*args, **kwargs)\n    return wrapper\n```")

    add("debug",
        "Why does pickling a lambda fail?",
        "Lambdas have no qualified name; pickle can't reference them. Use a regular function defined at module scope. Same for closures with non-picklable captures.")

    add("debug",
        "Why is my pytest test passing locally but failing in CI?",
        "Common causes: hardcoded paths, dependency on local env vars, hidden order dependency between tests, timezone differences, or a test polluting state that other tests rely on. Run with `pytest -p no:randomly --tb=short -v` locally to reproduce; consider the `pytest-randomly` plugin to surface order dependence.")

    add("debug",
        "Refactor: rewrite `lst[len(lst)-1]` more idiomatically.",
        "```python\nlst[-1]\n```\nNegative indexing handles the empty case the same way (raising IndexError). For 'last or default', `lst[-1] if lst else default`.")

    add("debug",
        "Why does my Flask app slow down after running for a day?",
        "Probably a memory leak: caches that grow without bound, references stuck in closures, or accumulated logging objects. Profile with `tracemalloc.start(); print(tracemalloc.take_snapshot().statistics('lineno')[:10])`. Most leaks come from holding objects in module-level lists or singletons.")

    add("debug",
        "Why is `Decimal` slower than `float` for my computation?",
        "Decimal is implemented in pure Python in older versions and uses arbitrary precision. It's ~100x slower than float. Use Decimal only when exact decimal arithmetic matters (currency); use float for general numeric work.")

    add("debug",
        "Refactor: replace `if x == True` and `if x == False` with idiomatic forms.",
        "```python\nif x:\n    ...\nif not x:\n    ...\n```\n`== True` is redundant and forces strict equality with the boolean True. Truthiness handles the common case; if you need exact identity use `if x is True`.")

    add("debug",
        "My `os.path.exists(path)` returns False but the file is there!",
        "Check for trailing whitespace, hidden chars (BOM), or symlink targets that don't exist. `pathlib.Path(path).resolve(strict=True)` will raise with the actual problem. Also confirm the process has permission to stat the parent directory.")

    add("debug",
        "Why does `exec` not see my function's local variables?",
        "`exec` executes in the caller's globals by default but in *its own* locals scope. Pass an explicit dict: `exec(code, my_globals, my_locals)`. Honestly, almost any use of `exec` should be reconsidered -- it's a security and debuggability nightmare.")

    add("debug",
        "Refactor: `if isinstance(x, list) or isinstance(x, tuple) or isinstance(x, set)`.",
        "```python\nif isinstance(x, (list, tuple, set)):\n    ...\n```\n`isinstance` accepts a tuple of types. For sequence-like behavior, `isinstance(x, collections.abc.Sequence)` is more general (covers user-defined sequences too).")

    add("debug",
        "Why does my regex with backreferences fail in Python?",
        "Backreferences in patterns use `\\1`, `\\2`. In replacement strings, both `\\1` and `\\\\1` may be interpreted. Use raw strings (`r'(\\w+) \\1'`) for patterns to avoid double-escaping.")

    add("debug",
        "Why does my SQL injection lint fail when I use f-strings in queries?",
        "Because string interpolation into SQL is the textbook injection vector. Use parameter binding:\n```python\ncursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))\n```\nNever f-string user input into SQL, even if you 'sanitize' it first.")

    add("debug",
        "Refactor: simplify `data['users'][0]['emails'][0]` chains that error on missing keys.",
        "Use a helper that walks safely:\n```python\ndef get_in(d, *keys, default=None):\n    for k in keys:\n        if isinstance(d, dict):\n            d = d.get(k)\n        elif isinstance(d, list) and isinstance(k, int) and 0 <= k < len(d):\n            d = d[k]\n        else:\n            return default\n        if d is None:\n            return default\n    return d\n```\nOr use `glom` / `dpath` for deeper traversal needs.")

    add("debug",
        "Why is my generator expression evaluated multiple times?",
        "Generators are single-use iterators. After one pass they're exhausted. To reuse, materialize to a list (`xs = list(gen)`) or wrap in a function that builds a fresh generator each call.")

    add("debug",
        "Refactor: replace `list(filter(None, xs))` with a comprehension.",
        "```python\n[x for x in xs if x]\n```\n`filter(None, xs)` filters out falsy values; the comprehension is more explicit and readable. Plus comprehensions are micro-optimized in CPython.")

    add("debug",
        "Why does my function with `async def` return a coroutine instead of the value?",
        "Calling an async function returns a coroutine object; you must `await` it inside another async function or run it via `asyncio.run`. From sync code: `result = asyncio.run(my_async_fn())`. From within an existing event loop, use `await`.")

    add("debug",
        "Refactor: simplify `sum([x*x for x in xs])` to use less memory.",
        "```python\nsum(x * x for x in xs)\n```\nGenerator expression instead of list -- same answer, no intermediate list allocation. Important for large iterables.")

    add("debug",
        "Why does `list.append(x)` return None?",
        "Most mutating methods on built-in types return None to discourage chaining and clarify that they mutate in place. Use the method, then access the variable separately.")

    add("debug",
        "My `pip install -r requirements.txt` succeeds but imports fail. Why?",
        "Check that `pip` and `python` point to the same environment: `which python && which pip` and `pip --version`. Mixing system pip with a venv python is a common cause of 'why isn't this installed?' confusion.")

    add("debug",
        "Refactor: combine multiple `.replace` calls into one pass.",
        "```python\nimport re\n\ndef multi_replace(text: str, mapping: dict[str, str]) -> str:\n    pattern = re.compile('|'.join(map(re.escape, mapping)))\n    return pattern.sub(lambda m: mapping[m.group(0)], text)\n```\nA single regex pass beats N sequential `.replace` calls when the input is large.")

    add("debug",
        "Why is my multiprocessing slower than a simple loop?",
        "Process creation, pickling overhead, and worker-pool synchronization all have fixed cost. For small workloads or quick functions, multiprocessing adds latency rather than removing it. Profile first; then consider threads (for IO-bound) or vectorized numpy (for CPU-bound numeric).")

    add("debug",
        "Refactor: replace `if foo == 'a' or foo == 'b' or foo == 'c'`.",
        "```python\nif foo in {'a', 'b', 'c'}:\n    ...\n```\nA set literal makes the intent obvious and is O(1) for membership.")

    add("debug",
        "Why does my numpy code with mixed types silently lose precision?",
        "Numpy upcasts mixed-dtype operations to a common type, which sometimes truncates. Be explicit:\n```python\narr = arr.astype(np.float64)\n```\nFor money-like calculations, never use floats -- use int (cents) or Decimal.")

    add("debug",
        "I'm hitting `ConnectionPool full, discarding connection` warnings. Why?",
        "You're issuing requests faster than your pool can recycle connections. Increase pool size on the adapter / Session, or batch requests, or use a circuit-breaker pattern. Treat the warning as a real signal; ignored connections often time out and confuse retries.")

    add("debug",
        "Why does my async function not run when I call it?",
        "Calling an async function returns a coroutine; you must `await` it or schedule it via `asyncio.create_task`. Bare `async_fn()` calls are a no-op. Lint with `pyflakes`/`ruff` (rule `F841` and friends) -- many catch the unused-coroutine mistake.")

    add("debug",
        "Refactor: replace `dict(a, **b)` with the modern syntax.",
        "```python\nmerged = {**a, **b}\n# Or, in Python 3.9+:\nmerged = a | b\n```\nThe `|` operator is the most readable form and matches set union semantics.")

    add("debug",
        "My `subprocess.run` is hanging on output. Why?",
        "If the child writes more than the OS pipe buffer can hold, it blocks. Use `capture_output=True` (which buffers safely) or read incrementally. Don't use `Popen.stdout.read()` after the process exits without `communicate()` -- it can deadlock.")

    add("debug",
        "Why does my CPython script hit 100% on one core only?",
        "The GIL allows only one Python thread at a time for CPU-bound work. Use `multiprocessing` for parallel CPU work, or move the hot loop into numpy/numba/cython where the GIL releases. For IO-bound code, threads still scale.")

    add("debug",
        "Refactor: rewrite this `for/else` construct more clearly.",
        "```python\n# old\nfor x in xs:\n    if matches(x):\n        result = x\n        break\nelse:\n    result = None\n# clearer:\nresult = next((x for x in xs if matches(x)), None)\n```\n`for/else` is a Python oddity that confuses most readers; `next()` with a generator is direct.")

    add("debug",
        "Why does my Flask error handler not run?",
        "Either the exception is being caught earlier in your code, or you're running in debug mode (which bypasses error handlers and shows the debugger). Use `app.config['PROPAGATE_EXCEPTIONS'] = False` or just turn off debug mode for testing handlers.")

    add("debug",
        "My JSON has trailing commas and Python's parser rejects it.",
        "Standard JSON disallows trailing commas. For lenient parsing, use `json5` or fix the JSON at the source. Hand-edited JSON files are a common source of this; consider TOML or YAML for human-edited config.")

    add("debug",
        "Refactor: simplify a nested try/except with multiple handlers.",
        "```python\ntry:\n    do_thing()\nexcept (FooError, BarError) as exc:\n    log.warning('expected', exc_info=exc)\nexcept Exception as exc:\n    log.exception('unexpected')\n    raise\n```\nGroup expected exceptions by tuple, log unexpected ones with `log.exception`, and let unknown errors propagate.")

    add("debug",
        "Why is `str(my_object)` showing `<__main__.X object at 0x...>`?",
        "You haven't defined `__str__` or `__repr__`. Add them:\n```python\nclass X:\n    def __init__(self, n): self.n = n\n    def __repr__(self) -> str:\n        return f'X(n={self.n!r})'\n```\n`__repr__` is the developer-facing view; `__str__` falls back to `__repr__` if undefined.")

    add("debug",
        "My matplotlib script creates a window but the script keeps running.",
        "Use `plt.show()` to block, or `plt.savefig(...)` if you don't need a window. For non-interactive backends (CI, servers), set `matplotlib.use('Agg')` before importing pyplot.")

    add("debug",
        "Refactor: simplify converting between 0-based and 1-based indices.",
        "Pick one convention and convert at the boundary:\n```python\n# Internally 0-based; convert when displaying or accepting external input.\ndef from_1based(i: int) -> int:\n    return i - 1\ndef to_1based(i: int) -> int:\n    return i + 1\n```\nNamed functions make the convention boundary explicit.")

    add("debug",
        "Why does `pathlib.Path('foo.txt').exists()` return False even when I see the file?",
        "Probably the cwd isn't what you think. `Path.cwd()` shows it. Or you're checking a relative path from a script run from a different directory. Use absolute paths or `Path(__file__).parent` to anchor.")

    add("debug",
        "My logging configuration is being overridden somewhere. How do I find the culprit?",
        "Set `logging.basicConfig(force=True)` to take over, then audit other places that call `basicConfig` or `dictConfig`. Many libraries call `logging.getLogger(__name__)` (good) but a few call `basicConfig` (bad) -- search for it across deps.")

    add("debug",
        "Refactor: replace `lambda x: x.attr` with a clearer alternative.",
        "```python\nfrom operator import attrgetter\n\nsorted(items, key=attrgetter('attr'))\n```\n`operator.attrgetter` reads better and is a tiny bit faster than a lambda for this purpose.")

    add("debug",
        "Why does `sorted([1, 'a'])` raise TypeError?",
        "Python 3 forbids comparing different types in sort. Provide a key that produces comparable values: `sorted(items, key=str)`. Or normalize the data so types are consistent before sorting.")

    add("debug",
        "Why does `csv.reader` give me strings even for numeric columns?",
        "CSV is plain text -- every field is a string. Cast as you load:\n```python\nimport csv\n\nwith open('data.csv') as f:\n    rows = [(int(r[0]), float(r[1]), r[2]) for r in csv.reader(f) if r and r[0] != 'id']\n```\nFor heavier work, pandas does this automatically with `dtype` hints.")

    add("debug",
        "Refactor: replace a fragile `eval(user_input)`.",
        "`eval` is unsafe on user input -- attackers can run arbitrary code. For arithmetic, use `ast.literal_eval` (safe for literals) or the `simpleeval` library:\n```python\nimport ast\nresult = ast.literal_eval(user_input)\n```\nFor expressions with variables, write a real parser; never eval untrusted strings.")

    add("debug",
        "Why does my code print `b'hello'` instead of `hello`?",
        "Bytes vs str distinction. Decode: `b'hello'.decode('utf-8')`. If you read a file in binary mode (`'rb'`), the result is bytes; switch to text mode (`'r'`) or decode explicitly.")

    add("debug",
        "Refactor: collapse this counter-keeping loop with `Counter`.",
        "```python\nfrom collections import Counter\n\ncounts = Counter(words)\n```\nReplaces `counts = {}; for w in words: counts[w] = counts.get(w, 0) + 1`. Counter also supports `most_common`, arithmetic between counters, and zero-by-default access.")

    add("debug",
        "Why does my context manager not run `__exit__` on certain exceptions?",
        "It always runs `__exit__`; the question is whether `__exit__` swallows the exception. Returning truthy from `__exit__` suppresses it; falsy/None re-raises. For 'cleanup but don't swallow', return None.")

    add("debug",
        "My SQLAlchemy session shows stale data after another process commits.",
        "Sessions cache queries. Either expire (`session.expire_all()`) or commit/refresh to see fresh data. For long-running scripts, also consider isolation levels and the autoflush behavior.")

    add("debug",
        "Refactor: replace explicit dict/list-building with comprehensions where appropriate.",
        "Use comprehensions when the loop body is a single conditional append. Keep an explicit `for` loop when the body is multiple statements or has side effects -- comprehensions get unreadable past one or two clauses. The principle: comprehensions for transformation, loops for procedure.")

    add("debug",
        "Why does `pickle.loads` execute code from arbitrary inputs?",
        "Pickle calls `__reduce__` during deserialization, which can execute any callable. Never unpickle data from untrusted sources. For network/disk transport between trust domains, use JSON, Protobuf, or MessagePack.")

    add("debug",
        "Why does `with open(...) as f:` leave the file open in some traceback frames?",
        "It doesn't -- the file closes on context exit. What you may be seeing is the OS still holding the descriptor briefly after Python releases it. If you need an immediately-released file, also call `f.flush(); os.fsync(f.fileno())` for durability.")

    add("debug",
        "Refactor: tighten this code that uses `sys.exit(1)` everywhere.",
        "Raise an exception at the failure site and have `main()` translate it to an exit code at the boundary:\n```python\nclass UserError(Exception): pass\n\ndef main() -> int:\n    try:\n        do_work()\n        return 0\n    except UserError as exc:\n        print(f'error: {exc}', file=sys.stderr)\n        return 1\n```\nFunctions stay testable; the exit-code translation lives in one place.")

    add("debug",
        "Why are my tests slow? They take 90 seconds for 200 tests.",
        "Profile with `pytest --durations=20`. Common culprits: hitting real DBs/networks, `time.sleep` in fixtures, parametrized fixtures rebuilding heavy state, or untargeted `freezegun`. Mark slow tests with `@pytest.mark.slow` and exclude them from your fast loop.")

    add("debug",
        "Refactor: simplify this code that pads a string with leading zeros.",
        "```python\nf'{n:08d}'\n```\nF-string format spec is the cleanest; use `str.zfill(8)` if `n` is already a string.")

    add("debug",
        "I see `RuntimeWarning: invalid value encountered in...` in numpy. What's wrong?",
        "Likely a NaN or Inf entered the computation (e.g., divide by zero). Find the source: `np.errstate(invalid='raise'): your_code()` raises on the offending op so you can catch it. Then either fix the input or handle the bad value with `np.nan_to_num` or masking.")

    add("debug",
        "Refactor: collapse this 'sentinel value' return into an Optional.",
        "```python\nfrom typing import Optional\n\ndef find(xs: list, target) -> Optional[int]:\n    try:\n        return xs.index(target)\n    except ValueError:\n        return None\n```\nReturning None for 'not found' is more idiomatic than -1, and the type hint forces callers to handle it.")

    # ---- testing ----
    add("test",
        "Write a basic pytest test for an `add(a, b)` function.",
        "```python\nfrom mymath import add\n\ndef test_add_basic():\n    assert add(2, 3) == 5\n\ndef test_add_negative():\n    assert add(-1, 1) == 0\n```\nKeep tests narrow; one assertion shape per test makes failures self-explanatory.")

    add("test",
        "Show how to parametrize a pytest test.",
        "```python\nimport pytest\nfrom mymath import add\n\n@pytest.mark.parametrize('a,b,want', [(1,1,2),(0,0,0),(-1,1,0)])\ndef test_add(a, b, want):\n    assert add(a, b) == want\n```\nOne function, many cases. Use `ids` for readable failure names: `pytest.mark.parametrize(..., ids=['ones','zeros','signs'])`.")

    add("test",
        "Show how to test that a function raises a specific exception.",
        "```python\nimport pytest\n\ndef test_raises():\n    with pytest.raises(ValueError, match='must be positive'):\n        my_function(-1)\n```\n`match` accepts a regex against the exception message; protects against catching wrong-but-same-type errors.")

    add("test",
        "Show how to use a fixture to share setup across tests.",
        "```python\nimport pytest\n\n@pytest.fixture\ndef sample_data():\n    return {'a': 1, 'b': 2}\n\ndef test_uses_data(sample_data):\n    assert sample_data['a'] == 1\n```\nFixtures inject values; pytest discovers them by parameter name.")

    add("test",
        "Show how to use a `tmp_path` fixture.",
        "```python\ndef test_writes_file(tmp_path):\n    target = tmp_path / 'out.txt'\n    target.write_text('hello')\n    assert target.read_text() == 'hello'\n```\n`tmp_path` is a `pathlib.Path` to a unique temp dir cleaned after the test.")

    add("test",
        "Show how to mock a function call with `unittest.mock`.",
        "```python\nfrom unittest.mock import patch\n\n@patch('mymodule.requests.get')\ndef test_fetches(mock_get):\n    mock_get.return_value.status_code = 200\n    mock_get.return_value.json.return_value = {'ok': True}\n    assert mymodule.fetch() == {'ok': True}\n```\nPatch where the name is *looked up*, not where it's defined.")

    add("test",
        "Show how to use `monkeypatch` to set an env var in a test.",
        "```python\ndef test_uses_env(monkeypatch):\n    monkeypatch.setenv('API_KEY', 'test-key')\n    assert get_api_key() == 'test-key'\n```\n`monkeypatch` automatically reverts after the test -- no global state pollution.")

    add("test",
        "Show how to share fixtures across files via conftest.py.",
        "```python\n# conftest.py\nimport pytest\n\n@pytest.fixture\ndef api_client():\n    return APIClient(base_url='http://test')\n```\nFixtures defined in conftest are visible to all tests in the same directory and below. No imports required.")

    add("test",
        "Show how to mark slow tests so you can skip them by default.",
        "```python\n# in pytest.ini or pyproject.toml\n# [tool.pytest.ini_options]\n# markers = ['slow: marks tests as slow']\n\nimport pytest\n\n@pytest.mark.slow\ndef test_full_pipeline():\n    ...\n# Run fast tests: pytest -m 'not slow'\n```\nKeep the inner-loop suite fast; expensive tests run only in CI.")

    add("test",
        "Show how to test an HTTP client by mocking with `responses`.",
        "```python\nimport responses\nimport requests\n\n@responses.activate\ndef test_get():\n    responses.add(responses.GET, 'https://api.example.com/x', json={'ok': True}, status=200)\n    r = requests.get('https://api.example.com/x', timeout=5)\n    assert r.json() == {'ok': True}\n    assert len(responses.calls) == 1\n```\nNo real HTTP; deterministic and fast.")

    add("test",
        "Show how to use Hypothesis for property-based tests.",
        "```python\nfrom hypothesis import given, strategies as st\n\n@given(st.lists(st.integers()))\ndef test_sort_idempotent(xs):\n    assert sorted(sorted(xs)) == sorted(xs)\n```\nHypothesis explores edge cases (empty, single, duplicates) you'd never enumerate manually.")

    add("test",
        "Show how to test a Flask app with the test client.",
        "```python\nfrom myapp import create_app\n\ndef test_health():\n    app = create_app()\n    client = app.test_client()\n    r = client.get('/health')\n    assert r.status_code == 200\n    assert r.get_json() == {'status': 'ok'}\n```\n`test_client` runs in-process; no socket required.")

    add("test",
        "Show how to test a FastAPI endpoint with TestClient.",
        "```python\nfrom fastapi.testclient import TestClient\nfrom myapp import app\n\nclient = TestClient(app)\n\ndef test_create_user():\n    r = client.post('/users', json={'name': 'Lakshman'})\n    assert r.status_code == 200\n    assert r.json()['name'] == 'Lakshman'\n```\nTestClient wraps the ASGI app -- no separate server.")

    add("test",
        "Show how to test that logs were emitted with `caplog`.",
        "```python\nimport logging\n\ndef test_logs(caplog):\n    with caplog.at_level(logging.WARNING):\n        do_thing()\n    assert any('expected message' in r.message for r in caplog.records)\n```\n`caplog.at_level` captures within a context; use `caplog.records` for fine-grained inspection.")

    add("test",
        "Show how to mock time with `freezegun`.",
        "```python\nfrom datetime import datetime\nfrom freezegun import freeze_time\n\n@freeze_time('2025-01-15')\ndef test_today():\n    assert datetime.now().date().isoformat() == '2025-01-15'\n```\nFor cleaner code long-term, inject a clock callable instead and pass a fake. Use freezegun for legacy code.")

    add("test",
        "Show how to use a fixture that yields setup and teardown.",
        "```python\nimport pytest\n\n@pytest.fixture\ndef temp_table(db):\n    db.execute('CREATE TABLE t(id INT)')\n    try:\n        yield 't'\n    finally:\n        db.execute('DROP TABLE t')\n```\nThe `yield`-style fixture cleanly handles cleanup even if the test fails.")

    add("test",
        "Show how to skip a test conditionally.",
        "```python\nimport pytest\nimport sys\n\n@pytest.mark.skipif(sys.platform == 'win32', reason='Linux/Mac only')\ndef test_unix_only():\n    ...\n```\nAlways pass a `reason` -- it shows up in the test summary.")

    add("test",
        "Show how to expect a test to fail.",
        "```python\nimport pytest\n\n@pytest.mark.xfail(reason='known bug, fixed in v2')\ndef test_known_issue():\n    assert False\n```\n`xfail` records the test as expected-failure. If it unexpectedly passes (`XPASS`), pytest flags it -- a useful bug-fix tripwire.")

    add("test",
        "Show how to test a database operation with a transactional fixture.",
        "```python\nimport pytest\n\n@pytest.fixture\ndef db_session(engine):\n    conn = engine.connect()\n    trans = conn.begin()\n    session = Session(bind=conn)\n    try:\n        yield session\n    finally:\n        session.close()\n        trans.rollback()\n        conn.close()\n```\nWrap each test in a transaction and roll back -- isolation without truncate scripts.")

    add("test",
        "Show how to use Hypothesis with a custom strategy.",
        "```python\nfrom hypothesis import given, strategies as st\n\nemails = st.builds(lambda u, d: f'{u}@{d}', st.text(min_size=1, max_size=20), st.sampled_from(['a.com','b.org']))\n\n@given(emails)\ndef test_email(addr):\n    assert '@' in addr\n```\n`builds` constructs values from sub-strategies; `sampled_from` chooses among fixed values.")

    add("test",
        "Show how to use pytest's `raises` for context-rich error checking.",
        "```python\nimport pytest\n\ndef test_error_chain():\n    with pytest.raises(ValueError) as exc_info:\n        do_thing()\n    assert exc_info.value.args[0] == 'expected'\n    assert isinstance(exc_info.value.__cause__, KeyError)\n```\n`exc_info.value` is the raised exception object; access `__cause__` / `__context__` to assert chained errors.")

    add("test",
        "Show how to use a class-scoped fixture.",
        "```python\nimport pytest\n\n@pytest.fixture(scope='class')\ndef expensive_resource():\n    return load_huge_thing()\n\nclass TestX:\n    def test_a(self, expensive_resource):\n        assert expensive_resource.ready\n    def test_b(self, expensive_resource):\n        assert expensive_resource.size > 0\n```\nScopes: function (default), class, module, session.")

    add("test",
        "Show how to make tests deterministic when randomness is involved.",
        "```python\nimport random\nimport pytest\n\n@pytest.fixture(autouse=True)\ndef _seed():\n    random.seed(42)\n```\n`autouse=True` applies the fixture to every test in the file. Combine with `numpy.random.seed` if numpy is in the mix.")

    add("test",
        "Show how to test exception chaining.",
        "```python\nimport pytest\n\ndef test_chain():\n    with pytest.raises(RuntimeError) as ei:\n        try:\n            int('x')\n        except ValueError as v:\n            raise RuntimeError('wrap') from v\n    assert isinstance(ei.value.__cause__, ValueError)\n```\n`raise X from Y` sets `__cause__`; bare `raise X` only sets `__context__`.")

    add("test",
        "Show how to assert approximate equality for floats.",
        "```python\nimport pytest\n\ndef test_close():\n    assert 0.1 + 0.2 == pytest.approx(0.3)\n```\n`approx` chooses sensible default tolerances; pass `rel=` / `abs=` to override.")

    add("test",
        "Show how to assert a Mock was called with specific args.",
        "```python\nfrom unittest.mock import Mock\n\nm = Mock()\nm(1, 2, key='v')\nm.assert_called_once_with(1, 2, key='v')\n```\nPrefer `assert_called_once_with` over `assert_called_with` -- the former also asserts call count.")

    add("test",
        "Show how to mock an async function.",
        "```python\nfrom unittest.mock import AsyncMock, patch\n\n@patch('mymodule.fetch', new_callable=AsyncMock)\nasync def test_uses_fetch(mock_fetch):\n    mock_fetch.return_value = {'ok': True}\n    assert await use_fetch() == {'ok': True}\n```\n`AsyncMock` returns awaitables from its calls. Use `pytest-asyncio` to run async tests.")

    add("test",
        "Show how to capture stdout in a test.",
        "```python\ndef test_print(capsys):\n    print('hello')\n    captured = capsys.readouterr()\n    assert captured.out == 'hello\\n'\n```\n`capsys` captures stdout/stderr per test; for byte-level capture use `capsysbinary`.")

    add("test",
        "Show how to make a custom assertion helper visible in pytest tracebacks.",
        "```python\ndef assert_user(u, *, name: str, active: bool):\n    __tracebackhide__ = True  # hide this frame on failure\n    assert u.name == name\n    assert u.active == active\n```\n`__tracebackhide__ = True` makes pytest skip the helper in the failure trace -- shows the test code instead.")

    add("test",
        "Show how to test with multiple async tasks using asyncio.",
        "```python\nimport asyncio\nimport pytest\n\n@pytest.mark.asyncio\nasync def test_concurrent():\n    results = await asyncio.gather(task_a(), task_b())\n    assert all(r['ok'] for r in results)\n```\nRequires `pytest-asyncio`. The decorator dispatches the coroutine.")

    add("test",
        "Show how to use `pytest-mock` to patch in a fixture style.",
        "```python\ndef test_patches(mocker):\n    mock_fn = mocker.patch('mymodule.do_work', return_value=42)\n    assert call_it() == 42\n    mock_fn.assert_called_once()\n```\n`mocker` is auto-cleaned between tests; less boilerplate than `unittest.mock.patch`.")

    add("test",
        "Show how to mark a parametrize entry as expected-fail.",
        "```python\nimport pytest\n\n@pytest.mark.parametrize('x,want', [\n    (1, 1),\n    pytest.param(2, 'broken', marks=pytest.mark.xfail),\n    (3, 9),\n])\ndef test_pow(x, want):\n    assert x * x == want\n```\n`pytest.param` lets you mark individual entries without splitting the parametrize.")

    add("test",
        "Show how to compare large structures with deep diffs.",
        "```python\nfrom deepdiff import DeepDiff\n\ndef test_complex():\n    diff = DeepDiff(actual, expected, ignore_order=True)\n    assert not diff, diff\n```\n`DeepDiff` reports the exact path of every difference; assert messages remain useful for big structures.")

    add("test",
        "Show how to use a session-scoped fixture for an expensive app instance.",
        "```python\nimport pytest\nfrom myapp import create_app\n\n@pytest.fixture(scope='session')\ndef app():\n    return create_app(testing=True)\n\n@pytest.fixture\ndef client(app):\n    return app.test_client()\n```\nOne app object per test session; fresh client per test for test independence.")

    add("test",
        "Show how to assert a SQLAlchemy query produced specific rows.",
        "```python\ndef test_query(db_session):\n    db_session.add(User(id=1, name='Lakshman'))\n    db_session.flush()\n    rows = db_session.query(User).all()\n    assert [(u.id, u.name) for u in rows] == [(1, 'Lakshman')]\n```\nFlush, don't commit, in tests -- the rollback in the fixture handles cleanup.")

    add("test",
        "Show how to test that a context manager cleans up on error.",
        "```python\nimport pytest\n\ndef test_cleanup_on_error(my_ctx):\n    with pytest.raises(ValueError):\n        with my_ctx() as resource:\n            assert resource.opened\n            raise ValueError('boom')\n    # external assertion: resource is closed\n    assert not resource.opened\n```\nAssert post-`with` state to verify cleanup actually happened.")

    add("test",
        "Show how to mock a class instance method.",
        "```python\nfrom unittest.mock import patch\n\n@patch.object(MyClass, 'do_work', return_value='mocked')\ndef test_instance_method(mock_method):\n    instance = MyClass()\n    assert instance.do_work() == 'mocked'\n    mock_method.assert_called_once()\n```\n`patch.object` is more targeted than string-based `patch` for class methods.")

    add("test",
        "Show how to use VCR.py to record HTTP fixtures.",
        "```python\nimport pytest\nimport vcr\nimport requests\n\n@vcr.use_cassette('tests/fixtures/api.yaml')\ndef test_api():\n    r = requests.get('https://api.example.com/x', timeout=5)\n    assert r.status_code == 200\n```\nFirst run records; subsequent runs replay from the cassette. Great for HTTP-heavy tests.")

    add("test",
        "Show how to test a CLI built with `click`.",
        "```python\nfrom click.testing import CliRunner\nfrom mycli import main\n\ndef test_cli():\n    runner = CliRunner()\n    result = runner.invoke(main, ['--name', 'world'])\n    assert result.exit_code == 0\n    assert 'hello world' in result.output\n```\n`CliRunner` invokes commands in-process and captures stdout; no shell needed.")

    add("test",
        "Show how to write a regression test for a fixed bug.",
        "```python\ndef test_handles_empty_list_no_crash():\n    # regression: previously raised IndexError on empty input\n    assert process([]) == []\n```\nName the test for the symptom; comment with the original bug. Future maintainers thank you.")

    add("test",
        "Show how to test thread-safety with concurrent execution.",
        "```python\nimport threading\n\ndef test_thread_safe():\n    counter = Counter()\n    def worker():\n        for _ in range(1000):\n            counter.increment()\n    threads = [threading.Thread(target=worker) for _ in range(10)]\n    for t in threads: t.start()\n    for t in threads: t.join()\n    assert counter.value == 10_000\n```\nThread tests are nondeterministic; run in CI a few times before declaring victory.")

    add("test",
        "Show how to use `pytest-xdist` for parallel test execution.",
        "```bash\npip install pytest-xdist\npytest -n auto\n```\n`-n auto` uses one worker per CPU. Tests must be independent; flaky tests show up immediately under parallelization.")

    add("test",
        "Show how to test pandas DataFrame equality.",
        "```python\nimport pandas as pd\nfrom pandas.testing import assert_frame_equal\n\ndef test_transform():\n    actual = transform(input_df)\n    expected = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})\n    assert_frame_equal(actual.reset_index(drop=True), expected)\n```\n`assert_frame_equal` reports exactly what differs; raw `==` returns a DataFrame, not a bool.")

    add("test",
        "Show how to test numpy array equality.",
        "```python\nimport numpy as np\n\ndef test_array():\n    np.testing.assert_array_equal(np.array([1, 2, 3]), expected)\n    np.testing.assert_allclose(np.array([1.0, 2.0]), other, rtol=1e-5)\n```\nUse `assert_array_equal` for exact, `assert_allclose` for floating-point.")

    add("test",
        "Show how to test a generator function.",
        "```python\ndef test_evens():\n    g = evens(10)\n    assert list(g) == [0, 2, 4, 6, 8]\n```\nMaterialize with `list()` for value-by-value comparison; for infinite generators take a slice with `itertools.islice`.")

    add("test",
        "Show how to organize tests in a folder mirroring source layout.",
        "```\nproject/\n  src/myapp/\n    routes.py\n    models.py\n  tests/\n    test_routes.py\n    test_models.py\n  pyproject.toml\n```\nMirroring source paths makes it obvious which tests cover which module. Configure pytest with `pythonpath = ['src']`.")

    add("test",
        "Show how to test that a deprecation warning is emitted.",
        "```python\nimport pytest\nimport warnings\n\ndef test_warns():\n    with pytest.warns(DeprecationWarning, match='use new_api'):\n        old_api()\n```\nWraps the same way as `pytest.raises`. `recwarn` is an alternative for asserting warning counts.")

    add("test",
        "Show how to use `tmp_path_factory` for cross-test temp dirs.",
        "```python\nimport pytest\n\n@pytest.fixture(scope='session')\ndef shared_data_dir(tmp_path_factory):\n    d = tmp_path_factory.mktemp('shared')\n    (d / 'data.txt').write_text('once')\n    return d\n```\n`tmp_path_factory` lives at session scope and cleans up at the end; useful for shared but throwaway data.")

    add("test",
        "Show how to write a smoke test for an entire app.",
        "```python\ndef test_app_starts(client):\n    r = client.get('/')\n    assert r.status_code in (200, 302)\n```\nSmoke tests catch the 'is anything wired up at all' class of bug; cheap and high-value.")

    add("test",
        "Show how to assert structured logs with structlog.",
        "```python\nimport structlog\n\ndef test_logs(caplog):\n    structlog.configure(processors=[structlog.testing.LogCapture()])\n    log = structlog.get_logger()\n    log.info('event', user='lak')\n    captured = structlog.testing.LogCapture().entries  # adapt per your fixture\n```\nFor a cleaner setup, follow structlog's `LogCapture` fixture pattern in their docs.")

    add("test",
        "Show how to share heavy fixtures between processes with pytest-xdist.",
        "Use `tmp_path_factory` plus `worker_id` to compute a unique dir per worker. For shared session-level state (e.g., a DB), build it once in a session fixture and gate creation with a file-based lock to avoid races between workers."  )

    add("test",
        "Show how to use `pytest-benchmark` to track perf regressions.",
        "```python\ndef test_perf(benchmark):\n    result = benchmark(my_function, 1000)\n    assert result == 500500\n```\nThe benchmark fixture reports min/max/mean time. Save baselines and compare in CI to catch regressions early.")

    add("test",
        "Show how to test an environment variable affects behavior.",
        "```python\ndef test_debug(monkeypatch):\n    monkeypatch.setenv('DEBUG', '1')\n    assert is_debug() is True\n    monkeypatch.delenv('DEBUG')\n    assert is_debug() is False\n```\n`monkeypatch` reverts after the test, so other tests aren't polluted.")

    add("test",
        "Show how to use `mock_open` to fake reading a file.",
        "```python\nfrom unittest.mock import patch, mock_open\n\ndef test_read():\n    with patch('builtins.open', mock_open(read_data='hello')):\n        assert read_first_line('any.txt') == 'hello'\n```\nGood for unit tests where you don't want a real file. For integration use `tmp_path`.")

    add("test",
        "Show how to fail a test with a custom message.",
        "```python\ndef test_value():\n    actual = compute()\n    assert actual == 42, f'expected 42, got {actual} (intermediate={debug_state()})'\n```\nThe `, msg` form preserves all the context you need to debug from the failure log.")
