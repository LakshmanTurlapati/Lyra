# SPDX-License-Identifier: MIT
"""Wave 7 part 5: debugging / refactoring (75) + testing (50)."""


def register_all(add):
    # ----- Debug (75) -----
    add("debug",
        "This loop runs forever: `while xs: process(xs[0])`. What's wrong?",
        "You're never removing items from `xs`. The loop should be `while xs: process(xs.pop(0))` -- or better, iterate directly: `for item in xs: process(item)`. The infinite-loop pattern usually appears when you copy a `while q:` queue idiom but forget the pop.")

    add("debug",
        "I get 'RecursionError: maximum recursion depth exceeded'. How do I fix it?",
        "Either rewrite recursion as iteration (use an explicit stack), add a base case you missed, or memoize so you don't revisit the same state. Bumping `sys.setrecursionlimit` is a band-aid -- the real fix is one of the three above. Most production CPython is happy below ~1000 deep frames.")

    add("debug",
        "My function returns None unexpectedly. How do I find why?",
        "Add a print right before the return, or use `pdb.set_trace()` (or `breakpoint()`). The classic cause is forgetting `return` in a recursive call (`fn(x)` instead of `return fn(x)`). Type hints with `-> int` and `mypy --strict` catch missing returns at lint time.")

    add("debug",
        "I see 'UnboundLocalError: local variable referenced before assignment'. Why?",
        "Python decides scope at compile time. Once you write `x = ...` anywhere in a function, every read of `x` is treated as local -- even reads before that assignment. Either don't reassign the outer name, or declare `nonlocal x` (or `global x`) at the top of the function.")

    add("debug",
        "My dict.get() returns None even when the key exists. How can that be?",
        "Three usual suspects: (1) the value really is `None`; (2) you're keying by a similar-looking but different type (e.g. `'1'` vs `1`); (3) whitespace or case. Print `repr(key)` and `list(d.keys())` to compare exactly.")

    add("debug",
        "Can you refactor this `if/elif` chain into a dispatch dict?",
        "```python\nHANDLERS = {\n    'create': handle_create,\n    'update': handle_update,\n    'delete': handle_delete,\n}\n\ndef route(action: str, payload: dict):\n    handler = HANDLERS.get(action)\n    if not handler:\n        raise ValueError(f'unknown action: {action}')\n    return handler(payload)\n```\nDict dispatch is faster than a long elif and easier to extend -- new actions are one line, not a new branch.")

    add("debug",
        "My pytest test passes locally but fails in CI. How do I debug?",
        "Most likely: (1) order dependence -- run with `pytest --randomly-seed=...` locally to reproduce; (2) timezone or locale difference; (3) a flaky network call; (4) Python version mismatch. Print `sys.version` and any time-sensitive values in the failing test to compare environments.")

    add("debug",
        "I refactored a class, now `pickle.load` fails. What happened?",
        "Pickle stores the fully-qualified class name; renaming the class or moving the module breaks loading. Either: (1) re-create the data; (2) keep a compatibility shim with the old import path; or (3) switch to a forward-compatible format like JSON for anything you'll deserialize months later.")

    add("debug",
        "How do I refactor this big function into smaller pieces?",
        "Look for clusters of lines that operate on the same variables -- those become a function. Name each by what it returns or what it mutates. After extraction, the top-level function should read like an outline of the algorithm. If a sub-function has more than ~5 parameters, you probably need a small dataclass to bundle them.")

    add("debug",
        "My code throws KeyError on a deeply-nested dict. How do I make it tolerant?",
        "Use `dict.get` with defaults, or pre-validate with Pydantic so you fail fast at the boundary. For deeply nested cases: `from functools import reduce; reduce(lambda d, k: d.get(k, {}), keys, data)` walks safely. Better -- model the data as Pydantic objects and let attribute access fail loudly when a field is missing.")

    add("debug",
        "I got a 'BlockingIOError' from an async call. What's wrong?",
        "You're calling a sync function inside an async context. Wrap the call in `asyncio.to_thread(sync_fn, ...)` (3.9+) or `loop.run_in_executor`. The classic offenders are `requests`, `time.sleep`, and stdlib file IO -- replace with their async equivalents (`httpx`, `asyncio.sleep`, `aiofiles`) when possible.")

    add("debug",
        "Why is my list comprehension slower than a `for` loop here?",
        "Comprehensions are fast for transformations and filters. They lose to explicit loops when: (1) you're mutating an outer variable; (2) the body has a try/except (Python builds a comprehension scope); or (3) you're calling Python-level functions that allocate a lot. Profile with `timeit` -- intuition lies more than the numbers.")

    add("debug",
        "Refactor: this function takes 8 arguments. How do I clean it up?",
        "Group related args into a dataclass: turn `(host, port, user, password, ssl, timeout, retries, pool_size)` into `ConnectionConfig(...)`. Single-purpose dataclasses also make tests cleaner -- you mock or build a config rather than juggle 8 positional values.")

    add("debug",
        "How do I track down a memory leak in a long-running script?",
        "1) `tracemalloc.start()` and snapshot every minute; 2) compare snapshots with `compare_to(prev, 'lineno')` to see growing allocators; 3) for Cython/C extensions, `objgraph.show_growth()` shows new live objects per type. Common culprits: unbounded caches, modules with module-level lists, callbacks holding closures.")

    add("debug",
        "Refactor this nested loop into a generator pipeline.",
        "Replace each step with a generator function and chain them: `for x in source(): for y in expand(x): for z in filter_(y): yield process(z)`. Pipelines are easier to test (each stage is a function) and stream-friendly (no intermediate lists). `itertools.chain.from_iterable` collapses one level of nesting.")

    add("debug",
        "My exception is being swallowed. How do I find where?",
        "Search for `except:` (bare) and `except Exception:` without `raise`. Add a `logger.exception(...)` to every catch site. Better: refactor so only the outermost layer catches generic exceptions, and only after logging the traceback. Bare `except` is almost always wrong -- it eats KeyboardInterrupt too.")

    add("debug",
        "I keep getting 'TypeError: X object is not iterable' on a single value. What am I doing wrong?",
        "Probably a function that sometimes returns a list and sometimes a single item. Standardize on always returning a list (even of length 1). The Python adage: 'if it might be a sequence, make it a sequence'. Defensive callers can do `[x] if not isinstance(x, list) else x`, but it's better to fix the producer.")

    add("debug",
        "Refactor: my class has a 200-line __init__. What should I do?",
        "Break it into staged factory methods: `MyThing.from_config(cfg)`, `MyThing.from_database(...)`, etc. Each factory does one parsing/loading job. Or extract a builder class that constructs the value step-by-step. The constructor should mostly do field assignment.")

    add("debug",
        "Why does `is` sometimes return False for equal small integers?",
        "Python caches small ints (-5 to 256), but only for some construction paths. `1000 is 1000` may be True at the REPL and False elsewhere. The rule: `is` checks identity, not equality. Use `==` for value comparisons; reserve `is` for `None`, `True`, `False`, and intentional singleton checks.")

    add("debug",
        "How do I refactor this code that opens files but never closes them?",
        "Wrap every open call in `with`: `with open(path) as f: ...` closes even on exceptions. For collections of files, use `contextlib.ExitStack` to manage a dynamic number. Long-lived handles? Move to a class with `__enter__`/`__exit__` so callers always pair acquire and release.")

    add("debug",
        "Refactor this code that uses global mutable state.",
        "Convert globals into class attributes or pass them through. If a function genuinely needs shared state, encapsulate it: `class Cache: def __init__(self): self._d = {}`. Globals make tests order-dependent and hide dependencies. The exception: real constants (uppercase names, immutable values) -- those are fine.")

    add("debug",
        "My test passes when run alone but fails in the suite. Why?",
        "State leak between tests. Look for: module-level mutable values, env vars set but not torn down, monkeypatches that escape fixture scope, shared database rows. Use `pytest --randomly` to find order dependencies fast. Convert leaky test setups into fixtures with proper teardown.")

    add("debug",
        "How do I make this slow regex faster?",
        "1) Compile it once with `re.compile`; 2) anchor it (`^foo` is dramatically faster than `foo`); 3) avoid backtracking traps -- replace `(.*)+` with `[^x]*`; 4) for fixed strings use `str.find` or `in`, not regex. For very hot paths consider `re2` (no backtracking) via the `google-re2` package.")

    add("debug",
        "Refactor this function so it's testable without the database.",
        "Split it: pure logic in one function, IO in another. The pure function takes data and returns data (no `db.session.query`); the IO wrapper does the fetch and calls the pure one. Tests cover the logic without a database; integration tests cover the wrapper.")

    add("debug",
        "I'm seeing different behavior in Python 3.10 vs 3.12. How do I bisect?",
        "1) Read the 'What's New' between versions for breaking changes; 2) run `python -W error` to surface DeprecationWarnings as exceptions; 3) for binary-search style bisection, try 3.11 first to halve the search space. Common 3.12 surprises: stricter `int(str)` whitespace handling, removed `distutils`.")

    add("debug",
        "How do I trace where a value is being mutated?",
        "Wrap it in a class that overrides `__setattr__`/`__setitem__` and logs the caller via `traceback.format_stack()`. Or use `gc.get_referrers(obj)` to find what's holding it. For dicts, subclass `dict` with a logging `__setitem__` and use that during debugging.")

    add("debug",
        "Refactor: the same try/except block is repeated in every method.",
        "Hoist it into a decorator: `@retry_on_db_error` wraps each method. Or use a single pre-request middleware/decorator at the framework level. Repeated try/except is a code smell -- the cross-cutting concern wants to live in one place.")

    add("debug",
        "My production logs show no traceback for a crash. Why?",
        "`logger.error('something failed')` with no `exc_info`/`exception()` doesn't include the traceback. Either call `logger.exception(...)` inside the except, or pass `exc_info=True` to `logger.error`. Better: install `sentry-sdk` so unhandled exceptions are captured automatically.")

    add("debug",
        "How do I find the slowest function in a script?",
        "`python -m cProfile -o out.prof script.py` then `snakeviz out.prof` for an interactive flame graph. For finer detail use `line_profiler` (`@profile` decorator + `kernprof -l`). Don't optimize without measuring -- the bottleneck is rarely where you'd guess.")

    add("debug",
        "Refactor this dict comprehension that's hard to read.",
        "If the key/value expressions are complex, hoist them into helper functions: `{key_for(item): value_for(item) for item in xs}`. If you're filtering AND transforming, two passes (one filter, one map) often read better than a single one-liner with both.")

    add("debug",
        "Why is `dict.update` slower than I expect on a big merge?",
        "`update` is O(n) on the right-hand side, but if you call it in a loop you get O(n*m). Use `{**a, **b, **c}` for a single combined merge, or `collections.ChainMap` if you don't need a real merge -- it's O(1) and gives lookup-time fallback.")

    add("debug",
        "How do I refactor a function that has too many return points?",
        "Sometimes more returns are clearer (early exits for invalid input). When they obscure the flow: store the result in a variable and return at the end, or factor the validation into its own function so the main body has one return path.")

    add("debug",
        "My JSON write produces an ImportError on `datetime` objects. How do I serialize them?",
        "`json.dumps(obj, default=str)` is the quick fix. Better: `json.dumps(obj, default=lambda o: o.isoformat() if hasattr(o, 'isoformat') else str(o))` so dates round-trip cleanly. For complex models, switch to Pydantic or `orjson` -- both handle datetime natively.")

    add("debug",
        "Refactor: my function uses `sys.exit()` deep inside. How should I clean it up?",
        "Raise a custom exception (`class CommandError(Exception): pass`) that the entry point catches and converts to an exit code. Library code should never call `sys.exit` -- it makes the function untestable and unusable from other scripts.")

    add("debug",
        "How do I tell if my script is leaking file descriptors?",
        "On Linux: `lsof -p <pid> | wc -l` -- watch it grow over time. In tests, `psutil.Process().num_fds()` before and after to assert no leak. Common cause: `open()` without `with`, or closing the wrapper but not the underlying socket.")

    add("debug",
        "My multiprocessing pool hangs. What might be wrong?",
        "Common causes: (1) the worker function imports something that re-runs code at import time; (2) the worker function references a non-picklable closure; (3) the pool is created inside `if __name__ == '__main__'` block but the worker function isn't at module top level. Make worker functions top-level and pickle-safe.")

    add("debug",
        "Refactor: the same dict literal appears in 12 tests.",
        "Extract a fixture or factory: `def make_user(**overrides): return {'id': 1, 'name': 'Test', **overrides}`. Tests call `make_user(name='Bob')` and only the relevant differences appear. `factory_boy` automates this for SQLAlchemy/Django models.")

    add("debug",
        "Why is my `assert` statement not running?",
        "Python's `-O` flag strips assertions. If your production runner uses `-O` or `PYTHONOPTIMIZE=1`, every assert is skipped. Don't put load-bearing checks (auth, validation) inside `assert` -- use `if not x: raise ValueError(...)` for runtime checks.")

    add("debug",
        "How do I refactor this monolithic Flask view?",
        "Extract the request parsing into a Pydantic model. Move the business logic into a service-layer function that takes the model and returns a result. The view then becomes: parse -> call service -> serialize. Each piece is testable in isolation.")

    add("debug",
        "Why is `for k in d` iterating in a different order than I expect?",
        "Python 3.7+ guarantees insertion order for dicts. If you're seeing different order, you might be: (1) constructing the dict from an unordered source (set, kwargs in older versions); (2) running on Python 3.6 where ordering was an implementation detail. If you need a specific sort, do it explicitly with `sorted(d.items())`.")

    add("debug",
        "Refactor this function that mutates its input list.",
        "Copy at the top: `xs = list(xs)`. Document that the function takes any iterable and returns a new list. Mutating arguments is a footgun -- callers don't expect it, especially when the type hint is `list[int]`.")

    add("debug",
        "My script suddenly produces nondeterministic output. Why?",
        "Look for: (1) `set` iteration (use `sorted()`); (2) `os.listdir` (sort it); (3) `dict.keys()` from a JSON load on Python 3.6; (4) `random` without a seed; (5) `time.time` in test paths. Add a `--seed` flag and pass it to every PRNG you use.")

    add("debug",
        "How do I refactor this code that uses `eval` on user input?",
        "Don't. Replace with `ast.literal_eval` for safe parsing of literals, or `json.loads` for JSON. For expression evaluation use a real expression library (`asteval`, `simpleeval`). `eval` on user input is a remote-code-execution vulnerability waiting to happen.")

    add("debug",
        "I see 'ResourceWarning: unclosed file' in tests. How do I fix?",
        "Find the open() that isn't using `with`. Run tests with `-W error::ResourceWarning` to turn the warning into a failure that points to the line. In long-running code, audit any class that holds a file handle -- it likely needs `__enter__`/`__exit__` or an explicit `close()`.")

    add("debug",
        "Refactor this `if x is not None` ladder into something cleaner.",
        "Use `and` short-circuiting: `value = obj and obj.field and obj.field.subfield`. For deep attribute walks, write a helper: `getattr_chain(obj, 'a.b.c', default=None)`. If you're querying a Pydantic model, accessing missing fields raises -- handle at the boundary, not via defensive checks at every read.")

    add("debug",
        "Why does my numpy code give different results from pure Python?",
        "Floating-point order matters: `(a + b) + c` may differ from `a + (b + c)` at the last bit. numpy reorders for SIMD; pure Python doesn't. For exact reproducibility either accept the tolerance or compute in a fixed order with `math.fsum`.")

    add("debug",
        "Refactor: my decorator stack is hard to debug.",
        "Apply `@functools.wraps(fn)` inside every decorator so `__name__` and `__doc__` are preserved. Add `print` statements during development; remove via a `DEBUG` flag. For complex cases use `decorator` package which preserves signatures faithfully.")

    add("debug",
        "How do I find which import is taking 10 seconds at startup?",
        "`python -X importtime script.py 2>import.log` prints a tree of import durations. The biggest leaf is your culprit. Common offenders: `pandas`, `tensorflow`, anything that auto-discovers plugins. Lazy-import those inside the functions that need them.")

    add("debug",
        "Refactor this function that has too many type-checking branches.",
        "Use `singledispatch` from `functools`: register one implementation per type, and the right one is called automatically. For runtime data, `match`/`case` (3.10+) reads better than nested isinstance checks.")

    add("debug",
        "Why does my SQLAlchemy session lose changes?",
        "You probably forgot `session.commit()`. `flush` writes SQL but stays in transaction; `commit` makes it permanent. Also: if you raise an exception, the session may auto-rollback in a context manager. Add explicit `session.commit()` in every write path and log when it runs.")

    add("debug",
        "Refactor: this function returns a dict with 15 fields.",
        "Replace with a dataclass or Pydantic model. Callers get attribute access (`result.user_id`) and IDE autocomplete. If the dict varies (some keys sometimes absent), switch to Optional fields rather than mixing keys-may-be-missing dict patterns.")

    add("debug",
        "I'm getting 'TypeError: Object of type X is not JSON serializable'. How?",
        "JSON only handles `dict, list, tuple, str, int, float, bool, None`. For custom objects, write a `default=` function or use `dataclasses.asdict`. For numpy: `obj.tolist()` or convert to native Python types first. For datetime: `o.isoformat()`. For Decimals: `str(o)` and parse on the other side.")

    add("debug",
        "How do I refactor this code that's swallowing all exceptions?",
        "Replace `except Exception:` (or worse, bare `except:`) with the specific exceptions you actually expect. Log unexpected ones with `logger.exception` and re-raise. Bare excepts hide bugs and make on-call painful -- the stack trace tells you what to fix.")

    add("debug",
        "Why does my asyncio.gather raise just one exception when several tasks failed?",
        "By default, `gather` raises the first exception it sees and cancels the rest. Pass `return_exceptions=True` to get back a list with each result or exception, then handle them yourself. For partial-success patterns this is essential.")

    add("debug",
        "Refactor: I have 50 lines of argparse boilerplate.",
        "Switch to `click` (decorators) or `typer` (uses type hints). Both reduce a 50-line argparse setup to ~10 lines and produce nicer help text. For very simple CLIs, `argparse` is still fine -- but once it grows, the maintenance cost flips.")

    add("debug",
        "How do I debug a flaky test that fails 1 in 50 runs?",
        "Loop it: `for i in {1..200}; do pytest -k flaky_test || break; done`. Add a fixture that captures and saves the random state. Once you've reproduced, look for: time-of-day code, race conditions in async, network calls, ordering assumptions. Determinism is non-negotiable for reliable CI.")

    add("debug",
        "My type hints don't catch a bug at runtime. Why not?",
        "Python type hints are not enforced -- they're documentation that mypy/pyright check statically. Run `mypy --strict` in CI to catch what runtime won't. For runtime enforcement use Pydantic models at boundaries; inside the function body, mypy is the right tool.")

    add("debug",
        "Refactor this code that uses string concatenation for SQL.",
        "Use parameterized queries: `cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))`. Or move to SQLAlchemy Core/ORM. Concatenating user input into SQL is the textbook SQL injection -- never do it, ever, even 'just for an internal tool'.")

    add("debug",
        "Why is my `requests.get` slow on the first call?",
        "DNS lookup + TCP + TLS handshake all happen once. Subsequent calls reuse the connection if you use a `Session`: `s = requests.Session(); s.get(url)`. For hot paths to a single host, sessions cut latency by 50-90%.")

    add("debug",
        "Refactor this large dict-of-dicts config.",
        "Convert to nested Pydantic models. Validation runs at load time, types are checked, and IDE autocomplete works. Optional fields with defaults make migration painless. If you really need dict access, `model.model_dump()` returns the original dict.")

    add("debug",
        "How do I track down which test is leaving the database in a bad state?",
        "Run with `pytest --randomly-seed=N` to fix the order, then bisect by deselecting halves: `pytest --deselect path/to/test::ID`. Or wrap each test in a savepoint that rolls back on teardown -- `pytest-postgresql` and similar fixtures handle this.")

    add("debug",
        "Refactor: I'm passing the same logger everywhere.",
        "Each module should call `logger = logging.getLogger(__name__)` at the top. The hierarchy gives you per-module log levels for free. Don't pass loggers as arguments -- it adds noise and ties callers to logging implementation details.")

    add("debug",
        "Why does `id(x) == id(y)` return True for objects that should be different?",
        "`id()` returns a memory address, but Python recycles addresses after objects are garbage-collected. `id(temp1) == id(temp2)` can be True for sequentially created throwaway objects. Use `is` for identity, but really -- prefer `==` and only reach for identity when comparing to `None`, `True`, `False`.")

    add("debug",
        "Refactor this code that has a bunch of `if foo: x = 1; else: x = 2` patterns.",
        "Use ternary: `x = 1 if foo else 2`. For more complex branches, `dict.get` works: `x = mapping.get(key, default)`. If you're picking from many cases based on type, `match`/`case` (Python 3.10+) is the cleanest pattern.")

    add("debug",
        "How do I find a memory growth bug across a long-running service?",
        "1) `tracemalloc.start(25)` (deep stack); 2) on a timer, snapshot and compare to baseline with `compare_to(baseline, 'lineno')`; 3) export the top 20 lines that grew. Pair with a heap dump tool like `pympler.asizeof` for occasional deep dives.")

    add("debug",
        "Why does my multi-threaded counter give wrong results?",
        "The GIL doesn't make `counter += 1` atomic -- it's read-modify-write. Use `threading.Lock` around increments, or `itertools.count()` (the iterator's `next` is thread-safe), or `collections.Counter` with a lock, or skip threads and use `multiprocessing` for CPU-bound work.")

    add("debug",
        "Refactor this code so the side effects are testable.",
        "Inject the dependencies that cause the side effects: a `clock`, a `db`, a `mailer`. Tests pass fakes/spies; production passes real implementations. The inversion makes side effects explicit at the call site -- often the bug is that you didn't realize a function was doing IO at all.")

    add("debug",
        "Why does my `is None` check trigger a Pylint warning?",
        "Probably you wrote `x == None` -- Pylint warns to use `is None` because `==` may be overridden (e.g. on a numpy array, `arr == None` returns a boolean array, not a bool). `is None` is identity, can't be overridden, and is the official idiom.")

    add("debug",
        "Refactor this code that opens the same connection on every call.",
        "Connection pooling: keep a single connection (or pool) at module level and reuse. For SQLAlchemy that's the engine; for redis it's `redis.Redis()`; for HTTP it's a `Session` or `httpx.Client`. Per-call connections destroy throughput.")

    add("debug",
        "How do I find what's importing a deprecated module?",
        "`python -W error::DeprecationWarning script.py` turns the warning into an exception, with a traceback pointing to the offending import. Or `python -X tracemalloc=25 -W default` to see the full stack of where any warning originates.")

    add("debug",
        "Refactor: my function has duplicated 'if config.debug: print' lines.",
        "Replace with proper logging at DEBUG level: `logger = logging.getLogger(__name__); logger.debug('msg')`. Configure the level once at startup. Logging is structured, levelable, and can go to multiple sinks; print-debugging in production code is a smell.")

    add("debug",
        "Refactor: my function has 6 boolean parameters.",
        "Replace booleans with an enum or a small dataclass of options. `do_thing(verbose=True, dry_run=False, force=True, ...)` becomes `do_thing(Options(verbose=True, force=True))`. Boolean parameter explosion is also a sign the function is doing too much -- consider splitting.")

    add("debug",
        "Why is my unicode string showing escape sequences when printed?",
        "You're probably looking at `repr()` output (interactive mode shows it for non-print returns) instead of `str()`. Print explicitly with `print(s)` to see the rendered form. For nested structures, `pprint` handles unicode correctly; `print(json.dumps(d, ensure_ascii=False))` is good for dicts.")

    add("debug",
        "How do I detect the slowest SQL query in my app?",
        "Enable SQLAlchemy's query logging via `echo=True` on the engine, or hook into the `before_cursor_execute` and `after_cursor_execute` events to time each query. For Postgres, `pg_stat_statements` aggregates by normalized query -- the source of truth in production.")

    add("debug",
        "Refactor this code that calls `len(x) > 0` everywhere.",
        "In Python, empty containers are falsy: `if x:` is the idiom for 'is non-empty'. Reserve `len(x) > 0` for cases where length itself is the meaningful threshold. Same with `if x is not None and len(x) > 0` -- if `None` is allowed, `if x:` covers both checks at once (with the intended semantics).")

    # ----- Test (50) -----
    add("test",
        "Write a pytest fixture that yields a temp directory.",
        "```python\nimport pytest\nfrom pathlib import Path\n\n@pytest.fixture\ndef tmp_dir(tmp_path: Path) -> Path:\n    return tmp_path\n```\n`tmp_path` is a built-in fixture that gives a unique pathlib.Path per test and auto-cleans. Wrap it if you want to add custom prep.")

    add("test",
        "Implement a parametrized pytest test for a math function.",
        "```python\nimport pytest\n\n@pytest.mark.parametrize('a, b, expected', [\n    (1, 2, 3),\n    (-1, 1, 0),\n    (0, 0, 0),\n])\ndef test_add(a, b, expected):\n    assert add(a, b) == expected\n```\nEach row becomes its own test ID; failures point to the exact row.")

    add("test",
        "Write a pytest test that mocks an HTTP call.",
        "```python\nimport responses\nimport requests\n\n@responses.activate\ndef test_fetch_user():\n    responses.add(responses.GET, 'https://api.example.com/user/1',\n                  json={'name': 'Ada'}, status=200)\n    user = fetch_user(1)\n    assert user['name'] == 'Ada'\n```\n`responses` decorates the test; calls inside that match the registered URLs return the canned response.")

    add("test",
        "Implement a pytest fixture that creates a fresh SQLite DB per test.",
        "```python\nimport pytest\nfrom sqlalchemy import create_engine\nfrom myapp.models import Base\n\n@pytest.fixture\ndef db():\n    engine = create_engine('sqlite:///:memory:')\n    Base.metadata.create_all(engine)\n    yield engine\n    engine.dispose()\n```\nIn-memory SQLite gives perfect isolation between tests. For Postgres-specific features, use `pytest-postgresql` instead.")

    add("test",
        "Write a pytest test that asserts a function raises a specific exception.",
        "```python\nimport pytest\n\ndef test_division_by_zero():\n    with pytest.raises(ZeroDivisionError, match='division by zero'):\n        divide(1, 0)\n```\nThe `match=` arg checks the error message via regex -- catches accidental message regressions.")

    add("test",
        "Implement a pytest fixture with module scope.",
        "```python\nimport pytest\n\n@pytest.fixture(scope='module')\ndef expensive_resource():\n    resource = build_resource()\n    yield resource\n    resource.close()\n```\nModule scope means built once per test file. Use sparingly -- shared state across tests can introduce order dependence.")

    add("test",
        "Write a pytest test using monkeypatch to override an env var.",
        "```python\ndef test_with_debug_env(monkeypatch):\n    monkeypatch.setenv('DEBUG', '1')\n    assert is_debug() is True\n```\n`monkeypatch` undoes the change at test teardown; safer than mutating `os.environ` directly.")

    add("test",
        "Implement a pytest test that mocks a function with unittest.mock.",
        "```python\nfrom unittest.mock import patch\n\n@patch('myapp.module.fetch_data')\ndef test_process(mock_fetch):\n    mock_fetch.return_value = {'value': 42}\n    assert process() == 42\n    mock_fetch.assert_called_once()\n```\nPatch where the function is *used*, not where it's defined -- a classic gotcha.")

    add("test",
        "Write a pytest test that captures stdout.",
        "```python\ndef test_print_output(capsys):\n    print('hello')\n    captured = capsys.readouterr()\n    assert captured.out == 'hello\\n'\n```\n`capsys` returns a tuple-like with `.out` and `.err`. Use `capfd` instead if you need to capture file descriptor output (subprocess, C extensions).")

    add("test",
        "Implement a pytest test that asserts a warning is raised.",
        "```python\nimport pytest\n\ndef test_deprecated_function():\n    with pytest.warns(DeprecationWarning, match='use new_func'):\n        old_func()\n```\nPair with `python -W error::DeprecationWarning` in CI to make sure new code doesn't silently use the deprecated path.")

    add("test",
        "Write a pytest fixture that creates and destroys a Docker container.",
        "```python\nimport pytest\nimport docker\n\n@pytest.fixture(scope='session')\ndef redis_container():\n    client = docker.from_env()\n    container = client.containers.run('redis:7', detach=True, ports={'6379/tcp': None})\n    try:\n        yield container\n    finally:\n        container.remove(force=True)\n```\nFor production use `testcontainers-python` which handles ports, health checks, and cleanup automatically.")

    add("test",
        "Implement a hypothesis property test.",
        "```python\nfrom hypothesis import given, strategies as st\n\n@given(st.lists(st.integers()))\ndef test_sort_idempotent(xs):\n    assert sorted(sorted(xs)) == sorted(xs)\n```\nProperty tests find edge cases you didn't think of -- empty lists, tuples with NaNs, huge integers.")

    add("test",
        "Write a pytest test that uses freeze_time.",
        "```python\nfrom freezegun import freeze_time\n\n@freeze_time('2024-01-01 12:00:00')\ndef test_today_is_jan_first():\n    assert today_iso() == '2024-01-01'\n```\nFreezing time makes any code path that calls `datetime.now()` deterministic. Better: inject a clock parameter and avoid `freezegun` altogether.")

    add("test",
        "Implement a pytest test for an async function.",
        "```python\nimport pytest\n\n@pytest.mark.asyncio\nasync def test_fetch_data():\n    result = await fetch_data()\n    assert result['status'] == 'ok'\n```\nNeeds `pytest-asyncio`. Set `asyncio_mode = 'auto'` in `pyproject.toml` to skip the marker on every test.")

    add("test",
        "Write a pytest test that mocks an HTTP API with httpx.",
        "```python\nimport respx\nimport httpx\n\n@respx.mock\ndef test_fetch_user():\n    respx.get('https://api.example.com/user/1').respond(json={'name': 'Ada'})\n    user = fetch_user(1)\n    assert user.name == 'Ada'\n```\n`respx` is to `httpx` what `responses` is to `requests`.")

    add("test",
        "Implement a pytest fixture that mocks a Redis client.",
        "```python\nimport pytest\nfrom fakeredis import FakeRedis\n\n@pytest.fixture\ndef redis():\n    return FakeRedis(decode_responses=True)\n```\n`fakeredis` is a pure-Python Redis simulator -- supports most commands and runs entirely in-process.")

    add("test",
        "Write a pytest test that checks log output.",
        "```python\nimport logging\n\ndef test_logs_warning(caplog):\n    with caplog.at_level(logging.WARNING):\n        do_thing()\n    assert 'unexpected value' in caplog.text\n```\n`caplog.at_level` ensures the test sees messages even if root logger is set higher.")

    add("test",
        "Implement a pytest configuration that runs slow tests on demand.",
        "```python\n# conftest.py\nimport pytest\n\ndef pytest_addoption(parser):\n    parser.addoption('--slow', action='store_true', help='run slow tests')\n\ndef pytest_collection_modifyitems(config, items):\n    if config.getoption('--slow'):\n        return\n    skip_slow = pytest.mark.skip(reason='need --slow')\n    for item in items:\n        if 'slow' in item.keywords:\n            item.add_marker(skip_slow)\n```\nMark tests with `@pytest.mark.slow` and run them with `pytest --slow` only.")

    add("test",
        "Write a pytest test that asserts a function is idempotent.",
        "```python\nfrom hypothesis import given, strategies as st\n\n@given(st.dictionaries(st.text(), st.integers()))\ndef test_normalize_idempotent(d):\n    once = normalize(d)\n    twice = normalize(once)\n    assert once == twice\n```\nRunning twice should equal running once -- a strong property to encode.")

    add("test",
        "Implement a pytest fixture that uses a factory pattern.",
        "```python\nimport pytest\n\n@pytest.fixture\ndef make_user():\n    def _make(**overrides):\n        defaults = {'id': 1, 'name': 'Test', 'email': 't@example.com'}\n        return defaults | overrides\n    return _make\n\ndef test_admin(make_user):\n    admin = make_user(role='admin')\n    assert admin['role'] == 'admin'\n```\nFactories let each test specify only what differs from the default.")

    add("test",
        "Write a pytest test that uses parametrize with ids.",
        "```python\nimport pytest\n\n@pytest.mark.parametrize('n, expected', [\n    pytest.param(0, 1, id='zero'),\n    pytest.param(5, 120, id='five'),\n    pytest.param(10, 3628800, id='ten'),\n])\ndef test_factorial(n, expected):\n    assert factorial(n) == expected\n```\nNamed IDs make `pytest -k zero` work and improve failure readability.")

    add("test",
        "Implement a pytest test using temporary file fixture.",
        "```python\ndef test_writes_csv(tmp_path):\n    out = tmp_path / 'out.csv'\n    write_csv(out, [{'a': 1}])\n    assert out.read_text() == 'a\\n1\\n'\n```\n`tmp_path` is per-test; use `tmp_path_factory` for session-scoped temps.")

    add("test",
        "Write a pytest test that validates Pydantic model behaviour.",
        "```python\nimport pytest\nfrom pydantic import ValidationError\nfrom myapp.schemas import User\n\ndef test_user_requires_email():\n    with pytest.raises(ValidationError) as exc:\n        User(name='Ada')\n    assert any(e['loc'] == ('email',) for e in exc.value.errors())\n```\nAssert on the structured error rather than the message text -- robust to phrasing changes.")

    add("test",
        "Implement a coverage configuration for pytest.",
        "```toml\n# pyproject.toml\n[tool.coverage.run]\nbranch = true\nsource = ['myapp']\nomit = ['*/tests/*', '*/__main__.py']\n\n[tool.coverage.report]\nfail_under = 80\nshow_missing = true\nexclude_lines = ['pragma: no cover', 'raise NotImplementedError']\n```\n`branch = true` reports both branch sides; `fail_under` makes CI fail on regression.")

    add("test",
        "Write a pytest test using subTests for similar checks.",
        "```python\nimport pytest\n\ndef test_validation_cases(subtests):\n    for value, expected in [(0, False), (1, True), (-1, False)]:\n        with subtests.test(value=value):\n            assert is_positive(value) == expected\n```\nNeeds `pytest-subtests`. Useful when failures are correlated and you want to see all of them.")

    add("test",
        "Implement a pytest fixture for a httpx test client.",
        "```python\nimport pytest\nimport httpx\nfrom myapp import app\n\n@pytest.fixture\nasync def client():\n    transport = httpx.ASGITransport(app=app)\n    async with httpx.AsyncClient(transport=transport, base_url='http://test') as c:\n        yield c\n```\nReuse across all FastAPI integration tests.")

    add("test",
        "Write a pytest test that checks for float equality.",
        "```python\nimport pytest\n\ndef test_compute_pi_approximation():\n    assert compute_pi(10000) == pytest.approx(3.14159, abs=1e-3)\n```\n`pytest.approx` handles floats, lists, dicts, and numpy arrays -- much better than custom epsilon math.")

    add("test",
        "Implement a pytest test using context manager fixtures.",
        "```python\nimport pytest\n\n@pytest.fixture\ndef opened_file(tmp_path):\n    path = tmp_path / 'data.txt'\n    path.write_text('hello')\n    with path.open() as f:\n        yield f\n```\nThe `with` block ensures cleanup even if the test raises.")

    add("test",
        "Write a pytest configuration for parallel testing with xdist.",
        "```toml\n# pyproject.toml\n[tool.pytest.ini_options]\naddopts = '-n auto --dist worksteal'\n```\n`-n auto` uses one worker per CPU; `worksteal` redistributes when fast workers finish first. Run with `pip install pytest-xdist`.")

    add("test",
        "Implement a pytest test for command-line scripts.",
        "```python\nimport subprocess\nimport sys\n\ndef test_cli_outputs_version():\n    result = subprocess.run([sys.executable, '-m', 'myapp', '--version'],\n                            capture_output=True, text=True, check=True)\n    assert 'myapp 1.0' in result.stdout\n```\nFor click/typer apps, use their built-in `runner.invoke` -- avoids subprocess overhead.")

    add("test",
        "Write a pytest test for a function with random behaviour.",
        "```python\nimport random\n\ndef test_shuffle_preserves_set(monkeypatch):\n    rng = random.Random(42)\n    monkeypatch.setattr('myapp.module.random', rng)\n    xs = [1, 2, 3]\n    shuffled = shuffle_list(xs)\n    assert sorted(shuffled) == [1, 2, 3]\n```\nInject a seeded RNG so the test is reproducible. Better -- accept the RNG as a parameter so monkeypatching isn't needed.")

    add("test",
        "Implement a pytest fixture that provides a mock SMTP server.",
        "```python\nimport pytest\nfrom aiosmtpd.controller import Controller\nfrom aiosmtpd.handlers import Sink\n\n@pytest.fixture\ndef smtp_server():\n    handler = Sink()\n    controller = Controller(handler, hostname='localhost', port=1025)\n    controller.start()\n    yield ('localhost', 1025)\n    controller.stop()\n```\nLocal SMTP listener swallows messages without sending; pair with a handler that records for assertion.")

    add("test",
        "Write a pytest test that uses a snapshot.",
        "```python\nfrom syrupy.assertion import SnapshotAssertion\n\ndef test_render_template(snapshot: SnapshotAssertion):\n    output = render('user_card', {'name': 'Ada'})\n    assert output == snapshot\n```\nNeeds `pytest-syrupy`. Snapshots auto-update with `--snapshot-update`; review the diff before committing.")

    add("test",
        "Implement a pytest test that asserts no warnings raised.",
        "```python\nimport warnings\n\ndef test_no_warnings():\n    with warnings.catch_warnings():\n        warnings.simplefilter('error')\n        my_function()\n```\nTurns any warning into a test failure -- great for catching DeprecationWarnings before they become errors.")

    add("test",
        "Write a pytest test with class-based grouping.",
        "```python\nimport pytest\n\nclass TestUserCreation:\n    @pytest.fixture\n    def user_data(self):\n        return {'name': 'Ada', 'email': 'ada@example.com'}\n\n    def test_creates_user(self, user_data):\n        u = create_user(user_data)\n        assert u.name == 'Ada'\n\n    def test_rejects_invalid_email(self, user_data):\n        with pytest.raises(ValueError):\n            create_user({**user_data, 'email': 'invalid'})\n```\nClass-scoped fixtures are shared across all methods in the class.")

    add("test",
        "Implement a pytest test with an autouse fixture.",
        "```python\nimport pytest\n\n@pytest.fixture(autouse=True)\ndef reset_state():\n    yield\n    GlobalRegistry.clear()\n```\n`autouse=True` runs the fixture for every test in scope without explicit injection -- great for cleanup.")

    add("test",
        "Write a pytest test that uses tmp_path_factory for session.",
        "```python\nimport pytest\n\n@pytest.fixture(scope='session')\ndef shared_dir(tmp_path_factory):\n    path = tmp_path_factory.mktemp('shared')\n    seed_data(path)\n    return path\n```\nOne expensive setup, shared across all tests in the run.")

    add("test",
        "Implement a pytest plugin via conftest.py.",
        "```python\n# conftest.py\nimport pytest\n\n@pytest.fixture\ndef api_client():\n    return APIClient(base_url='http://test')\n\ndef pytest_collection_modifyitems(config, items):\n    for item in items:\n        if 'integration' in item.nodeid:\n            item.add_marker(pytest.mark.integration)\n```\n`conftest.py` is auto-discovered; no setup required.")

    add("test",
        "Write a pytest test for a Flask app using test_client.",
        "```python\nimport pytest\nfrom myapp import create_app\n\n@pytest.fixture\ndef client():\n    app = create_app('config.Test')\n    with app.test_client() as c:\n        yield c\n\ndef test_index(client):\n    r = client.get('/')\n    assert r.status_code == 200\n```\nApp factory + `test_client` is the canonical Flask testing setup.")

    add("test",
        "Implement a pytest test with a custom marker.",
        "```python\n# conftest.py\ndef pytest_configure(config):\n    config.addinivalue_line('markers', 'slow: mark test as slow')\n\n# test_file.py\nimport pytest\n\n@pytest.mark.slow\ndef test_huge_dataset():\n    ...\n```\nRegister markers to avoid the 'unknown marker' warning. Run only slow tests: `pytest -m slow`.")

    add("test",
        "Write a pytest test that validates a CLI app with click.",
        "```python\nfrom click.testing import CliRunner\nfrom myapp.cli import cli\n\ndef test_greet():\n    runner = CliRunner()\n    result = runner.invoke(cli, ['greet', '--name', 'Ada'])\n    assert result.exit_code == 0\n    assert 'Hello, Ada' in result.output\n```\nClick's runner is fast and gives full output capture.")

    add("test",
        "Implement a pytest test that monkeypatches a class method.",
        "```python\ndef test_with_fake_clock(monkeypatch):\n    monkeypatch.setattr('myapp.module.Clock.now',\n                        classmethod(lambda cls: datetime(2024, 1, 1)))\n    assert get_today() == '2024-01-01'\n```\n`monkeypatch.setattr` reverts at teardown; safe alternative to mutating classes directly.")

    add("test",
        "Write a pytest test that uses pytest-mock.",
        "```python\ndef test_with_mocker(mocker):\n    mock_db = mocker.patch('myapp.module.db')\n    mock_db.fetch.return_value = {'id': 1}\n    assert get_user(1)['id'] == 1\n    mock_db.fetch.assert_called_once_with(1)\n```\n`mocker` is `unittest.mock.patch` with automatic teardown -- much cleaner than nested context managers.")

    add("test",
        "Implement a pytest test with a session-scoped fixture and cleanup.",
        "```python\nimport pytest\n\n@pytest.fixture(scope='session')\ndef shared_resource(request):\n    res = SharedResource()\n    request.addfinalizer(res.close)\n    return res\n```\n`addfinalizer` runs at the end of the scope; equivalent to `yield` form but useful when you can't restructure as a generator.")

    add("test",
        "Write a pytest test that covers async context managers.",
        "```python\nimport pytest\n\n@pytest.mark.asyncio\nasync def test_db_session():\n    async with db_session() as session:\n        result = await session.execute('SELECT 1')\n        assert result.scalar() == 1\n```\nThe context manager handles open/close; the test focuses on the work.")

    add("test",
        "Implement a pytest test for a function that uses requests.",
        "```python\nimport pytest\nimport responses\n\n@responses.activate\ndef test_fetch_users():\n    responses.add(responses.GET, 'https://api.example.com/users',\n                  json=[{'id': 1}], status=200)\n    users = fetch_users()\n    assert len(users) == 1\n    assert responses.calls[0].request.url == 'https://api.example.com/users'\n```\nVerify both the result and that the right URL was hit.")

    add("test",
        "Write a pytest test that captures and asserts on log records.",
        "```python\nimport logging\n\ndef test_warns_on_retry(caplog):\n    with caplog.at_level(logging.WARNING, logger='myapp'):\n        retry_request()\n    records = [r for r in caplog.records if 'retry' in r.message.lower()]\n    assert len(records) == 3\n```\nFilter by logger name to isolate from noisy dependencies.")

    add("test",
        "Implement a pytest test using a database transaction rollback.",
        "```python\nimport pytest\nfrom sqlalchemy.orm import Session\n\n@pytest.fixture\ndef db_session(engine):\n    connection = engine.connect()\n    transaction = connection.begin()\n    session = Session(bind=connection)\n    try:\n        yield session\n    finally:\n        session.close()\n        transaction.rollback()\n        connection.close()\n```\nEvery test runs in a savepoint and rolls back -- perfect isolation, no schema reset between tests.")

    add("test",
        "Write a pytest test that checks for resource cleanup with weakref.",
        "```python\nimport gc\nimport weakref\n\ndef test_no_lingering_references():\n    obj = MyClass()\n    ref = weakref.ref(obj)\n    del obj\n    gc.collect()\n    assert ref() is None\n```\nCatches accidental retention via global lists, caches, or callbacks.")

    add("test",
        "Implement a pytest fixture that sets up FastAPI dependency overrides.",
        "```python\nimport pytest\nfrom fastapi.testclient import TestClient\nfrom myapp import app\nfrom myapp.deps import get_db\n\n@pytest.fixture\ndef client(test_db):\n    app.dependency_overrides[get_db] = lambda: test_db\n    yield TestClient(app)\n    app.dependency_overrides.clear()\n```\nSwap real dependencies for test fakes without touching production code.")
