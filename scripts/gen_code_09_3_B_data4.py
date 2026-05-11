# SPDX-License-Identifier: MIT
"""Debugging / refactoring entries (~85 total)."""


def register_all(add):
    add("debug",
        "This raises 'TypeError: unhashable type: list'. What's wrong?\n\n```python\ncache = {}\ncache[[1, 2]] = 'value'\n```",
        "Lists are mutable and therefore unhashable -- you can't use them as dict keys. Use a tuple instead:\n\n```python\ncache = {}\ncache[(1, 2)] = 'value'\n```\nIf the key really needs to be a sequence, `frozenset` works for unordered, `tuple` for ordered.")

    add("debug",
        "Why does this print '[1, 2, 3]' twice?\n\n```python\ndef append_one(lst=[]):\n    lst.append(1)\n    return lst\n\nprint(append_one())\nprint(append_one())\nprint(append_one())\n```",
        "The default argument `lst=[]` is evaluated *once* at function definition, then mutated on every call. Fix:\n\n```python\ndef append_one(lst=None):\n    if lst is None:\n        lst = []\n    lst.append(1)\n    return lst\n```\nThis is the most-asked-about Python gotcha. Never use mutable defaults.")

    add("debug",
        "This script hangs forever. What's wrong?\n\n```python\nimport multiprocessing as mp\n\ndef worker(x): return x * x\n\npool = mp.Pool(4)\nresults = pool.map(worker, range(10))\nprint(results)\n```",
        "On macOS/Windows, multiprocessing requires the entry point to be guarded with `if __name__ == '__main__':`. Otherwise child processes re-execute the module top-to-bottom and spawn more children:\n\n```python\nif __name__ == '__main__':\n    with mp.Pool(4) as pool:\n        results = pool.map(worker, range(10))\n    print(results)\n```\nUse the `with` block so the pool is properly closed.")

    add("debug",
        "I get 'IndexError: list index out of range' here. Why?\n\n```python\nxs = [1, 2, 3]\nfor i in range(len(xs) + 1):\n    print(xs[i])\n```",
        "`range(len(xs) + 1)` produces 0, 1, 2, 3 but `xs` only has indices 0..2. Drop the +1:\n\n```python\nfor i in range(len(xs)):\n    print(xs[i])\n# better:\nfor x in xs:\n    print(x)\n```\nIterate the list directly when you don't need the index.")

    add("debug",
        "This async function never finishes. Why?\n\n```python\nimport asyncio\n\nasync def fetch():\n    return 42\n\nfetch()  # nothing happens\n```",
        "Calling an async function returns a coroutine object -- it doesn't run. You need to await it from another coroutine, or use `asyncio.run`:\n\n```python\nresult = asyncio.run(fetch())\nprint(result)\n```\nA bare coroutine call without `await` or `run` is almost always a bug.")

    add("debug",
        "Why is the count zero?\n\n```python\nclass Counter:\n    n = 0\n    def inc(self):\n        self.n + 1\n```",
        "`self.n + 1` evaluates and discards the result. You need an assignment:\n\n```python\nclass Counter:\n    def __init__(self):\n        self.n = 0\n    def inc(self):\n        self.n += 1\n```\nAlso move `n` into `__init__` so each instance has its own counter rather than sharing the class attribute.")

    add("debug",
        "I get a UnicodeDecodeError reading this file:\n\n```python\nwith open('data.txt') as f:\n    text = f.read()\n```",
        "The default encoding is locale-dependent and often fails on UTF-8 with non-ASCII bytes. Always pass an explicit encoding:\n\n```python\nwith open('data.txt', encoding='utf-8') as f:\n    text = f.read()\n```\nFrom Python 3.10 you can also set `PYTHONUTF8=1` in the environment to force UTF-8 mode globally.")

    add("debug",
        "This regex doesn't match what I expect:\n\n```python\nimport re\nre.match(r'\\d+', 'abc123').group()\n```",
        "`re.match` only matches at the start of the string. Use `re.search` to find anywhere:\n\n```python\nimport re\nre.search(r'\\d+', 'abc123').group()  # '123'\n```\nThe distinction is one of the most common regex gotchas.")

    add("debug",
        "Refactor this for readability:\n\n```python\ndef p(x):\n    if x > 0:\n        return True\n    else:\n        return False\n```",
        "The condition is already a bool -- return it directly:\n\n```python\ndef is_positive(x: float) -> bool:\n    return x > 0\n```\nAlso renamed the function and added a type hint.")

    add("debug",
        "Why is my list reversed in the wrong way?\n\n```python\nxs = [1, 2, 3]\nxs.reverse()\nprint(xs[::-1])\n```",
        "`list.reverse()` mutates in place; you then reverse it again with the slice, putting it back to the original order. Pick one:\n\n```python\nxs = [1, 2, 3]\nprint(xs[::-1])     # non-mutating reverse\n# or\nxs.reverse(); print(xs)  # in-place reverse\n```\nSlicing returns a new list; the method modifies the original.")

    add("debug",
        "Refactor this nested if for clarity:\n\n```python\ndef process(user):\n    if user is not None:\n        if user.active:\n            if user.email:\n                send_email(user)\n```",
        "Use early returns and combine conditions:\n\n```python\ndef process(user):\n    if user is None or not user.active or not user.email:\n        return\n    send_email(user)\n```\nGuard clauses flatten the function and put the happy path at the bottom.")

    add("debug",
        "This code is very slow. Why?\n\n```python\nresult = ''\nfor s in strings:\n    result += s\n```",
        "String concatenation in a loop is O(n^2) because Python may copy the entire result on each `+=`. Use `join`:\n\n```python\nresult = ''.join(strings)\n```\nO(n) and far faster. Same fix for bytes (`b''.join(...)`).")

    add("debug",
        "Refactor to use a comprehension:\n\n```python\nsquares = []\nfor x in range(10):\n    squares.append(x * x)\n```",
        "List comprehension is more idiomatic:\n\n```python\nsquares = [x * x for x in range(10)]\n```\nFor lazy evaluation, use a generator: `(x * x for x in range(10))`.")

    add("debug",
        "Why does my comparison return False?\n\n```python\n0.1 + 0.2 == 0.3\n```",
        "Floating-point addition isn't exact -- `0.1 + 0.2` is `0.30000000000000004`. Use `math.isclose` for floats:\n\n```python\nimport math\nmath.isclose(0.1 + 0.2, 0.3)  # True\n```\nOr the `decimal` module if you need exact decimal arithmetic.")

    add("debug",
        "Refactor this to avoid repeated dict lookups:\n\n```python\nfor i in range(100):\n    if config.get('debug'):\n        print(config.get('prefix'), i)\n```",
        "Read once outside the loop:\n\n```python\ndebug = config.get('debug')\nprefix = config.get('prefix')\nfor i in range(100):\n    if debug:\n        print(prefix, i)\n```\nDict lookups are fast but still have overhead -- hoisting them is a small win that adds up in tight loops.")

    add("debug",
        "Why does this raise 'AttributeError: 'NoneType' object has no attribute 'split''?\n\n```python\ndef get_words(s):\n    s.strip()\n    return s.split()\n\nresult = get_words(name)\n```",
        "Either `name` is `None`, or `s.strip()` was meant to mutate (it returns a new string). The fix:\n\n```python\ndef get_words(s):\n    if s is None:\n        return []\n    return s.strip().split()\n```\nAlso reassign or chain so the strip actually has effect.")

    add("debug",
        "Refactor with f-strings:\n\n```python\nname = 'alice'; age = 30\nmsg = 'Hello ' + name + ', age=' + str(age)\n```",
        "f-strings are clearer and faster:\n\n```python\nname = 'alice'; age = 30\nmsg = f'Hello {name}, age={age}'\n```\nNo type conversions needed; the formatter handles it.")

    add("debug",
        "Why is the second item missing?\n\n```python\nxs = [1, 2, 3, 4]\nfor x in xs:\n    if x == 2:\n        xs.remove(x)\n```",
        "Mutating a list while iterating shifts indices, so the iterator skips the next element. Iterate over a copy or filter:\n\n```python\nxs = [x for x in xs if x != 2]\n```\nNever modify a collection during a for-loop over it.")

    add("debug",
        "Refactor this for-else loop for clarity:\n\n```python\nfor x in xs:\n    if x < 0:\n        found = True\n        break\nelse:\n    found = False\n```",
        "`any` makes the intent obvious:\n\n```python\nfound = any(x < 0 for x in xs)\n```\nfor-else is a real Python feature but it confuses most readers.")

    add("debug",
        "Why does this test fail intermittently?\n\n```python\ndef test_order():\n    items = {'a': 1, 'b': 2}\n    assert list(items.keys()) == ['a', 'b']\n```",
        "Modern Python preserves insertion order, but if you build the dict from an unordered source (set, kwargs in 3.6, etc.) the order can vary. Compare as sets or sort first:\n\n```python\nassert sorted(items.keys()) == sorted(['a', 'b'])\n```\nFlaky tests that depend on order are technical debt -- be explicit about whether order matters.")

    add("debug",
        "Refactor this to use dataclass:\n\n```python\nclass Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def __repr__(self):\n        return f'Point(x={self.x}, y={self.y})'\n    def __eq__(self, other):\n        return self.x == other.x and self.y == other.y\n```",
        "Dataclass gives you all that for free:\n\n```python\nfrom dataclasses import dataclass\n\n@dataclass\nclass Point:\n    x: float\n    y: float\n```\nAdd `frozen=True` if you want immutability and hashability.")

    add("debug",
        "Why does this print numbers in random order?\n\n```python\nimport threading\n\nresults = []\ndef worker(i):\n    results.append(i)\n\nts = [threading.Thread(target=worker, args=(i,)) for i in range(10)]\n[t.start() for t in ts]\n[t.join() for t in ts]\nprint(results)\n```",
        "Thread scheduling is nondeterministic, so `append` order doesn't match thread spawn order. If order matters, build an output array indexed by thread id:\n\n```python\nresults = [None] * 10\ndef worker(i):\n    results[i] = i\n```\nList.append is thread-safe in CPython (GIL) but order isn't guaranteed.")

    add("debug",
        "Refactor to remove duplication:\n\n```python\nif x > 0:\n    log.info(f'positive: {x}')\n    metrics.inc('positive')\nelif x < 0:\n    log.info(f'negative: {x}')\n    metrics.inc('negative')\nelse:\n    log.info(f'zero: {x}')\n    metrics.inc('zero')\n```",
        "Compute the label first, then act once:\n\n```python\nlabel = 'positive' if x > 0 else 'negative' if x < 0 else 'zero'\nlog.info(f'{label}: {x}')\nmetrics.inc(label)\n```\nThe action structure is now data-driven and easy to extend.")

    add("debug",
        "Why does this open file leak?\n\n```python\ndef read_first_line(path):\n    return open(path).readline()\n```",
        "The file object isn't closed because there's no `with`. Use a context manager:\n\n```python\ndef read_first_line(path):\n    with open(path, encoding='utf-8') as f:\n        return f.readline()\n```\nGarbage collection eventually closes it but you can hit 'too many open files' in long-running processes.")

    add("debug",
        "This list comprehension is hard to read. Refactor it.\n\n```python\nresult = [x*y for x in range(10) for y in range(10) if x != y if x + y > 5]\n```",
        "Multi-clause comprehensions are quicker to write than to read. Break it apart:\n\n```python\ndef pairs():\n    for x in range(10):\n        for y in range(10):\n            if x == y or x + y <= 5:\n                continue\n            yield x * y\n\nresult = list(pairs())\n```\nThe explicit version is easier to debug and scales to more conditions.")

    add("debug",
        "Why doesn't this lambda capture work as expected?\n\n```python\nfns = [lambda: i for i in range(3)]\nprint([f() for f in fns])\n# expected [0, 1, 2], got [2, 2, 2]\n```",
        "The lambda closes over the variable `i`, not its value at creation. By the time you call them, `i == 2`. Use a default-argument trick:\n\n```python\nfns = [lambda i=i: i for i in range(3)]\nprint([f() for f in fns])\n```\nDefault args are evaluated at definition time, so each lambda gets its own `i`.")

    add("debug",
        "Refactor to avoid mutating the input.\n\n```python\ndef remove_negatives(xs):\n    for x in list(xs):\n        if x < 0:\n            xs.remove(x)\n    return xs\n```",
        "Return a new list and leave the input alone:\n\n```python\ndef remove_negatives(xs):\n    return [x for x in xs if x >= 0]\n```\nFunctions that mutate arguments are surprising; documenting 'this returns a new list' is friendlier.")

    add("debug",
        "I get 'RecursionError: maximum recursion depth exceeded'. Why?\n\n```python\ndef sum_range(n):\n    if n == 0: return 0\n    return n + sum_range(n - 1)\n\nsum_range(2000)\n```",
        "Python's default recursion limit is 1000. Either iterate, or raise the limit:\n\n```python\ndef sum_range(n):\n    return sum(range(n + 1))\n```\nDon't bump `sys.setrecursionlimit` unless you really need recursion -- you'll just hit a real stack overflow eventually.")

    add("debug",
        "Refactor this with `functools.reduce`:\n\n```python\ntotal = 0\nfor x in xs:\n    total += x\n```",
        "Don't -- use `sum`:\n\n```python\ntotal = sum(xs)\n```\n`reduce` is rarely the right tool when a builtin or comprehension exists. Reach for it only when accumulating a non-trivial value with no specialized helper.")

    add("debug",
        "Why does `==` not work for these dataclasses?\n\n```python\nclass A:\n    def __init__(self, x): self.x = x\n\nA(1) == A(1)  # False\n```",
        "Default `__eq__` checks identity, not contents. Use a dataclass:\n\n```python\nfrom dataclasses import dataclass\n\n@dataclass\nclass A:\n    x: int\n\nA(1) == A(1)  # True\n```\nOr write `__eq__` manually if you need custom semantics.")

    add("debug",
        "Refactor to avoid the global:\n\n```python\ncounter = 0\ndef tick():\n    global counter\n    counter += 1\n    return counter\n```",
        "Use a class or a closure:\n\n```python\nfrom itertools import count\n\nticker = count(1)\nnext(ticker)  # 1, 2, 3, ...\n```\n`itertools.count` already implements an unbounded counter without state floating in module scope.")

    add("debug",
        "Why does `json.dumps` raise on this dict?\n\n```python\nimport json\nimport datetime as dt\n\njson.dumps({'when': dt.datetime.now()})\n```",
        "Datetimes aren't natively JSON-serializable. Convert to ISO strings:\n\n```python\nimport json\nimport datetime as dt\n\njson.dumps({'when': dt.datetime.now().isoformat()})\n# or pass a custom default:\njson.dumps({'when': dt.datetime.now()}, default=str)\n```\nFor consistent serialization across a project, register a custom encoder.")

    add("debug",
        "Refactor this nested try/except.\n\n```python\ntry:\n    x = int(s)\n    try:\n        y = 1 / x\n    except ZeroDivisionError:\n        y = float('inf')\nexcept ValueError:\n    x, y = 0, 0\n```",
        "Combine the except clauses:\n\n```python\ntry:\n    x = int(s)\n    y = 1 / x\nexcept ValueError:\n    x = y = 0\nexcept ZeroDivisionError:\n    y = float('inf')\n```\nFlatter; same semantics.")

    add("debug",
        "Why does this assertion sometimes fail in production?\n\n```python\nassert user.is_admin, 'access denied'\n```",
        "Python's `-O` flag strips assertions. Never rely on `assert` for security or runtime checks; raise an explicit exception:\n\n```python\nif not user.is_admin:\n    raise PermissionError('access denied')\n```\nAsserts are for invariants you'd never expect to fail.")

    add("debug",
        "Refactor to avoid calling len() in a tight loop:\n\n```python\nfor i in range(len(xs)):\n    if i < len(xs) - 1 and xs[i] == xs[i+1]:\n        ...\n```",
        "Hoist the length and use pairwise:\n\n```python\nfrom itertools import pairwise\n\nfor a, b in pairwise(xs):\n    if a == b:\n        ...\n```\n`itertools.pairwise` (3.10+) avoids both the manual indexing and the repeated `len`.")

    add("debug",
        "Why is this faster on the second call?\n\n```python\nimport re\nfor _ in range(10000):\n    re.match(r'\\d+', '123')\n```",
        "`re` caches compiled patterns internally, but compiling once is still cleaner:\n\n```python\nimport re\nDIGITS = re.compile(r'\\d+')\nfor _ in range(10000):\n    DIGITS.match('123')\n```\nExplicit compilation also lets you reuse flags and document intent.")

    add("debug",
        "Refactor to handle exceptions with else/finally clearly.\n\n```python\ndef load(path):\n    try:\n        f = open(path)\n        data = f.read()\n        f.close()\n        return data\n    except IOError:\n        return None\n```",
        "Use a context manager and let the exception propagate up to the right boundary:\n\n```python\ndef load(path):\n    try:\n        with open(path, encoding='utf-8') as f:\n            return f.read()\n    except OSError:\n        return None\n```\n`OSError` is the modern parent of IOError in Python 3.")

    add("debug",
        "Refactor this loop to use enumerate.\n\n```python\ni = 0\nfor item in items:\n    print(i, item)\n    i += 1\n```",
        "`enumerate` is the canonical idiom:\n\n```python\nfor i, item in enumerate(items):\n    print(i, item)\n```\nUse `enumerate(items, start=1)` for 1-based numbering.")

    add("debug",
        "Why does my generator run twice and produce nothing on the second loop?\n\n```python\ngen = (x * 2 for x in range(5))\nfor x in gen: print(x)\nfor x in gen: print(x)  # nothing\n```",
        "Generators are single-pass iterators -- once exhausted they don't restart. If you need to iterate multiple times, materialize a list or wrap in a function:\n\n```python\ndef gen():\n    return (x * 2 for x in range(5))\n\nfor x in gen(): print(x)\nfor x in gen(): print(x)\n```\nA generator function gives you a fresh iterator on each call.")

    add("debug",
        "Refactor to use `pathlib`:\n\n```python\nimport os\n\npath = os.path.join('logs', '2026-01', 'app.log')\nif os.path.exists(path):\n    with open(path) as f:\n        print(f.read())\n```",
        "`pathlib` is more readable:\n\n```python\nfrom pathlib import Path\n\npath = Path('logs') / '2026-01' / 'app.log'\nif path.exists():\n    print(path.read_text(encoding='utf-8'))\n```\nThe `/` operator joins paths; `read_text` handles open/read/close.")

    add("debug",
        "Why does `sorted` give wrong order with custom objects?\n\n```python\nclass User:\n    def __init__(self, age): self.age = age\n\nsorted([User(30), User(20)])  # TypeError\n```",
        "Without `__lt__`, sorted doesn't know how to compare. Use a key:\n\n```python\nsorted(users, key=lambda u: u.age)\n# or define ordering on the class:\nfrom dataclasses import dataclass\n@dataclass(order=True)\nclass User:\n    age: int\n```\nThe key approach is preferred -- it's explicit about which field controls ordering.")

    add("debug",
        "Refactor to use `dict.get` properly.\n\n```python\nif 'k' in d:\n    v = d['k']\nelse:\n    v = 'default'\n```",
        "One line:\n\n```python\nv = d.get('k', 'default')\n```\nFor a missing key that should fall back to a freshly computed value, use `setdefault` if you also want to insert it.")

    add("debug",
        "Why is `is` failing for short strings sometimes but not others?\n\n```python\na = 'hello'\nb = 'hello'\na is b  # True\nc = 'hello world!'\nd = 'hello world!'\nc is d  # may be False\n```",
        "Python interns short, identifier-like strings. Don't rely on it. Always use `==` for value equality and reserve `is` for identity (None, True, False, sentinel objects):\n\n```python\nif a == b:\n    ...\n```\nString interning is a CPython implementation detail.")

    add("debug",
        "Refactor to remove the broad except:\n\n```python\ntry:\n    process(item)\nexcept Exception:\n    pass\n```",
        "Catch only what you handle:\n\n```python\ntry:\n    process(item)\nexcept (ValueError, KeyError) as e:\n    log.warning('skipping %s: %s', item, e)\n```\nBroad except hides bugs and breaks Ctrl+C; if you must catch everything, log the exception and re-raise.")

    add("debug",
        "Why does this dict comprehension overwrite earlier keys?\n\n```python\nd = {x % 3: x for x in range(10)}\n```",
        "Later keys overwrite earlier ones. If you want a list of values per key, use a defaultdict:\n\n```python\nfrom collections import defaultdict\nd = defaultdict(list)\nfor x in range(10):\n    d[x % 3].append(x)\n```\nThe comprehension's behavior is correct, just probably not what you wanted.")

    add("debug",
        "Refactor to use `collections.Counter`:\n\n```python\ncounts = {}\nfor x in xs:\n    if x in counts:\n        counts[x] += 1\n    else:\n        counts[x] = 1\n```",
        "One line:\n\n```python\nfrom collections import Counter\ncounts = Counter(xs)\n```\nCounter has handy methods like `most_common(k)` and arithmetic between counters.")

    add("debug",
        "Why does this fail with 'TypeError: 'int' object is not iterable'?\n\n```python\nfor x in 42:\n    print(x)\n```",
        "Integers aren't iterable. You probably meant `range(42)`:\n\n```python\nfor x in range(42):\n    print(x)\n```\nOr if you wanted to act on a single value, just use it directly without a loop.")

    add("debug",
        "Refactor to use `argparse`:\n\n```python\nimport sys\nif sys.argv[1] == '--verbose':\n    verbose = True\n```",
        "argparse handles defaults, types, and help text:\n\n```python\nimport argparse\n\np = argparse.ArgumentParser()\np.add_argument('--verbose', action='store_true')\nargs = p.parse_args()\nif args.verbose:\n    ...\n```\n`-h/--help` is generated for free.")

    add("debug",
        "Why does this fail with 'ModuleNotFoundError'?\n\n```python\n# In foo/bar.py:\nfrom utils import helper\n```",
        "Relative imports require either an explicit relative path or a fully-qualified package path:\n\n```python\nfrom .utils import helper      # if utils is a sibling module\nfrom foo.utils import helper   # if foo is a package on sys.path\n```\nMake sure your package has `__init__.py` (or is a namespace package) and that you run scripts via `python -m foo.bar`, not as a loose file.")

    add("debug",
        "Refactor to use `with contextlib.suppress`:\n\n```python\ntry:\n    os.remove(path)\nexcept FileNotFoundError:\n    pass\n```",
        "Cleaner with suppress:\n\n```python\nimport contextlib\nimport os\n\nwith contextlib.suppress(FileNotFoundError):\n    os.remove(path)\n```\nUse sparingly -- swallowing exceptions is rarely the right answer outside of cleanup paths.")

    add("debug",
        "Why is this regex slow on long inputs?\n\n```python\nimport re\nre.match(r'(a+)+b', 'a' * 30)\n```",
        "Catastrophic backtracking -- the regex tries every way to partition the a's between the inner and outer groups. Rewrite to be unambiguous:\n\n```python\nimport re\nre.match(r'a+b', 'a' * 30)\n```\nNested unbounded quantifiers like `(a+)+` should set off alarm bells.")

    add("debug",
        "Refactor to avoid manual sentinel.\n\n```python\nresult = None\nfor x in xs:\n    if predicate(x):\n        result = x\n        break\n```",
        "`next` does this in one line:\n\n```python\nresult = next((x for x in xs if predicate(x)), None)\n```\nA second argument provides the default when the iterator is exhausted.")

    add("debug",
        "Why does logging output appear twice?\n\n```python\nimport logging\nlogging.basicConfig()\nlogging.basicConfig(level=logging.INFO)\nlogging.info('hi')\n```",
        "`basicConfig` only configures the root logger if it has no handlers. The second call is a no-op, but if you've added handlers elsewhere (libraries often do), you can end up with duplicates. Fix:\n\n```python\nimport logging\nlogging.basicConfig(level=logging.INFO, force=True)\n```\n`force=True` (3.8+) clears existing handlers first.")

    add("debug",
        "Refactor this to use functools.cache:\n\n```python\n_memo = {}\ndef fib(n):\n    if n < 2: return n\n    if n in _memo: return _memo[n]\n    _memo[n] = fib(n-1) + fib(n-2)\n    return _memo[n]\n```",
        "`functools.cache` removes the boilerplate:\n\n```python\nfrom functools import cache\n\n@cache\ndef fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n```\nUnbounded; use `lru_cache(maxsize=...)` if you want to cap memory.")

    add("debug",
        "Why does my pickle load fail with `ModuleNotFoundError`?",
        "Pickle stores the fully-qualified class path. If you renamed the module or moved the class, loading breaks. Either keep a compatibility shim with the old import path, or migrate to a self-describing format like JSON or msgpack. For long-lived data, never use pickle -- it's tied to the exact code at write time.")

    add("debug",
        "Refactor to use `dict(**a, **b)` instead of `dict.update`:\n\n```python\nresult = a.copy()\nresult.update(b)\n```",
        "PEP 448 gives a one-liner:\n\n```python\nresult = {**a, **b}\n```\nor in Python 3.9+:\n\n```python\nresult = a | b\n```\nKeys in `b` win on collision -- same as `update`.")

    add("debug",
        "Why does this DataFrame chained assignment warning appear?\n\n```python\ndf[df['a'] > 0]['b'] = 1\n```",
        "`df[df['a'] > 0]` returns a copy in some cases and a view in others -- the assignment may silently affect nothing. Use `.loc`:\n\n```python\ndf.loc[df['a'] > 0, 'b'] = 1\n```\nThis is the canonical fix and pandas 3.0 will require it.")

    add("debug",
        "Refactor to avoid manual locking:\n\n```python\nimport threading\nlock = threading.Lock()\nlock.acquire()\ntry:\n    do_work()\nfinally:\n    lock.release()\n```",
        "Locks are context managers:\n\n```python\nimport threading\nlock = threading.Lock()\nwith lock:\n    do_work()\n```\nGuarantees release even on exception.")

    add("debug",
        "Why is my datetime in the wrong timezone?\n\n```python\nimport datetime as dt\nnow = dt.datetime.now()\n```",
        "`datetime.now()` returns a naive datetime in local time. For UTC and tz-aware:\n\n```python\nimport datetime as dt\nnow = dt.datetime.now(dt.UTC)\n```\nNaive datetimes cause subtle bugs at DST boundaries; always make them tz-aware in production code.")

    add("debug",
        "Refactor to use `enumerate` and `zip` instead of indices.\n\n```python\nfor i in range(len(xs)):\n    if xs[i] == ys[i]:\n        ...\n```",
        "`zip` directly:\n\n```python\nfor x, y in zip(xs, ys):\n    if x == y:\n        ...\n```\nFor strict length matching: `zip(xs, ys, strict=True)` (3.10+).")

    # ---- 5-msg follow-ups ----
    add("debug",
        "Why is my asyncio code single-threaded but still appearing to deadlock?\n\n```python\nimport asyncio\n\nasync def main():\n    loop = asyncio.get_event_loop()\n    loop.run_until_complete(other_coro())\n```",
        "You can't call `run_until_complete` from inside a running coroutine -- the loop is already running. Just `await`:\n\n```python\nasync def main():\n    await other_coro()\n```\nMixing sync 'run_X' calls with async code is one of the most common asyncio mistakes.",
        "When is run_until_complete actually appropriate?",
        "Only at the top-level entry point, before any coroutine has started. Modern code shouldn't reach for it at all -- `asyncio.run(main())` handles loop creation, running the coroutine, and clean shutdown. Reserve `run_until_complete` for cases where you're already managing a loop manually (rare, e.g. integrating with another async framework).")

    add("debug",
        "Refactor to avoid duplicate log handlers:\n\n```python\nimport logging\n\ndef setup_log(name):\n    log = logging.getLogger(name)\n    h = logging.StreamHandler()\n    log.addHandler(h)\n    return log\n```",
        "Guard against double-setup:\n\n```python\nimport logging\n\ndef setup_log(name):\n    log = logging.getLogger(name)\n    if not log.handlers:\n        log.addHandler(logging.StreamHandler())\n    return log\n```\nOr move handler setup to a single boot-time function and never call `addHandler` from library code.",
        "Why is double-setup so common?",
        "Reloading a module (Jupyter notebooks, hot-reload servers) re-executes top-level code, which re-runs your handler-attachment code. Each pass adds another handler, so logs get printed N times. The cleanest pattern: configure logging once at the application entry point, and have library modules just call `logging.getLogger(__name__)` without touching handlers.")

    add("debug",
        "Refactor to avoid the magic strings:\n\n```python\nif role == 'admin':\n    ...\nelif role == 'user':\n    ...\n```",
        "Use an `Enum` so the values are checked statically:\n\n```python\nfrom enum import StrEnum\n\nclass Role(StrEnum):\n    ADMIN = 'admin'\n    USER = 'user'\n\nif role == Role.ADMIN:\n    ...\n```\n`StrEnum` (3.11+) inherits from str so JSON/serialization just works.",
        "What if I'm on Python 3.10 and don't have StrEnum?",
        "Use the older idiom: `class Role(str, Enum): ...` -- multiple inheritance gives you the same str-comparable behavior. Or use `Enum` plus a `.value` attribute and explicit string conversion at the boundaries. Pydantic models also handle Enum-to-string serialization automatically, so if you're already using Pydantic the choice matters less.")

    add("debug",
        "Why does my decorator break introspection?\n\n```python\ndef logged(f):\n    def wrapper(*a, **k):\n        return f(*a, **k)\n    return wrapper\n```",
        "The wrapper hides the original function's name and docstring. Use `functools.wraps`:\n\n```python\nfrom functools import wraps\n\ndef logged(f):\n    @wraps(f)\n    def wrapper(*a, **k):\n        return f(*a, **k)\n    return wrapper\n```\nIDE help, `help()`, and most introspection now work as expected.",
        "Does wraps copy *everything*?",
        "By default it copies `__module__`, `__name__`, `__qualname__`, `__doc__`, `__dict__`, and `__wrapped__`. `__signature__` is also exposed via `inspect.signature` through `__wrapped__`. It does NOT copy type hints (`__annotations__`) -- those come along automatically because `__dict__` is updated. For most decorators this is sufficient. Add `assigned=` or `updated=` if you need finer control.")

    add("debug",
        "Refactor to use `match` (3.10+) instead of nested ifs.\n\n```python\nif isinstance(x, int) and x > 0:\n    print('positive int')\nelif isinstance(x, str):\n    print('string')\nelif isinstance(x, list) and len(x) == 0:\n    print('empty list')\nelse:\n    print('other')\n```",
        "Pattern matching is cleaner:\n\n```python\nmatch x:\n    case int() if x > 0:\n        print('positive int')\n    case str():\n        print('string')\n    case []:\n        print('empty list')\n    case _:\n        print('other')\n```\nGuards (`if x > 0`) and structural patterns combine well.",
        "Are there cases where match hurts readability?",
        "Yes -- for binary or trinary branching, plain ifs are clearer. Match shines when you're destructuring complex objects (parsing AST nodes, JSON shapes, message types). For shallow cases, a dict-dispatch or chained `if/elif` reads more naturally to most Python programmers.")

    add("debug",
        "Why is this exception not caught?\n\n```python\ntry:\n    open('missing.txt')\nexcept IOError:\n    print('missing')\n```",
        "Subtle: in Python 3, `IOError` is an alias for `OSError`, and `FileNotFoundError` is a subclass. The code as written should work, but if you're catching something more specific upstream you might miss it. Be explicit:\n\n```python\ntry:\n    open('missing.txt')\nexcept FileNotFoundError:\n    print('missing')\n```\nUse the most specific exception class for clarity.",
        "What's the rule of thumb for exception specificity?",
        "Catch the narrowest type that you actually know how to handle. Catching `Exception` says 'I handle all errors' which is almost never true. `OSError` is reasonable when you genuinely want to handle any I/O issue. `FileNotFoundError` says 'I have a fallback when this file is missing'. The narrower the type, the more confident future readers can be that the handler is correct.")

    add("debug",
        "Refactor to use the walrus operator:\n\n```python\nlines = []\nline = f.readline()\nwhile line:\n    lines.append(line)\n    line = f.readline()\n```",
        "PEP 572 walrus:\n\n```python\nlines = []\nwhile (line := f.readline()):\n    lines.append(line)\n```\nOr just `lines = list(f)` -- iterating a file gives lines for free.",
        "When is the walrus actually a win vs. a code smell?",
        "Win: avoiding repeated computation in conditionals (`if (m := re.match(...)): use(m)`), tightening read-loops, and cleaning up comprehensions where you'd otherwise compute the same value twice. Smell: when readers have to mentally track the assignment alongside the condition. If the walrus saves more than two lines, keep it; if it just compresses two lines into one, consider whether plain assignment would be clearer.")

    add("debug",
        "Why does my pandas operation on a slice not stick?\n\n```python\ndf2 = df[df['x'] > 0]\ndf2['y'] = df2['x'] * 2  # SettingWithCopyWarning\n```",
        "`df2` may be a view into `df`. To make a clean copy, be explicit:\n\n```python\ndf2 = df[df['x'] > 0].copy()\ndf2['y'] = df2['x'] * 2\n```\nOr use `.loc` on the original DataFrame to do the assignment in place.",
        "Why does pandas leave this ambiguous in the first place?",
        "Historical reasons -- early pandas tried to be smart about views vs copies for memory efficiency, but the behavior depended on the specific indexing path used and was confusing. Pandas 3.0 will adopt 'Copy-on-Write' as the default, eliminating the ambiguity: writes always create independent copies. For now, `.copy()` and `.loc` are the safe idioms.")

    add("debug",
        "This shows GIL contention. Refactor for true parallelism.\n\n```python\nfrom concurrent.futures import ThreadPoolExecutor\n\ndef cpu_work(x): return sum(i*i for i in range(x))\n\nwith ThreadPoolExecutor() as pool:\n    list(pool.map(cpu_work, [10**6] * 8))\n```",
        "ThreadPool doesn't help CPU-bound work because of the GIL. Use a process pool:\n\n```python\nfrom concurrent.futures import ProcessPoolExecutor\n\nif __name__ == '__main__':\n    with ProcessPoolExecutor() as pool:\n        list(pool.map(cpu_work, [10**6] * 8))\n```\nThreads are still the right answer for I/O-bound work where the GIL is released.",
        "Will Python 3.13's free-threaded mode make this obsolete?",
        "Eventually, yes -- PEP 703 introduces a no-GIL build which lets threads run CPU-bound code in parallel. Today (3.13) it's an opt-in build; many C extensions aren't yet thread-safe under it, so production code shouldn't rely on it. For the next year or two, processes remain the safe default for CPU parallelism in Python.")

    add("debug",
        "Refactor this hand-rolled iteration for clarity.\n\n```python\nresult = []\ni = 0\nwhile i < len(xs):\n    if xs[i] is not None:\n        result.append(xs[i])\n    i += 1\n```",
        "Comprehension:\n\n```python\nresult = [x for x in xs if x is not None]\n```\nClearer, faster, idiomatic. Avoid manual indexing whenever you can iterate directly.",
        "Are there cases where the manual version is preferable?",
        "Yes -- when you need lookahead (peeking at xs[i+1]) or want to alter the index based on the value, an indexed loop is fine. For most filter/map/reduce work, the comprehension or built-in function is faster (the iteration runs in C) and clearer. If readability is hurt by a too-dense comprehension, break it apart with intermediate variables; don't switch back to manual indexing.")

    add("debug",
        "Refactor this Singleton pattern.\n\n```python\nclass Config:\n    _instance = None\n    def __new__(cls):\n        if cls._instance is None:\n            cls._instance = super().__new__(cls)\n        return cls._instance\n```",
        "In Python, modules are already singletons. Use module-level state:\n\n```python\n# config.py\nDEBUG = False\nDB_URL = 'sqlite:///app.db'\n```\nimport `config` from anywhere; you get the same module object. The classic Singleton pattern is rarely needed in Python.",
        "What if I need lazy initialization?",
        "Wrap in `functools.lru_cache` or use a module-level function: `@cache\\ndef get_config(): return load_from_disk()`. The first caller pays the cost, subsequent callers get the cached result. Cleaner than a class with `__new__` magic.")

    add("debug",
        "Why does my `lru_cache` on an instance method leak memory?\n\n```python\nclass Big:\n    @lru_cache(maxsize=None)\n    def compute(self, x):\n        return x * 2\n```",
        "The cache holds a reference to `self`, so instances never get garbage collected. Use `functools.cache` on a free function or use `weakref` per-instance:\n\n```python\nfrom functools import cached_property\n\nclass Big:\n    @cached_property\n    def heavy(self):\n        return load()\n```\nFor single values per instance, `cached_property` is the right tool.",
        "What if I need a parameterized cache per instance?",
        "Move the cache into `__init__` as a per-instance dict, or use `functools.cache` inside a closure created in `__init__`. There's also `methodtools.lru_cache` (third-party) which keys correctly on a weak reference to `self`. Direct `lru_cache` on a method is almost always wrong because of the leak.")

    add("debug",
        "Refactor to avoid the deepcopy hot path.\n\n```python\nimport copy\nfor item in items:\n    work = copy.deepcopy(template)\n    work.update(item)\n```",
        "Deepcopy is slow and overkill if the template is mostly read-only. Try a shallow merge:\n\n```python\nfor item in items:\n    work = {**template, **item}\n```\nIf the template contains nested dicts that you also need to mutate independently, you do need a deeper copy -- but consider an immutable representation (tuple, frozen dataclass) instead.",
        "What's the fastest deep-copy alternative?",
        "For pure JSON-like data, `json.loads(json.dumps(x))` is often faster than `copy.deepcopy` because it avoids the visit-and-cache machinery. For more structured data (dataclasses, numpy arrays), use type-specific copy methods (`dataclasses.replace`, `arr.copy()`). Profile before assuming deepcopy is the bottleneck -- often it's the surrounding Python loops.")

    add("debug",
        "Why does this loop-based code allocate so much?\n\n```python\nresult = []\nfor x in big_list:\n    result.append(transform(x))\n    result = result[:1000]  # keep only the latest 1000\n```",
        "Slicing in a loop creates a new list each time -- O(n^2) total work. Use a deque with maxlen:\n\n```python\nfrom collections import deque\n\nresult: deque = deque(maxlen=1000)\nfor x in big_list:\n    result.append(transform(x))\n```\n`deque(maxlen=...)` automatically drops the oldest items when full.",
        "When is a deque the wrong data structure?",
        "Random access by index is O(n) on a deque (vs O(1) on a list). If you need both bounded size and random access, a list combined with explicit `del result[0]` after each append is fine but slower than deque for the rolling-window use case. For sorted bounded windows, `sortedcontainers.SortedList` is a better fit.")
