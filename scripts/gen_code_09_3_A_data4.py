# SPDX-License-Identifier: MIT
"""Debugging / refactoring entries (75 total)."""


def register_all(add):
    # ---- 3-msg (45) ----
    add("debug",
        "This code raises 'TypeError: list indices must be integers'. Why?\n```python\ndata = ['a', 'b', 'c']\nprint(data['0'])\n```",
        "You're indexing with the string `'0'` instead of the integer `0`. Lists only accept integer indices and slices.\n```python\ndata = ['a', 'b', 'c']\nprint(data[0])\n```\nIf the index comes from user input, convert with `int()` first.")

    add("debug",
        "Why does this print None?\n```python\ndef add_one(x):\n    x = x + 1\n\nprint(add_one(5))\n```",
        "The function never returns the result, so it implicitly returns `None`. Add an explicit `return`.\n```python\ndef add_one(x):\n    return x + 1\n\nprint(add_one(5))  # 6\n```\nA common gotcha when refactoring from in-place mutation to a pure function.")

    add("debug",
        "This loop only modifies the first element. Why?\n```python\nnums = [1, 2, 3]\nfor n in nums:\n    n *= 2\nprint(nums)\n```",
        "`n` is just a binding to each value. Reassigning it doesn't touch the list. Mutate by index instead.\n```python\nnums = [1, 2, 3]\nfor i in range(len(nums)):\n    nums[i] *= 2\nprint(nums)  # [2, 4, 6]\n```\nOr build a new list: `nums = [n * 2 for n in nums]`.")

    add("debug",
        "This raises 'IndexError: list index out of range'. What's wrong?\n```python\nxs = [1, 2, 3]\nfor i in range(1, len(xs) + 1):\n    print(xs[i])\n```",
        "The range goes from 1 to `len(xs)` inclusive, so `i` reaches 3 — past the last valid index of 2. Use 0-based indexing.\n```python\nxs = [1, 2, 3]\nfor i in range(len(xs)):\n    print(xs[i])\n```\nBetter: `for x in xs: print(x)` — Python iterates directly.")

    add("debug",
        "Why does this give 'AttributeError: 'NoneType' object has no attribute 'lower''?\n```python\ndef normalize(s):\n    s.lower()\n    return s\n\nprint(normalize(None).lower())\n```",
        "Two issues. First, `s.lower()` doesn't mutate; you need to assign. Second, you're passing `None`, which has no `.lower()`. Guard against it.\n```python\ndef normalize(s):\n    if s is None:\n        return ''\n    return s.lower()\n```\nReturn an empty string (or raise) on `None` rather than letting it propagate.")

    add("debug",
        "This sometimes fails with 'KeyError: 'name''. Why?\n```python\ndef greet(user):\n    return f\"Hello, {user['name']}\"\n```",
        "The 'name' key isn't always present. Use `.get` with a default.\n```python\ndef greet(user):\n    return f\"Hello, {user.get('name', 'friend')}\"\n```\nOr validate the input upfront — `user['name']` is appropriate when 'name' is a hard requirement.")

    add("debug",
        "This raises 'ZeroDivisionError'. Fix it.\n```python\ndef average(xs):\n    return sum(xs) / len(xs)\n```",
        "Empty list divides by zero. Handle the edge case.\n```python\ndef average(xs):\n    if not xs:\n        return 0.0\n    return sum(xs) / len(xs)\n```\nReturning 0 is one choice; raising `ValueError('empty')` is another. Pick what makes sense for your callers.")

    add("debug",
        "Why does this overwrite my list?\n```python\ndef extend(default=[]):\n    default.append(1)\n    return default\n\nprint(extend())  # [1]\nprint(extend())  # [1, 1] -- bug!\n```",
        "Mutable default arguments are evaluated once at function definition. Use `None` as a sentinel.\n```python\ndef extend(default=None):\n    if default is None:\n        default = []\n    default.append(1)\n    return default\n```\nThis is the most-cited Python pitfall.")

    add("debug",
        "This raises 'UnboundLocalError'. Why?\n```python\ncount = 0\n\ndef tick():\n    count += 1\n\ntick()\n```",
        "Assigning to `count` inside `tick` makes it a local variable, but it has no value yet. Declare `global` or pass it in.\n```python\ncount = 0\n\ndef tick():\n    global count\n    count += 1\n```\nFor most cases, prefer returning the new value rather than mutating module state.")

    add("debug",
        "This regex doesn't match 'Hello World'. Why?\n```python\nimport re\nprint(re.match(r'World', 'Hello World'))\n```",
        "`re.match` anchors at the start of the string. Use `re.search` if the match can be anywhere.\n```python\nimport re\nprint(re.search(r'World', 'Hello World'))\n```\nThis catches a lot of folks coming from JavaScript or Perl.")

    add("debug",
        "Why does this comparison return False?\n```python\nprint(0.1 + 0.2 == 0.3)\n```",
        "Floating-point arithmetic is approximate. `0.1 + 0.2 == 0.30000000000000004`. Use a tolerance.\n```python\nimport math\nprint(math.isclose(0.1 + 0.2, 0.3))\n```\nFor financial work, use `decimal.Decimal` to avoid the issue entirely.")

    add("debug",
        "This file isn't being closed properly. Refactor it.\n```python\nf = open('data.txt')\ncontents = f.read()\nprocess(contents)\nf.close()\n```",
        "If `process` raises, `f.close()` never runs. Use a context manager.\n```python\nwith open('data.txt') as f:\n    contents = f.read()\nprocess(contents)\n```\n`with` guarantees the file closes whether the block succeeds or raises.")

    add("debug",
        "This exception swallows useful context. Improve it.\n```python\ntry:\n    parse(s)\nexcept Exception:\n    raise ValueError('bad input')\n```",
        "Chain the original exception so you don't lose the traceback.\n```python\ntry:\n    parse(s)\nexcept Exception as e:\n    raise ValueError('bad input') from e\n```\n`from e` records the cause; without it the original error is hidden as 'During handling of...'.")

    add("debug",
        "This is_admin check is broken. Why?\n```python\nis_admin = user.role == 'admin' or 'superuser'\n```",
        "Operator precedence and truthiness make `'superuser'` the result whenever `role != 'admin'` (since the string is truthy). Use a tuple membership test.\n```python\nis_admin = user.role in ('admin', 'superuser')\n```\n`x == a or b` is the most common version of this bug.")

    add("debug",
        "This async code never prints. What's wrong?\n```python\nimport asyncio\n\nasync def hello():\n    print('hi')\n\nhello()\n```",
        "Calling an async function doesn't run it — it returns a coroutine. You have to schedule it on an event loop.\n```python\nimport asyncio\n\nasync def hello():\n    print('hi')\n\nasyncio.run(hello())\n```\nFor scripts, `asyncio.run` is the standard entry point.")

    add("debug",
        "This shows duplicate log lines. Fix it.\n```python\nimport logging\n\ndef setup():\n    logger = logging.getLogger('app')\n    handler = logging.StreamHandler()\n    logger.addHandler(handler)\n    return logger\n\nlog = setup()\nlog = setup()\nlog.info('hello')\n```",
        "Calling `setup` twice attaches two handlers to the same logger. Guard against it or only configure once.\n```python\nimport logging\n\ndef setup():\n    logger = logging.getLogger('app')\n    if logger.handlers:\n        return logger\n    logger.addHandler(logging.StreamHandler())\n    return logger\n```\nLoggers are global by name, so subsequent `getLogger('app')` calls return the same object.")

    add("debug",
        "This recursive function blows the stack on long lists. Refactor.\n```python\ndef sum_list(xs):\n    if not xs:\n        return 0\n    return xs[0] + sum_list(xs[1:])\n```",
        "Slicing creates a new list every recursion (O(n^2) memory) and Python's recursion limit caps you around 1000. Use a loop or `sum()`.\n```python\ndef sum_list(xs):\n    return sum(xs)\n```\nIf you need to do something more than sum, use a loop with an accumulator.")

    add("debug",
        "This can deadlock. Why?\n```python\nimport threading\n\nlock = threading.Lock()\n\ndef work():\n    with lock:\n        with lock:\n            print('hi')\n```",
        "A standard `Lock` is non-reentrant — acquiring it twice in the same thread blocks forever. Use `RLock` if you genuinely need re-entry.\n```python\nimport threading\n\nlock = threading.RLock()\n\ndef work():\n    with lock:\n        with lock:\n            print('hi')\n```\nUsually the better fix is to not nest locks at all.")

    add("debug",
        "This logs the wrong line number. Why?\n```python\nimport logging\n\ndef log_msg(msg):\n    logging.info(msg)\n\nlog_msg('hello')\n```",
        "The default formatter shows `log_msg`'s frame. Use `stacklevel=2` to point at the caller.\n```python\nimport logging\n\ndef log_msg(msg):\n    logging.info(msg, stacklevel=2)\n```\nOr just call `logging.info` directly at the call site.")

    add("debug",
        "This dictionary lookup is slow. Speed it up.\n```python\ndef contains(items, target):\n    return target in [item.id for item in items]\n```",
        "Building a list each call is O(n). Build a set once.\n```python\ndef contains(items, target):\n    return target in {item.id for item in items}\n```\nIf `contains` is called many times with the same `items`, hoist the set outside.")

    add("debug",
        "This breaks when items contain commas. Fix it.\n```python\ndef parse_csv(line):\n    return line.split(',')\n```",
        "`split(',')` doesn't handle quoted fields. Use the `csv` module.\n```python\nimport csv\nimport io\n\ndef parse_csv(line):\n    return next(csv.reader(io.StringIO(line)))\n```\nIt handles quoting, escaped quotes, and newlines properly.")

    add("debug",
        "This date comparison is backwards.\n```python\nfrom datetime import date\nif date.today() < '2023-01-01':\n    print('before')\n```",
        "You're comparing a `date` object to a string. Compare two `date`s.\n```python\nfrom datetime import date\nif date.today() < date(2023, 1, 1):\n    print('before')\n```\nPython 3 forbids comparing different types via `<`/`>`.")

    add("debug",
        "This decorator drops the function's name. Fix it.\n```python\ndef trace(fn):\n    def wrapper(*a, **kw):\n        print(f'calling {fn.__name__}')\n        return fn(*a, **kw)\n    return wrapper\n\n@trace\ndef hello():\n    pass\n\nprint(hello.__name__)  # 'wrapper'\n```",
        "Use `functools.wraps` to preserve metadata.\n```python\nfrom functools import wraps\n\ndef trace(fn):\n    @wraps(fn)\n    def wrapper(*a, **kw):\n        print(f'calling {fn.__name__}')\n        return fn(*a, **kw)\n    return wrapper\n```\nThis also preserves docstrings and signatures, which tools like Sphinx need.")

    add("debug",
        "This subprocess call hangs. Why?\n```python\nimport subprocess\nsubprocess.run(['cat', 'huge.log'])\n```",
        "If `cat`'s output overflows the OS pipe buffer (typically 64KB), the process blocks waiting for someone to read. Capture or redirect.\n```python\nimport subprocess\nresult = subprocess.run(['cat', 'huge.log'], capture_output=True, text=True)\n```\nOr stream with `Popen.stdout` if the output is huge.")

    add("debug",
        "This is_truthy gives wrong results.\n```python\ndef is_truthy(value):\n    if value == True:\n        return True\n    return False\n```",
        "`==` is overly strict — `1 == True` is True, but `'hello' == True` is False even though it's truthy. Use `bool()` or just the value.\n```python\ndef is_truthy(value):\n    return bool(value)\n```\nIn most cases you don't even need the function — `if value:` is the idiom.")

    add("debug",
        "This is_palindrome handles only ASCII. Fix it for unicode.\n```python\ndef is_palindrome(s):\n    return s == s[::-1]\n```",
        "Slice-reverse works for unicode code points but not for graphemes (e.g., emoji with skin-tone modifiers). For most uses, normalize and lowercase first.\n```python\nimport unicodedata\n\ndef is_palindrome(s):\n    n = unicodedata.normalize('NFC', s.casefold())\n    return n == n[::-1]\n```\nFor full grapheme correctness, use the `regex` module's `\\X` to iterate graphemes.")

    add("debug",
        "This raises 'json.decoder.JSONDecodeError'. What's wrong?\n```python\nimport json\nprint(json.loads(\"{'name': 'alice'}\"))\n```",
        "JSON requires double-quoted strings; that's a Python dict literal. If you need to parse Python-style data, use `ast.literal_eval`.\n```python\nimport ast\nprint(ast.literal_eval(\"{'name': 'alice'}\"))\n```\nOr fix the source to be valid JSON: `'{\"name\": \"alice\"}'`.")

    add("debug",
        "This list comprehension uses too much memory.\n```python\ntotal = sum([x**2 for x in range(10**8)])\n```",
        "Switch to a generator expression — no intermediate list.\n```python\ntotal = sum(x**2 for x in range(10**8))\n```\nDropping the brackets turns it into a generator. `sum`, `min`, `max`, `any`, `all` all accept generators.")

    add("debug",
        "Why does this print 'after' before 'before'?\n```python\nimport asyncio\n\nasync def main():\n    print('before')\n    await asyncio.sleep(1)\n    print('after')\n\nasyncio.run(main())\nprint('top')\n```",
        "Looks fine actually — but if you're seeing odd interleaving, it's because you're calling `asyncio.run` from inside an event loop already. Use `await main()` from within an existing async context.\n```python\n# In a Jupyter cell or another async context:\nawait main()\n```\nDon't nest `asyncio.run` calls — they each spin up a fresh event loop.")

    add("debug",
        "This ints fail to compare correctly:\n```python\nprint('10' > '9')  # False!\n```",
        "You're comparing strings, which use lexicographic order. Convert to int.\n```python\nprint(int('10') > int('9'))  # True\n```\nFor numeric data coming from CSVs, convert columns at parse time rather than at comparison.")

    add("debug",
        "This recursive function returns None instead of the result.\n```python\ndef find(tree, target):\n    if tree['value'] == target:\n        return tree\n    for child in tree.get('children', []):\n        find(child, target)\n```",
        "The recursive call's result is discarded. Return it when found.\n```python\ndef find(tree, target):\n    if tree['value'] == target:\n        return tree\n    for child in tree.get('children', []):\n        result = find(child, target)\n        if result is not None:\n            return result\n    return None\n```\nThe missing 'plumbing' of recursive returns is the most common bug in tree-walking code.")

    add("debug",
        "Refactor this nested if-else into something cleaner.\n```python\ndef get_label(score):\n    if score >= 90:\n        return 'A'\n    else:\n        if score >= 80:\n            return 'B'\n        else:\n            if score >= 70:\n                return 'C'\n            else:\n                return 'F'\n```",
        "Drop the `else` after each `return` (early-return style).\n```python\ndef get_label(score):\n    if score >= 90:\n        return 'A'\n    if score >= 80:\n        return 'B'\n    if score >= 70:\n        return 'C'\n    return 'F'\n```\nNo dangling indentation, easier to add new bands later.")

    add("debug",
        "This check is broken for empty strings:\n```python\ndef has_content(s):\n    return s != None\n```",
        "Two issues. Use `is not None` for None checks, and consider whether empty strings should count.\n```python\ndef has_content(s):\n    return s is not None and s != ''\n```\nOr more idiomatically: `bool(s)` returns False for both `None` and `''`.")

    add("debug",
        "Why does this not raise on a missing key?\n```python\nsettings = {'mode': 'fast'}\nvalue = settings.get('missing')\nprint(value.upper())  # AttributeError\n```",
        "`get` returns `None` by default for missing keys. Either provide a sane default or check explicitly.\n```python\nsettings = {'mode': 'fast'}\nvalue = settings.get('missing', '')\nprint(value.upper())\n```\nOr use `settings['missing']` if the key truly must exist (raises `KeyError`).")

    add("debug",
        "This eats CPU forever. Why?\n```python\nimport time\n\nwhile True:\n    if check_done():\n        break\n```",
        "There's no sleep, so it spins as fast as it can. Add a small delay.\n```python\nimport time\n\nwhile not check_done():\n    time.sleep(0.1)\n```\n100ms keeps responsiveness while dropping CPU usage to near zero.")

    add("debug",
        "This iteration mutates the list and skips items.\n```python\nxs = [1, 2, 3, 4, 5]\nfor x in xs:\n    if x % 2 == 0:\n        xs.remove(x)\nprint(xs)  # [1, 3, 5] -- but is unsafe\n```",
        "Modifying a list during iteration is undefined behavior. Build a new list.\n```python\nxs = [1, 2, 3, 4, 5]\nxs = [x for x in xs if x % 2 != 0]\n```\nSometimes you'll get lucky; other times you'll silently skip elements.")

    add("debug",
        "This boolean is hard to read. Refactor.\n```python\nif (x is not None and x != '' and x != 0 and x != [] and x != {}):\n    process(x)\n```",
        "All those checks reduce to truthiness.\n```python\nif x:\n    process(x)\n```\nIn Python, `None`, `0`, `''`, `[]`, `{}` are all falsy. The original code is `if x:` written verbosely.")

    add("debug",
        "Refactor this 100-line function. (Pseudocode)\n```python\ndef process(data):\n    # parse...\n    # validate...\n    # transform...\n    # save...\n```",
        "Extract each phase into its own function so each is testable and composable.\n```python\ndef process(data):\n    parsed = parse(data)\n    validate(parsed)\n    transformed = transform(parsed)\n    save(transformed)\n```\nThe rule of thumb: a comment that introduces a section is a hint that section should be a function.")

    add("debug",
        "This function has too many parameters. Group them.\n```python\ndef create_user(name, email, age, country, city, postal, role, active, plan):\n    ...\n```",
        "Use a dataclass.\n```python\nfrom dataclasses import dataclass\n\n@dataclass\nclass UserSpec:\n    name: str\n    email: str\n    age: int\n    country: str\n    city: str\n    postal: str\n    role: str\n    active: bool = True\n    plan: str = 'free'\n\ndef create_user(spec: UserSpec):\n    ...\n```\nCallers can construct, mutate, and pass `UserSpec` clearly. Defaults move into one place.")

    add("debug",
        "This function does two things. Split it.\n```python\ndef get_and_save_users(url, path):\n    users = requests.get(url).json()\n    with open(path, 'w') as f:\n        json.dump(users, f)\n    return users\n```",
        "Separation of concerns: fetching, persisting, and returning.\n```python\nimport json\nimport requests\n\ndef get_users(url):\n    return requests.get(url, timeout=10).json()\n\ndef save_users(users, path):\n    with open(path, 'w') as f:\n        json.dump(users, f)\n\n# At the call site:\nusers = get_users(url)\nsave_users(users, path)\n```\nNow each function is independently testable.")

    add("debug",
        "Why is this slow on large lists?\n```python\nresult = []\nfor x in big_list:\n    if x not in result:\n        result.append(x)\n```",
        "Membership test on a list is O(n), making the whole thing O(n^2). Use a set or `dict.fromkeys`.\n```python\nresult = list(dict.fromkeys(big_list))\n```\nO(n) time and preserves insertion order (Python 3.7+).")

    add("debug",
        "This always prints zeros. Why?\n```python\nimport random\nrandom.seed(0)\nfor _ in range(3):\n    random.seed(0)\n    print(random.randint(0, 100))\n```",
        "Re-seeding the RNG inside the loop makes it deterministic from the same point each time. Seed once.\n```python\nimport random\nrandom.seed(0)\nfor _ in range(3):\n    print(random.randint(0, 100))\n```\nSeed at the start of a run for reproducibility, never inside loops.")

    add("debug",
        "This raises 'RecursionError'. What's the fix?\n```python\ndef factorial(n):\n    return n * factorial(n - 1)\n```",
        "There's no base case — recursion never stops. Add one.\n```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n```\nFor large `n`, Python's recursion limit will still kick in. Use a loop or `math.factorial` instead.")

    add("debug",
        "Why does the cache not work between calls?\n```python\ndef expensive(x):\n    cache = {}\n    if x in cache:\n        return cache[x]\n    result = compute(x)\n    cache[x] = result\n    return result\n```",
        "`cache` is created fresh on every call. Move it outside, or use `functools.lru_cache`.\n```python\nfrom functools import lru_cache\n\n@lru_cache(maxsize=None)\ndef expensive(x):\n    return compute(x)\n```\nFar simpler and thread-safe.")

    # ---- 5-msg with follow-ups (30) ----
    add("debug",
        "This raises 'ValueError: invalid literal for int()'. Why?\n```python\nx = int('3.14')\n```",
        "`int()` parses integers, not floats. Convert through `float` first if the input has a decimal.\n```python\nx = int(float('3.14'))  # 3\n```\nOr use `int('3.14'.split('.')[0])` if you want only the integer part with no rounding.",
        "What if the string is sometimes scientific notation like '1e3'?",
        "```python\ndef parse_int(s: str) -> int:\n    return int(float(s))\n\nprint(parse_int('1e3'))   # 1000\nprint(parse_int('3.14'))  # 3\n```\nGoing through `float` handles `'1e3'`, `'3.14'`, and even `'inf'` (which would then raise `OverflowError` on the int cast — usually what you want).")

    add("debug",
        "This dict comprehension fails:\n```python\nresult = {k: v for k, v in [1, 2, 3]}\n```",
        "You're trying to unpack each integer into two variables. Either fix the data or rethink the comprehension.\n```python\nresult = {k: v for k, v in [(1, 'a'), (2, 'b'), (3, 'c')]}\n```\nThe input must be an iterable of 2-tuples (or any 2-element iterable).",
        "What if I want to use index as key and value as value?",
        "```python\nresult = {i: v for i, v in enumerate([1, 2, 3])}\n# {0: 1, 1: 2, 2: 3}\n```\n`enumerate` pairs each element with its index — exactly what's needed for the indexed dict.")

    add("debug",
        "This file path doesn't work on Windows.\n```python\npath = 'C:\\users\\bob\\data.txt'\n```",
        "`\\u` is interpreted as a unicode escape. Either use raw strings or forward slashes.\n```python\npath = r'C:\\users\\bob\\data.txt'  # raw\npath = 'C:/users/bob/data.txt'    # forward slashes work on Windows too\n```\nForward slashes are accepted by Windows APIs and avoid the escape gotcha entirely.",
        "What's the most portable approach?",
        "```python\nfrom pathlib import Path\n\npath = Path('C:/users/bob/data.txt')\nprint(path)  # auto-renders correctly per OS\n```\n`pathlib` abstracts away the separator entirely. Construct with `/` operator: `Path('C:') / 'users' / 'bob' / 'data.txt'`.")

    add("debug",
        "Why does this print '10' instead of 10?\n```python\nfrom collections import Counter\nc = Counter('hello')\nprint(c['l'])\n```",
        "Actually that prints `2` correctly. But if you're seeing string output, your variable was a `Counter` of strings of digits. Show me the actual code.\n```python\n# Confirmed correct behavior:\nfrom collections import Counter\nc = Counter('hello')\nprint(c['l'])  # 2\n```\nDouble-check that the upstream code isn't running `str()` on the count somewhere.",
        "OK I lied — what I really meant: this prints '00' when I expected 0.\n```python\nprint(0 + '0')\n```",
        "That actually raises `TypeError: unsupported operand types`. The output `'00'` comes from string concatenation in JavaScript-land thinking. In Python:\n```python\nprint(int('0') + 0)   # 0\nprint(str(0) + '0')   # '00'\n```\nPython is strict about types here, which is a feature.")

    add("debug",
        "This works but is too slow:\n```python\nresult = ''\nfor x in items:\n    result += str(x)\n```",
        "String concatenation in a loop is O(n^2) — each `+=` builds a new string. Use `join`.\n```python\nresult = ''.join(str(x) for x in items)\n```\nO(n) and idiomatic.",
        "And if I want a separator between items?",
        "```python\nresult = ', '.join(str(x) for x in items)\n```\n`str.join` takes the separator from the string it's called on. The classic way to build comma-separated lists.")

    add("debug",
        "This raises a 'ModuleNotFoundError'. The package is installed.\n```python\nimport mypackage\n```",
        "You probably installed it in a different Python environment than the one running. Check:\n```python\nimport sys\nprint(sys.executable)\n```\nThen `pip install` against that exact interpreter: `<that python> -m pip install mypackage`.",
        "I'm using a venv but still hit this — what next?",
        "Make sure the venv is activated *and* your IDE/notebook is using its interpreter:\n```bash\nsource .venv/bin/activate\nwhich python  # should point inside .venv\npython -m pip list | grep mypackage\n```\nIn VSCode/Jupyter, explicitly select the venv's Python via the interpreter picker — IDEs often default to the system one.")

    add("debug",
        "This is_admin bug bit us in production:\n```python\nis_admin = (user.role == 'admin' or 'superuser')\n```",
        "Operator precedence: `('admin' or 'superuser')` is just `'admin'` (truthy). Then `user.role == 'admin'`, which is correct for admins, but the `or 'superuser'` is dead code. Use a tuple.\n```python\nis_admin = user.role in ('admin', 'superuser')\n```\nThe original 'works' for admin role but never matches 'superuser'.",
        "How would I add tests so this doesn't happen again?",
        "```python\nimport pytest\n\ndef test_is_admin():\n    assert is_admin(User(role='admin'))\n    assert is_admin(User(role='superuser'))\n    assert not is_admin(User(role='user'))\n    assert not is_admin(User(role='guest'))\n```\nA parametrized test forcing every relevant role to be exercised — the `superuser` case would have caught the bug instantly.")

    add("debug",
        "This shows duplicate items:\n```python\nfor item in items:\n    print(item)\n    process(item)\n```",
        "Looks fine alone. If you're seeing duplicates, it's because `items` already has duplicates or `process` re-adds. Show me upstream code.\n```python\nfrom collections import Counter\nprint(Counter(items))  # which item appears multiple times?\n```\nDiagnose the data first.",
        "Counter shows each item appears twice. Let me skip dupes.",
        "```python\nseen = set()\nfor item in items:\n    if item in seen:\n        continue\n    seen.add(item)\n    print(item)\n    process(item)\n```\nFor hashable items, a `set` of seen ids is the cleanest dedupe. If items are dicts, key on a stable id field.")

    add("debug",
        "This crashes after a few iterations with 'RuntimeError: dictionary changed size during iteration'.\n```python\nfor key in mydict:\n    if mydict[key] is None:\n        del mydict[key]\n```",
        "You can't mutate a dict while iterating its keys. Iterate a snapshot.\n```python\nfor key in list(mydict):\n    if mydict[key] is None:\n        del mydict[key]\n```\n`list(mydict)` materializes the keys before the loop begins.",
        "Alternative without copying the keys?",
        "```python\nmydict = {k: v for k, v in mydict.items() if v is not None}\n```\nReassigning a freshly-built dict is often cleaner than mutating in place. Same memory cost, more readable.")

    add("debug",
        "This compares lists wrong:\n```python\nlist_a = [1, 2, 3]\nlist_b = [3, 2, 1]\nprint(list_a == list_b)  # False, but I want True\n```",
        "Lists are ordered; equality is element-wise. If order doesn't matter, compare as sorted lists or sets.\n```python\nprint(sorted(list_a) == sorted(list_b))     # True\nprint(set(list_a) == set(list_b))            # True (loses duplicates info)\n```\nUse sorted for multisets (counts matter), set for set semantics.",
        "What if duplicates matter?",
        "```python\nfrom collections import Counter\nprint(Counter(list_a) == Counter(list_b))  # True, with duplicate counts\n```\n`Counter` equality checks both presence and multiplicity — the right answer for multisets.")

    add("debug",
        "Why does this not call the override?\n```python\nclass Base:\n    def __init__(self):\n        self.greet()\n    def greet(self):\n        print('hello from base')\n\nclass Sub(Base):\n    def greet(self):\n        print('hello from sub')\n\nSub()  # prints 'hello from sub'\n```",
        "Actually it does call the subclass override — Python uses dynamic dispatch even from `__init__`. Output is `'hello from sub'`. Are you sure?\n```python\n# As written, output is:\n# hello from sub\n```\nIf you're seeing 'hello from base', check that `Sub.greet` is actually defined (no typo) and that you're instantiating `Sub`, not `Base`.",
        "Then how would I prevent the override from being called during init in the parent?",
        "```python\nclass Base:\n    def __init__(self):\n        Base.greet(self)  # explicit base-class call\n    def greet(self):\n        print('hello from base')\n```\nReferring to `Base.greet` bypasses the MRO, so subclass overrides won't intercept. This is sometimes called the 'unbound method' technique.")

    add("debug",
        "This regex matches too much:\n```python\nimport re\nprint(re.findall(r'<.+>', '<a><b><c>'))\n# ['<a><b><c>']\n```",
        "`.+` is greedy and grabs as much as possible. Use `.+?` (non-greedy) or `[^>]+`.\n```python\nimport re\nprint(re.findall(r'<.+?>', '<a><b><c>'))   # ['<a>', '<b>', '<c>']\nprint(re.findall(r'<[^>]+>', '<a><b><c>'))  # ['<a>', '<b>', '<c>']\n```\nThe negated character class is usually faster than non-greedy quantifiers.",
        "When should I prefer `.+?` over `[^>]+`?",
        "Use `[^>]+` when the terminator is a single character — it's faster and avoids backtracking. Use `.+?` when the terminator is a multi-character pattern, e.g.:\n```python\nre.findall(r'<!--.+?-->', html, re.DOTALL)  # match HTML comments\n```\nThere, you can't easily express 'anything except `-->`' in a character class, so non-greedy is the right tool.")

    add("debug",
        "This fails when items have different types:\n```python\nresult = sorted([1, 'a', 2, 'b'])\n# TypeError\n```",
        "Python 3 won't compare different types. Sort separately or convert to a common type.\n```python\nresult = sorted([1, 'a', 2, 'b'], key=str)  # all-as-string ordering\n```\nThis gives `['1', '2', 'a', 'b']` (compared as strings).",
        "What if I want all numbers first, then strings, both sorted within type?",
        "```python\ndef key(x):\n    return (0, x) if isinstance(x, (int, float)) else (1, x)\n\nresult = sorted([1, 'a', 2, 'b'], key=key)\n# [1, 2, 'a', 'b']\n```\nThe leading `0`/`1` puts numbers before strings; the second tuple element sorts within each type.")

    add("debug",
        "Why does this break when path contains spaces?\n```python\nimport subprocess\nsubprocess.run(f'ls {path}', shell=True)\n```",
        "With `shell=True`, `path` isn't quoted, so spaces split it into multiple arguments. Pass a list and skip the shell.\n```python\nimport subprocess\nsubprocess.run(['ls', path])\n```\nList form makes spaces and special characters safe automatically.",
        "But I need shell features like glob expansion.",
        "```python\nimport glob\nimport subprocess\n\nfiles = glob.glob('/some/dir/*.txt')\nsubprocess.run(['cat', *files])\n```\nDo the expansion in Python, then pass the resulting paths as a list. Avoids shell-injection risk and keeps quoting under your control.")

    add("debug",
        "This open() leaks file descriptors:\n```python\nfor name in filenames:\n    f = open(name)\n    process(f.read())\n```",
        "`f` is never closed. CPython usually closes on garbage-collection, but PyPy and other implementations may not. Use `with`.\n```python\nfor name in filenames:\n    with open(name) as f:\n        process(f.read())\n```\nDeterministic cleanup is the right discipline regardless of GC behavior.",
        "Now with proper error handling for missing files.",
        "```python\nimport logging\n\nfor name in filenames:\n    try:\n        with open(name) as f:\n            process(f.read())\n    except FileNotFoundError:\n        logging.warning('skipping missing file: %s', name)\n        continue\n```\nLog and continue is appropriate when one bad file shouldn't abort the batch. Use `raise` instead if you want strict behavior.")

    add("debug",
        "Function 'process' modifies the input list. Make it pure.\n```python\ndef process(items):\n    items.sort()\n    items.append('done')\n    return items\n```",
        "Operate on a copy.\n```python\ndef process(items):\n    out = sorted(items)\n    out.append('done')\n    return out\n```\n`sorted` returns a new list; `list.sort` mutates. Reaching for `sorted` is the easiest signal of pure-function intent.",
        "How would I document this so callers know it doesn't mutate?",
        "```python\ndef process(items: list[str]) -> list[str]:\n    \"\"\"Return a sorted copy of *items* with 'done' appended.\n\n    Does not modify the input list.\n    \"\"\"\n    out = sorted(items)\n    out.append('done')\n    return out\n```\nA short docstring with a 'Does not modify' line is the conventional way. Type hints showing in/out as `list[str]` reinforce that the input is treated as input only.")

    add("debug",
        "Refactor: this function is doing too much.\n```python\ndef report(filename):\n    data = json.load(open(filename))\n    cleaned = [x for x in data if x['active']]\n    grouped = defaultdict(list)\n    for x in cleaned:\n        grouped[x['team']].append(x)\n    print(f'Teams: {len(grouped)}')\n    for team, members in grouped.items():\n        print(f'  {team}: {len(members)}')\n```",
        "Split into pure functions plus a thin printer.\n```python\nimport json\nfrom collections import defaultdict\n\ndef load(filename):\n    with open(filename) as f:\n        return json.load(f)\n\ndef active(data):\n    return [x for x in data if x['active']]\n\ndef by_team(items):\n    grouped = defaultdict(list)\n    for x in items:\n        grouped[x['team']].append(x)\n    return grouped\n\ndef render(grouped):\n    print(f'Teams: {len(grouped)}')\n    for team, members in grouped.items():\n        print(f'  {team}: {len(members)}')\n\ndef report(filename):\n    render(by_team(active(load(filename))))\n```\nEach piece is testable in isolation; `report` becomes a narrative.",
        "How would I unit-test by_team?",
        "```python\nfrom collections import defaultdict\n\ndef test_by_team():\n    items = [\n        {'team': 'a', 'active': True},\n        {'team': 'a', 'active': True},\n        {'team': 'b', 'active': True},\n    ]\n    result = by_team(items)\n    assert dict(result) == {'a': items[:2], 'b': [items[2]]}\n```\nWith `by_team` extracted, the test just hands it a list and inspects the dict — no file I/O, no print capture.")

    add("debug",
        "Why does this default mutable behave weirdly?\n```python\nclass Box:\n    def __init__(self, items=[]):\n        self.items = items\n```",
        "Same issue as functions: the list is created once. Two Boxes share it.\n```python\nclass Box:\n    def __init__(self, items=None):\n        self.items = list(items) if items else []\n```\n`list(items)` also defensively copies a passed-in list so the caller can't mutate ours later.",
        "What about with dataclasses?",
        "```python\nfrom dataclasses import dataclass, field\n\n@dataclass\nclass Box:\n    items: list = field(default_factory=list)\n```\n`field(default_factory=list)` calls `list()` for each new instance — the dataclass-native way to handle mutable defaults.")

    add("debug",
        "This counts wrong:\n```python\ndef count_words(text):\n    return len(text.split(' '))\n```",
        "Splitting on a single space gives extra empties for multiple spaces. Use no-arg `split`.\n```python\ndef count_words(text):\n    return len(text.split())\n```\n`split()` collapses any whitespace runs and trims.",
        "What if I want to count unique words case-insensitively?",
        "```python\ndef count_unique_words(text):\n    return len({w.lower() for w in text.split()})\n```\nLower-casing inside the set comprehension folds 'Hello' and 'hello' into one. For more aggressive normalization, also strip punctuation with `re.sub(r'[^\\w\\s]', '', text)` first.")

    add("debug",
        "This logs nothing:\n```python\nimport logging\nlogging.info('hello')\n```",
        "Default level is WARNING, so INFO messages are dropped. Configure first.\n```python\nimport logging\nlogging.basicConfig(level=logging.INFO)\nlogging.info('hello')\n```\n`basicConfig` only takes effect on the first call; safe to put at module top.",
        "How would I split logs by level — info to stdout, errors to stderr?",
        "```python\nimport logging\nimport sys\n\nlogger = logging.getLogger()\nlogger.setLevel(logging.DEBUG)\n\nout = logging.StreamHandler(sys.stdout)\nout.setLevel(logging.DEBUG)\nout.addFilter(lambda r: r.levelno < logging.WARNING)\n\nerr = logging.StreamHandler(sys.stderr)\nerr.setLevel(logging.WARNING)\n\nlogger.addHandler(out)\nlogger.addHandler(err)\n```\nThe filter on the stdout handler keeps WARNING+ off it; errors go only to stderr — what most CLIs expect.")

    add("debug",
        "This raises 'TypeError: unhashable type: 'list'':\n```python\nseen = set()\nfor item in items:\n    if item in seen:\n        continue\n    seen.add(item)\n```",
        "Lists can't be hashed. Convert to a tuple if items are sequences.\n```python\nseen = set()\nfor item in items:\n    key = tuple(item) if isinstance(item, list) else item\n    if key in seen:\n        continue\n    seen.add(key)\n```\nTuples are hashable when their elements are, which usually covers nested data shallowly.",
        "What if items are dicts?",
        "```python\nimport json\nseen = set()\nfor item in items:\n    key = json.dumps(item, sort_keys=True) if isinstance(item, dict) else item\n    if key in seen:\n        continue\n    seen.add(key)\n```\nSerializing with `sort_keys=True` produces a stable string per logical content. For deep structures, use `frozenset` of items if the dict is shallow.")

    add("debug",
        "This is_email check rejects valid addresses:\n```python\nimport re\n_EMAIL = re.compile(r'^[a-z]+@[a-z]+\\.[a-z]+$')\n\ndef is_email(s):\n    return bool(_EMAIL.match(s))\n```",
        "Too restrictive — no digits, no dots in local part, no plus addressing, only lowercase. Loosen it.\n```python\nimport re\n_EMAIL = re.compile(r\"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$\")\n\ndef is_email(s):\n    return bool(_EMAIL.match(s))\n```\nStill not RFC-compliant, but covers practical cases.",
        "I have to validate against the actual RFC. What's the right approach?",
        "Don't try to write the regex — use a library.\n```python\nfrom email_validator import validate_email, EmailNotValidError\n\ndef is_email(s):\n    try:\n        validate_email(s)\n        return True\n    except EmailNotValidError:\n        return False\n```\n`email_validator` checks both syntax (per the spec) and DNS, and can normalize the address. The pragmatic approach for any non-trivial use.")

    add("debug",
        "This raises 'AttributeError' on Python 3.12 but not 3.10:\n```python\nimport asyncio\nloop = asyncio.get_event_loop()\n```",
        "`get_event_loop` is deprecated in 3.10+ and behaves differently outside a running loop. Use `asyncio.run` or `asyncio.new_event_loop`.\n```python\nimport asyncio\nasyncio.run(main())  # for top-level entry\n```\nFor library code that just needs the running loop:\n```python\nloop = asyncio.get_running_loop()  # only inside an async context\n```",
        "What if I'm in a sync context and need to schedule async work?",
        "```python\nimport asyncio\n\ndef sync_caller():\n    loop = asyncio.new_event_loop()\n    try:\n        return loop.run_until_complete(some_async_fn())\n    finally:\n        loop.close()\n```\nManually creating and closing a loop is the standard pattern when you can't restructure to use `asyncio.run` (e.g., already inside another framework's loop).")

    add("debug",
        "Refactor this if-elif chain that maps codes to handlers.\n```python\ndef handle(code, payload):\n    if code == 'ping':\n        return ping(payload)\n    elif code == 'echo':\n        return echo(payload)\n    elif code == 'shutdown':\n        return shutdown(payload)\n    else:\n        raise ValueError(f'unknown: {code}')\n```",
        "Use a dict of handlers.\n```python\nHANDLERS = {\n    'ping': ping,\n    'echo': echo,\n    'shutdown': shutdown,\n}\n\ndef handle(code, payload):\n    fn = HANDLERS.get(code)\n    if fn is None:\n        raise ValueError(f'unknown: {code}')\n    return fn(payload)\n```\nAdd handlers without touching dispatch code.",
        "How would I let plugins register additional handlers?",
        "```python\nHANDLERS = {}\n\ndef register(code):\n    def deco(fn):\n        HANDLERS[code] = fn\n        return fn\n    return deco\n\n@register('ping')\ndef ping(payload):\n    return 'pong'\n```\nThe decorator pattern lets each module register its own handlers at import time. For external plugins, expose `register` as part of your public API.")

    add("debug",
        "This shows wrong tz:\n```python\nfrom datetime import datetime\nprint(datetime.utcnow())\n```",
        "`utcnow` returns a naive datetime — no tzinfo attached, even though the value is UTC. Mistakes follow. Use `now(timezone.utc)`.\n```python\nfrom datetime import datetime, timezone\nprint(datetime.now(timezone.utc))\n```\nThis returns a timezone-aware UTC datetime that prints with `+00:00`.",
        "How do I parse an incoming UTC string back to an aware datetime?",
        "```python\nfrom datetime import datetime, timezone\n\ndef parse_utc(s: str) -> datetime:\n    if s.endswith('Z'):\n        s = s[:-1] + '+00:00'\n    return datetime.fromisoformat(s).astimezone(timezone.utc)\n```\nThe Z-to-`+00:00` swap is needed because pre-3.11 `fromisoformat` doesn't accept `Z`. Then `astimezone(timezone.utc)` normalizes any offset to UTC.")

    add("debug",
        "Why does this not stop on the first match?\n```python\ndef find_first_even(xs):\n    result = None\n    for x in xs:\n        if x % 2 == 0:\n            result = x\n    return result\n```",
        "It finds the *last* even, not the first. Return immediately.\n```python\ndef find_first_even(xs):\n    for x in xs:\n        if x % 2 == 0:\n            return x\n    return None\n```\nReturning early is faster and clearer.",
        "Make it a one-liner.",
        "```python\ndef find_first_even(xs):\n    return next((x for x in xs if x % 2 == 0), None)\n```\n`next(generator, default)` is the canonical 'find-first' idiom.")

    add("debug",
        "This pickle file is 100MB but contains only 1000 records. Why?\n```python\nimport pickle\nwith open('data.pkl', 'wb') as f:\n    pickle.dump([record for record in stream()], f)\n```",
        "If `record` holds references back to a parent (e.g., logger, DB connection, large config), pickle drags those in. Inspect what gets serialized.\n```python\nimport pickle\nimport pickletools\n\nwith open('data.pkl', 'rb') as f:\n    pickletools.dis(f)\n```\nThe disassembly shows you exactly which classes/objects are being referenced.",
        "How can I strip those references before pickling?",
        "```python\nimport pickle\n\ndef strip(record):\n    return {k: v for k, v in record.items() if k in ('id', 'name', 'data')}\n\nwith open('data.pkl', 'wb') as f:\n    pickle.dump([strip(r) for r in stream()], f)\n```\nExplicitly pick the fields you want. Or implement `__getstate__` on your class to control serialization.")

    add("debug",
        "Why does this UnicodeDecodeError happen on a CSV?\n```python\nimport pandas as pd\ndf = pd.read_csv('data.csv')\n```",
        "The file isn't UTF-8 (the default). Detect or specify the encoding.\n```python\nimport pandas as pd\ndf = pd.read_csv('data.csv', encoding='latin-1')\n```\n`latin-1` accepts any byte without erroring, which is sometimes the right answer; `cp1252` is common for Excel-exported CSVs from Windows.",
        "How can I detect the encoding programmatically?",
        "```python\nfrom charset_normalizer import from_path\n\ndef detect(path):\n    return from_path(path).best().encoding\n\nencoding = detect('data.csv')\nimport pandas as pd\ndf = pd.read_csv('data.csv', encoding=encoding)\n```\n`charset-normalizer` is the modern replacement for `chardet` and is already a `requests` dependency.")

    add("debug",
        "Refactor this for-loop that builds two lists.\n```python\npass_list = []\nfail_list = []\nfor item in items:\n    if item.passed:\n        pass_list.append(item)\n    else:\n        fail_list.append(item)\n```",
        "Same logic but with a partition helper.\n```python\nfrom itertools import filterfalse\n\nitems_a, items_b = list(items), list(items)\npass_list = list(filter(lambda x: x.passed, items_a))\nfail_list = list(filterfalse(lambda x: x.passed, items_b))\n```\nNot necessarily better — the original is fine. The single-pass version below avoids iterating twice.",
        "Show me the single-pass version.",
        "```python\ndef partition(items, pred):\n    yes, no = [], []\n    for x in items:\n        (yes if pred(x) else no).append(x)\n    return yes, no\n\npass_list, fail_list = partition(items, lambda x: x.passed)\n```\nOne pass, both lists in order, easy to reuse.")

    add("debug",
        "Why does this division give 0?\n```python\nresult = 1 / 2\n```",
        "That gives `0.5` in Python 3. If you're seeing 0, you're either on Python 2 (please don't) or one operand is a string `'1'/'2'`. Check.\n```python\nprint(1 / 2)    # 0.5\nprint(1 // 2)   # 0  (integer division)\n```\nThe integer-division operator `//` is what gives 0.",
        "Then how do I emulate Python 2 behavior intentionally?",
        "```python\nfrom __future__ import division  # not needed in Py3\nprint(1 // 2)  # explicit integer division\n```\nUse `//` when you want floor division. The `from __future__` import is a no-op on Python 3 but harmless if your code targets both.")
