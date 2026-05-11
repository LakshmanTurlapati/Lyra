# SPDX-License-Identifier: MIT
"""Algorithms / data structures / utilities entries (150 total). Wave 2.
60 have follow-ups (5-msg), 90 are 3-msg.
"""


def register_all(add):
    # ---- 3-msg (90) ----
    add("algo",
        "Write a function that returns the running maximum of a list as a new list.",
        "```python\ndef running_max(xs: list[int]) -> list[int]:\n    out = []\n    cur = float('-inf')\n    for x in xs:\n        cur = max(cur, x)\n        out.append(cur)\n    return out\n```\nSingle pass, O(n). For NumPy arrays use `np.maximum.accumulate` instead.")

    add("algo",
        "Write a function that returns True if a list is sorted in non-decreasing order.",
        "```python\ndef is_sorted(xs: list[int]) -> bool:\n    return all(a <= b for a, b in zip(xs, xs[1:]))\n```\n`zip(xs, xs[1:])` is the canonical pairwise idiom. Python 3.10+ also offers `itertools.pairwise`.")

    add("algo",
        "Implement a function that rotates a list to the right by k positions.",
        "```python\ndef rotate(xs: list[int], k: int) -> list[int]:\n    if not xs:\n        return xs\n    k %= len(xs)\n    return xs[-k:] + xs[:-k]\n```\nThe modulo handles k larger than the list length gracefully. Returns a new list rather than mutating.")

    add("algo",
        "Write a function that returns the intersection of two lists, preserving order from the first.",
        "```python\ndef intersect(a: list, b: list) -> list:\n    bset = set(b)\n    seen = set()\n    out = []\n    for x in a:\n        if x in bset and x not in seen:\n            out.append(x); seen.add(x)\n    return out\n```\nO(n + m). Using a set for `b` makes membership checks O(1).")

    add("algo",
        "Write a function that returns the symmetric difference of two iterables as a sorted list.",
        "```python\ndef sym_diff(a, b) -> list:\n    return sorted(set(a) ^ set(b))\n```\n`^` is Python's set XOR. Sorting at the end gives a deterministic order.")

    add("algo",
        "Implement a stack with O(1) get_min using two stacks.",
        "```python\nclass MinStack:\n    def __init__(self) -> None:\n        self._main: list[int] = []\n        self._mins: list[int] = []\n    def push(self, x: int) -> None:\n        self._main.append(x)\n        self._mins.append(x if not self._mins else min(x, self._mins[-1]))\n    def pop(self) -> int:\n        self._mins.pop()\n        return self._main.pop()\n    def get_min(self) -> int:\n        return self._mins[-1]\n```\nThe parallel stack tracks the minimum at each level.")

    add("algo",
        "Write a function that converts a list of tuples into a dict, last value wins on duplicate keys.",
        "```python\ndef to_dict(pairs: list[tuple]) -> dict:\n    return dict(pairs)\n```\n`dict()` already does last-wins on duplicates. No need for a manual loop.")

    add("algo",
        "Write a function that splits a list into chunks of size n.",
        "```python\ndef chunks(xs: list, n: int) -> list[list]:\n    if n <= 0:\n        raise ValueError('n must be positive')\n    return [xs[i:i+n] for i in range(0, len(xs), n)]\n```\nThe last chunk may be shorter than n. For lazy iteration use `itertools.islice` in a loop.")

    add("algo",
        "Implement a Caesar cipher that shifts letters by k, preserving case and non-letters.",
        "```python\ndef caesar(s: str, k: int) -> str:\n    out = []\n    for c in s:\n        if 'a' <= c <= 'z':\n            out.append(chr((ord(c) - 97 + k) % 26 + 97))\n        elif 'A' <= c <= 'Z':\n            out.append(chr((ord(c) - 65 + k) % 26 + 65))\n        else:\n            out.append(c)\n    return ''.join(out)\n```\nModulo 26 handles wraparound and negative shifts.")

    add("algo",
        "Write a function that counts the number of set bits in an integer.",
        "```python\ndef popcount(n: int) -> int:\n    return bin(n & 0xFFFFFFFFFFFFFFFF).count('1') if n >= 0 else bin(n).count('1')\n```\nPython 3.10+ has `int.bit_count()` which is faster. Mask negative values if you want two's-complement semantics.")

    add("algo",
        "Implement a function that returns True if n is prime.",
        "```python\ndef is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    if n < 4:\n        return True\n    if n % 2 == 0:\n        return False\n    i = 3\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 2\n    return True\n```\nSquare-root bound and skipping even divisors keeps it fast for moderate n.")

    add("algo",
        "Write a function that returns the n-th triangular number.",
        "```python\ndef triangular(n: int) -> int:\n    return n * (n + 1) // 2\n```\nClosed form avoids any iteration. Use floor-division to keep the result an int.")

    add("algo",
        "Write a function that decodes a string like 'a3b2' back into 'aaabb'.",
        "```python\nimport re\n\ndef rle_decode(s: str) -> str:\n    return ''.join(ch * int(n) for ch, n in re.findall(r'([A-Za-z])(\\d+)', s))\n```\nSingle regex captures (letter, digits) pairs, then expand each.")

    add("algo",
        "Implement a circular buffer of fixed size with append and to_list.",
        "```python\nfrom collections import deque\n\nclass Ring:\n    def __init__(self, size: int) -> None:\n        self._buf: deque = deque(maxlen=size)\n    def append(self, x) -> None:\n        self._buf.append(x)\n    def to_list(self) -> list:\n        return list(self._buf)\n```\n`deque(maxlen=...)` already implements ring-buffer semantics.")

    add("algo",
        "Write a function that returns the longest run of consecutive equal characters.",
        "```python\nfrom itertools import groupby\n\ndef longest_run(s: str) -> tuple[str, int]:\n    if not s:\n        return ('', 0)\n    return max(((c, sum(1 for _ in g)) for c, g in groupby(s)), key=lambda t: t[1])\n```\n`groupby` packages consecutive runs; pick the longest by count.")

    add("algo",
        "Write a function that converts a Roman numeral to an integer.",
        "```python\ndef roman_to_int(s: str) -> int:\n    vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n    total = 0\n    prev = 0\n    for ch in reversed(s):\n        v = vals[ch]\n        total += -v if v < prev else v\n        prev = v\n    return total\n```\nWalking right-to-left lets you subtract when a smaller numeral precedes a larger one.")

    add("algo",
        "Write a function that returns the dot product of two equal-length vectors.",
        "```python\ndef dot(a: list[float], b: list[float]) -> float:\n    if len(a) != len(b):\n        raise ValueError('length mismatch')\n    return sum(x * y for x, y in zip(a, b))\n```\n`sum(... for ...)` keeps it lazy. For numeric work, prefer `numpy.dot`.")

    add("algo",
        "Implement Fisher-Yates shuffle in place.",
        "```python\nimport random\n\ndef shuffle_inplace(xs: list) -> None:\n    for i in range(len(xs) - 1, 0, -1):\n        j = random.randint(0, i)\n        xs[i], xs[j] = xs[j], xs[i]\n```\nUnbiased shuffle in O(n). The standard library's `random.shuffle` does exactly this.")

    add("algo",
        "Write a function that returns the moving average over a window of size k.",
        "```python\nfrom collections import deque\n\ndef moving_average(xs: list[float], k: int) -> list[float]:\n    out, win, total = [], deque(), 0.0\n    for x in xs:\n        win.append(x); total += x\n        if len(win) > k:\n            total -= win.popleft()\n        if len(win) == k:\n            out.append(total / k)\n    return out\n```\nDeque + running sum gives O(n).")

    add("algo",
        "Implement a function that determines whether parentheses in a string are balanced.",
        "```python\ndef balanced(s: str) -> bool:\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack: list[str] = []\n    for c in s:\n        if c in '([{':\n            stack.append(c)\n        elif c in pairs:\n            if not stack or stack.pop() != pairs[c]:\n                return False\n    return not stack\n```\nStack-based check is the canonical solution.")

    add("algo",
        "Write a function that returns the nth row of Pascal's triangle.",
        "```python\ndef pascal_row(n: int) -> list[int]:\n    row = [1]\n    for k in range(n):\n        row.append(row[-1] * (n - k) // (k + 1))\n    return row\n```\nIterative formula avoids computing factorials directly.")

    add("algo",
        "Implement a function that finds the missing number in a list containing 0..n with one missing.",
        "```python\ndef missing(xs: list[int]) -> int:\n    n = len(xs)\n    return n * (n + 1) // 2 - sum(xs)\n```\nO(n), no extra storage. Subtract observed sum from expected sum.")

    add("algo",
        "Write a function that converts seconds to a 'HH:MM:SS' string.",
        "```python\ndef fmt_seconds(secs: int) -> str:\n    h, rem = divmod(secs, 3600)\n    m, s = divmod(rem, 60)\n    return f'{h:02d}:{m:02d}:{s:02d}'\n```\n`divmod` keeps the arithmetic clean. f-string padding handles the leading zeros.")

    add("algo",
        "Write a function that returns True if two strings are rotations of each other.",
        "```python\ndef is_rotation(a: str, b: str) -> bool:\n    return len(a) == len(b) and b in (a + a)\n```\nClassic trick: every rotation of `a` is a substring of `a + a`.")

    add("algo",
        "Implement a function that interleaves two lists element by element.",
        "```python\nfrom itertools import zip_longest\n\ndef interleave(a: list, b: list, fill=None) -> list:\n    out = []\n    for x, y in zip_longest(a, b, fillvalue=fill):\n        out.append(x); out.append(y)\n    return [v for v in out if v is not fill]\n```\n`zip_longest` handles unequal lengths; filter the fill at the end.")

    add("algo",
        "Write a function that returns all prime factors of n with multiplicity.",
        "```python\ndef prime_factors(n: int) -> list[int]:\n    out = []\n    d = 2\n    while d * d <= n:\n        while n % d == 0:\n            out.append(d); n //= d\n        d += 1\n    if n > 1:\n        out.append(n)\n    return out\n```\nTrial division up to sqrt(n) is fine for n under ~1e12.")

    add("algo",
        "Write a function that returns True if a number is a perfect square.",
        "```python\nimport math\n\ndef is_square(n: int) -> bool:\n    if n < 0:\n        return False\n    r = math.isqrt(n)\n    return r * r == n\n```\n`math.isqrt` is exact for arbitrarily large integers, unlike `sqrt`.")

    add("algo",
        "Implement a function that returns the unique elements of a list while preserving order.",
        "```python\ndef unique(xs: list) -> list:\n    seen = set()\n    out = []\n    for x in xs:\n        if x not in seen:\n            seen.add(x); out.append(x)\n    return out\n```\nFor hashable items only. For non-hashable, fall back to a list-membership check.")

    add("algo",
        "Write a function that computes n! iteratively.",
        "```python\ndef factorial(n: int) -> int:\n    if n < 0:\n        raise ValueError('n must be non-negative')\n    out = 1\n    for k in range(2, n + 1):\n        out *= k\n    return out\n```\nPython integers are arbitrary precision so no overflow worries.")

    add("algo",
        "Write a function that returns the cumulative sum of a list.",
        "```python\nfrom itertools import accumulate\n\ndef cumsum(xs: list[float]) -> list[float]:\n    return list(accumulate(xs))\n```\n`itertools.accumulate` is the right primitive; it also accepts a custom binary op.")

    add("algo",
        "Implement a function that finds the maximum subarray sum (Kadane's algorithm).",
        "```python\ndef max_subarray(xs: list[int]) -> int:\n    best = cur = xs[0]\n    for x in xs[1:]:\n        cur = max(x, cur + x)\n        best = max(best, cur)\n    return best\n```\nO(n) and O(1) memory. Assumes the list is non-empty.")

    add("algo",
        "Write a function that compresses consecutive duplicates: [1,1,2,3,3,3] -> [1,2,3].",
        "```python\nfrom itertools import groupby\n\ndef compress(xs: list) -> list:\n    return [k for k, _ in groupby(xs)]\n```\n`groupby` keys give exactly one element per consecutive run.")

    add("algo",
        "Write a function that returns the median of a list.",
        "```python\nimport statistics\n\ndef median(xs: list[float]) -> float:\n    return statistics.median(xs)\n```\nThe stdlib already does this. Don't roll your own unless you have a constraint that forces it.")

    add("algo",
        "Implement a function that returns the transpose of a 2D list.",
        "```python\ndef transpose(m: list[list]) -> list[list]:\n    return [list(row) for row in zip(*m)]\n```\n`zip(*m)` is the canonical transpose. Wrap in `list` to materialize the rows.")

    add("algo",
        "Write a function that returns True if two intervals (a, b) and (c, d) overlap.",
        "```python\ndef overlap(a: int, b: int, c: int, d: int) -> bool:\n    return max(a, c) <= min(b, d)\n```\nSingle-line overlap test. Inclusive endpoints; switch to `<` for half-open intervals.")

    add("algo",
        "Write a function that merges overlapping intervals.",
        "```python\ndef merge_intervals(ivs: list[tuple[int, int]]) -> list[tuple[int, int]]:\n    if not ivs:\n        return []\n    ivs = sorted(ivs)\n    out = [ivs[0]]\n    for s, e in ivs[1:]:\n        if s <= out[-1][1]:\n            out[-1] = (out[-1][0], max(out[-1][1], e))\n        else:\n            out.append((s, e))\n    return out\n```\nSort, then sweep -- standard interval-merge pattern.")

    add("algo",
        "Implement bubble sort with an early-exit when no swaps occurred.",
        "```python\ndef bubble_sort(xs: list[int]) -> list[int]:\n    xs = xs[:]\n    n = len(xs)\n    for i in range(n):\n        swapped = False\n        for j in range(n - 1 - i):\n            if xs[j] > xs[j+1]:\n                xs[j], xs[j+1] = xs[j+1], xs[j]\n                swapped = True\n        if not swapped:\n            break\n    return xs\n```\nThe early exit makes it adaptive: nearly-sorted input runs in O(n).")

    add("algo",
        "Write a function that returns True if n is a power of two.",
        "```python\ndef is_power_of_two(n: int) -> bool:\n    return n > 0 and (n & (n - 1)) == 0\n```\nClassic bit trick: only powers of two have a single set bit.")

    add("algo",
        "Implement a function that converts a base-10 integer to a binary string without prefix.",
        "```python\ndef to_binary(n: int) -> str:\n    if n == 0:\n        return '0'\n    return bin(abs(n))[2:] if n > 0 else '-' + bin(abs(n))[2:]\n```\n`bin()` returns '0b...' so we slice off the prefix.")

    add("algo",
        "Write a function that finds the longest substring without repeating characters.",
        "```python\ndef longest_unique_substr(s: str) -> str:\n    last: dict[str, int] = {}\n    start = best_lo = best_hi = 0\n    for i, c in enumerate(s):\n        if c in last and last[c] >= start:\n            start = last[c] + 1\n        last[c] = i\n        if i - start > best_hi - best_lo:\n            best_lo, best_hi = start, i\n    return s[best_lo:best_hi+1]\n```\nSliding-window with a map of last-seen indices.")

    add("algo",
        "Write a function that converts an integer to its English words representation up to 999.",
        "```python\nDIG = ['', 'one','two','three','four','five','six','seven','eight','nine']\nTEEN = ['ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen']\nTENS = ['','','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety']\n\ndef in_words(n: int) -> str:\n    if n == 0: return 'zero'\n    parts = []\n    if n >= 100:\n        parts.append(f'{DIG[n//100]} hundred'); n %= 100\n    if 10 <= n < 20:\n        parts.append(TEEN[n-10])\n    else:\n        if n >= 20:\n            parts.append(TENS[n//10]); n %= 10\n        if n > 0:\n            parts.append(DIG[n])\n    return ' '.join(parts)\n```\nBreak the number into hundreds, tens, ones; assemble with conditional joins.")

    add("algo",
        "Implement a function that finds all duplicates in a list (any element appearing more than once).",
        "```python\nfrom collections import Counter\n\ndef duplicates(xs: list) -> list:\n    return [x for x, c in Counter(xs).items() if c > 1]\n```\nCounter + filter is the cleanest expression.")

    add("algo",
        "Write a function that returns True if a Sudoku row (length 9) is valid.",
        "```python\ndef sudoku_row_ok(row: list[int]) -> bool:\n    seen = [x for x in row if x != 0]\n    return len(seen) == len(set(seen)) and all(1 <= x <= 9 for x in seen)\n```\nIgnore zeros (blanks); confirm remaining digits are unique and in 1..9.")

    add("algo",
        "Implement a function that counts vowels in a string.",
        "```python\ndef count_vowels(s: str) -> int:\n    return sum(1 for c in s.lower() if c in 'aeiou')\n```\nLowercase once, then a generator-comprehension sum. O(n).")

    add("algo",
        "Write a function that returns the list reversed without using slicing or built-in reverse.",
        "```python\ndef reverse(xs: list) -> list:\n    out = []\n    for i in range(len(xs) - 1, -1, -1):\n        out.append(xs[i])\n    return out\n```\nManual reverse for didactic purposes. Real code should just use `xs[::-1]` or `reversed(xs)`.")

    add("algo",
        "Write a function that returns True if a string contains only digits.",
        "```python\ndef all_digits(s: str) -> bool:\n    return s.isdigit() and bool(s)\n```\n`str.isdigit()` returns False for the empty string in Python; we still guard explicitly to be unambiguous.")

    add("algo",
        "Implement a function that flattens a dict of dicts into dotted keys.",
        "```python\ndef flatten_dict(d: dict, prefix: str = '') -> dict:\n    out = {}\n    for k, v in d.items():\n        key = f'{prefix}.{k}' if prefix else k\n        if isinstance(v, dict):\n            out.update(flatten_dict(v, key))\n        else:\n            out[key] = v\n    return out\n```\nRecursive merge. Useful for flattening JSON config trees.")

    add("algo",
        "Write a function that returns the symmetric pair counts: how many indices i,j have xs[i]+xs[j]==target with i<j.",
        "```python\nfrom collections import Counter\n\ndef pair_count(xs: list[int], target: int) -> int:\n    seen: Counter = Counter()\n    n = 0\n    for x in xs:\n        n += seen[target - x]\n        seen[x] += 1\n    return n\n```\nOne pass with a counter -- O(n).")

    add("algo",
        "Write a function that returns the maximum depth of a nested list.",
        "```python\ndef max_depth(xs) -> int:\n    if not isinstance(xs, list):\n        return 0\n    return 1 + max((max_depth(x) for x in xs), default=0)\n```\n`default=0` handles the empty-list case cleanly.")

    add("algo",
        "Implement a function that converts a Python list to a comma-separated string with 'and' before the last item.",
        "```python\ndef oxford_join(xs: list) -> str:\n    if not xs:\n        return ''\n    if len(xs) == 1:\n        return str(xs[0])\n    if len(xs) == 2:\n        return f'{xs[0]} and {xs[1]}'\n    return ', '.join(str(x) for x in xs[:-1]) + f', and {xs[-1]}'\n```\nHandle the 0/1/2/3+ cases explicitly so the punctuation is right.")

    add("algo",
        "Write a function that returns True if a number is an Armstrong number.",
        "```python\ndef is_armstrong(n: int) -> bool:\n    digits = [int(c) for c in str(n)]\n    p = len(digits)\n    return sum(d ** p for d in digits) == n\n```\nConvert to string for digit extraction; raise each digit to the digit-count power.")

    add("algo",
        "Implement a function that returns the difference between consecutive elements.",
        "```python\ndef diffs(xs: list[int]) -> list[int]:\n    return [b - a for a, b in zip(xs, xs[1:])]\n```\nDiscrete-difference operator. For NumPy use `np.diff`.")

    add("algo",
        "Write a function that produces a power set of a list.",
        "```python\nfrom itertools import combinations, chain\n\ndef power_set(xs: list) -> list[tuple]:\n    return list(chain.from_iterable(combinations(xs, r) for r in range(len(xs) + 1)))\n```\nChain `combinations` of all sizes 0..n. Returns 2**n tuples, so only practical for small n.")

    add("algo",
        "Implement a function that returns the index of the first element greater than a threshold or -1.",
        "```python\ndef first_gt(xs: list[int], t: int) -> int:\n    return next((i for i, x in enumerate(xs) if x > t), -1)\n```\n`next(..., -1)` returns the default when the iterator is exhausted.")

    add("algo",
        "Write a function that returns True if any two numbers in a list sum to zero.",
        "```python\ndef has_zero_pair(xs: list[int]) -> bool:\n    seen: set[int] = set()\n    for x in xs:\n        if -x in seen:\n            return True\n        seen.add(x)\n    return False\n```\nOne-pass with a set. Watch the case where the list contains a zero -- requires a second zero to form a pair.")

    add("algo",
        "Write a function that returns the n-th prime number.",
        "```python\ndef nth_prime(n: int) -> int:\n    if n < 1:\n        raise ValueError('n must be >= 1')\n    primes: list[int] = []\n    cand = 2\n    while len(primes) < n:\n        if all(cand % p for p in primes if p * p <= cand):\n            primes.append(cand)\n        cand += 1\n    return primes[-1]\n```\nFor large n use a sieve; this trial-division approach is fine for n up to a few thousand.")

    add("algo",
        "Implement a function that converts a 1D index into 2D (row, col) given a width.",
        "```python\ndef to_rc(idx: int, width: int) -> tuple[int, int]:\n    return divmod(idx, width)\n```\n`divmod` already returns `(quotient, remainder)` -- exactly `(row, col)` for row-major layout.")

    add("algo",
        "Write a function that returns True if a string of brackets `<>{}[]()` is balanced.",
        "```python\ndef balanced_brackets(s: str) -> bool:\n    pairs = {')': '(', ']': '[', '}': '{', '>': '<'}\n    stack: list[str] = []\n    for c in s:\n        if c in '([{<':\n            stack.append(c)\n        elif c in pairs:\n            if not stack or stack.pop() != pairs[c]:\n                return False\n    return not stack\n```\nSame stack idiom as parens-only, just with one more pair.")

    add("algo",
        "Write a function that returns the count of unique paths in an MxN grid moving only right or down.",
        "```python\nfrom math import comb\n\ndef unique_paths(m: int, n: int) -> int:\n    return comb(m + n - 2, m - 1)\n```\nCombinatorics: choose which of the m+n-2 moves are 'down'.")

    add("algo",
        "Implement an LRU cache using OrderedDict.",
        "```python\nfrom collections import OrderedDict\n\nclass LRU:\n    def __init__(self, capacity: int) -> None:\n        self.cap = capacity\n        self._d: OrderedDict = OrderedDict()\n    def get(self, key):\n        if key not in self._d:\n            return None\n        self._d.move_to_end(key)\n        return self._d[key]\n    def put(self, key, value) -> None:\n        if key in self._d:\n            self._d.move_to_end(key)\n        self._d[key] = value\n        if len(self._d) > self.cap:\n            self._d.popitem(last=False)\n```\n`move_to_end` and `popitem(last=False)` give O(1) operations. Or just use `functools.lru_cache`.")

    add("algo",
        "Write a function that pads a list with a value to a target length.",
        "```python\ndef pad(xs: list, n: int, fill=None) -> list:\n    return xs + [fill] * max(0, n - len(xs))\n```\n`max(0, ...)` keeps it idempotent when `xs` is already long enough.")

    add("algo",
        "Write a function that returns the index of the maximum element of a list.",
        "```python\ndef argmax(xs: list[float]) -> int:\n    return max(range(len(xs)), key=xs.__getitem__)\n```\nUse the `key` argument to compare positions by their values. For NumPy use `np.argmax`.")

    add("algo",
        "Implement a function that converts a column index (0-based) to spreadsheet letters: 0->A, 25->Z, 26->AA.",
        "```python\ndef col_letters(n: int) -> str:\n    out = ''\n    n += 1\n    while n:\n        n, r = divmod(n - 1, 26)\n        out = chr(65 + r) + out\n    return out\n```\nBijective base-26 -- the +1/-1 dance accounts for there being no 'zero digit'.")

    add("algo",
        "Write a function that returns the average of a list, or None if empty.",
        "```python\ndef average(xs: list[float]) -> float | None:\n    return sum(xs) / len(xs) if xs else None\n```\nThe `if xs` check is a clean way to avoid `ZeroDivisionError`.")

    add("algo",
        "Write a function that swaps the case of every letter in a string without using str.swapcase.",
        "```python\ndef swap_case(s: str) -> str:\n    return ''.join(c.lower() if c.isupper() else c.upper() if c.islower() else c for c in s)\n```\nNon-letters fall through unchanged. Built-in `str.swapcase` is preferred in real code.")

    add("algo",
        "Implement a binary tree node and an in-order traversal returning a list.",
        "```python\nfrom dataclasses import dataclass, field\nfrom typing import Optional\n\n@dataclass\nclass Node:\n    val: int\n    left: Optional['Node'] = None\n    right: Optional['Node'] = None\n\ndef inorder(root: Optional[Node]) -> list[int]:\n    out: list[int] = []\n    def walk(n):\n        if n is None: return\n        walk(n.left); out.append(n.val); walk(n.right)\n    walk(root)\n    return out\n```\nClassic LNR traversal. Iterative with a stack avoids recursion-limit issues.")

    add("algo",
        "Write a function that converts hex string to a list of bytes.",
        "```python\ndef hex_to_bytes(s: str) -> bytes:\n    return bytes.fromhex(s)\n```\nUse the built-in. Strips spaces between pairs automatically.")

    add("algo",
        "Write a function that finds the median of two sorted arrays in O(n+m).",
        "```python\nimport heapq\n\ndef median_of_two(a: list[int], b: list[int]) -> float:\n    merged = list(heapq.merge(a, b))\n    n = len(merged)\n    if n == 0:\n        raise ValueError('empty')\n    if n % 2:\n        return float(merged[n // 2])\n    return (merged[n // 2 - 1] + merged[n // 2]) / 2\n```\n`heapq.merge` is the right tool for merging pre-sorted iterables.")

    add("algo",
        "Implement a function that splits a list into N approximately-equal chunks.",
        "```python\ndef nsplit(xs: list, n: int) -> list[list]:\n    if n <= 0:\n        raise ValueError('n must be positive')\n    k, r = divmod(len(xs), n)\n    out: list[list] = []\n    i = 0\n    for j in range(n):\n        size = k + (1 if j < r else 0)\n        out.append(xs[i:i+size]); i += size\n    return out\n```\n`divmod` distributes the leftover items across the first `r` chunks.")

    add("algo",
        "Write a function that returns the most frequent k elements.",
        "```python\nfrom collections import Counter\n\ndef top_k(xs: list, k: int) -> list:\n    return [x for x, _ in Counter(xs).most_common(k)]\n```\n`Counter.most_common(k)` runs in O(n log k) using a heap internally.")

    add("algo",
        "Write a function that converts a list of integers to a single integer (digits joined).",
        "```python\ndef digits_to_int(ds: list[int]) -> int:\n    return int(''.join(str(d) for d in ds))\n```\nString join is the most readable; arithmetic accumulation also works but reads less clearly.")

    add("algo",
        "Implement a function that returns the longest increasing subsequence length.",
        "```python\nfrom bisect import bisect_left\n\ndef lis_length(xs: list[int]) -> int:\n    tails: list[int] = []\n    for x in xs:\n        i = bisect_left(tails, x)\n        if i == len(tails):\n            tails.append(x)\n        else:\n            tails[i] = x\n    return len(tails)\n```\nO(n log n) patience-sorting algorithm. `tails[i]` = smallest tail among LIS of length i+1.")

    add("algo",
        "Write a function that converts a snake_case string to camelCase.",
        "```python\ndef snake_to_camel(s: str) -> str:\n    head, *rest = s.split('_')\n    return head + ''.join(p.capitalize() for p in rest)\n```\nUnpack first part separately so it stays lowercase.")

    add("algo",
        "Write a function that pads each row of a 2D list to the same width with a fill value.",
        "```python\ndef pad_rows(rows: list[list], fill=None) -> list[list]:\n    w = max((len(r) for r in rows), default=0)\n    return [r + [fill] * (w - len(r)) for r in rows]\n```\nSingle pass to find max width; single pass to pad.")

    add("algo",
        "Implement a deque-based BFS that returns the level order of a binary tree as a list of lists.",
        "```python\nfrom collections import deque\n\ndef level_order(root) -> list[list]:\n    if root is None:\n        return []\n    out: list[list] = []\n    q = deque([root])\n    while q:\n        level = []\n        for _ in range(len(q)):\n            n = q.popleft()\n            level.append(n.val)\n            if n.left: q.append(n.left)\n            if n.right: q.append(n.right)\n        out.append(level)\n    return out\n```\nBatching by `len(q)` separates levels cleanly.")

    add("algo",
        "Write a function that returns the count of substrings with all unique characters.",
        "```python\ndef unique_substr_count(s: str) -> int:\n    seen: set[str] = set()\n    left = 0\n    total = 0\n    for right, c in enumerate(s):\n        while c in seen:\n            seen.discard(s[left]); left += 1\n        seen.add(c)\n        total += right - left + 1\n    return total\n```\nSliding window counts unique-char substrings ending at each `right`.")

    add("algo",
        "Write a function that counts how many times a pattern occurs in a string (overlaps allowed).",
        "```python\ndef count_overlap(s: str, pat: str) -> int:\n    if not pat:\n        return 0\n    n = 0\n    i = 0\n    while True:\n        i = s.find(pat, i)\n        if i == -1:\n            return n\n        n += 1; i += 1\n```\n`str.count` does NOT count overlaps -- you need a manual scan.")

    add("algo",
        "Implement a function that determines whether one list is a subsequence of another.",
        "```python\ndef is_subseq(needle: list, hay: list) -> bool:\n    it = iter(hay)\n    return all(any(x == y for y in it) for x in needle)\n```\n`iter` keeps the cursor advancing through `hay` so each match consumes characters.")

    add("algo",
        "Write a function that converts a dict of lists into a list of dicts (column-wise to row-wise).",
        "```python\ndef cols_to_rows(d: dict) -> list[dict]:\n    keys = list(d.keys())\n    return [dict(zip(keys, vals)) for vals in zip(*d.values())]\n```\n`zip(*d.values())` walks columns in lockstep.")

    add("algo",
        "Write a function that returns True if a string is a valid integer (with optional sign).",
        "```python\ndef is_int(s: str) -> bool:\n    if not s:\n        return False\n    body = s[1:] if s[0] in '+-' else s\n    return body.isdigit() and len(body) > 0\n```\nManual check avoids exceptions and is faster than `try: int(s)` for tight loops.")

    add("algo",
        "Implement Conway's Game of Life single-step on a list-of-lists grid.",
        "```python\ndef step(g: list[list[int]]) -> list[list[int]]:\n    h, w = len(g), len(g[0])\n    out = [[0]*w for _ in range(h)]\n    for i in range(h):\n        for j in range(w):\n            n = sum(g[i+di][j+dj]\n                    for di in (-1,0,1) for dj in (-1,0,1)\n                    if (di or dj) and 0 <= i+di < h and 0 <= j+dj < w)\n            out[i][j] = 1 if n == 3 or (g[i][j] and n == 2) else 0\n    return out\n```\nStandard rule: birth on 3 neighbors, survival on 2 or 3.")

    add("algo",
        "Write a function that returns the elements of two iterables zipped strictly (raise on length mismatch).",
        "```python\ndef strict_zip(a, b):\n    return list(zip(a, b, strict=True))\n```\nPython 3.10+ accepts `strict=True`. On older versions, manually compare lengths.")

    add("algo",
        "Write a function that returns the count of ones in the binary representation of a list of ints.",
        "```python\ndef total_ones(xs: list[int]) -> int:\n    return sum(int.bit_count(x) if x >= 0 else bin(x).count('1') for x in xs)\n```\n`int.bit_count` (3.10+) is faster than `bin(...).count('1')`.")

    add("algo",
        "Implement a function that returns the element at index i, supporting negative indexing without using Python's built-in.",
        "```python\ndef at(xs: list, i: int):\n    n = len(xs)\n    if not -n <= i < n:\n        raise IndexError(i)\n    return xs[i + n] if i < 0 else xs[i]\n```\nManual normalization makes the bounds check explicit.")

    add("algo",
        "Write a function that finds the smallest positive integer missing from a list.",
        "```python\ndef missing_positive(xs: list[int]) -> int:\n    s = set(xs)\n    i = 1\n    while i in s:\n        i += 1\n    return i\n```\nSet lookup is O(1); the loop runs at most n+1 times so the algorithm is O(n).")

    # ---- 5-msg with follow-up (60) ----
    add("algo",
        "Write a function that finds the longest palindrome substring.",
        "```python\ndef longest_palindrome(s: str) -> str:\n    if not s:\n        return ''\n    lo = hi = 0\n    def expand(l, r):\n        while l >= 0 and r < len(s) and s[l] == s[r]:\n            l -= 1; r += 1\n        return l + 1, r - 1\n    for i in range(len(s)):\n        for l, r in (expand(i, i), expand(i, i+1)):\n            if r - l > hi - lo:\n                lo, hi = l, r\n    return s[lo:hi+1]\n```\nExpand-around-center handles odd and even palindromes uniformly.",
        "What's the time complexity, and is there a faster algorithm?",
        "Expand-around-center is O(n^2) time and O(1) extra space. For O(n) you can use Manacher's algorithm, which transforms the string (e.g. `^#a#b#a#$`) so all palindromes have odd length, then maintains a 'right boundary' to reuse previously-computed palindrome radii. Manacher is rarely worth the complexity unless n is huge -- the n^2 version is fast enough for most inputs.")

    add("algo",
        "Implement insertion sort.",
        "```python\ndef insertion_sort(xs: list[int]) -> list[int]:\n    xs = xs[:]\n    for i in range(1, len(xs)):\n        cur = xs[i]\n        j = i - 1\n        while j >= 0 and xs[j] > cur:\n            xs[j+1] = xs[j]; j -= 1\n        xs[j+1] = cur\n    return xs\n```\nO(n^2) worst case but adaptive: nearly-sorted inputs run in O(n).",
        "When would you actually pick this over Timsort?",
        "Almost never as a top-level sort -- Python's built-in `sorted` (Timsort) is optimal for general use. Insertion sort still appears as a base case in many hybrid sorts (Timsort itself uses it for small runs of size <= 32). It's also a reasonable choice when the data is known to be very small (say n < 16) or already nearly sorted, because the constant factor is tiny.")

    add("algo",
        "Write a function that finds all anagram groups in a list of strings.",
        "```python\nfrom collections import defaultdict\n\ndef group_anagrams(words: list[str]) -> list[list[str]]:\n    groups: dict[str, list[str]] = defaultdict(list)\n    for w in words:\n        groups[''.join(sorted(w))].append(w)\n    return list(groups.values())\n```\nSorting each word gives a canonical key shared by all its anagrams.",
        "Can I use a different key to avoid the sort?",
        "Yes -- you can use a 26-tuple of letter counts: `tuple(Counter(w).get(c, 0) for c in string.ascii_lowercase)`. That's O(n) per word vs O(n log n) for sorting, but the tuple construction has a higher constant factor, so for short words sorting often wins in practice. Profile both before committing.")

    add("algo",
        "Implement a function that returns the diameter of a binary tree (longest path between any two nodes).",
        "```python\ndef diameter(root) -> int:\n    best = 0\n    def depth(n) -> int:\n        nonlocal best\n        if n is None:\n            return 0\n        l = depth(n.left); r = depth(n.right)\n        best = max(best, l + r)\n        return 1 + max(l, r)\n    depth(root)\n    return best\n```\nAt each node, the longest path through it is left depth + right depth.",
        "Why use `nonlocal best` instead of returning a tuple?",
        "Both work. The nonlocal version keeps the recursive return type to a single int (depth), which is slightly cleaner. The tuple version `(depth, best_so_far)` is more functional and easier to test. In production code I'd lean toward the tuple, since closures capturing mutable state are a common source of bugs in larger codebases.")

    add("algo",
        "Write a function that solves the coin change problem (minimum coins to make amount, -1 if impossible).",
        "```python\ndef coin_change(coins: list[int], amount: int) -> int:\n    INF = amount + 1\n    dp = [0] + [INF] * amount\n    for a in range(1, amount + 1):\n        for c in coins:\n            if c <= a:\n                dp[a] = min(dp[a], dp[a-c] + 1)\n    return dp[amount] if dp[amount] != INF else -1\n```\nClassic 1D DP. O(amount * len(coins)).",
        "Does this work with negative or zero coin values?",
        "No -- you need to filter coins to positive values first, otherwise zero coins create infinite loops conceptually (though here the `c <= a` guard prevents the obvious bad case) and negatives violate the recurrence. Add `coins = [c for c in coins if c > 0]` at the top, and consider raising on empty input.")

    add("algo",
        "Implement a function that returns the maximum profit from at most one stock buy/sell.",
        "```python\ndef max_profit(prices: list[int]) -> int:\n    best = 0\n    lo = float('inf')\n    for p in prices:\n        lo = min(lo, p)\n        best = max(best, p - lo)\n    return best\n```\nO(n), one pass. Track the running minimum and the best profit seen so far.",
        "How does this generalize to k transactions?",
        "For unlimited transactions, sum every positive day-over-day gain -- O(n). For at most k transactions, use 2D DP: `dp[t][i] = max(dp[t][i-1], prices[i] + max(dp[t-1][j] - prices[j] for j < i))`. Maintain the inner max as a running variable to keep it O(k*n).")

    add("algo",
        "Write a function that performs topological sort on a DAG given as adjacency dict.",
        "```python\nfrom collections import deque\n\ndef topo_sort(adj: dict) -> list:\n    indeg = {u: 0 for u in adj}\n    for u in adj:\n        for v in adj[u]:\n            indeg[v] = indeg.get(v, 0) + 1\n            indeg.setdefault(u, indeg.get(u, 0))\n    q = deque([u for u, d in indeg.items() if d == 0])\n    order = []\n    while q:\n        u = q.popleft(); order.append(u)\n        for v in adj.get(u, []):\n            indeg[v] -= 1\n            if indeg[v] == 0:\n                q.append(v)\n    if len(order) != len(indeg):\n        raise ValueError('graph has a cycle')\n    return order\n```\nKahn's algorithm. Detects cycles by checking that all nodes were drained.",
        "Can I do it with DFS instead?",
        "Yes -- DFS post-order gives reverse topological order. Mark nodes white/gray/black; if you ever encounter a gray node during a DFS edge walk, you've found a back edge (cycle). DFS uses the call stack so it's slightly easier to write recursively but harder on very deep graphs. Kahn's is iterative by construction and easier to parallelize.")

    add("algo",
        "Implement Dijkstra's shortest path on a weighted graph.",
        "```python\nimport heapq\n\ndef dijkstra(adj: dict, src) -> dict:\n    dist = {src: 0}\n    pq = [(0, src)]\n    while pq:\n        d, u = heapq.heappop(pq)\n        if d > dist[u]:\n            continue\n        for v, w in adj.get(u, []):\n            nd = d + w\n            if nd < dist.get(v, float('inf')):\n                dist[v] = nd\n                heapq.heappush(pq, (nd, v))\n    return dist\n```\nLazy deletion (the `d > dist[u]` skip) avoids needing a decrease-key.",
        "What if some edges have negative weights?",
        "Dijkstra is incorrect with negative edges -- it commits to a node's distance the first time it's popped, but a later path through a negative edge could reach it more cheaply. Use Bellman-Ford for negative weights (O(V*E), detects negative cycles), or Johnson's algorithm if you need all-pairs shortest paths with negative weights but no negative cycles.")

    add("algo",
        "Write a function that returns the longest common substring of two strings.",
        "```python\ndef lcs_substring(a: str, b: str) -> str:\n    if not a or not b:\n        return ''\n    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]\n    end_i = best = 0\n    for i, ca in enumerate(a, 1):\n        for j, cb in enumerate(b, 1):\n            if ca == cb:\n                dp[i][j] = dp[i-1][j-1] + 1\n                if dp[i][j] > best:\n                    best, end_i = dp[i][j], i\n    return a[end_i - best:end_i]\n```\nO(len(a) * len(b)). Track the best run and where it ended.",
        "How is this different from longest common subsequence?",
        "Substring requires contiguity -- the matched characters must appear consecutively in both strings. Subsequence allows gaps -- you can skip characters as long as relative order is preserved. The DP recurrences look similar but in subsequence you carry forward the maximum of left/up cells when characters differ, instead of resetting to zero.")

    add("algo",
        "Implement a function that detects a cycle in a linked list using Floyd's algorithm.",
        "```python\ndef has_cycle(head) -> bool:\n    slow = fast = head\n    while fast and fast.next:\n        slow = slow.next\n        fast = fast.next.next\n        if slow is fast:\n            return True\n    return False\n```\nTwo pointers; if they meet, there's a cycle. O(1) extra space.",
        "How would you find the start of the cycle?",
        "After slow and fast meet inside the cycle, reset one pointer to the head. Move both one step at a time -- they'll meet again at the cycle's entry node. The math: when they first meet, the distance from head to cycle-start equals the distance from meeting-point to cycle-start (going around the cycle).")

    add("algo",
        "Write a function that solves N-Queens and returns one valid placement.",
        "```python\ndef n_queens(n: int) -> list[int] | None:\n    cols: list[int] = []\n    def place(r: int) -> bool:\n        if r == n:\n            return True\n        for c in range(n):\n            if all(c != cc and abs(c - cc) != r - rr for rr, cc in enumerate(cols)):\n                cols.append(c)\n                if place(r + 1):\n                    return True\n                cols.pop()\n        return False\n    return cols if place(0) else None\n```\nReturns a list where `cols[r]` is the queen's column on row `r`.",
        "Can you make this faster with bit manipulation?",
        "Yes. Track three bitmasks: `cols`, `diag1` (r+c), `diag2` (r-c). At each row the available columns are `~(cols | diag1 | diag2) & ((1 << n) - 1)`. Pick a bit, recurse with updated masks. This avoids the O(r) check at each step and runs noticeably faster for n >= 12.")

    add("algo",
        "Implement union-find with path compression and union by rank.",
        "```python\nclass UF:\n    def __init__(self, n: int) -> None:\n        self.parent = list(range(n))\n        self.rank = [0] * n\n    def find(self, x: int) -> int:\n        while self.parent[x] != x:\n            self.parent[x] = self.parent[self.parent[x]]\n            x = self.parent[x]\n        return x\n    def union(self, a: int, b: int) -> bool:\n        ra, rb = self.find(a), self.find(b)\n        if ra == rb:\n            return False\n        if self.rank[ra] < self.rank[rb]:\n            ra, rb = rb, ra\n        self.parent[rb] = ra\n        if self.rank[ra] == self.rank[rb]:\n            self.rank[ra] += 1\n        return True\n```\nNear-amortized O(1) per op. Returns False from union if already connected.",
        "When is union-find the right choice over BFS/DFS?",
        "Use union-find when you need to incrementally maintain connectivity as edges arrive (Kruskal's MST, dynamic connectivity, image-segment merging). BFS/DFS are better when the graph is static and you want full traversal data (paths, distances). Union-find gives no path information, only equivalence-class membership.")

    add("algo",
        "Write a function that returns the kth largest element in O(n) average using quickselect.",
        "```python\nimport random\n\ndef quickselect(xs: list[int], k: int) -> int:\n    xs = xs[:]\n    lo, hi = 0, len(xs) - 1\n    target = len(xs) - k\n    while lo < hi:\n        p = random.randint(lo, hi)\n        xs[p], xs[hi] = xs[hi], xs[p]\n        store = lo\n        for i in range(lo, hi):\n            if xs[i] < xs[hi]:\n                xs[i], xs[store] = xs[store], xs[i]; store += 1\n        xs[store], xs[hi] = xs[hi], xs[store]\n        if store == target:\n            return xs[store]\n        if store < target:\n            lo = store + 1\n        else:\n            hi = store - 1\n    return xs[lo]\n```\nO(n) expected, O(n^2) worst case. Random pivot avoids degenerate inputs.",
        "What's the alternative if I need worst-case O(n)?",
        "Median-of-medians (BFPRT) gives deterministic O(n) but with a much larger constant -- in practice it's slower than randomized quickselect on real inputs. For most cases, `heapq.nlargest(k, xs)[-1]` is the pragmatic choice: O(n log k), simple, and uses a battle-tested implementation.")

    add("algo",
        "Implement a function that performs Karatsuba multiplication of two large integers.",
        "```python\ndef karatsuba(x: int, y: int) -> int:\n    if x < 1 << 32 or y < 1 << 32:\n        return x * y\n    n = max(x.bit_length(), y.bit_length())\n    half = (n + 1) // 2\n    mask = (1 << half) - 1\n    x1, x0 = x >> half, x & mask\n    y1, y0 = y >> half, y & mask\n    z0 = karatsuba(x0, y0)\n    z2 = karatsuba(x1, y1)\n    z1 = karatsuba(x0 + x1, y0 + y1) - z0 - z2\n    return (z2 << (2 * half)) + (z1 << half) + z0\n```\nThree subproducts instead of four -- O(n^1.585).",
        "When does this actually beat Python's built-in multiplication?",
        "It usually doesn't. CPython's int multiplication uses an optimized C implementation that switches to Karatsuba (and even better algorithms for larger ints) automatically. A pure-Python Karatsuba is purely educational here; for real performance you'd write it in C, Rust, or use libraries like gmpy2.")

    add("algo",
        "Write a function that returns the longest path in a DAG.",
        "```python\nfrom collections import defaultdict\n\ndef longest_dag_path(edges: list[tuple]) -> int:\n    g = defaultdict(list)\n    indeg: dict = {}\n    for u, v in edges:\n        g[u].append(v)\n        indeg[v] = indeg.get(v, 0) + 1\n        indeg.setdefault(u, 0)\n    order = []\n    stack = [u for u, d in indeg.items() if d == 0]\n    while stack:\n        u = stack.pop(); order.append(u)\n        for v in g[u]:\n            indeg[v] -= 1\n            if indeg[v] == 0:\n                stack.append(v)\n    dist = {u: 0 for u in indeg}\n    for u in order:\n        for v in g[u]:\n            dist[v] = max(dist[v], dist[u] + 1)\n    return max(dist.values(), default=0)\n```\nTopo-sort then relax edges in order. O(V+E).",
        "Why doesn't Dijkstra work here?",
        "Dijkstra finds shortest paths and assumes non-negative weights. For longest paths in a general graph the problem is NP-hard, but on a DAG you can negate weights and run shortest-path -- or, more simply, do a topological sort and relax edges in order, which is exactly what this function does.")

    add("algo",
        "Implement a trie with insert and starts_with.",
        "```python\nclass Trie:\n    def __init__(self) -> None:\n        self.root: dict = {}\n    def insert(self, word: str) -> None:\n        n = self.root\n        for c in word:\n            n = n.setdefault(c, {})\n        n['$'] = True\n    def starts_with(self, prefix: str) -> bool:\n        n = self.root\n        for c in prefix:\n            if c not in n:\n                return False\n            n = n[c]\n        return True\n```\nNested dict keeps the implementation tiny. Use a sentinel like `'$'` to mark word ends.",
        "How would you support deletion?",
        "Walk down to the word's terminal node, remove the `'$'` sentinel, then walk back up removing nodes that are now both empty (no children) and not terminal. Be careful not to remove a node that's a prefix of another stored word -- that's why you check both conditions before deleting.")

    add("algo",
        "Write a function that solves the knapsack problem (0/1) with weights and values.",
        "```python\ndef knapsack(weights: list[int], values: list[int], capacity: int) -> int:\n    dp = [0] * (capacity + 1)\n    for w, v in zip(weights, values):\n        for c in range(capacity, w - 1, -1):\n            dp[c] = max(dp[c], dp[c-w] + v)\n    return dp[capacity]\n```\n1D rolling DP. The reverse iteration over `c` ensures each item is used at most once.",
        "What changes for unbounded knapsack?",
        "Iterate `c` forward instead of backward: `for c in range(w, capacity + 1)`. Forward iteration lets `dp[c-w]` already include the current item, so you can pick it multiple times. The base case (dp = [0]*(capacity+1)) stays the same.")

    add("algo",
        "Implement a function that returns all subsets of a list using bit enumeration.",
        "```python\ndef subsets(xs: list) -> list[list]:\n    n = len(xs)\n    out = []\n    for mask in range(1 << n):\n        out.append([xs[i] for i in range(n) if mask >> i & 1])\n    return out\n```\nEach bitmask maps to one subset. O(n * 2^n) which is optimal since the output is that big.",
        "How would you generate them in lexicographic order by length?",
        "Use `itertools.combinations` in a loop over r: `chain.from_iterable(combinations(xs, r) for r in range(len(xs)+1))`. That gives all 0-element subsets, then 1-element, etc. Each group is already in lex order if the input is sorted.")

    add("algo",
        "Write a function that finds strongly connected components using Tarjan's algorithm.",
        "```python\ndef tarjan_scc(adj: dict) -> list[list]:\n    idx: dict = {}\n    low: dict = {}\n    on_stack: set = set()\n    stack: list = []\n    counter = [0]\n    sccs: list[list] = []\n    def strong(u):\n        idx[u] = low[u] = counter[0]; counter[0] += 1\n        stack.append(u); on_stack.add(u)\n        for v in adj.get(u, []):\n            if v not in idx:\n                strong(v); low[u] = min(low[u], low[v])\n            elif v in on_stack:\n                low[u] = min(low[u], idx[v])\n        if low[u] == idx[u]:\n            comp = []\n            while True:\n                w = stack.pop(); on_stack.discard(w); comp.append(w)\n                if w == u: break\n            sccs.append(comp)\n    for u in list(adj):\n        if u not in idx:\n            strong(u)\n    return sccs\n```\nO(V+E). Each node's `low` value tracks the smallest index reachable through its DFS subtree.",
        "Tarjan vs Kosaraju -- which would you use?",
        "Both are O(V+E). Kosaraju is conceptually simpler: DFS once on G to produce a finish-order stack, transpose G, then DFS in reverse finish-order on G^T -- each tree is one SCC. Tarjan does it in a single pass with no transpose, which is faster in practice. I usually reach for Kosaraju when explaining the concept and Tarjan when writing production code.")

    add("algo",
        "Implement a function that finds bridges in an undirected graph.",
        "```python\ndef bridges(adj: dict) -> list[tuple]:\n    timer = [0]\n    tin: dict = {}\n    low: dict = {}\n    out: list = []\n    def dfs(u, parent):\n        tin[u] = low[u] = timer[0]; timer[0] += 1\n        for v in adj.get(u, []):\n            if v == parent:\n                continue\n            if v in tin:\n                low[u] = min(low[u], tin[v])\n            else:\n                dfs(v, u)\n                low[u] = min(low[u], low[v])\n                if low[v] > tin[u]:\n                    out.append((u, v))\n    for u in list(adj):\n        if u not in tin:\n            dfs(u, None)\n    return out\n```\nA bridge `(u,v)` exists iff `low[v] > tin[u]` -- v's subtree can't reach above u without using this edge.",
        "What's the difference from finding articulation points?",
        "Articulation points use `low[v] >= tin[u]` plus a special root case (root is articulation iff it has >=2 DFS children). Bridges use strict `>`. Conceptually: a bridge edge has no alternate path; an articulation point has no alternate path through any single edge but the node can still be on multiple paths via different edges.")

    add("algo",
        "Write a function that performs RLE compression on a bytes object.",
        "```python\ndef rle_bytes(data: bytes) -> bytes:\n    out = bytearray()\n    i = 0\n    while i < len(data):\n        j = i\n        while j < len(data) and data[j] == data[i] and j - i < 255:\n            j += 1\n        out.append(j - i); out.append(data[i])\n        i = j\n    return bytes(out)\n```\nCount byte capped at 255 so each run fits in a single byte.",
        "When is RLE actually a good compression choice?",
        "Only when input has long runs of identical bytes -- think monochrome bitmaps, simple icons, or sparse buffers. For arbitrary text or binary, RLE often makes the output larger. Modern formats use RLE as a preprocessing pass (e.g. PackBits in TIFF) before a stronger entropy coder, never as the only compression stage.")

    add("algo",
        "Implement a min-heap from scratch with push and pop.",
        "```python\nclass MinHeap:\n    def __init__(self) -> None:\n        self._h: list = []\n    def push(self, x) -> None:\n        self._h.append(x)\n        i = len(self._h) - 1\n        while i and self._h[(i-1)//2] > self._h[i]:\n            self._h[(i-1)//2], self._h[i] = self._h[i], self._h[(i-1)//2]\n            i = (i-1)//2\n    def pop(self):\n        if not self._h:\n            raise IndexError('empty')\n        top = self._h[0]\n        last = self._h.pop()\n        if self._h:\n            self._h[0] = last\n            i = 0; n = len(self._h)\n            while True:\n                l, r = 2*i+1, 2*i+2\n                small = i\n                if l < n and self._h[l] < self._h[small]: small = l\n                if r < n and self._h[r] < self._h[small]: small = r\n                if small == i: break\n                self._h[i], self._h[small] = self._h[small], self._h[i]\n                i = small\n        return top\n```\nArray-backed heap with sift-up on push and sift-down on pop.",
        "Why does Python's heapq only do min-heaps?",
        "It's a deliberate API choice -- the algorithms are identical for max-heaps, you just flip the comparison. Standard idiom: push `(-priority, item)` tuples for a max-heap, or define a wrapper class with reversed `__lt__`. The simplicity of one canonical primitive is worth the small inconvenience.")

    add("algo",
        "Write a function that finds the maximum sum path in a triangle (top to bottom).",
        "```python\ndef triangle_max(tri: list[list[int]]) -> int:\n    dp = tri[-1][:]\n    for r in range(len(tri) - 2, -1, -1):\n        for c in range(len(tri[r])):\n            dp[c] = tri[r][c] + max(dp[c], dp[c+1])\n    return dp[0]\n```\nBottom-up DP using a single rolling array.",
        "Can this go top-down instead?",
        "Yes -- top-down works but needs care because each cell at row r has two predecessors at row r-1 (except the edges). It's also slightly less elegant: you need to handle boundary cells specially. Bottom-up is the canonical approach for this problem; the recurrence is a clean `dp[c] = tri[r][c] + max(dp[c], dp[c+1])`.")

    add("algo",
        "Implement a function that returns all combinations summing to a target with unlimited reuse.",
        "```python\ndef combination_sum(cands: list[int], target: int) -> list[list[int]]:\n    cands = sorted(set(cands))\n    out: list[list[int]] = []\n    def back(start: int, remain: int, path: list[int]):\n        if remain == 0:\n            out.append(path[:]); return\n        for i in range(start, len(cands)):\n            if cands[i] > remain:\n                break\n            path.append(cands[i])\n            back(i, remain - cands[i], path)\n            path.pop()\n    back(0, target, [])\n    return out\n```\nSorting + the `> remain: break` cuts off many branches early.",
        "How does this change for 'each candidate used at most once'?",
        "Pass `i + 1` (instead of `i`) to the recursive call so the next pick must come from a strictly later index. Also dedupe within a level: `if i > start and cands[i] == cands[i-1]: continue` to skip duplicates that would produce the same combination.")

    add("algo",
        "Write a function that returns the edit distance with three operations: insert, delete, replace.",
        "```python\ndef edit_distance(a: str, b: str) -> int:\n    if len(a) < len(b):\n        a, b = b, a\n    prev = list(range(len(b) + 1))\n    for i, ca in enumerate(a, 1):\n        cur = [i]\n        for j, cb in enumerate(b, 1):\n            cur.append(min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (ca != cb)))\n        prev = cur\n    return prev[-1]\n```\nTwo-row DP for O(min(m, n)) memory.",
        "What changes if I want to also allow transpose (swap of two adjacent characters)?",
        "That's the Damerau-Levenshtein distance. Add a fourth case to the recurrence: when characters allow a transpose -- i.e. `a[i-1]==b[j-2]` and `a[i-2]==b[j-1]` -- consider `dp[i-2][j-2] + 1`. You'll need to keep two previous rows, not one, so the rolling-array trick gets uglier.")

    add("algo",
        "Implement a function that performs binary lifting for LCA on a tree.",
        "```python\nimport math\n\nclass LCA:\n    def __init__(self, n: int, edges: list[tuple], root: int = 0) -> None:\n        self.LOG = max(1, math.ceil(math.log2(n)))\n        adj: dict = {i: [] for i in range(n)}\n        for u, v in edges:\n            adj[u].append(v); adj[v].append(u)\n        self.up = [[-1] * n for _ in range(self.LOG)]\n        self.depth = [0] * n\n        order = [root]; seen = {root}\n        for u in order:\n            for v in adj[u]:\n                if v not in seen:\n                    seen.add(v); self.depth[v] = self.depth[u] + 1\n                    self.up[0][v] = u; order.append(v)\n        for k in range(1, self.LOG):\n            for v in range(n):\n                p = self.up[k-1][v]\n                self.up[k][v] = self.up[k-1][p] if p != -1 else -1\n```\nPrecompute 2^k-th ancestors. Query in O(log n) by lifting both nodes to the same depth, then jumping in lockstep.",
        "What about Euler-tour + sparse-table RMQ?",
        "That gives O(n log n) preprocessing and O(1) query, vs binary lifting's O(n log n) / O(log n). RMQ-LCA is faster for query-heavy workloads but more code. For most use cases binary lifting wins on simplicity, and the log factor at query time is rarely the bottleneck.")

    add("algo",
        "Write a function that performs matrix exponentiation to compute Fibonacci(n) in O(log n).",
        "```python\ndef matmul(a, b):\n    return [[a[0][0]*b[0][0]+a[0][1]*b[1][0], a[0][0]*b[0][1]+a[0][1]*b[1][1]],\n            [a[1][0]*b[0][0]+a[1][1]*b[1][0], a[1][0]*b[0][1]+a[1][1]*b[1][1]]]\n\ndef fib_fast(n: int) -> int:\n    if n == 0: return 0\n    base, result = [[1,1],[1,0]], [[1,0],[0,1]]\n    e = n\n    while e:\n        if e & 1:\n            result = matmul(result, base)\n        base = matmul(base, base)\n        e >>= 1\n    return result[0][1]\n```\nThe matrix [[1,1],[1,0]]^n encodes Fibonacci. Exponentiation by squaring gives O(log n) multiplications.",
        "Is there an even faster way?",
        "The 'fast doubling' identities give the same O(log n) but with fewer operations: F(2k) = F(k)*(2*F(k+1) - F(k)) and F(2k+1) = F(k+1)^2 + F(k)^2. Each doubling costs two multiplications and a few adds, vs eight in the matrix version.")

    add("algo",
        "Implement a function that solves the rod-cutting problem.",
        "```python\ndef rod_cut(prices: list[int], n: int) -> int:\n    dp = [0] * (n + 1)\n    for length in range(1, n + 1):\n        best = 0\n        for cut in range(1, min(length, len(prices)) + 1):\n            best = max(best, prices[cut - 1] + dp[length - cut])\n        dp[length] = best\n    return dp[n]\n```\n1D DP, O(n^2). `prices[i]` is the price for a piece of length `i+1`.",
        "How would I also recover the cut sequence?",
        "Maintain a parallel `choice` array where `choice[length]` records the first cut that produced `dp[length]`. To recover, start at `n`, append `choice[n]`, then move to `dp[n - choice[n]]` and repeat until you reach 0. That gives the multiset of cut lengths.")

    add("algo",
        "Write a function that performs depth-first traversal of a graph and returns reachable nodes.",
        "```python\ndef dfs_reachable(adj: dict, start) -> set:\n    seen: set = set()\n    stack = [start]\n    while stack:\n        u = stack.pop()\n        if u in seen:\n            continue\n        seen.add(u)\n        stack.extend(adj.get(u, []))\n    return seen\n```\nIterative DFS with an explicit stack avoids Python's recursion limit on large graphs.",
        "When should I use BFS instead?",
        "Use BFS when you need shortest-path distances in an unweighted graph, level-order processing (e.g. tree levels), or when memory matters and the graph is wide rather than deep. DFS is better for cycle detection, topological sort, finding connected components when you don't care about paths, and when the graph is deep but narrow.")

    add("algo",
        "Implement a function that returns minimum window substring containing all chars of t.",
        "```python\nfrom collections import Counter\n\ndef min_window(s: str, t: str) -> str:\n    if not t:\n        return ''\n    need = Counter(t)\n    have: Counter = Counter()\n    required = len(need)\n    formed = 0\n    l = 0\n    best = (float('inf'), 0, 0)\n    for r, c in enumerate(s):\n        have[c] += 1\n        if c in need and have[c] == need[c]:\n            formed += 1\n        while formed == required:\n            if r - l + 1 < best[0]:\n                best = (r - l + 1, l, r)\n            have[s[l]] -= 1\n            if s[l] in need and have[s[l]] < need[s[l]]:\n                formed -= 1\n            l += 1\n    return '' if best[0] == float('inf') else s[best[1]:best[2]+1]\n```\nClassic sliding-window, O(|s| + |t|).",
        "What if I need the *maximum* window with at most k distinct characters?",
        "Same shape: expand `r`; when `len(have)` exceeds k, shrink `l` until back at k. Track the best window length seen while `len(have) <= k`. The hard part is exclusively in the shrink step -- decrement `have[s[l]]` and `del have[s[l]]` when it hits zero, otherwise `len(have)` won't reflect distinct counts.")

    add("algo",
        "Write a function that solves the partition equal-sum subset problem.",
        "```python\ndef can_partition(xs: list[int]) -> bool:\n    total = sum(xs)\n    if total % 2:\n        return False\n    target = total // 2\n    dp = {0}\n    for x in xs:\n        dp = dp | {s + x for s in dp if s + x <= target}\n    return target in dp\n```\nSet-based DP -- much smaller in memory than a full boolean array when many sums aren't reachable.",
        "How does this compare to the boolean-array DP?",
        "The boolean array is `dp[s] = True` for each reachable sum, iterated `s` from `target` down to `x`. Time is O(n * target) in both versions. The set version has smaller memory in sparse cases (many unreachable sums) but worse cache behavior. For dense inputs the boolean array is faster; for sparse, the set wins.")

    add("algo",
        "Implement a function that simulates a producer-consumer with bounded queue using threading.",
        "```python\nimport queue, threading, time\n\ndef run(n_items: int, n_workers: int) -> list:\n    q: queue.Queue = queue.Queue(maxsize=8)\n    out: list = []\n    lock = threading.Lock()\n    def producer():\n        for i in range(n_items):\n            q.put(i)\n        for _ in range(n_workers):\n            q.put(None)\n    def worker():\n        while True:\n            item = q.get()\n            if item is None:\n                q.task_done(); return\n            with lock:\n                out.append(item * 2)\n            q.task_done()\n    threads = [threading.Thread(target=worker) for _ in range(n_workers)]\n    [t.start() for t in threads]\n    threading.Thread(target=producer).start()\n    [t.join() for t in threads]\n    return out\n```\n`queue.Queue` is thread-safe; sentinel `None` signals workers to stop.",
        "Why use a sentinel instead of just calling `q.join()`?",
        "`q.join()` waits for all queued tasks to be processed but doesn't tell workers to exit -- they'd block forever on `q.get()`. The sentinel pattern is the simplest way to give each worker an explicit shutdown signal. Alternatives: a `threading.Event` flag checked alongside `q.get(timeout=...)`, or `concurrent.futures` which handles shutdown for you.")

    add("algo",
        "Write a function that computes the convex hull of a set of points using Graham scan.",
        "```python\ndef convex_hull(pts: list[tuple]) -> list[tuple]:\n    pts = sorted(set(pts))\n    if len(pts) <= 1:\n        return pts\n    def cross(o, a, b):\n        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])\n    lower = []\n    for p in pts:\n        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:\n            lower.pop()\n        lower.append(p)\n    upper = []\n    for p in reversed(pts):\n        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:\n            upper.pop()\n        upper.append(p)\n    return lower[:-1] + upper[:-1]\n```\nMonotone-chain variant. O(n log n) -- the sort dominates.",
        "What changes for collinear points on the hull?",
        "If you want collinear points kept on the boundary, change `<= 0` to `< 0` so the algorithm only pops when there's a strict right turn, allowing equal cross-products (collinear) to remain. This produces a 'closed' hull including all border points.")

    add("algo",
        "Implement a function that performs RSA key generation (toy version).",
        "```python\nfrom math import gcd\nfrom random import randrange\n\ndef miller_rabin(n: int, k: int = 10) -> bool:\n    if n < 4: return n in (2, 3)\n    if n % 2 == 0: return False\n    r, d = 0, n - 1\n    while d % 2 == 0:\n        r += 1; d //= 2\n    for _ in range(k):\n        a = randrange(2, n - 2)\n        x = pow(a, d, n)\n        if x in (1, n - 1): continue\n        for _ in range(r - 1):\n            x = pow(x, 2, n)\n            if x == n - 1: break\n        else:\n            return False\n    return True\n\ndef gen_prime(bits: int) -> int:\n    while True:\n        n = randrange(1 << (bits-1), 1 << bits) | 1\n        if miller_rabin(n): return n\n\ndef rsa_keys(bits: int = 512):\n    p, q = gen_prime(bits), gen_prime(bits)\n    n = p * q; phi = (p-1) * (q-1)\n    e = 65537\n    if gcd(e, phi) != 1:\n        raise RuntimeError('retry')\n    d = pow(e, -1, phi)\n    return (n, e), (n, d)\n```\nReal-world RSA needs careful padding (OAEP), constant-time arithmetic, and validated parameters -- never use this in production.",
        "Why is e=65537 the standard public exponent?",
        "It's prime (so always coprime with phi), it has only two set bits (binary 10000000000000001), which makes encryption with public exponent very fast via binary exponentiation, and it's large enough to defeat low-exponent attacks like Coppersmith's attack on small e with related plaintexts.")

    add("algo",
        "Write a function that finds all primes up to n using the Sieve of Eratosthenes.",
        "```python\ndef sieve(n: int) -> list[int]:\n    if n < 2:\n        return []\n    mark = [True] * (n + 1)\n    mark[0] = mark[1] = False\n    for i in range(2, int(n**0.5) + 1):\n        if mark[i]:\n            for j in range(i*i, n + 1, i):\n                mark[j] = False\n    return [i for i, ok in enumerate(mark) if ok]\n```\nStart marking at i*i since smaller multiples were marked by smaller primes already.",
        "What's the memory-efficient alternative for very large n?",
        "Segmented sieve: precompute small primes up to sqrt(n) with a normal sieve, then process [n0, n0+SEGMENT) blocks at a time, using the small primes to mark composites within each block. This keeps memory bounded by SEGMENT (e.g. a few MB) while still running in roughly O(n log log n) total time.")

    add("algo",
        "Implement a function that performs reservoir sampling of size k from a stream.",
        "```python\nimport random\nfrom typing import Iterable\n\ndef reservoir(stream: Iterable, k: int) -> list:\n    out: list = []\n    for i, x in enumerate(stream):\n        if i < k:\n            out.append(x)\n        else:\n            j = random.randint(0, i)\n            if j < k:\n                out[j] = x\n    return out\n```\nSingle pass, O(n). Each element ends up in the reservoir with probability k/n.",
        "Why is each element's selection probability exactly k/n?",
        "By induction. After seeing n elements, P(item i is in the reservoir) = (k/i) * product over j>i of P(item j doesn't kick item i out) = (k/i) * prod_{j=i+1..n}((j-1)/j) = (k/i) * (i/n) = k/n. Each item from the first k starts in the reservoir with probability 1, and gets thinned to k/n by the same logic.")

    add("algo",
        "Write a function that returns the longest path in a binary tree from root to leaf.",
        "```python\ndef longest_root_to_leaf(root) -> list:\n    if root is None:\n        return []\n    if root.left is None and root.right is None:\n        return [root.val]\n    l = longest_root_to_leaf(root.left)\n    r = longest_root_to_leaf(root.right)\n    return [root.val] + (l if len(l) >= len(r) else r)\n```\nRecursive with the path returned up the call stack.",
        "Will this blow up the stack on a degenerate tree?",
        "Yes -- a tree shaped like a linked list (each node has only a left child) has recursion depth O(n), and Python's default limit is 1000. Either raise the limit with `sys.setrecursionlimit`, or convert to iterative DFS using an explicit stack. For really deep trees the iterative version is the only safe choice.")

    add("algo",
        "Implement a function that does string interning using a class-level cache.",
        "```python\nclass Interned:\n    _pool: dict[str, 'Interned'] = {}\n    def __new__(cls, s: str):\n        if s in cls._pool:\n            return cls._pool[s]\n        obj = super().__new__(cls)\n        obj.s = s\n        cls._pool[s] = obj\n        return obj\n    def __repr__(self) -> str:\n        return f'Interned({self.s!r})'\n```\nDe-duplicates equal strings so identity comparison works.",
        "How is this different from sys.intern?",
        "`sys.intern(s)` returns Python's interned copy of `s` (real `str`); equal interned strings are physically the same object. This wrapper class instead reuses *Interned* objects keyed by their string content -- it lets you attach extra attributes per unique string. Use `sys.intern` when you just want fast `is` comparison; use a wrapper when you need to carry metadata.")

    add("algo",
        "Write a function that implements memoization with a max cache size.",
        "```python\nfrom functools import lru_cache\n\n@lru_cache(maxsize=128)\ndef fib(n: int) -> int:\n    return n if n < 2 else fib(n-1) + fib(n-2)\n```\n`functools.lru_cache` is the canonical answer. `maxsize=None` for unbounded; an integer caps the cache.",
        "When does memoization with a max size hurt you?",
        "When call patterns aren't temporally clustered -- e.g. accessing rare keys repeatedly will keep evicting each other. A small LRU cache can degrade to no caching at all in that case. Profile cache hit rate (`fib.cache_info()`) before assuming a small cache is helping.")

    add("algo",
        "Implement a function that performs run-length decoding of a list of (value, count) tuples.",
        "```python\ndef rle_decode_pairs(pairs: list[tuple]) -> list:\n    out = []\n    for v, c in pairs:\n        out.extend([v] * c)\n    return out\n```\nSimple and idiomatic. `extend` with a multiplied list is faster than appending in a loop.",
        "How can I make this lazy?",
        "Use a generator: `def rle_decode_lazy(pairs): return (v for v, c in pairs for _ in range(c))`. That returns a generator yielding one element at a time, useful when the decoded sequence is huge or you just want to stream it through a downstream consumer.")

    add("algo",
        "Write a function that finds peaks in a list (elements greater than both neighbors).",
        "```python\ndef peaks(xs: list[int]) -> list[int]:\n    out = []\n    for i in range(1, len(xs) - 1):\n        if xs[i-1] < xs[i] > xs[i+1]:\n            out.append(i)\n    return out\n```\nThe chained comparison reads like math: 'left < center > right'.",
        "How does scipy.signal.find_peaks compare?",
        "It does the same core idea but adds optional filters: minimum height, minimum prominence, distance between peaks, width thresholds, and a wlen for prominence calculation. For real signal-processing use that. The hand-rolled version is fine for textbook 'strict local maximum' definitions.")

    add("algo",
        "Implement a function that simulates a leaky bucket rate limiter.",
        "```python\nimport time\n\nclass LeakyBucket:\n    def __init__(self, capacity: float, leak_per_sec: float) -> None:\n        self.cap = capacity\n        self.leak = leak_per_sec\n        self.level = 0.0\n        self.last = time.monotonic()\n    def allow(self, cost: float = 1.0) -> bool:\n        now = time.monotonic()\n        self.level = max(0.0, self.level - (now - self.last) * self.leak)\n        self.last = now\n        if self.level + cost > self.cap:\n            return False\n        self.level += cost\n        return True\n```\nLeak proportional to elapsed time; reject if the bucket would overflow.",
        "How is this different from a token bucket?",
        "Token bucket adds tokens at the leak rate up to capacity and consumes one per request -- a request is allowed if at least one token is available. Leaky bucket tracks 'level' that drains over time -- a request is allowed if adding cost doesn't exceed capacity. They're mathematically equivalent (one is the dual of the other) but token bucket allows controlled bursts more naturally; leaky bucket smooths output more strictly.")

    add("algo",
        "Write a function that returns the kth permutation of [1..n] in lex order without enumerating all.",
        "```python\nfrom math import factorial\n\ndef kth_perm(n: int, k: int) -> list[int]:\n    nums = list(range(1, n + 1))\n    out = []\n    k -= 1\n    while nums:\n        f = factorial(len(nums) - 1) if len(nums) > 1 else 1\n        idx, k = divmod(k, f)\n        out.append(nums.pop(idx))\n    return out\n```\nFactorial number system: each digit picks which of the remaining numbers to take.",
        "What if k is given 0-indexed?",
        "Just remove the `k -= 1`. The standard math convention is 0-indexed (the 'kth' permutation usually means the kth in 0..n!-1), but a lot of competitive-programming problems use 1-indexed. Either works -- just be explicit in the function's docstring.")

    add("algo",
        "Implement a function that returns the LCM of a list of integers.",
        "```python\nfrom math import gcd\nfrom functools import reduce\n\ndef lcm_list(xs: list[int]) -> int:\n    if not xs:\n        return 1\n    return reduce(lambda a, b: a * b // gcd(a, b), xs)\n```\n`math.lcm` (3.9+) accepts variadic args; this is the manual reduction.",
        "Why does an empty list return 1?",
        "It's the multiplicative identity, and it makes `lcm_list([] + xs) == lcm_list(xs)` hold for any xs (associative composition). The convention matches `math.lcm()` (no args) which also returns 1, and `math.gcd()` with no args returns 0 -- the identity for that operation.")

    add("algo",
        "Write a function that returns the maximum number of overlapping intervals at any point.",
        "```python\ndef max_overlap(ivs: list[tuple[int, int]]) -> int:\n    events = []\n    for s, e in ivs:\n        events.append((s, 1))\n        events.append((e, -1))\n    events.sort()\n    cur = best = 0\n    for _, delta in events:\n        cur += delta\n        best = max(best, cur)\n    return best\n```\nSweep-line: +1 at start, -1 at end. Sort and accumulate.",
        "What if endpoints are inclusive vs half-open?",
        "For half-open `[s, e)`, sort `(s, +1)` before `(e, -1)` -- the default tuple sort already does this since +1 > -1, but you'd want to flip if intervals are inclusive `[s, e]` and you want to count touching intervals as overlapping. Tie-break with `-1` before `+1` (process ends before starts) to merge touching, or `+1` before `-1` to keep them separate.")

    add("algo",
        "Implement a function that determines if a number is happy (eventually reaches 1 by sum-of-squares-of-digits).",
        "```python\ndef is_happy(n: int) -> bool:\n    seen: set[int] = set()\n    while n != 1 and n not in seen:\n        seen.add(n)\n        n = sum(int(c) ** 2 for c in str(n))\n    return n == 1\n```\nLoop until either hitting 1 or revisiting a number (cycle detected).",
        "Can you do it without storing all visited numbers?",
        "Yes -- use Floyd's cycle detection with two pointers. One advances by one step, the other by two; they meet inside the cycle (if any). Memory is O(1) regardless of how many numbers were visited.")

    add("algo",
        "Write a function that decompresses gzipped data from bytes.",
        "```python\nimport gzip\n\ndef gunzip(data: bytes) -> bytes:\n    return gzip.decompress(data)\n```\n`gzip.decompress` is the simplest interface for in-memory data.",
        "What if I have a large file I want to stream?",
        "Use `gzip.open(path, 'rb')` and iterate in chunks: `while chunk := f.read(64*1024): process(chunk)`. Avoids loading the entire decompressed content into memory. For pipelining through other tools, `gzip.GzipFile` wraps any binary file-like object including `sys.stdin.buffer`.")

    add("algo",
        "Implement a function that returns the maximum XOR of any two elements in a list using a trie.",
        "```python\ndef max_xor(xs: list[int]) -> int:\n    BITS = max(x.bit_length() for x in xs) if xs else 0\n    root: dict = {}\n    best = 0\n    for x in xs:\n        node = greedy = root\n        cur = 0\n        for i in range(BITS - 1, -1, -1):\n            b = (x >> i) & 1\n            if 1 - b in greedy:\n                cur |= 1 << i; greedy = greedy[1 - b]\n            elif b in greedy:\n                greedy = greedy[b]\n            else:\n                break\n        for i in range(BITS - 1, -1, -1):\n            b = (x >> i) & 1\n            node = node.setdefault(b, {})\n        best = max(best, cur)\n    return best\n```\nFor each new number, greedily take the opposite bit at each level if available.",
        "What's the time complexity?",
        "O(n * B) where B is the bit-width of the largest number -- typically 32 or 64. Each insert and each query traverses at most B levels of the trie. Compare to the naive O(n^2) pairwise XOR; the trie wins for n > a few hundred.")

    add("algo",
        "Write a function that performs interpolation search on a uniformly-distributed sorted array.",
        "```python\ndef interp_search(xs: list[int], target: int) -> int:\n    lo, hi = 0, len(xs) - 1\n    while lo <= hi and xs[lo] <= target <= xs[hi]:\n        if xs[lo] == xs[hi]:\n            return lo if xs[lo] == target else -1\n        pos = lo + (target - xs[lo]) * (hi - lo) // (xs[hi] - xs[lo])\n        if xs[pos] == target:\n            return pos\n        if xs[pos] < target:\n            lo = pos + 1\n        else:\n            hi = pos - 1\n    return -1\n```\nO(log log n) on uniform data, O(n) worst case on skewed input.",
        "When does interpolation search beat plain binary search?",
        "When the data is approximately uniformly distributed and very large -- think indexed numeric keys in a database. The expected log log n is dramatically better than log n at scale (e.g. 5 vs 32 for n=2^32). On non-uniform data (logarithmic, exponential), interpolation can degrade to O(n), so binary search is the safer default.")

    add("algo",
        "Implement a function that returns a deep copy of a graph (adjacency dict).",
        "```python\ndef deep_copy_graph(adj: dict) -> dict:\n    return {u: list(neighbors) for u, neighbors in adj.items()}\n```\nThis copies the outer dict and each neighbor list. If neighbors carry mutable data, recurse or use `copy.deepcopy`.",
        "Why not just use copy.deepcopy directly?",
        "You could -- it's correct and concise. The hand-rolled version is faster for simple structures because deepcopy uses a memo dict and reflection to handle cycles and arbitrary types, which adds overhead. For a flat adjacency dict with primitive node IDs, the comprehension is roughly an order of magnitude faster on benchmarks.")

    add("algo",
        "Write a function that simulates dice rolls and returns the empirical distribution.",
        "```python\nimport random\nfrom collections import Counter\n\ndef roll_dist(n: int, sides: int = 6) -> dict[int, float]:\n    rolls = [random.randint(1, sides) for _ in range(n)]\n    counts = Counter(rolls)\n    return {face: counts[face] / n for face in range(1, sides + 1)}\n```\nReturn a dict mapping face -> empirical probability.",
        "How would I assess if the die is fair?",
        "Run a chi-squared goodness-of-fit test against the uniform expected counts: `scipy.stats.chisquare(observed, expected)`. A p-value below your threshold (commonly 0.05) suggests bias. For small samples, results are noisy -- you usually need at least a few hundred rolls per face for the chi-squared approximation to be valid.")

    add("algo",
        "Implement a function that performs the Floyd-Warshall all-pairs shortest path.",
        "```python\nfrom math import inf\n\ndef floyd_warshall(n: int, edges: list[tuple]) -> list[list[float]]:\n    dist = [[inf] * n for _ in range(n)]\n    for i in range(n):\n        dist[i][i] = 0\n    for u, v, w in edges:\n        dist[u][v] = min(dist[u][v], w)\n    for k in range(n):\n        for i in range(n):\n            for j in range(n):\n                if dist[i][k] + dist[k][j] < dist[i][j]:\n                    dist[i][j] = dist[i][k] + dist[k][j]\n    return dist\n```\nO(n^3). Detects negative cycles by checking `dist[i][i] < 0` after the run.",
        "When is Floyd-Warshall the right choice over running Dijkstra n times?",
        "For dense graphs where E is close to V^2, Floyd-Warshall's O(V^3) is comparable to Dijkstra-with-binary-heap's O(V * E log V) and the constant factor is much smaller -- no heap operations, just three nested loops, very cache-friendly. For sparse graphs, Dijkstra-from-each-node wins. Floyd-Warshall also handles negative weights (no negative cycles).")

    add("algo",
        "Write a function that returns the longest zigzag subsequence length.",
        "```python\ndef longest_zigzag(xs: list[int]) -> int:\n    if len(xs) < 2:\n        return len(xs)\n    up = down = 1\n    for i in range(1, len(xs)):\n        if xs[i] > xs[i-1]:\n            up = down + 1\n        elif xs[i] < xs[i-1]:\n            down = up + 1\n    return max(up, down)\n```\nO(n) DP with two state variables.",
        "How would I extend this to allow equal-element runs as 'flat' steps?",
        "Add a `flat` state. When `xs[i] == xs[i-1]`, `flat = max(up, down) + 1`. When transitioning from a flat run to up or down, allow `flat` as a predecessor: `up = max(down, flat) + 1`. The semantics depend on whether you want flats to count as part of zigzag length or just be skipped -- pin that down before coding.")

    add("algo",
        "Implement a function that performs fast modular exponentiation.",
        "```python\ndef mod_pow(base: int, exp: int, mod: int) -> int:\n    return pow(base, exp, mod)\n```\nPython's three-arg `pow` is the fastest implementation -- it's in C and uses sliding-window exponentiation.",
        "What if I need to roll my own for educational purposes?",
        "Standard right-to-left binary exponentiation: `result = 1; base %= mod; while exp: if exp & 1: result = result * base % mod; base = base * base % mod; exp >>= 1`. The `% mod` after each multiplication keeps numbers bounded. For very large exponents (cryptographic sizes) the built-in is dramatically faster.")

    add("algo",
        "Write a function that returns the longest valid parentheses substring length.",
        "```python\ndef longest_valid_parens(s: str) -> int:\n    stack: list[int] = [-1]\n    best = 0\n    for i, c in enumerate(s):\n        if c == '(':\n            stack.append(i)\n        else:\n            stack.pop()\n            if not stack:\n                stack.append(i)\n            else:\n                best = max(best, i - stack[-1])\n    return best\n```\nThe stack stores indices; the bottom is always 'last unmatched index'.",
        "Is there an O(1)-space solution?",
        "Yes -- two passes, each tracking opens/closes counters. Left-to-right: increment on '(', decrement on ')'; if equal, update best with `2*opens`; if closes > opens, reset both to 0. Right-to-left mirror handles the case where the string ends with unmatched '('. Correct, neat, and uses O(1) extra space.")

    add("algo",
        "Implement a function that performs Boyer-Moore majority vote.",
        "```python\ndef majority(xs: list[int]) -> int | None:\n    cand = None; count = 0\n    for x in xs:\n        if count == 0:\n            cand = x; count = 1\n        elif x == cand:\n            count += 1\n        else:\n            count -= 1\n    if cand is None or xs.count(cand) <= len(xs) // 2:\n        return None\n    return cand\n```\nO(n) time, O(1) space. Second pass verifies the candidate is actually a majority.",
        "Does this work for finding the top-k frequent elements?",
        "There's a generalization (Misra-Gries) that finds all elements appearing more than n/(k+1) times using k counters. It's used in approximate streaming algorithms. For exact top-k by frequency you can't beat hash-based counting in general -- O(n) time with O(distinct) space.")

    add("algo",
        "Write a function that decodes a string with k[encoded] notation: '3[a]2[bc]' -> 'aaabcbc'.",
        "```python\ndef decode_str(s: str) -> str:\n    stack: list = []\n    cur = ''\n    k = 0\n    for c in s:\n        if c.isdigit():\n            k = k * 10 + int(c)\n        elif c == '[':\n            stack.append((cur, k))\n            cur = ''; k = 0\n        elif c == ']':\n            prev, repeat = stack.pop()\n            cur = prev + cur * repeat\n        else:\n            cur += c\n    return cur\n```\nA stack of (prefix, multiplier) pairs handles nested brackets.",
        "How would I make this resistant to ridiculous inputs like '999999999[a]'?",
        "Add a maximum-output-length guard. Track the running expanded size and raise an exception if it exceeds a budget. Otherwise an attacker can craft tiny inputs that explode into gigabytes. Same defensive thinking applies to recursive expansions in `re`, JSON unmarshaling, etc.")
