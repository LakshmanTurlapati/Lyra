# SPDX-License-Identifier: MIT
"""Algorithms / data structures / utilities entries (150 total).
60 of these have follow-ups (5-msg), 90 are 3-msg.
"""


def register_all(add):
    # ---- 3-msg (90) ----
    add("algo",
        "Write a function that converts camelCase or PascalCase strings to snake_case.",
        "```python\nimport re\n\ndef camel_to_snake(name: str) -> str:\n    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\\1_\\2', name)\n    return re.sub(r'([a-z0-9])([A-Z])', r'\\1_\\2', s1).lower()\n```\nTwo regex passes handle both `HTTPRequest` and `getHTTPResponse` styles cleanly.")

    add("algo",
        "Write a function that returns the n-th Fibonacci number iteratively.",
        "```python\ndef fib(n: int) -> int:\n    if n < 0:\n        raise ValueError('n must be non-negative')\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```\nO(n) time, O(1) space. Iteration avoids the recursion-depth limit you'd hit around n=1000.")

    add("algo",
        "Implement binary search over a sorted list. Return the index or -1.",
        "```python\ndef binary_search(arr: list[int], target: int) -> int:\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        if arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n```\nUsing `(lo + hi) // 2` is safe in Python (no integer overflow) and clearer than the bit-shift trick.")

    add("algo",
        "Write a function that flattens an arbitrarily nested list of integers.",
        "```python\ndef flatten(items):\n    out = []\n    for x in items:\n        if isinstance(x, list):\n            out.extend(flatten(x))\n        else:\n            out.append(x)\n    return out\n```\nRecursive descent keeps the code small. For very deep nesting use an explicit stack to avoid hitting Python's recursion limit.")

    add("algo",
        "Write a one-liner that returns the most common element in a list.",
        "```python\nfrom collections import Counter\n\ndef most_common(xs):\n    return Counter(xs).most_common(1)[0][0]\n```\n`Counter.most_common(1)` is O(n) and handles ties by insertion order.")

    add("algo",
        "Implement a function that checks whether a string is a palindrome, ignoring case and non-alphanumeric characters.",
        "```python\ndef is_palindrome(s: str) -> bool:\n    cleaned = [c.lower() for c in s if c.isalnum()]\n    return cleaned == cleaned[::-1]\n```\nList comprehension + slice reversal is the most readable approach. For huge strings, compare two pointers from the ends instead.")

    add("algo",
        "Write a function that returns the longest common prefix of a list of strings.",
        "```python\ndef longest_common_prefix(strs: list[str]) -> str:\n    if not strs:\n        return ''\n    shortest = min(strs, key=len)\n    for i, ch in enumerate(shortest):\n        if any(s[i] != ch for s in strs):\n            return shortest[:i]\n    return shortest\n```\nIterating over the shortest string bounds the work and avoids index errors.")

    add("algo",
        "Write a function that reverses the words in a sentence but keeps individual words intact.",
        "```python\ndef reverse_words(s: str) -> str:\n    return ' '.join(reversed(s.split()))\n```\n`split()` (no args) collapses runs of whitespace and trims edges, which is usually what you want.")

    add("algo",
        "Implement run-length encoding for a string. Example: 'aaabbc' -> 'a3b2c1'.",
        "```python\nfrom itertools import groupby\n\ndef rle(s: str) -> str:\n    return ''.join(f'{ch}{sum(1 for _ in grp)}' for ch, grp in groupby(s))\n```\n`itertools.groupby` is built for this — it groups consecutive equal characters.")

    add("algo",
        "Write a function that returns the GCD of two integers using Euclid's algorithm.",
        "```python\ndef gcd(a: int, b: int) -> int:\n    a, b = abs(a), abs(b)\n    while b:\n        a, b = b, a % b\n    return a\n```\nThe standard library has `math.gcd`; this implementation is here for clarity.")

    add("algo",
        "Write a Python function that returns all permutations of a string as a list.",
        "```python\nfrom itertools import permutations\n\ndef all_perms(s: str) -> list[str]:\n    return [''.join(p) for p in permutations(s)]\n```\n`itertools.permutations` returns tuples; join them back into strings. This is O(n!) so only use it for small inputs.")

    add("algo",
        "Write a function that finds two numbers in a list that sum to a target. Return their indices.",
        "```python\ndef two_sum(nums: list[int], target: int) -> tuple[int, int] | None:\n    seen: dict[int, int] = {}\n    for i, x in enumerate(nums):\n        if target - x in seen:\n            return seen[target - x], i\n        seen[x] = i\n    return None\n```\nOne pass, O(n) time. The dict maps value -> earliest index.")

    add("algo",
        "Implement a function that checks whether two strings are anagrams.",
        "```python\nfrom collections import Counter\n\ndef are_anagrams(a: str, b: str) -> bool:\n    return Counter(a) == Counter(b)\n```\nCounter equality is O(n) and reads well. If you need to ignore case/whitespace, normalize first.")

    add("algo",
        "Write a function that returns the first non-repeating character in a string, or None.",
        "```python\nfrom collections import Counter\n\ndef first_unique(s: str) -> str | None:\n    counts = Counter(s)\n    return next((c for c in s if counts[c] == 1), None)\n```\nPython 3.7+ dicts preserve insertion order, so we can scan `s` directly.")

    add("algo",
        "Write a function that computes the Levenshtein distance between two strings.",
        "```python\ndef levenshtein(a: str, b: str) -> int:\n    if len(a) < len(b):\n        a, b = b, a\n    prev = list(range(len(b) + 1))\n    for i, ca in enumerate(a, 1):\n        curr = [i]\n        for j, cb in enumerate(b, 1):\n            curr.append(min(\n                prev[j] + 1,\n                curr[j-1] + 1,\n                prev[j-1] + (ca != cb),\n            ))\n        prev = curr\n    return prev[-1]\n```\nOnly two rows are needed at a time, keeping memory at O(min(len(a), len(b))).")

    add("algo",
        "Write a function that merges two sorted lists into one sorted list.",
        "```python\ndef merge(a: list[int], b: list[int]) -> list[int]:\n    out = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    out.extend(a[i:])\n    out.extend(b[j:])\n    return out\n```\nO(n+m). For >2 sorted iterables, prefer `heapq.merge`.")

    add("algo",
        "Implement quicksort in Python with a clear pivot strategy.",
        "```python\nimport random\n\ndef quicksort(xs: list[int]) -> list[int]:\n    if len(xs) <= 1:\n        return xs\n    pivot = xs[random.randrange(len(xs))]\n    lo = [x for x in xs if x < pivot]\n    eq = [x for x in xs if x == pivot]\n    hi = [x for x in xs if x > pivot]\n    return quicksort(lo) + eq + quicksort(hi)\n```\nRandom pivot avoids worst-case O(n^2) on already-sorted input. Three-way partition handles duplicates well.")

    add("algo",
        "Write a function that finds the k smallest numbers in an unsorted list.",
        "```python\nimport heapq\n\ndef k_smallest(xs: list[int], k: int) -> list[int]:\n    return heapq.nsmallest(k, xs)\n```\n`nsmallest` runs in O(n log k) — much better than sorting the whole list when k is small.")

    add("algo",
        "Implement a stack with O(1) min() lookup.",
        "```python\nclass MinStack:\n    def __init__(self):\n        self._stack: list[int] = []\n        self._mins: list[int] = []\n\n    def push(self, x: int) -> None:\n        self._stack.append(x)\n        self._mins.append(x if not self._mins else min(x, self._mins[-1]))\n\n    def pop(self) -> int:\n        self._mins.pop()\n        return self._stack.pop()\n\n    def min(self) -> int:\n        return self._mins[-1]\n```\nThe parallel `_mins` stack tracks the running minimum so every operation is O(1).")

    add("algo",
        "Write a function that detects a cycle in a singly linked list.",
        "```python\nclass Node:\n    def __init__(self, val, nxt=None):\n        self.val = val\n        self.nxt = nxt\n\ndef has_cycle(head: Node | None) -> bool:\n    slow = fast = head\n    while fast and fast.nxt:\n        slow = slow.nxt\n        fast = fast.nxt.nxt\n        if slow is fast:\n            return True\n    return False\n```\nFloyd's tortoise-and-hare uses O(1) extra space. If there's a cycle the fast pointer eventually laps the slow one.")

    add("algo",
        "Write a function that returns the unique values from a list while preserving order.",
        "```python\ndef unique(xs):\n    seen = set()\n    out = []\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\n```\nSince Python 3.7 you can also do `list(dict.fromkeys(xs))`, which is just as ordered and a bit shorter.")

    add("algo",
        "Write a function that returns True if a number is prime.",
        "```python\ndef is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    if n % 2 == 0:\n        return n == 2\n    i = 3\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 2\n    return True\n```\nTrial division up to sqrt(n) skipping evens. For very large n, use Miller-Rabin.")

    add("algo",
        "Implement the Sieve of Eratosthenes to list primes up to n.",
        "```python\ndef primes_up_to(n: int) -> list[int]:\n    if n < 2:\n        return []\n    sieve = [True] * (n + 1)\n    sieve[0] = sieve[1] = False\n    for i in range(2, int(n**0.5) + 1):\n        if sieve[i]:\n            for j in range(i*i, n + 1, i):\n                sieve[j] = False\n    return [i for i, ok in enumerate(sieve) if ok]\n```\nStarting the inner loop at i*i is the classic small optimization — smaller multiples have already been crossed out.")

    add("algo",
        "Write a function that rotates a list to the right by k places.",
        "```python\ndef rotate(xs: list, k: int) -> list:\n    if not xs:\n        return xs\n    k %= len(xs)\n    return xs[-k:] + xs[:-k]\n```\nSlicing makes this a one-liner; the modulo keeps oversized k values from doing unnecessary work.")

    add("algo",
        "Write a function that chunks an iterable into lists of size n.",
        "```python\nfrom itertools import islice\n\ndef chunks(iterable, n: int):\n    it = iter(iterable)\n    while True:\n        batch = list(islice(it, n))\n        if not batch:\n            return\n        yield batch\n```\nA generator works for any iterable (not just sequences) and lazily yields chunks without buffering the whole input.")

    add("algo",
        "Write a function that returns the longest substring without repeating characters.",
        "```python\ndef longest_unique_substring(s: str) -> str:\n    last: dict[str, int] = {}\n    start = best_start = best_len = 0\n    for i, ch in enumerate(s):\n        if ch in last and last[ch] >= start:\n            start = last[ch] + 1\n        last[ch] = i\n        if i - start + 1 > best_len:\n            best_len = i - start + 1\n            best_start = start\n    return s[best_start:best_start + best_len]\n```\nClassic sliding window. `last` tracks the most recent index of each character so we can jump `start` forward in O(1).")

    add("algo",
        "Implement a basic LRU cache class with `get` and `put` operations.",
        "```python\nfrom collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity: int):\n        self.cap = capacity\n        self.data: OrderedDict = OrderedDict()\n\n    def get(self, key):\n        if key not in self.data:\n            return None\n        self.data.move_to_end(key)\n        return self.data[key]\n\n    def put(self, key, value) -> None:\n        if key in self.data:\n            self.data.move_to_end(key)\n        self.data[key] = value\n        if len(self.data) > self.cap:\n            self.data.popitem(last=False)\n```\n`OrderedDict.move_to_end` and `popitem(last=False)` give you LRU semantics in O(1) per operation.")

    add("algo",
        "Write a function that returns the power set of a list.",
        "```python\nfrom itertools import combinations\n\ndef power_set(xs):\n    return [list(c) for r in range(len(xs) + 1) for c in combinations(xs, r)]\n```\nThe power set has 2^n elements — feasible only for small n.")

    add("algo",
        "Write a function to validate a balanced-parentheses string supporting (), [], {}.",
        "```python\ndef is_balanced(s: str) -> bool:\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack: list[str] = []\n    for ch in s:\n        if ch in '([{':\n            stack.append(ch)\n        elif ch in pairs:\n            if not stack or stack.pop() != pairs[ch]:\n                return False\n    return not stack\n```\nClassic stack pattern. Non-bracket characters are simply ignored, which mirrors how most balance-checkers behave.")

    add("algo",
        "Write a function that returns the depth of a nested dictionary.",
        "```python\ndef dict_depth(d) -> int:\n    if not isinstance(d, dict) or not d:\n        return 0\n    return 1 + max(dict_depth(v) for v in d.values())\n```\nBase case is non-dict or empty dict (depth 0). Recursion handles arbitrary nesting up to Python's recursion limit.")

    add("algo",
        "Write a function that returns the integer square root of n (floor).",
        "```python\ndef isqrt(n: int) -> int:\n    if n < 0:\n        raise ValueError('n must be non-negative')\n    x = n\n    y = (x + 1) // 2\n    while y < x:\n        x = y\n        y = (x + n // x) // 2\n    return x\n```\nNewton's method converges quadratically. Python 3.8+ also ships `math.isqrt` — prefer that in production.")

    add("algo",
        "Implement a function that counts the number of inversions in a list using merge sort.",
        "```python\ndef count_inversions(xs: list[int]) -> int:\n    def merge(a, b):\n        i = j = inv = 0\n        out = []\n        while i < len(a) and j < len(b):\n            if a[i] <= b[j]:\n                out.append(a[i]); i += 1\n            else:\n                out.append(b[j]); j += 1\n                inv += len(a) - i\n        out.extend(a[i:]); out.extend(b[j:])\n        return out, inv\n\n    def sort(xs):\n        if len(xs) <= 1:\n            return xs, 0\n        m = len(xs) // 2\n        l, il = sort(xs[:m])\n        r, ir = sort(xs[m:])\n        merged, im = merge(l, r)\n        return merged, il + ir + im\n\n    return sort(xs)[1]\n```\nCounting during merge gives O(n log n) — far better than the O(n^2) brute force.")

    add("algo",
        "Write a function that returns the Cartesian product of two iterables as a list of tuples.",
        "```python\nfrom itertools import product\n\ndef cartesian(a, b):\n    return list(product(a, b))\n```\n`itertools.product` handles any number of arguments and accepts a `repeat=` argument for self-products.")

    add("algo",
        "Write a function that converts an integer to a base between 2 and 36.",
        "```python\ndef to_base(n: int, base: int) -> str:\n    if not 2 <= base <= 36:\n        raise ValueError('base must be in [2, 36]')\n    if n == 0:\n        return '0'\n    digits = '0123456789abcdefghijklmnopqrstuvwxyz'\n    sign = '-' if n < 0 else ''\n    n = abs(n)\n    out = []\n    while n:\n        out.append(digits[n % base])\n        n //= base\n    return sign + ''.join(reversed(out))\n```\nThe digit alphabet supports up to base-36 (the conventional cap for alphanumeric digit characters).")

    add("algo",
        "Write a function that returns the moving average of a list with window size k.",
        "```python\nfrom collections import deque\n\ndef moving_average(xs: list[float], k: int) -> list[float]:\n    if k <= 0:\n        raise ValueError('k must be positive')\n    window: deque[float] = deque(maxlen=k)\n    out = []\n    s = 0.0\n    for x in xs:\n        if len(window) == k:\n            s -= window[0]\n        window.append(x)\n        s += x\n        if len(window) == k:\n            out.append(s / k)\n    return out\n```\nO(n) using a rolling sum. Output length is `len(xs) - k + 1`.")

    add("algo",
        "Implement a function that returns the n-th row of Pascal's triangle.",
        "```python\ndef pascal_row(n: int) -> list[int]:\n    row = [1]\n    for k in range(n):\n        row.append(row[-1] * (n - k) // (k + 1))\n    return row\n```\nUses the multiplicative recurrence C(n, k+1) = C(n, k) * (n-k) / (k+1) to avoid recomputing factorials.")

    add("algo",
        "Write a Python implementation of bubble sort.",
        "```python\ndef bubble_sort(xs: list[int]) -> list[int]:\n    a = list(xs)\n    n = len(a)\n    for i in range(n):\n        swapped = False\n        for j in range(n - i - 1):\n            if a[j] > a[j + 1]:\n                a[j], a[j + 1] = a[j + 1], a[j]\n                swapped = True\n        if not swapped:\n            break\n    return a\n```\nThe `swapped` flag lets the loop short-circuit on already-sorted input. Useful as a teaching example, but use `sorted()` in real code.")

    add("algo",
        "Write a function that compresses consecutive duplicates of a list into [(value, count)] pairs.",
        "```python\nfrom itertools import groupby\n\ndef compress(xs):\n    return [(k, sum(1 for _ in g)) for k, g in groupby(xs)]\n```\n`groupby` only groups consecutive equal items, which matches what 'compress' usually means.")

    add("algo",
        "Implement a class that yields the first n primes lazily.",
        "```python\nclass Primes:\n    def __init__(self, count: int):\n        self.count = count\n\n    def __iter__(self):\n        n = 2\n        found = 0\n        while found < self.count:\n            if all(n % p != 0 for p in range(2, int(n**0.5) + 1)):\n                yield n\n                found += 1\n            n += 1\n\nprimes_10 = list(Primes(10))\n```\nGenerator semantics give callers control over how many to consume. For high counts, switch to a sieve.")

    add("algo",
        "Write a function that finds the longest increasing subsequence length of a list.",
        "```python\nfrom bisect import bisect_left\n\ndef lis_length(xs: list[int]) -> int:\n    tails: list[int] = []\n    for x in xs:\n        i = bisect_left(tails, x)\n        if i == len(tails):\n            tails.append(x)\n        else:\n            tails[i] = x\n    return len(tails)\n```\n`tails[i]` holds the smallest possible tail for any increasing subsequence of length i+1. O(n log n).")

    add("algo",
        "Write a function that converts a list of dicts into a CSV string.",
        "```python\nimport csv\nimport io\n\ndef dicts_to_csv(rows: list[dict]) -> str:\n    if not rows:\n        return ''\n    buf = io.StringIO()\n    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))\n    writer.writeheader()\n    writer.writerows(rows)\n    return buf.getvalue()\n```\nUsing `csv.DictWriter` plus an in-memory `StringIO` buffer keeps the function pure (no file I/O).")

    add("algo",
        "Write a function that deep-merges two dictionaries.",
        "```python\ndef deep_merge(a: dict, b: dict) -> dict:\n    out = dict(a)\n    for k, v in b.items():\n        if k in out and isinstance(out[k], dict) and isinstance(v, dict):\n            out[k] = deep_merge(out[k], v)\n        else:\n            out[k] = v\n    return out\n```\nSecond argument wins on leaf collisions. Returns a new dict so caller's inputs aren't mutated.")

    add("algo",
        "Write a function that returns the Hamming distance between two equal-length strings.",
        "```python\ndef hamming(a: str, b: str) -> int:\n    if len(a) != len(b):\n        raise ValueError('strings must have equal length')\n    return sum(x != y for x, y in zip(a, b))\n```\n`zip` + boolean sum is the cleanest expression. For DNA at scale, prefer numpy.")

    add("algo",
        "Write a function that converts seconds to a 'HH:MM:SS' string.",
        "```python\ndef format_seconds(s: int) -> str:\n    h, rem = divmod(s, 3600)\n    m, s = divmod(rem, 60)\n    return f'{h:02d}:{m:02d}:{s:02d}'\n```\n`divmod` plus zero-padded f-strings keeps the formatting tight. For >24h values, hours just keep growing past 24 — change the format if you'd rather wrap.")

    add("algo",
        "Write a function that produces a frequency histogram (dict) of words in a string, normalized to lowercase.",
        "```python\nimport re\nfrom collections import Counter\n\ndef word_freq(text: str) -> dict[str, int]:\n    return dict(Counter(re.findall(r\"[a-z']+\", text.lower())))\n```\nThe regex `[a-z']+` keeps contractions together (e.g., 'don't' stays one token).")

    add("algo",
        "Write a function that interleaves two lists element-by-element. Extra trailing items go at the end.",
        "```python\nfrom itertools import zip_longest\n\n_SENTINEL = object()\n\ndef interleave(a, b):\n    out = []\n    for x, y in zip_longest(a, b, fillvalue=_SENTINEL):\n        if x is not _SENTINEL:\n            out.append(x)\n        if y is not _SENTINEL:\n            out.append(y)\n    return out\n```\nUsing a sentinel rather than `None` lets the inputs themselves contain `None` values without ambiguity.")

    add("algo",
        "Write a function that checks if a list is monotonic (entirely non-decreasing or non-increasing).",
        "```python\ndef is_monotonic(xs: list) -> bool:\n    inc = all(a <= b for a, b in zip(xs, xs[1:]))\n    dec = all(a >= b for a, b in zip(xs, xs[1:]))\n    return inc or dec\n```\nTwo passes but very readable. If memory matters, fold both checks into a single loop.")

    add("algo",
        "Write a function that returns the maximum subarray sum (Kadane's algorithm).",
        "```python\ndef max_subarray(xs: list[int]) -> int:\n    if not xs:\n        raise ValueError('empty input')\n    best = curr = xs[0]\n    for x in xs[1:]:\n        curr = max(x, curr + x)\n        best = max(best, curr)\n    return best\n```\nO(n) and elegant. The recurrence is: best ending here = max(start fresh, extend previous).")

    add("algo",
        "Write a function that swaps the keys and values of a dict, raising on duplicate values.",
        "```python\ndef invert(d: dict) -> dict:\n    out: dict = {}\n    for k, v in d.items():\n        if v in out:\n            raise ValueError(f'duplicate value: {v!r}')\n        out[v] = k\n    return out\n```\nDuplicates would silently lose data, so an explicit error is safer than `{v: k for k, v in d.items()}`.")

    add("algo",
        "Write a function that pads a string with spaces to a given width and centers it.",
        "```python\ndef center(s: str, width: int) -> str:\n    return s.center(width)\n```\n`str.center` is built in. If you want a non-space fill, pass it as the second arg: `s.center(width, '-')`.")

    add("algo",
        "Write a function that applies a function to each value of a dict, returning a new dict.",
        "```python\ndef map_values(d: dict, fn):\n    return {k: fn(v) for k, v in d.items()}\n```\nDict comprehension is the idiomatic answer. Generic enough to handle any callable.")

    add("algo",
        "Write a function that pairs adjacent elements of a list: [1,2,3,4] -> [(1,2),(2,3),(3,4)].",
        "```python\nfrom itertools import tee\n\ndef pairwise(xs):\n    a, b = tee(xs)\n    next(b, None)\n    return list(zip(a, b))\n```\nOn Python 3.10+, prefer `itertools.pairwise(xs)` directly.")

    add("algo",
        "Write a function that sorts a list of dicts by a given key.",
        "```python\nfrom operator import itemgetter\n\ndef sort_by(rows: list[dict], key: str) -> list[dict]:\n    return sorted(rows, key=itemgetter(key))\n```\n`operator.itemgetter` is faster than a lambda and reads as 'sort by this field'.")

    add("algo",
        "Write a function that returns True if two lists contain the same elements regardless of order.",
        "```python\nfrom collections import Counter\n\ndef same_multiset(a, b) -> bool:\n    return Counter(a) == Counter(b)\n```\nUsing `Counter` (not `set`) ensures duplicate counts must match too.")

    add("algo",
        "Write a function that returns the index of the first occurrence of a sublist in a list, or -1.",
        "```python\ndef find_sublist(haystack: list, needle: list) -> int:\n    if not needle:\n        return 0\n    n = len(needle)\n    for i in range(len(haystack) - n + 1):\n        if haystack[i:i+n] == needle:\n            return i\n    return -1\n```\nO(n*m) worst case but very simple. For large inputs use KMP or similar.")

    add("algo",
        "Write a function that converts a hex color like '#ff8800' to an (r, g, b) tuple.",
        "```python\ndef hex_to_rgb(color: str) -> tuple[int, int, int]:\n    s = color.lstrip('#')\n    if len(s) != 6:\n        raise ValueError('expected 6-digit hex color')\n    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)\n```\n`lstrip('#')` makes the leading hash optional.")

    add("algo",
        "Write a function that converts an (r, g, b) tuple to a hex color string.",
        "```python\ndef rgb_to_hex(rgb: tuple[int, int, int]) -> str:\n    r, g, b = rgb\n    return f'#{r:02x}{g:02x}{b:02x}'\n```\nThe `:02x` format spec produces two-digit lowercase hex with zero padding.")

    add("algo",
        "Write a function that picks N items uniformly at random from an iterable in one pass (reservoir sampling).",
        "```python\nimport random\n\ndef reservoir_sample(iterable, n: int) -> list:\n    sample = []\n    for i, item in enumerate(iterable):\n        if i < n:\n            sample.append(item)\n        else:\n            j = random.randrange(i + 1)\n            if j < n:\n                sample[j] = item\n    return sample\n```\nWorks for streams of unknown length. Each item ends up in the sample with probability n/total.")

    add("algo",
        "Write a function that returns the depth-first traversal of a tree represented as nested dicts {value, children}.",
        "```python\ndef dfs(node) -> list:\n    out = [node['value']]\n    for child in node.get('children', []):\n        out.extend(dfs(child))\n    return out\n```\nPre-order traversal. Switch the order of append/recurse for post-order.")

    add("algo",
        "Write a function that returns the breadth-first traversal of a tree represented as nested dicts.",
        "```python\nfrom collections import deque\n\ndef bfs(root) -> list:\n    out = []\n    q = deque([root])\n    while q:\n        node = q.popleft()\n        out.append(node['value'])\n        q.extend(node.get('children', []))\n    return out\n```\n`collections.deque` gives O(1) popleft, which a list does not.")

    add("algo",
        "Write a function that determines whether a year is a leap year.",
        "```python\ndef is_leap(year: int) -> bool:\n    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)\n```\nDivisible by 4, except centuries unless also divisible by 400. The standard rule.")

    add("algo",
        "Write a function that flattens a dict with nested dicts using dot-separated keys.",
        "```python\ndef flatten_dict(d: dict, prefix: str = '') -> dict:\n    out = {}\n    for k, v in d.items():\n        key = f'{prefix}.{k}' if prefix else k\n        if isinstance(v, dict):\n            out.update(flatten_dict(v, key))\n        else:\n            out[key] = v\n    return out\n```\nUseful for shipping nested config to systems that only accept flat keys (like some metric backends).")

    add("algo",
        "Write a function that unflattens a dot-separated dict back into nested form.",
        "```python\ndef unflatten(d: dict) -> dict:\n    out: dict = {}\n    for key, value in d.items():\n        parts = key.split('.')\n        node = out\n        for p in parts[:-1]:\n            node = node.setdefault(p, {})\n        node[parts[-1]] = value\n    return out\n```\nThe inverse of `flatten_dict`. `setdefault` builds intermediate levels lazily.")

    add("algo",
        "Implement a function to compute factorial without recursion.",
        "```python\ndef factorial(n: int) -> int:\n    if n < 0:\n        raise ValueError('n must be non-negative')\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n```\nLoop avoids stack overflow for big n. The standard library's `math.factorial` is even faster for large n.")

    add("algo",
        "Write a function that converts a list of (key, value) pairs into a dict-of-lists, grouping by key.",
        "```python\nfrom collections import defaultdict\n\ndef group_pairs(pairs):\n    out = defaultdict(list)\n    for k, v in pairs:\n        out[k].append(v)\n    return dict(out)\n```\n`defaultdict(list)` auto-creates empty lists on first access. Convert back to a plain dict on return.")

    add("algo",
        "Write a function that returns the n-th triangular number.",
        "```python\ndef triangular(n: int) -> int:\n    if n < 0:\n        raise ValueError('n must be non-negative')\n    return n * (n + 1) // 2\n```\nClosed-form. Integer division ensures we return an int even though one of n, n+1 is even.")

    add("algo",
        "Write a function that returns whether a list contains duplicate values.",
        "```python\ndef has_duplicates(xs) -> bool:\n    seen = set()\n    for x in xs:\n        if x in seen:\n            return True\n        seen.add(x)\n    return False\n```\nShort-circuits on the first duplicate. `len(set(xs)) != len(xs)` is shorter but always scans the whole list.")

    add("algo",
        "Write a function that returns a list of indices where a predicate holds.",
        "```python\nfrom typing import Callable, TypeVar\n\nT = TypeVar('T')\n\ndef indices_where(xs: list[T], pred: Callable[[T], bool]) -> list[int]:\n    return [i for i, x in enumerate(xs) if pred(x)]\n```\nThe predicate signature is generic so it works for any element type.")

    add("algo",
        "Write a function that returns the transpose of a 2D list.",
        "```python\ndef transpose(matrix):\n    return [list(row) for row in zip(*matrix)]\n```\n`zip(*matrix)` is the classic transpose trick. For numpy arrays, prefer `arr.T`.")

    add("algo",
        "Write a function that rotates a square matrix 90 degrees clockwise.",
        "```python\ndef rotate90(matrix):\n    return [list(row) for row in zip(*matrix[::-1])]\n```\nReverse the rows, then transpose. For in-place rotation, do it in O(1) extra space with four-cell swaps.")

    add("algo",
        "Write a function that returns all subsets of size k from a list.",
        "```python\nfrom itertools import combinations\n\ndef subsets_of_size(xs, k: int):\n    return [list(c) for c in combinations(xs, k)]\n```\n`combinations` gives lexicographically ordered tuples; convert to lists if mutability matters.")

    add("algo",
        "Write a function that returns the Cartesian product of N iterables.",
        "```python\nfrom itertools import product\n\ndef cartesian(*iters):\n    return list(product(*iters))\n```\n`*` unpacks the iterables; each tuple in the result has one element per input.")

    add("algo",
        "Write a function that finds all duplicate elements in a list.",
        "```python\nfrom collections import Counter\n\ndef duplicates(xs):\n    return [x for x, c in Counter(xs).items() if c > 1]\n```\nOrder of the result follows insertion order (Python 3.7+).")

    add("algo",
        "Write a function that returns the symmetric difference of two iterables.",
        "```python\ndef sym_diff(a, b):\n    return list(set(a) ^ set(b))\n```\nThe `^` operator on sets is symmetric difference: items in either set but not both.")

    add("algo",
        "Write a function that joins consecutive integers into 'ranges': [1,2,3,5,6,8] -> ['1-3', '5-6', '8'].",
        "```python\ndef compact_ranges(xs):\n    if not xs:\n        return []\n    xs = sorted(set(xs))\n    out = []\n    start = end = xs[0]\n    for n in xs[1:]:\n        if n == end + 1:\n            end = n\n        else:\n            out.append(f'{start}-{end}' if start != end else str(start))\n            start = end = n\n    out.append(f'{start}-{end}' if start != end else str(start))\n    return out\n```\nDeduplicates first so repeated values don't break the consecutive check.")

    add("algo",
        "Write a function that returns the longest run of consecutive equal values in a list with its length.",
        "```python\nfrom itertools import groupby\n\ndef longest_run(xs):\n    if not xs:\n        return None, 0\n    runs = [(k, sum(1 for _ in g)) for k, g in groupby(xs)]\n    return max(runs, key=lambda r: r[1])\n```\nReturns `(value, length)`. Empty input gets `(None, 0)`.")

    add("algo",
        "Write a function that converts a Roman numeral string to an integer.",
        "```python\ndef roman_to_int(s: str) -> int:\n    vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n    total = 0\n    prev = 0\n    for ch in reversed(s):\n        v = vals[ch]\n        total += v if v >= prev else -v\n        prev = v\n    return total\n```\nIterate right-to-left and subtract anything smaller than the running max — this handles 'IV', 'IX', 'XL', etc. uniformly.")

    add("algo",
        "Write a function that returns whether one string is a rotation of another.",
        "```python\ndef is_rotation(a: str, b: str) -> bool:\n    return len(a) == len(b) and b in (a + a)\n```\nNeat trick: every rotation of `a` appears as a substring of `a + a`.")

    add("algo",
        "Write a function that converts an integer to English words for values 0-999.",
        "```python\nONES = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',\n        'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',\n        'seventeen', 'eighteen', 'nineteen']\nTENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']\n\ndef int_to_words(n: int) -> str:\n    if not 0 <= n <= 999:\n        raise ValueError('only 0-999 supported')\n    if n == 0:\n        return 'zero'\n    parts = []\n    if n >= 100:\n        parts.append(f'{ONES[n // 100]} hundred')\n        n %= 100\n    if n >= 20:\n        t = TENS[n // 10]\n        o = ONES[n % 10]\n        parts.append(f'{t}-{o}' if o else t)\n    elif n > 0:\n        parts.append(ONES[n])\n    return ' '.join(parts)\n```\nThe special-case range 10-19 is what makes English number words tricky.")

    add("algo",
        "Write a function that detects whether a string is a valid IPv4 address.",
        "```python\ndef is_ipv4(s: str) -> bool:\n    parts = s.split('.')\n    if len(parts) != 4:\n        return False\n    for p in parts:\n        if not p.isdigit() or (len(p) > 1 and p[0] == '0'):\n            return False\n        if not 0 <= int(p) <= 255:\n            return False\n    return True\n```\nWe also reject leading zeros (e.g., `01.0.0.0`) which most parsers consider non-canonical.")

    add("algo",
        "Write a function that splits an iterable into two lists based on a predicate.",
        "```python\ndef partition(xs, pred):\n    yes, no = [], []\n    for x in xs:\n        (yes if pred(x) else no).append(x)\n    return yes, no\n```\nSingle pass, preserves order in both halves.")

    add("algo",
        "Write a function that wraps an iterator with retries on failure.",
        "```python\nimport time\nfrom typing import Callable, TypeVar\n\nT = TypeVar('T')\n\ndef retry(fn: Callable[..., T], *args, attempts: int = 3, delay: float = 0.5, **kwargs) -> T:\n    last: Exception | None = None\n    for i in range(attempts):\n        try:\n            return fn(*args, **kwargs)\n        except Exception as e:\n            last = e\n            if i < attempts - 1:\n                time.sleep(delay * (2 ** i))\n    raise last  # type: ignore[misc]\n```\nExponential backoff. For richer policy, look at `tenacity`.")

    add("algo",
        "Write a function that timestamps each item in an iterable.",
        "```python\nimport time\n\ndef with_timestamps(iterable):\n    for x in iterable:\n        yield time.time(), x\n```\nGenerator, so the caller controls iteration speed and the timestamps reflect actual yield time.")

    add("algo",
        "Write a function that returns whether a Sudoku board (9x9, 0 = empty) has any rule violations.",
        "```python\ndef sudoku_valid(board):\n    def ok(group):\n        nums = [n for n in group if n != 0]\n        return len(nums) == len(set(nums))\n    rows = [board[r] for r in range(9)]\n    cols = [[board[r][c] for r in range(9)] for c in range(9)]\n    boxes = [[board[3*br+r][3*bc+c] for r in range(3) for c in range(3)]\n             for br in range(3) for bc in range(3)]\n    return all(ok(g) for g in rows + cols + boxes)\n```\nValid means: no duplicates among non-zero entries in any row, column, or 3x3 box.")

    add("algo",
        "Write a function that converts a duration like '1h30m' to seconds.",
        "```python\nimport re\n\ndef parse_duration(s: str) -> int:\n    units = {'h': 3600, 'm': 60, 's': 1}\n    total = 0\n    for num, unit in re.findall(r'(\\d+)([hms])', s):\n        total += int(num) * units[unit]\n    return total\n```\nHandles any subset of `h`, `m`, `s` in any order. Empty input returns 0.")

    add("algo",
        "Write a function that returns indices that would sort a list (like numpy.argsort).",
        "```python\ndef argsort(xs):\n    return sorted(range(len(xs)), key=lambda i: xs[i])\n```\nSort the index range using values as the key — `sorted` is a stable sort, so ties preserve input order.")

    add("algo",
        "Write a function that yields strings from a stream split on a delimiter byte, handling partial buffers.",
        "```python\ndef split_stream(chunks, delim: bytes = b'\\n'):\n    buf = b''\n    for chunk in chunks:\n        buf += chunk\n        while delim in buf:\n            line, buf = buf.split(delim, 1)\n            yield line\n    if buf:\n        yield buf\n```\nClassic line-splitting pattern for sockets / subprocess pipes where chunks don't align to record boundaries.")

    # ---- 5-msg with follow-ups (60) ----
    add("algo",
        "Write a function that returns the n-th Fibonacci number using memoization.",
        "```python\nfrom functools import lru_cache\n\n@lru_cache(maxsize=None)\ndef fib(n: int) -> int:\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n```\n`lru_cache` memoizes calls so repeated computation is cheap.",
        "Can you make it iterative instead, to avoid the recursion limit?",
        "```python\ndef fib(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```\nO(n) time, O(1) space, and no risk of hitting the recursion limit even for very large n.")

    add("algo",
        "Write a function that finds duplicates in a list and returns them.",
        "```python\nfrom collections import Counter\n\ndef duplicates(xs):\n    return [x for x, c in Counter(xs).items() if c > 1]\n```\n`Counter` tallies in one pass; the comprehension keeps insertion order (Python 3.7+).",
        "What if the list is huge and I only need to know if any duplicates exist?",
        "```python\ndef has_duplicates(xs) -> bool:\n    seen = set()\n    for x in xs:\n        if x in seen:\n            return True\n        seen.add(x)\n    return False\n```\nThis short-circuits at the first duplicate, so in the best case it's O(1) memory and one comparison.")

    add("algo",
        "Write a function that returns the GCD of a list of integers.",
        "```python\nfrom functools import reduce\nfrom math import gcd\n\ndef gcd_list(xs: list[int]) -> int:\n    return reduce(gcd, xs)\n```\n`functools.reduce` folds the binary `math.gcd` over the list.",
        "Add a sensible behavior for empty input.",
        "```python\nfrom functools import reduce\nfrom math import gcd\n\ndef gcd_list(xs: list[int]) -> int:\n    if not xs:\n        return 0\n    return reduce(gcd, xs)\n```\nReturning 0 is mathematically consistent — `gcd(0, x) == abs(x)`, so 0 is the identity for gcd.")

    add("algo",
        "Write a function that returns whether a string is a palindrome.",
        "```python\ndef is_palindrome(s: str) -> bool:\n    return s == s[::-1]\n```\nSlice-reverse is the simplest correct form for ASCII.",
        "Make it ignore case, spaces, and punctuation.",
        "```python\ndef is_palindrome(s: str) -> bool:\n    cleaned = [c.lower() for c in s if c.isalnum()]\n    return cleaned == cleaned[::-1]\n```\n`isalnum()` strips punctuation and whitespace; `lower()` removes case sensitivity.")

    add("algo",
        "Write a function that splits a list into n approximately equal chunks.",
        "```python\ndef split_into(xs, n: int):\n    k, m = divmod(len(xs), n)\n    return [xs[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]\n```\nThe `min(i, m)` trick distributes the leftover `m` items one each over the first `m` chunks.",
        "Can you turn it into a generator that streams chunks lazily?",
        "```python\ndef split_into(xs, n: int):\n    k, m = divmod(len(xs), n)\n    start = 0\n    for i in range(n):\n        size = k + (1 if i < m else 0)\n        yield xs[start:start + size]\n        start += size\n```\nSame semantics, but yields chunks one at a time rather than building the whole list.")

    add("algo",
        "Write a function that returns the longest word in a sentence.",
        "```python\ndef longest_word(sentence: str) -> str:\n    return max(sentence.split(), key=len, default='')\n```\n`max(default=...)` covers the empty-input case without an explicit branch.",
        "Now return all words tied for longest.",
        "```python\ndef longest_words(sentence: str) -> list[str]:\n    words = sentence.split()\n    if not words:\n        return []\n    n = max(len(w) for w in words)\n    return [w for w in words if len(w) == n]\n```\nOne pass to find the max length, one pass to filter.")

    add("algo",
        "Write a function to compute n-choose-k.",
        "```python\nfrom math import comb\n\ndef n_choose_k(n: int, k: int) -> int:\n    return comb(n, k)\n```\nPython 3.8+ has `math.comb` which is fast and exact for arbitrary integers.",
        "Implement it from scratch without using math.comb.",
        "```python\ndef n_choose_k(n: int, k: int) -> int:\n    if k < 0 or k > n:\n        return 0\n    k = min(k, n - k)\n    out = 1\n    for i in range(k):\n        out = out * (n - i) // (i + 1)\n    return out\n```\nComputing in a single loop using the multiplicative recurrence avoids huge intermediate factorials.")

    add("algo",
        "Write a function that converts a list of integers to a comma-separated string.",
        "```python\ndef join_ints(xs: list[int]) -> str:\n    return ','.join(str(x) for x in xs)\n```\nGenerator expression keeps memory low; `str.join` does one allocation.",
        "Add a configurable separator and an optional thousands separator within each number.",
        "```python\ndef join_ints(xs: list[int], sep: str = ',', thousands: str | None = None) -> str:\n    fmt = (lambda x: f'{x:,}'.replace(',', thousands)) if thousands else str\n    return sep.join(fmt(x) for x in xs)\n```\nThe default `f'{x:,}'` formatter gives English-style thousands; the replace lets callers pick another separator.")

    add("algo",
        "Implement a simple FIFO queue using collections.deque.",
        "```python\nfrom collections import deque\n\nclass Queue:\n    def __init__(self):\n        self._dq: deque = deque()\n\n    def enqueue(self, x):\n        self._dq.append(x)\n\n    def dequeue(self):\n        return self._dq.popleft()\n\n    def __len__(self):\n        return len(self._dq)\n```\n`deque` gives O(1) appends and popleft, unlike lists.",
        "Make it thread-safe.",
        "```python\nimport queue\n\nclass Queue:\n    def __init__(self):\n        self._q: queue.Queue = queue.Queue()\n\n    def enqueue(self, x):\n        self._q.put(x)\n\n    def dequeue(self):\n        return self._q.get()\n\n    def __len__(self):\n        return self._q.qsize()\n```\n`queue.Queue` is already thread-safe; reusing it is far better than rolling your own locking.")

    add("algo",
        "Write a function that finds the index of the maximum value in a list.",
        "```python\ndef argmax(xs):\n    return max(range(len(xs)), key=xs.__getitem__)\n```\nUsing `xs.__getitem__` as the key avoids an extra lambda and tends to be a hair faster.",
        "What if the list is empty?",
        "```python\ndef argmax(xs):\n    if not xs:\n        raise ValueError('empty input')\n    return max(range(len(xs)), key=xs.__getitem__)\n```\nRaising matches the behavior of `max([])` and is more honest than returning `None` (a caller might forget to check).")

    add("algo",
        "Write a function that returns prime factors of n with multiplicity.",
        "```python\ndef prime_factors(n: int) -> list[int]:\n    factors = []\n    d = 2\n    while d * d <= n:\n        while n % d == 0:\n            factors.append(d)\n            n //= d\n        d += 1\n    if n > 1:\n        factors.append(n)\n    return factors\n```\nClassic trial division. The final `if n > 1` catches a leftover large prime.",
        "Can you return them grouped as (prime, exponent) pairs?",
        "```python\nfrom collections import Counter\n\ndef prime_factorization(n: int) -> list[tuple[int, int]]:\n    counts = Counter(prime_factors(n))\n    return sorted(counts.items())\n```\nReuses `prime_factors` and counts multiplicities. Sorted output is stable across runs.")

    add("algo",
        "Write a function that returns whether a number is a power of two.",
        "```python\ndef is_power_of_two(n: int) -> bool:\n    return n > 0 and (n & (n - 1)) == 0\n```\nThe `n & (n-1)` trick clears the lowest set bit; if the result is zero, only one bit was set.",
        "Generalize it to powers of any base.",
        "```python\nimport math\n\ndef is_power_of(n: int, base: int) -> bool:\n    if n < 1 or base < 2:\n        return False\n    log = math.log(n, base)\n    return math.isclose(log, round(log)) and base ** round(log) == n\n```\nLog gives a candidate exponent; the round-trip check rules out floating-point edge cases.")

    add("algo",
        "Write a function that takes a list of dicts and groups them by a key.",
        "```python\nfrom collections import defaultdict\n\ndef group_by(rows: list[dict], key: str) -> dict:\n    out = defaultdict(list)\n    for row in rows:\n        out[row[key]].append(row)\n    return dict(out)\n```\n`defaultdict(list)` keeps the loop body tight.",
        "Allow the key to be a callable, not just a field name.",
        "```python\nfrom collections import defaultdict\nfrom typing import Callable, Union\n\ndef group_by(rows: list[dict], key: Union[str, Callable[[dict], object]]) -> dict:\n    fn = key if callable(key) else (lambda r: r[key])\n    out = defaultdict(list)\n    for row in rows:\n        out[fn(row)].append(row)\n    return dict(out)\n```\nMatches the API of `itertools.groupby` and is more flexible for composite keys.")

    add("algo",
        "Write a function that converts a list of (timestamp, value) tuples into a dict, keeping the latest value per timestamp.",
        "```python\ndef latest_per_ts(events):\n    out = {}\n    for ts, val in sorted(events):\n        out[ts] = val\n    return out\n```\nSorting ensures the later (timestamp, value) wins; dict assignment naturally keeps the last write.",
        "What if the timestamps are strings in ISO format that aren't lexicographically sortable in all timezones?",
        "```python\nfrom datetime import datetime\n\ndef latest_per_ts(events):\n    parsed = [(datetime.fromisoformat(ts), ts, val) for ts, val in events]\n    parsed.sort()\n    out = {}\n    for _dt, ts, val in parsed:\n        out[ts] = val\n    return out\n```\nParsing once into `datetime` makes the comparison correct across offsets while preserving the original timestamp string as the key.")

    add("algo",
        "Write a function that returns the median of a list of numbers.",
        "```python\nimport statistics\n\ndef median(xs: list[float]) -> float:\n    return statistics.median(xs)\n```\nThe stdlib's `statistics.median` handles even and odd lengths.",
        "Implement it without using statistics.",
        "```python\ndef median(xs: list[float]) -> float:\n    if not xs:\n        raise ValueError('empty input')\n    s = sorted(xs)\n    n = len(s)\n    mid = n // 2\n    if n % 2 == 1:\n        return s[mid]\n    return (s[mid - 1] + s[mid]) / 2\n```\nSort costs O(n log n); for huge inputs use a quickselect-based selection algorithm in O(n) average time.")

    add("algo",
        "Write a function that converts seconds since epoch to an ISO 8601 string in UTC.",
        "```python\nfrom datetime import datetime, timezone\n\ndef to_iso_utc(seconds: float) -> str:\n    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()\n```\nPassing `tz=timezone.utc` makes the result timezone-aware and unambiguous.",
        "Truncate to seconds and replace the offset with a 'Z' suffix.",
        "```python\nfrom datetime import datetime, timezone\n\ndef to_iso_utc(seconds: float) -> str:\n    dt = datetime.fromtimestamp(int(seconds), tz=timezone.utc)\n    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')\n```\nThe explicit `strftime` avoids `+00:00` and gives the more compact `Z` form many APIs expect.")

    add("algo",
        "Write a function that checks if a number is in a list using binary search.",
        "```python\nfrom bisect import bisect_left\n\ndef contains(arr: list[int], target: int) -> bool:\n    i = bisect_left(arr, target)\n    return i < len(arr) and arr[i] == target\n```\n`bisect_left` is the canonical 'find insertion point' primitive.",
        "How would you handle the list being unsorted?",
        "```python\ndef contains(arr: list[int], target: int) -> bool:\n    return target in arr\n```\nIf the list isn't sorted, sorting first (O(n log n)) is more expensive than a single linear scan (O(n)) just to answer membership. Use a `set` for repeated lookups.")

    add("algo",
        "Write a function that returns the second-largest distinct value in a list.",
        "```python\ndef second_largest(xs):\n    distinct = sorted(set(xs), reverse=True)\n    return distinct[1] if len(distinct) >= 2 else None\n```\nSorting after deduplication keeps it short.",
        "Do it in a single pass.",
        "```python\ndef second_largest(xs):\n    largest = second = None\n    for x in xs:\n        if largest is None or x > largest:\n            second = largest\n            largest = x\n        elif x != largest and (second is None or x > second):\n            second = x\n    return second\n```\nOne pass, O(n) time, O(1) extra space.")

    add("algo",
        "Write a function that converts a string to title case but keeps short words like 'and', 'of', 'the' lowercase (unless they're the first word).",
        "```python\nSMALL = {'a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 'in', 'of', 'on', 'or', 'the', 'to'}\n\ndef smart_title(s: str) -> str:\n    words = s.split()\n    out = []\n    for i, w in enumerate(words):\n        out.append(w.lower() if i and w.lower() in SMALL else w.capitalize())\n    return ' '.join(out)\n```\nThe first word is always capitalized regardless of which set it belongs to.",
        "What if the string contains hyphenated words like 'state-of-the-art'?",
        "```python\nSMALL = {'a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 'in', 'of', 'on', 'or', 'the', 'to'}\n\ndef smart_title(s: str) -> str:\n    def cap_token(token: str, is_first: bool) -> str:\n        parts = token.split('-')\n        return '-'.join(\n            (p.lower() if (i or not is_first) and p.lower() in SMALL else p.capitalize())\n            for i, p in enumerate(parts)\n        )\n    words = s.split()\n    return ' '.join(cap_token(w, i == 0) for i, w in enumerate(words))\n```\nWe split each whitespace-token on hyphens and apply the same rule, but only the very first sub-piece of the very first word is forced-capitalized.")

    add("algo",
        "Write a function that converts a list of records to JSON Lines (jsonl) format.",
        "```python\nimport json\n\ndef to_jsonl(rows) -> str:\n    return '\\n'.join(json.dumps(r, ensure_ascii=False) for r in rows)\n```\n`ensure_ascii=False` preserves non-ASCII characters as-is rather than escaping them.",
        "Make it write directly to a file path, opening in append mode if needed.",
        "```python\nimport json\nfrom pathlib import Path\n\ndef write_jsonl(rows, path: str | Path, append: bool = False) -> None:\n    mode = 'a' if append else 'w'\n    with open(path, mode, encoding='utf-8') as f:\n        for r in rows:\n            f.write(json.dumps(r, ensure_ascii=False))\n            f.write('\\n')\n```\nWriting line-by-line rather than building one big string keeps memory bounded for large inputs.")

    add("algo",
        "Write a function that returns the n-th element of an iterator without materializing the whole thing.",
        "```python\nfrom itertools import islice\n\ndef nth(iterable, n: int):\n    return next(islice(iterable, n, None), None)\n```\n`islice` skips the first n items lazily; `next(..., None)` gives a clean default.",
        "Raise IndexError instead of returning None when out of range.",
        "```python\nfrom itertools import islice\n\n_MISSING = object()\n\ndef nth(iterable, n: int):\n    val = next(islice(iterable, n, None), _MISSING)\n    if val is _MISSING:\n        raise IndexError(f'iterator has fewer than {n+1} items')\n    return val\n```\nUsing a sentinel object distinguishes 'absent' from 'present and equal to None'.")

    add("algo",
        "Write a function to encode a string with a Caesar cipher (shift n letters).",
        "```python\ndef caesar(s: str, n: int) -> str:\n    out = []\n    for ch in s:\n        if 'a' <= ch <= 'z':\n            out.append(chr((ord(ch) - ord('a') + n) % 26 + ord('a')))\n        elif 'A' <= ch <= 'Z':\n            out.append(chr((ord(ch) - ord('A') + n) % 26 + ord('A')))\n        else:\n            out.append(ch)\n    return ''.join(out)\n```\nNon-letters pass through unchanged. Negative `n` decrypts.",
        "Add a function to brute-force decrypt by trying all 25 shifts.",
        "```python\ndef caesar_brute_force(ciphertext: str) -> list[tuple[int, str]]:\n    return [(n, caesar(ciphertext, -n)) for n in range(1, 26)]\n```\nReturns each candidate shift and its plaintext; a human can scan for the readable one (or you can score with letter frequencies).")

    add("algo",
        "Write a function that decodes URL-percent-encoded strings.",
        "```python\nfrom urllib.parse import unquote\n\ndef decode(s: str) -> str:\n    return unquote(s)\n```\nAlways prefer `urllib.parse` over rolling your own — it handles edge cases like UTF-8 multi-byte sequences correctly.",
        "Now write the encoder for the same characters.",
        "```python\nfrom urllib.parse import quote\n\ndef encode(s: str) -> str:\n    return quote(s, safe='')\n```\nThe default `safe='/'` keeps slashes unescaped, which is wrong for query-string values; passing `safe=''` escapes everything except letters/digits.")

    add("algo",
        "Write a function that returns the angle (in degrees) between the hour and minute hands of a clock.",
        "```python\ndef clock_angle(hour: int, minute: int) -> float:\n    h = (hour % 12) * 30 + minute * 0.5\n    m = minute * 6\n    diff = abs(h - m)\n    return min(diff, 360 - diff)\n```\nHour hand: 30 degrees per hour plus 0.5 per minute. Minute hand: 6 degrees per minute. Take the smaller of the two arcs.",
        "Solve the inverse: given an angle, what minute reads that angle at hour h?",
        "```python\ndef minute_for_angle(hour: int, angle: float) -> list[float]:\n    base = (hour % 12) * 30\n    candidates = []\n    for target in (angle, -angle):\n        m = (target + base) / 5.5  # 6 - 0.5 = 5.5 deg/min relative speed\n        if 0 <= m < 60:\n            candidates.append(m)\n    return sorted(set(candidates))\n```\nThe minute hand gains on the hour hand at 5.5 degrees per minute; solve for both signs and keep solutions in [0, 60).")

    add("algo",
        "Write a function that splits a list at a value, like 'split' on strings.",
        "```python\ndef split_list(xs, sep):\n    out = [[]]\n    for x in xs:\n        if x == sep:\n            out.append([])\n        else:\n            out[-1].append(x)\n    return out\n```\nLike `'a,b,,c'.split(',')`, this preserves empty groups for consecutive separators.",
        "Add a maxsplit parameter mirroring str.split.",
        "```python\ndef split_list(xs, sep, maxsplit: int = -1):\n    out = [[]]\n    splits = 0\n    for x in xs:\n        if x == sep and (maxsplit < 0 or splits < maxsplit):\n            out.append([])\n            splits += 1\n        else:\n            out[-1].append(x)\n    return out\n```\nNegative maxsplit means unlimited (matching the stdlib's convention).")

    add("algo",
        "Write a function that returns the running sum of a list.",
        "```python\nfrom itertools import accumulate\n\ndef running_sum(xs):\n    return list(accumulate(xs))\n```\n`itertools.accumulate` defaults to addition; pass `func=` for other operators.",
        "Make it lazy.",
        "```python\nfrom itertools import accumulate\n\ndef running_sum(xs):\n    yield from accumulate(xs)\n```\nNow it's a generator — useful for huge or infinite inputs.")

    add("algo",
        "Write a function to find missing numbers from 1..n in a list.",
        "```python\ndef missing(xs: list[int], n: int) -> list[int]:\n    present = set(xs)\n    return [i for i in range(1, n + 1) if i not in present]\n```\nO(n) using a set lookup.",
        "What if n is very large and most numbers are present?",
        "```python\ndef missing(xs: list[int], n: int) -> list[int]:\n    present = [False] * (n + 1)\n    for x in xs:\n        if 1 <= x <= n:\n            present[x] = True\n    return [i for i in range(1, n + 1) if not present[i]]\n```\nA boolean array uses ~1 byte per element — tighter than a hash set when n is in the millions.")

    add("algo",
        "Write a function that returns the n-th Catalan number.",
        "```python\nfrom math import comb\n\ndef catalan(n: int) -> int:\n    return comb(2 * n, n) // (n + 1)\n```\nClosed form using `math.comb`. Catalan numbers count balanced parentheses, BSTs, etc.",
        "Avoid math.comb and compute it iteratively.",
        "```python\ndef catalan(n: int) -> int:\n    c = 1\n    for i in range(n):\n        c = c * 2 * (2 * i + 1) // (i + 2)\n    return c\n```\nUses the multiplicative recurrence C(n+1) = C(n) * 2*(2n+1)/(n+2).")

    add("algo",
        "Write a function that finds the kth largest element using a heap.",
        "```python\nimport heapq\n\ndef kth_largest(xs: list[int], k: int) -> int:\n    return heapq.nlargest(k, xs)[-1]\n```\n`nlargest` is O(n log k), and the kth largest is the last element of the result.",
        "Implement it with a manual min-heap to avoid the helper.",
        "```python\nimport heapq\n\ndef kth_largest(xs: list[int], k: int) -> int:\n    heap: list[int] = []\n    for x in xs:\n        heapq.heappush(heap, x)\n        if len(heap) > k:\n            heapq.heappop(heap)\n    return heap[0]\n```\nMaintaining a size-k min-heap keeps the smallest of the top-k at the root, which is the kth largest overall.")

    add("algo",
        "Write a function that picks a weighted random element from a list of (value, weight) pairs.",
        "```python\nimport random\n\ndef weighted_choice(items):\n    values, weights = zip(*items)\n    return random.choices(values, weights=weights, k=1)[0]\n```\n`random.choices` accepts weights directly and handles normalization internally.",
        "What if I'm doing thousands of draws from the same distribution?",
        "```python\nimport bisect\nimport random\n\ndef make_picker(items):\n    values, weights = zip(*items)\n    cum = []\n    s = 0.0\n    for w in weights:\n        s += w\n        cum.append(s)\n    total = cum[-1]\n    def pick():\n        return values[bisect.bisect_left(cum, random.random() * total)]\n    return pick\n```\nPrecomputing the cumulative weights gives each draw O(log n) using `bisect`.")

    add("algo",
        "Write a function that converts a string of digits to integer (without int()).",
        "```python\ndef parse_int(s: str) -> int:\n    if not s:\n        raise ValueError('empty')\n    sign = 1\n    i = 0\n    if s[0] in '+-':\n        sign = -1 if s[0] == '-' else 1\n        i = 1\n    n = 0\n    for ch in s[i:]:\n        if not '0' <= ch <= '9':\n            raise ValueError(f'invalid digit: {ch!r}')\n        n = n * 10 + (ord(ch) - ord('0'))\n    return sign * n\n```\nClassic teaching exercise. Real code should just call `int(s)`.",
        "Add support for a base argument up to 36.",
        "```python\ndef parse_int(s: str, base: int = 10) -> int:\n    if not 2 <= base <= 36:\n        raise ValueError('base must be in [2, 36]')\n    digits = '0123456789abcdefghijklmnopqrstuvwxyz'\n    s = s.strip()\n    sign = 1\n    i = 0\n    if s[:1] in ('+', '-'):\n        sign = -1 if s[0] == '-' else 1\n        i = 1\n    n = 0\n    for ch in s[i:].lower():\n        d = digits.find(ch)\n        if d == -1 or d >= base:\n            raise ValueError(f'invalid digit {ch!r} for base {base}')\n        n = n * base + d\n    return sign * n\n```\n`digits.find` cleanly handles letter digits A-Z (case-insensitive).")

    add("algo",
        "Write a function that returns whether a graph (adjacency list) is a tree.",
        "```python\ndef is_tree(adj: dict) -> bool:\n    if not adj:\n        return True\n    visited = set()\n    stack = [(next(iter(adj)), None)]\n    while stack:\n        node, parent = stack.pop()\n        if node in visited:\n            return False  # cycle\n        visited.add(node)\n        for nb in adj.get(node, []):\n            if nb != parent:\n                stack.append((nb, node))\n    return visited == set(adj)\n```\nTree iff connected and acyclic. We DFS and reject if we revisit a non-parent node.",
        "Modify it to return a meaningful error message instead of just False.",
        "```python\ndef check_tree(adj: dict) -> tuple[bool, str]:\n    if not adj:\n        return True, 'empty graph is a trivial tree'\n    visited = set()\n    stack = [(next(iter(adj)), None)]\n    while stack:\n        node, parent = stack.pop()\n        if node in visited:\n            return False, f'cycle detected at node {node!r}'\n        visited.add(node)\n        for nb in adj.get(node, []):\n            if nb != parent:\n                stack.append((nb, node))\n    if visited != set(adj):\n        missing = set(adj) - visited\n        return False, f'graph is disconnected; unreachable: {missing}'\n    return True, 'is a tree'\n```\nReturning a `(bool, message)` tuple makes failures actionable for callers.")

    add("algo",
        "Write a function that finds the longest palindromic substring.",
        "```python\ndef longest_palindrome(s: str) -> str:\n    if not s:\n        return ''\n    def expand(l: int, r: int) -> str:\n        while l >= 0 and r < len(s) and s[l] == s[r]:\n            l -= 1\n            r += 1\n        return s[l+1:r]\n    best = ''\n    for i in range(len(s)):\n        for cand in (expand(i, i), expand(i, i+1)):\n            if len(cand) > len(best):\n                best = cand\n    return best\n```\nExpand around every center (odd and even). O(n^2) time.",
        "Can you do it in O(n)?",
        "```python\ndef longest_palindrome(s: str) -> str:\n    # Manacher's algorithm\n    t = '^#' + '#'.join(s) + '#$'\n    p = [0] * len(t)\n    c = r = 0\n    for i in range(1, len(t) - 1):\n        mirror = 2 * c - i\n        if i < r:\n            p[i] = min(r - i, p[mirror])\n        while t[i + p[i] + 1] == t[i - p[i] - 1]:\n            p[i] += 1\n        if i + p[i] > r:\n            c, r = i, i + p[i]\n    k = max(range(len(p)), key=p.__getitem__)\n    start = (k - p[k]) // 2\n    return s[start:start + p[k]]\n```\nManacher's runs in linear time; the `^` and `$` sentinels remove the need for boundary checks.")

    add("algo",
        "Write a function that returns the Fibonacci numbers below a limit.",
        "```python\ndef fibs_below(limit: int) -> list[int]:\n    out = []\n    a, b = 0, 1\n    while a < limit:\n        out.append(a)\n        a, b = b, a + b\n    return out\n```\nLoop until we cross the limit.",
        "Convert it into an infinite generator I can take from.",
        "```python\ndef fibs():\n    a, b = 0, 1\n    while True:\n        yield a\n        a, b = b, a + b\n```\nNow callers can do `list(itertools.islice(fibs(), 10))` or `itertools.takewhile(lambda x: x < 1000, fibs())`.")

    add("algo",
        "Write a function that returns the dot product of two equal-length vectors.",
        "```python\ndef dot(a, b):\n    if len(a) != len(b):\n        raise ValueError('vectors must have equal length')\n    return sum(x * y for x, y in zip(a, b))\n```\nFor large vectors prefer numpy's `np.dot` — it's vectorized.",
        "Show the numpy version.",
        "```python\nimport numpy as np\n\ndef dot(a, b) -> float:\n    return float(np.dot(a, b))\n```\nWrapping with `float()` ensures we return a Python float rather than a numpy scalar.")

    add("algo",
        "Write a function that returns whether a number is an Armstrong number (sum of own digits to the power of digit count equals the number).",
        "```python\ndef is_armstrong(n: int) -> bool:\n    digits = str(abs(n))\n    p = len(digits)\n    return sum(int(d) ** p for d in digits) == n\n```\nA 3-digit example: 153 = 1^3 + 5^3 + 3^3.",
        "Generate all Armstrong numbers up to a limit.",
        "```python\ndef armstrong_up_to(limit: int) -> list[int]:\n    return [n for n in range(limit + 1) if is_armstrong(n)]\n```\nBrute force is fine for small limits; there are only 88 Armstrong numbers below 10^39.")

    add("algo",
        "Write a function that decodes a base-N integer given a custom digit alphabet.",
        "```python\ndef decode(s: str, alphabet: str) -> int:\n    base = len(alphabet)\n    pos = {c: i for i, c in enumerate(alphabet)}\n    n = 0\n    for ch in s:\n        if ch not in pos:\n            raise ValueError(f'invalid digit {ch!r}')\n        n = n * base + pos[ch]\n    return n\n```\nThe `pos` dict gives O(1) digit-to-value lookup.",
        "Now write the encoder.",
        "```python\ndef encode(n: int, alphabet: str) -> str:\n    if n < 0:\n        raise ValueError('negative not supported')\n    base = len(alphabet)\n    if n == 0:\n        return alphabet[0]\n    out = []\n    while n:\n        out.append(alphabet[n % base])\n        n //= base\n    return ''.join(reversed(out))\n```\nClassic 'collect remainders, reverse'. Together with `decode`, this is the basis of base62/base58 IDs.")

    add("algo",
        "Write a function that finds the median of two sorted arrays.",
        "```python\ndef median_two_sorted(a, b) -> float:\n    merged = sorted(a + b)\n    n = len(merged)\n    if n == 0:\n        raise ValueError('empty')\n    mid = n // 2\n    return merged[mid] if n % 2 else (merged[mid - 1] + merged[mid]) / 2\n```\nO((n+m) log(n+m)) — simple and clear. Skip the manual partition unless performance demands it.",
        "Can you do it in O(log(min(n, m)))?",
        "```python\ndef median_two_sorted(a, b) -> float:\n    if len(a) > len(b):\n        a, b = b, a\n    n, m = len(a), len(b)\n    if n + m == 0:\n        raise ValueError('empty')\n    half = (n + m + 1) // 2\n    lo, hi = 0, n\n    while lo <= hi:\n        i = (lo + hi) // 2\n        j = half - i\n        a_left = a[i-1] if i else float('-inf')\n        a_right = a[i] if i < n else float('inf')\n        b_left = b[j-1] if j else float('-inf')\n        b_right = b[j] if j < m else float('inf')\n        if a_left <= b_right and b_left <= a_right:\n            if (n + m) % 2:\n                return max(a_left, b_left)\n            return (max(a_left, b_left) + min(a_right, b_right)) / 2\n        if a_left > b_right:\n            hi = i - 1\n        else:\n            lo = i + 1\n    raise RuntimeError('unreachable')\n```\nBinary search the partition of `a`; the partition of `b` is determined by the half-length constraint.")

    add("algo",
        "Write a function that detects whether two rectangles overlap.",
        "```python\ndef overlap(a, b) -> bool:\n    ax1, ay1, ax2, ay2 = a\n    bx1, by1, bx2, by2 = b\n    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2\n```\nRectangles are `(x1, y1, x2, y2)` with x1<=x2, y1<=y2. Strict `<` means edges-touching is not overlap.",
        "Treat shared edges as overlapping.",
        "```python\ndef overlap(a, b) -> bool:\n    ax1, ay1, ax2, ay2 = a\n    bx1, by1, bx2, by2 = b\n    return ax1 <= bx2 and bx1 <= ax2 and ay1 <= by2 and by1 <= ay2\n```\nFlipping `<` to `<=` is the only change — semantics flow from the boundary policy.")

    add("algo",
        "Write a function that returns the closest pair of points by Euclidean distance.",
        "```python\nimport math\nfrom itertools import combinations\n\ndef closest_pair(points):\n    return min(combinations(points, 2),\n               key=lambda pair: math.dist(pair[0], pair[1]))\n```\nO(n^2) brute force. `math.dist` ships in Python 3.8+.",
        "I have 100k points — make it scale.",
        "```python\nimport math\n\ndef closest_pair(points):\n    pts = sorted(points)\n    def rec(pts):\n        if len(pts) < 4:\n            return min(((a, b) for i, a in enumerate(pts) for b in pts[i+1:]),\n                       key=lambda p: math.dist(*p))\n        mid = len(pts) // 2\n        midx = pts[mid][0]\n        l = rec(pts[:mid])\n        r = rec(pts[mid:])\n        best = min(l, r, key=lambda p: math.dist(*p))\n        d = math.dist(*best)\n        strip = sorted([p for p in pts if abs(p[0] - midx) < d], key=lambda p: p[1])\n        for i in range(len(strip)):\n            for j in range(i+1, min(i+8, len(strip))):\n                if strip[j][1] - strip[i][1] >= d:\n                    break\n                if math.dist(strip[i], strip[j]) < d:\n                    best = (strip[i], strip[j])\n                    d = math.dist(*best)\n        return best\n    return rec(pts)\n```\nDivide-and-conquer in O(n log^2 n). The middle 'strip' check handles pairs that span the dividing line.")

    add("algo",
        "Write a function that computes the area of a polygon given its vertices.",
        "```python\ndef polygon_area(verts) -> float:\n    n = len(verts)\n    s = 0.0\n    for i in range(n):\n        x1, y1 = verts[i]\n        x2, y2 = verts[(i + 1) % n]\n        s += x1 * y2 - x2 * y1\n    return abs(s) / 2\n```\nThe shoelace formula. Works for simple polygons (no self-intersections), CW or CCW order.",
        "Tell me whether the vertices were given clockwise or counter-clockwise.",
        "```python\ndef signed_area(verts) -> float:\n    n = len(verts)\n    s = 0.0\n    for i in range(n):\n        x1, y1 = verts[i]\n        x2, y2 = verts[(i + 1) % n]\n        s += x1 * y2 - x2 * y1\n    return s / 2  # positive: CCW, negative: CW\n\ndef orientation(verts) -> str:\n    return 'ccw' if signed_area(verts) > 0 else 'cw'\n```\nSkipping the `abs(...)` keeps the sign, which encodes orientation.")

    add("algo",
        "Write a function that detects whether a point is inside a polygon (ray casting).",
        "```python\ndef point_in_polygon(point, verts) -> bool:\n    x, y = point\n    inside = False\n    n = len(verts)\n    j = n - 1\n    for i in range(n):\n        xi, yi = verts[i]\n        xj, yj = verts[j]\n        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:\n            inside = not inside\n        j = i\n    return inside\n```\nCast a horizontal ray; flip `inside` once per crossing. The 'edge half-open' convention avoids double-counting horizontal edges.",
        "What if the polygon has thousands of vertices and I'm querying many points?",
        "```python\nfrom matplotlib.path import Path\n\ndef build_query(verts):\n    path = Path(verts)\n    def query(points):\n        return path.contains_points(points)\n    return query\n```\n`matplotlib.path.Path.contains_points` is C-optimized and vectorized. For full performance, look into shapely's prepared geometries.")

    add("algo",
        "Write a function that returns the determinant of a 3x3 matrix.",
        "```python\ndef det3(m) -> float:\n    a, b, c = m[0]\n    d, e, f = m[1]\n    g, h, i = m[2]\n    return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)\n```\nCofactor expansion along the first row. Simple and exact for small fixed sizes.",
        "Generalize to NxN.",
        "```python\nimport numpy as np\n\ndef det(m) -> float:\n    return float(np.linalg.det(np.asarray(m)))\n```\nFor general NxN, lean on numpy: it's LU-based, numerically stable, and far faster than recursive cofactor expansion.")

    add("algo",
        "Write a function that returns the longest common subsequence of two strings (the actual subsequence, not just length).",
        "```python\ndef lcs(a: str, b: str) -> str:\n    n, m = len(a), len(b)\n    dp = [[0]*(m+1) for _ in range(n+1)]\n    for i in range(n):\n        for j in range(m):\n            if a[i] == b[j]:\n                dp[i+1][j+1] = dp[i][j] + 1\n            else:\n                dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])\n    out = []\n    i, j = n, m\n    while i > 0 and j > 0:\n        if a[i-1] == b[j-1]:\n            out.append(a[i-1]); i -= 1; j -= 1\n        elif dp[i-1][j] >= dp[i][j-1]:\n            i -= 1\n        else:\n            j -= 1\n    return ''.join(reversed(out))\n```\nClassic DP table + backtracking. O(n*m) time and space.",
        "How would you reduce the memory?",
        "```python\ndef lcs_length(a: str, b: str) -> int:\n    n, m = len(a), len(b)\n    if n < m:\n        a, b = b, a\n        n, m = m, n\n    prev = [0] * (m + 1)\n    for i in range(n):\n        curr = [0] * (m + 1)\n        for j in range(m):\n            curr[j+1] = prev[j] + 1 if a[i] == b[j] else max(curr[j], prev[j+1])\n        prev = curr\n    return prev[m]\n```\nIf you only need the length, two rolling rows give O(min(n, m)) space. Reconstructing the actual LCS in O(min) space requires Hirschberg's algorithm.")

    add("algo",
        "Write a function to check if a graph (adjacency list) is bipartite.",
        "```python\nfrom collections import deque\n\ndef is_bipartite(adj: dict) -> bool:\n    color: dict = {}\n    for start in adj:\n        if start in color:\n            continue\n        color[start] = 0\n        q = deque([start])\n        while q:\n            u = q.popleft()\n            for v in adj.get(u, []):\n                if v not in color:\n                    color[v] = 1 - color[u]\n                    q.append(v)\n                elif color[v] == color[u]:\n                    return False\n    return True\n```\nBFS and 2-color. Disconnected components are handled by the outer loop.",
        "Return the two-coloring when it is bipartite.",
        "```python\nfrom collections import deque\n\ndef bipartition(adj: dict):\n    color: dict = {}\n    for start in adj:\n        if start in color:\n            continue\n        color[start] = 0\n        q = deque([start])\n        while q:\n            u = q.popleft()\n            for v in adj.get(u, []):\n                if v not in color:\n                    color[v] = 1 - color[u]\n                    q.append(v)\n                elif color[v] == color[u]:\n                    return None\n    a = [n for n, c in color.items() if c == 0]\n    b = [n for n, c in color.items() if c == 1]\n    return a, b\n```\nReturning `None` on failure makes the success/failure cases easy to distinguish.")

    add("algo",
        "Write a function that returns the topological order of a DAG.",
        "```python\nfrom collections import deque\n\ndef topo_sort(adj: dict) -> list:\n    indeg: dict = {n: 0 for n in adj}\n    for u in adj:\n        for v in adj[u]:\n            indeg[v] = indeg.get(v, 0) + 1\n            indeg.setdefault(u, indeg[u])\n    q = deque([n for n, d in indeg.items() if d == 0])\n    out = []\n    while q:\n        u = q.popleft()\n        out.append(u)\n        for v in adj.get(u, []):\n            indeg[v] -= 1\n            if indeg[v] == 0:\n                q.append(v)\n    if len(out) != len(indeg):\n        raise ValueError('cycle detected')\n    return out\n```\nKahn's algorithm. The cycle check is the leftover non-zero in-degrees.",
        "If there's a cycle, return one node on the cycle so I can debug.",
        "```python\ndef find_cycle_node(adj: dict):\n    WHITE, GRAY, BLACK = 0, 1, 2\n    state: dict = {n: WHITE for n in adj}\n    parent: dict = {}\n    def dfs(u):\n        state[u] = GRAY\n        for v in adj.get(u, []):\n            if state.get(v, WHITE) == WHITE:\n                state.setdefault(v, WHITE)\n                parent[v] = u\n                hit = dfs(v)\n                if hit is not None:\n                    return hit\n            elif state[v] == GRAY:\n                return v  # found a back edge\n        state[u] = BLACK\n        return None\n    for n in list(adj):\n        if state[n] == WHITE:\n            hit = dfs(n)\n            if hit is not None:\n                return hit\n    return None\n```\nClassic three-color DFS. The first GRAY node we revisit lies on a cycle.")

    add("algo",
        "Write a function that returns the n-th prime number.",
        "```python\nfrom sympy import prime\n\ndef nth_prime(n: int) -> int:\n    return prime(n)\n```\nIf you can take the dependency, sympy is the fast path.",
        "Implement it without sympy.",
        "```python\ndef nth_prime(n: int) -> int:\n    if n < 1:\n        raise ValueError('n must be >= 1')\n    primes = [2]\n    candidate = 3\n    while len(primes) < n:\n        is_p = True\n        for p in primes:\n            if p * p > candidate:\n                break\n            if candidate % p == 0:\n                is_p = False\n                break\n        if is_p:\n            primes.append(candidate)\n        candidate += 2\n    return primes[-1]\n```\nIncremental sieve: trial-divide each odd candidate by primes already found, stopping at sqrt.")

    add("algo",
        "Write a function that returns whether a string has all unique characters.",
        "```python\ndef all_unique(s: str) -> bool:\n    return len(set(s)) == len(s)\n```\nClean and clear. Builds the full set, but for short strings this is fine.",
        "Avoid building the full set if there's an early duplicate.",
        "```python\ndef all_unique(s: str) -> bool:\n    seen = set()\n    for ch in s:\n        if ch in seen:\n            return False\n        seen.add(ch)\n    return True\n```\nShort-circuits at the first duplicate, saving work for long inputs.")

    add("algo",
        "Write a function that returns the index of an element if it exists, raising KeyError otherwise.",
        "```python\ndef index_or_raise(xs, target):\n    try:\n        return xs.index(target)\n    except ValueError:\n        raise KeyError(target) from None\n```\nThe `from None` suppresses the chained `ValueError` for cleaner tracebacks.",
        "Now do the same but case-insensitive for strings.",
        "```python\ndef index_or_raise_ci(xs, target: str) -> int:\n    t = target.lower()\n    for i, x in enumerate(xs):\n        if isinstance(x, str) and x.lower() == t:\n            return i\n    raise KeyError(target)\n```\nManual loop because `list.index` doesn't take a key function.")

    add("algo",
        "Write a function that returns whether a list is sorted (ascending).",
        "```python\ndef is_sorted(xs) -> bool:\n    return all(a <= b for a, b in zip(xs, xs[1:]))\n```\nOne pass via `zip(xs, xs[1:])`. `all` short-circuits on the first violation.",
        "Make it accept a key function like sorted().",
        "```python\nfrom typing import Callable\n\ndef is_sorted(xs, key: Callable | None = None) -> bool:\n    keys = list(map(key, xs)) if key else xs\n    return all(a <= b for a, b in zip(keys, keys[1:]))\n```\nMaterializing the keys once avoids recomputing them during the comparison loop.")

    add("algo",
        "Write a function that returns the smallest positive integer not in a list.",
        "```python\ndef first_missing_positive(xs):\n    s = set(xs)\n    i = 1\n    while i in s:\n        i += 1\n    return i\n```\nO(n) time and O(n) space.",
        "Can you do it in O(1) extra space?",
        "```python\ndef first_missing_positive(xs):\n    n = len(xs)\n    i = 0\n    a = list(xs)\n    while i < n:\n        v = a[i]\n        if 1 <= v <= n and a[v - 1] != v:\n            a[v - 1], a[i] = a[i], a[v - 1]\n        else:\n            i += 1\n    for j in range(n):\n        if a[j] != j + 1:\n            return j + 1\n    return n + 1\n```\nIn-place 'cyclic sort': put each value v in slot v-1. Whichever slot still doesn't match is the answer.")

    add("algo",
        "Write a function that walks a directory and yields all .py files.",
        "```python\nfrom pathlib import Path\n\ndef python_files(root: str):\n    yield from Path(root).rglob('*.py')\n```\n`Path.rglob` recursively matches the pattern.",
        "Skip hidden directories (starting with '.').",
        "```python\nfrom pathlib import Path\n\ndef python_files(root: str):\n    root_path = Path(root)\n    for p in root_path.rglob('*.py'):\n        rel_parts = p.relative_to(root_path).parts\n        if any(part.startswith('.') for part in rel_parts):\n            continue\n        yield p\n```\nWe skip a path if any segment of it (relative to root) is hidden — covers `.git`, `.venv`, etc.")

    add("algo",
        "Write a function that converts a query-string into a dict.",
        "```python\nfrom urllib.parse import parse_qs\n\ndef parse_query(qs: str) -> dict:\n    return {k: v[0] for k, v in parse_qs(qs).items()}\n```\n`parse_qs` returns lists (since keys can repeat). We take the first to flatten.",
        "Preserve repeated keys instead of dropping them.",
        "```python\nfrom urllib.parse import parse_qs\n\ndef parse_query(qs: str) -> dict:\n    return parse_qs(qs)\n```\nReturning `parse_qs` directly preserves multi-values; callers can do `vals[0]` themselves when they want a single value.")

    add("algo",
        "Write a function that returns the most frequent N words from a long string.",
        "```python\nimport re\nfrom collections import Counter\n\ndef top_words(text: str, n: int) -> list[tuple[str, int]]:\n    words = re.findall(r\"[A-Za-z']+\", text.lower())\n    return Counter(words).most_common(n)\n```\nLowercasing folds case; `[A-Za-z']+` keeps contractions intact.",
        "Add an optional stopwords filter.",
        "```python\nimport re\nfrom collections import Counter\n\ndef top_words(text: str, n: int, stopwords: set[str] | None = None) -> list[tuple[str, int]]:\n    stop = stopwords or set()\n    words = (w for w in re.findall(r\"[A-Za-z']+\", text.lower()) if w not in stop)\n    return Counter(words).most_common(n)\n```\nLazy filter via generator avoids building a separate filtered list.")

    add("algo",
        "Write a function to return the maximum profit from a single buy/sell of a stock-price series.",
        "```python\ndef max_profit(prices: list[float]) -> float:\n    if not prices:\n        return 0\n    min_so_far = prices[0]\n    best = 0\n    for p in prices[1:]:\n        best = max(best, p - min_so_far)\n        min_so_far = min(min_so_far, p)\n    return best\n```\nO(n). Track the running min and the best profit relative to it.",
        "What if I'm allowed unlimited transactions?",
        "```python\ndef max_profit_unlimited(prices: list[float]) -> float:\n    return sum(max(0, b - a) for a, b in zip(prices, prices[1:]))\n```\nSum every positive day-over-day difference — that's the optimal strategy when you can buy and sell freely.")

    add("algo",
        "Write a function that produces a histogram (counts per bucket) of values.",
        "```python\ndef histogram(values, bins: int):\n    if not values:\n        return []\n    lo, hi = min(values), max(values)\n    if lo == hi:\n        return [(lo, hi, len(values))]\n    width = (hi - lo) / bins\n    counts = [0] * bins\n    for v in values:\n        idx = min(int((v - lo) / width), bins - 1)\n        counts[idx] += 1\n    return [(lo + i * width, lo + (i + 1) * width, counts[i]) for i in range(bins)]\n```\nReturns `(low_edge, high_edge, count)` triples. The `min(idx, bins-1)` handles the maximum value.",
        "Use numpy and return arrays instead.",
        "```python\nimport numpy as np\n\ndef histogram(values, bins: int):\n    counts, edges = np.histogram(values, bins=bins)\n    return counts, edges\n```\n`np.histogram` is the standard tool — well-tested and orders of magnitude faster on large inputs.")

    add("algo",
        "Write a function that takes a string and returns it with vowels removed.",
        "```python\ndef remove_vowels(s: str) -> str:\n    return ''.join(c for c in s if c not in 'aeiouAEIOU')\n```\nGenerator expression keeps it short and memory-friendly.",
        "Make it work for accented vowels too.",
        "```python\nimport unicodedata\n\nVOWEL_BASES = set('aeiou')\n\ndef remove_vowels(s: str) -> str:\n    out = []\n    for ch in s:\n        decomp = unicodedata.normalize('NFD', ch)\n        base = decomp[0].lower() if decomp else ''\n        if base not in VOWEL_BASES:\n            out.append(ch)\n    return ''.join(out)\n```\nNormalizing to NFD splits accents off; the base character tells us if it's a vowel regardless of accent or case.")

    add("algo",
        "Write a function that returns the standard deviation of a list of numbers.",
        "```python\nimport statistics\n\ndef stdev(xs: list[float]) -> float:\n    return statistics.stdev(xs)\n```\n`statistics.stdev` uses the sample (n-1) formula. Use `pstdev` for the population formula.",
        "Implement it from scratch using the two-pass algorithm.",
        "```python\nfrom math import sqrt\n\ndef stdev(xs: list[float]) -> float:\n    n = len(xs)\n    if n < 2:\n        raise ValueError('need at least 2 values')\n    mean = sum(xs) / n\n    var = sum((x - mean) ** 2 for x in xs) / (n - 1)\n    return sqrt(var)\n```\nTwo-pass is more numerically stable than one-pass `E[X^2] - E[X]^2` for floats.")

    add("algo",
        "Write a function that finds the smallest window in a string that contains all characters of a target string.",
        "```python\nfrom collections import Counter\n\ndef min_window(s: str, t: str) -> str:\n    if not t:\n        return ''\n    need = Counter(t)\n    have = Counter()\n    required = len(need)\n    formed = 0\n    l = 0\n    best = (float('inf'), 0, 0)\n    for r, ch in enumerate(s):\n        have[ch] += 1\n        if have[ch] == need[ch]:\n            formed += 1\n        while formed == required:\n            if r - l + 1 < best[0]:\n                best = (r - l + 1, l, r)\n            have[s[l]] -= 1\n            if have[s[l]] < need[s[l]]:\n                formed -= 1\n            l += 1\n    return '' if best[0] == float('inf') else s[best[1]:best[2]+1]\n```\nClassic sliding window with a 'formed' counter that tracks how many distinct character requirements are satisfied.",
        "Walk me through the time complexity.",
        "Each character of `s` is added to the window at most once and removed at most once, so the inner `while` loop runs O(n) total across the whole iteration. The outer loop is O(n). Counter operations are O(1) on average. So overall it's O(n + m) where n=len(s) and m=len(t). Space is O(unique chars in s + t), which is bounded by the alphabet size.")

    add("algo",
        "Write a function that finds the kth smallest element using quickselect.",
        "```python\nimport random\n\ndef quickselect(xs: list[int], k: int) -> int:\n    if not 0 <= k < len(xs):\n        raise IndexError('k out of range')\n    a = list(xs)\n    def select(lo: int, hi: int) -> int:\n        if lo == hi:\n            return a[lo]\n        pivot = a[random.randint(lo, hi)]\n        i, j = lo, hi\n        while i <= j:\n            while a[i] < pivot:\n                i += 1\n            while a[j] > pivot:\n                j -= 1\n            if i <= j:\n                a[i], a[j] = a[j], a[i]\n                i += 1\n                j -= 1\n        if k <= j:\n            return select(lo, j)\n        if k >= i:\n            return select(i, hi)\n        return a[k]\n    return select(0, len(a) - 1)\n```\nAverage O(n), worst-case O(n^2) (use median-of-medians for guaranteed O(n)).",
        "Why is the worst case O(n^2)?",
        "If the random pivot is consistently the smallest or largest element, each partition only shrinks the search range by 1 instead of roughly halving it. That gives a recurrence T(n) = T(n-1) + O(n), which solves to O(n^2). The randomized pivot makes this exponentially unlikely on adversarial inputs, and for guaranteed O(n) you can use median-of-medians to choose a pivot whose rank is provably between 30% and 70%.")

    add("algo",
        "Write a function that returns the n-th row of a memoized table for the coin change count problem.",
        "```python\ndef coin_change_ways(coins: list[int], target: int) -> int:\n    dp = [0] * (target + 1)\n    dp[0] = 1\n    for c in coins:\n        for x in range(c, target + 1):\n            dp[x] += dp[x - c]\n    return dp[target]\n```\nIterating coins outside and amounts inside counts each combination once — flipping the loop order would count permutations.",
        "What if I want the minimum number of coins instead?",
        "```python\ndef coin_change_min(coins: list[int], target: int) -> int:\n    INF = float('inf')\n    dp = [INF] * (target + 1)\n    dp[0] = 0\n    for x in range(1, target + 1):\n        for c in coins:\n            if c <= x and dp[x - c] + 1 < dp[x]:\n                dp[x] = dp[x - c] + 1\n    return -1 if dp[target] == INF else dp[target]\n```\nDifferent recurrence: take the minimum over all coins. Returning -1 signals unreachable amounts.")

    add("algo",
        "Write a function that returns whether two intervals overlap.",
        "```python\ndef overlaps(a, b) -> bool:\n    return a[0] < b[1] and b[0] < a[1]\n```\nIntervals are `(start, end)` half-open. Strict `<` means touching endpoints don't overlap, which is the standard convention.",
        "Now write a function that merges a list of intervals.",
        "```python\ndef merge_intervals(intervals):\n    if not intervals:\n        return []\n    s = sorted(intervals)\n    out = [list(s[0])]\n    for start, end in s[1:]:\n        if start <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], end)\n        else:\n            out.append([start, end])\n    return [tuple(iv) for iv in out]\n```\nSort by start, then sweep. The `<=` here treats touching intervals as mergeable, which is usually what users expect for ranges.")

    add("algo",
        "Write a function that wraps long text to a given width.",
        "```python\nimport textwrap\n\ndef wrap(text: str, width: int) -> str:\n    return textwrap.fill(text, width=width)\n```\nThe `textwrap` module handles word boundaries and edge cases correctly.",
        "Preserve existing paragraph breaks.",
        "```python\nimport textwrap\n\ndef wrap(text: str, width: int) -> str:\n    paragraphs = text.split('\\n\\n')\n    return '\\n\\n'.join(textwrap.fill(p, width=width) for p in paragraphs)\n```\nWrapping each paragraph independently keeps blank lines between them.")

    add("algo",
        "Write a function that returns the running max of a list.",
        "```python\nfrom itertools import accumulate\n\ndef running_max(xs):\n    return list(accumulate(xs, max))\n```\n`accumulate` takes any binary function; passing `max` gives the running maximum.",
        "Now do running max over a sliding window of size k.",
        "```python\nfrom collections import deque\n\ndef sliding_max(xs, k: int):\n    dq: deque = deque()\n    out = []\n    for i, x in enumerate(xs):\n        while dq and dq[0] <= i - k:\n            dq.popleft()\n        while dq and xs[dq[-1]] <= x:\n            dq.pop()\n        dq.append(i)\n        if i >= k - 1:\n            out.append(xs[dq[0]])\n    return out\n```\nMonotonic deque trick — each element enters and leaves at most once, giving amortized O(n).")
