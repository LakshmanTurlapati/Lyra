# SPDX-License-Identifier: MIT
"""Algorithms / data structures / utilities entries (100). Wave 5 part 1."""


def register_all(add):
    add("algo",
        "Write a function that flattens a nested list one level deep.",
        "```python\nfrom itertools import chain\n\ndef flatten1(xs: list) -> list:\n    return list(chain.from_iterable(xs))\n```\n`chain.from_iterable` is C-implemented and avoids the cost of `sum(xs, [])` (which is O(n^2)).")

    add("algo",
        "Write a function that flattens an arbitrarily nested list.",
        "```python\ndef flatten(xs):\n    out = []\n    stack = [iter(xs)]\n    while stack:\n        for x in stack[-1]:\n            if isinstance(x, list):\n                stack.append(iter(x))\n                break\n            out.append(x)\n        else:\n            stack.pop()\n    return out\n```\nIterative with an explicit stack avoids recursion limits on deeply nested input.")

    add("algo",
        "Implement a function that returns the intersection of two lists preserving order.",
        "```python\ndef ordered_intersection(a: list, b: list) -> list:\n    bset = set(b)\n    seen: set = set()\n    out = []\n    for x in a:\n        if x in bset and x not in seen:\n            out.append(x)\n            seen.add(x)\n    return out\n```\nO(n + m) using a hash set; preserves first-seen order from `a`.")

    add("algo",
        "Write a function that returns the union of two sorted lists as a sorted list.",
        "```python\ndef sorted_union(a: list, b: list) -> list:\n    i = j = 0\n    out = []\n    while i < len(a) and j < len(b):\n        if a[i] < b[j]:\n            out.append(a[i]); i += 1\n        elif a[i] > b[j]:\n            out.append(b[j]); j += 1\n        else:\n            out.append(a[i]); i += 1; j += 1\n    out.extend(a[i:]); out.extend(b[j:])\n    return out\n```\nClassic merge step from merge sort; O(n + m).")

    add("algo",
        "Implement binary search on a sorted list.",
        "```python\ndef bsearch(xs: list[int], target: int) -> int:\n    lo, hi = 0, len(xs) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if xs[mid] == target:\n            return mid\n        if xs[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n```\nFor production code reach for `bisect.bisect_left` -- it's already correct and fast.")

    add("algo",
        "Write a function that returns the longest common prefix of a list of strings.",
        "```python\ndef common_prefix(xs: list[str]) -> str:\n    if not xs:\n        return ''\n    s = min(xs); t = max(xs)\n    for i, ch in enumerate(s):\n        if ch != t[i]:\n            return s[:i]\n    return s\n```\nLexicographic min and max bracket the prefix in O(n) string comparisons.")

    add("algo",
        "Implement a function that checks whether a string is a palindrome ignoring case and non-alphanumerics.",
        "```python\ndef is_palindrome(s: str) -> bool:\n    cleaned = [c.lower() for c in s if c.isalnum()]\n    return cleaned == cleaned[::-1]\n```\nList comprehension + slice reversal is the most readable in Python; for huge inputs use a two-pointer loop.")

    add("algo",
        "Write a function that returns the n-th Fibonacci number iteratively.",
        "```python\ndef fib(n: int) -> int:\n    if n < 0:\n        raise ValueError('non-negative only')\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```\nO(n) time, O(1) space; tuple-swap avoids a temp variable.")

    add("algo",
        "Implement a function that counts the number of set bits in an integer.",
        "```python\ndef popcount(x: int) -> int:\n    return bin(x & ((1 << x.bit_length()) - 1) if x >= 0 else 0).count('1') if x < 0 else bin(x).count('1')\n```\nSimpler if you accept non-negatives only:\n```python\ndef popcount(x: int) -> int:\n    return bin(x).count('1')\n```\nIn Python 3.10+, `int.bit_count()` is the canonical answer.")

    add("algo",
        "Write a function that returns all subsets (the power set) of a list.",
        "```python\nfrom itertools import chain, combinations\n\ndef power_set(xs: list) -> list[tuple]:\n    return list(chain.from_iterable(combinations(xs, r) for r in range(len(xs) + 1)))\n```\nLifted straight from the `itertools` recipes section; 2^n entries, so guard input size.")

    add("algo",
        "Implement a stack using a Python list.",
        "```python\nclass Stack:\n    def __init__(self):\n        self._data: list = []\n    def push(self, x):\n        self._data.append(x)\n    def pop(self):\n        if not self._data:\n            raise IndexError('pop from empty stack')\n        return self._data.pop()\n    def peek(self):\n        return self._data[-1] if self._data else None\n    def __len__(self):\n        return len(self._data)\n```\n`list.append` and `list.pop()` are amortized O(1) -- a list IS a stack in Python.")

    add("algo",
        "Implement a queue using collections.deque.",
        "```python\nfrom collections import deque\n\nclass Queue:\n    def __init__(self):\n        self._d: deque = deque()\n    def enqueue(self, x):\n        self._d.append(x)\n    def dequeue(self):\n        if not self._d:\n            raise IndexError('dequeue from empty queue')\n        return self._d.popleft()\n    def __len__(self):\n        return len(self._d)\n```\n`deque.popleft` is O(1); `list.pop(0)` would be O(n).")

    add("algo",
        "Write a function that returns the GCD of two integers.",
        "```python\nfrom math import gcd\n\ndef my_gcd(a: int, b: int) -> int:\n    return gcd(a, b)\n```\n`math.gcd` is C-implemented and handles negatives. If you want it manually:\n```python\ndef my_gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return abs(a)\n```")

    add("algo",
        "Implement a function that returns the LCM of two integers.",
        "```python\nfrom math import gcd\n\ndef lcm(a: int, b: int) -> int:\n    if a == 0 or b == 0:\n        return 0\n    return abs(a * b) // gcd(a, b)\n```\nPython 3.9+ has `math.lcm` built in; this version works on older runtimes.")

    add("algo",
        "Write a function that returns whether a number is prime.",
        "```python\ndef is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    if n % 2 == 0:\n        return n == 2\n    i = 3\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 2\n    return True\n```\nO(sqrt n); for cryptographic sizes use Miller-Rabin instead.")

    add("algo",
        "Implement the Sieve of Eratosthenes.",
        "```python\ndef sieve(n: int) -> list[int]:\n    if n < 2:\n        return []\n    mark = [True] * (n + 1)\n    mark[0] = mark[1] = False\n    for i in range(2, int(n**0.5) + 1):\n        if mark[i]:\n            for j in range(i * i, n + 1, i):\n                mark[j] = False\n    return [i for i, ok in enumerate(mark) if ok]\n```\nO(n log log n); start the inner loop at `i*i` since smaller multiples are already crossed.")

    add("algo",
        "Write a function that reverses the words in a sentence in place (string version).",
        "```python\ndef reverse_words(s: str) -> str:\n    return ' '.join(s.split()[::-1])\n```\n`split()` collapses runs of whitespace; if you need to preserve exact spacing use `re.split`.")

    add("algo",
        "Implement a function that returns the histogram (count) of characters in a string.",
        "```python\nfrom collections import Counter\n\ndef histogram(s: str) -> dict[str, int]:\n    return dict(Counter(s))\n```\n`Counter` is the right tool; converting back to plain dict is optional.")

    add("algo",
        "Write a function that finds the first non-repeating character in a string.",
        "```python\nfrom collections import Counter\n\ndef first_unique(s: str) -> str | None:\n    counts = Counter(s)\n    for ch in s:\n        if counts[ch] == 1:\n            return ch\n    return None\n```\nTwo passes, O(n); preserves original order.")

    add("algo",
        "Implement an LRU cache with a fixed capacity.",
        "```python\nfrom collections import OrderedDict\n\nclass LRU:\n    def __init__(self, capacity: int):\n        self._cap = capacity\n        self._d: OrderedDict = OrderedDict()\n    def get(self, key):\n        if key not in self._d:\n            return None\n        self._d.move_to_end(key)\n        return self._d[key]\n    def put(self, key, value):\n        if key in self._d:\n            self._d.move_to_end(key)\n        self._d[key] = value\n        if len(self._d) > self._cap:\n            self._d.popitem(last=False)\n```\nFor function memoization just use `functools.lru_cache`.")

    add("algo",
        "Write a function that merges two sorted iterables into a sorted iterator.",
        "```python\nimport heapq\n\ndef merge_sorted(*iterables):\n    yield from heapq.merge(*iterables)\n```\n`heapq.merge` is lazy and works with any number of inputs; perfect for streaming.")

    add("algo",
        "Implement a min-heap-based priority queue.",
        "```python\nimport heapq\n\nclass PriorityQueue:\n    def __init__(self):\n        self._h: list = []\n        self._counter = 0\n    def push(self, priority, item):\n        heapq.heappush(self._h, (priority, self._counter, item))\n        self._counter += 1\n    def pop(self):\n        if not self._h:\n            raise IndexError('empty')\n        priority, _, item = heapq.heappop(self._h)\n        return item\n    def __len__(self):\n        return len(self._h)\n```\nThe counter avoids comparing un-orderable items when priorities tie.")

    add("algo",
        "Write a function that returns the longest run of identical elements.",
        "```python\nfrom itertools import groupby\n\ndef longest_run(xs: list) -> tuple:\n    if not xs:\n        return (None, 0)\n    val, length = max(((k, sum(1 for _ in g)) for k, g in groupby(xs)), key=lambda t: t[1])\n    return val, length\n```\n`groupby` collapses consecutive equals; `max` picks the longest streak.")

    add("algo",
        "Implement a function that returns the running maximum of a list.",
        "```python\nfrom itertools import accumulate\n\ndef running_max(xs: list[int]) -> list[int]:\n    return list(accumulate(xs, max))\n```\n`accumulate` with a binary function is the cleanest path; works for sum, min, max, mul, etc.")

    add("algo",
        "Write a function that splits a list into chunks of size n.",
        "```python\ndef chunks(xs: list, n: int) -> list[list]:\n    if n <= 0:\n        raise ValueError('n must be positive')\n    return [xs[i:i+n] for i in range(0, len(xs), n)]\n```\nFor an iterator version that doesn't require a sized input, use `itertools.batched` (3.12+).")

    add("algo",
        "Implement a function that returns whether two strings are anagrams.",
        "```python\nfrom collections import Counter\n\ndef is_anagram(a: str, b: str) -> bool:\n    return Counter(a) == Counter(b)\n```\nCounter-equality handles all character counts in one pass; simpler than sorted-string comparison.")

    add("algo",
        "Write a function that returns the cumulative sum of a list.",
        "```python\nfrom itertools import accumulate\n\ndef cumsum(xs: list[float]) -> list[float]:\n    return list(accumulate(xs))\n```\nFor numeric arrays prefer `numpy.cumsum`; same idea, vectorized.")

    add("algo",
        "Implement a function that removes duplicates while preserving order.",
        "```python\ndef dedupe(xs: list) -> list:\n    seen: set = set()\n    return [x for x in xs if not (x in seen or seen.add(x))]\n```\nThe `seen.add(x)` returns None which is falsy, so the side effect is safe inside the comprehension.")

    add("algo",
        "Write a function that returns the symmetric difference of two iterables.",
        "```python\ndef sym_diff(a, b) -> set:\n    return set(a) ^ set(b)\n```\n`^` on sets is symmetric difference -- in either but not both.")

    add("algo",
        "Implement a function that returns whether one string is a rotation of another.",
        "```python\ndef is_rotation(a: str, b: str) -> bool:\n    return len(a) == len(b) and b in (a + a)\n```\nClassic trick: if `b` is any rotation of `a`, it's a substring of `a + a`.")

    add("algo",
        "Write a function that converts a flat dict to a nested dict by dotted keys.",
        "```python\ndef nest(flat: dict) -> dict:\n    out: dict = {}\n    for key, val in flat.items():\n        d = out\n        parts = key.split('.')\n        for p in parts[:-1]:\n            d = d.setdefault(p, {})\n        d[parts[-1]] = val\n    return out\n```\nUseful for turning env-var or config-file flat keys into structured config.")

    add("algo",
        "Implement a function that flattens a nested dict to dotted keys.",
        "```python\ndef flatten_dict(d: dict, prefix: str = '') -> dict:\n    out = {}\n    for k, v in d.items():\n        key = f'{prefix}.{k}' if prefix else k\n        if isinstance(v, dict):\n            out.update(flatten_dict(v, key))\n        else:\n            out[key] = v\n    return out\n```\nInverse of the nesting function above; recursion depth is bounded by dict depth.")

    add("algo",
        "Write a function that compares two version strings like '1.2.10' vs '1.2.9'.",
        "```python\ndef compare_versions(a: str, b: str) -> int:\n    pa = [int(x) for x in a.split('.')]\n    pb = [int(x) for x in b.split('.')]\n    for x, y in zip(pa, pb):\n        if x != y:\n            return (x > y) - (x < y)\n    return (len(pa) > len(pb)) - (len(pa) < len(pb))\n```\nFor PEP 440 versions reach for `packaging.version.Version`.")

    add("algo",
        "Implement a function that returns the Levenshtein distance between two strings.",
        "```python\ndef levenshtein(a: str, b: str) -> int:\n    if len(a) < len(b):\n        a, b = b, a\n    prev = list(range(len(b) + 1))\n    for i, ca in enumerate(a, 1):\n        curr = [i]\n        for j, cb in enumerate(b, 1):\n            cost = 0 if ca == cb else 1\n            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j-1] + cost))\n        prev = curr\n    return prev[-1]\n```\nO(n*m) DP with a single row of state; use `rapidfuzz` for production.")

    add("algo",
        "Write a function that finds all pairs in a list summing to a target.",
        "```python\ndef pair_sums(xs: list[int], target: int) -> list[tuple[int, int]]:\n    seen: set = set()\n    out = []\n    for x in xs:\n        if target - x in seen:\n            out.append((target - x, x))\n        seen.add(x)\n    return out\n```\nSingle pass with a hash set; O(n).")

    add("algo",
        "Implement a function that returns the moving average of a list with window k.",
        "```python\nfrom collections import deque\n\ndef moving_average(xs: list[float], k: int) -> list[float]:\n    if k <= 0:\n        raise ValueError('k must be positive')\n    out, window, total = [], deque(), 0.0\n    for x in xs:\n        window.append(x); total += x\n        if len(window) > k:\n            total -= window.popleft()\n        if len(window) == k:\n            out.append(total / k)\n    return out\n```\nO(n) using a running sum; cheaper than `mean(xs[i:i+k])` per step.")

    add("algo",
        "Write a function that returns the longest increasing subsequence length.",
        "```python\nfrom bisect import bisect_left\n\ndef lis_length(xs: list[int]) -> int:\n    tails: list[int] = []\n    for x in xs:\n        i = bisect_left(tails, x)\n        if i == len(tails):\n            tails.append(x)\n        else:\n            tails[i] = x\n    return len(tails)\n```\nO(n log n) patience-sorting algorithm; `tails` is not the LIS itself but its length.")

    add("algo",
        "Implement a function that returns whether a list contains duplicates within window k.",
        "```python\ndef has_close_duplicate(xs: list[int], k: int) -> bool:\n    seen: dict[int, int] = {}\n    for i, x in enumerate(xs):\n        if x in seen and i - seen[x] <= k:\n            return True\n        seen[x] = i\n    return False\n```\nSliding-index map; O(n) time, O(n) space.")

    add("algo",
        "Write a function that returns the n-th triangular number.",
        "```python\ndef triangular(n: int) -> int:\n    if n < 0:\n        raise ValueError('non-negative only')\n    return n * (n + 1) // 2\n```\nClosed form is O(1); the `for i in range(...)` accumulator version is a bug magnet.")

    add("algo",
        "Implement a function that converts an integer to its binary string without the 0b prefix.",
        "```python\ndef to_binary(n: int) -> str:\n    if n < 0:\n        return '-' + bin(-n)[2:]\n    return bin(n)[2:]\n```\nOr use `format(n, 'b')` -- no slicing needed and easier to read.")

    add("algo",
        "Write a function that returns the maximum product of two integers in a list.",
        "```python\ndef max_product(xs: list[int]) -> int:\n    if len(xs) < 2:\n        raise ValueError('need at least two elements')\n    a = b = float('-inf')\n    c = d = float('inf')\n    for x in xs:\n        if x >= a:\n            b = a; a = x\n        elif x > b:\n            b = x\n        if x <= c:\n            d = c; c = x\n        elif x < d:\n            d = x\n    return max(a * b, c * d)\n```\nTwo largest or two smallest (negatives) win. O(n) one-pass.")

    add("algo",
        "Implement a function that finds the longest substring without repeating characters.",
        "```python\ndef longest_unique_substr(s: str) -> int:\n    last: dict[str, int] = {}\n    start = best = 0\n    for i, ch in enumerate(s):\n        if ch in last and last[ch] >= start:\n            start = last[ch] + 1\n        last[ch] = i\n        best = max(best, i - start + 1)\n    return best\n```\nSliding window with last-seen index; O(n).")

    add("algo",
        "Write a function that returns the depth of nested parentheses in a string.",
        "```python\ndef paren_depth(s: str) -> int:\n    depth = best = 0\n    for ch in s:\n        if ch == '(':\n            depth += 1; best = max(best, depth)\n        elif ch == ')':\n            depth -= 1\n            if depth < 0:\n                raise ValueError('unbalanced parentheses')\n    if depth != 0:\n        raise ValueError('unbalanced parentheses')\n    return best\n```\nLinear scan with a single counter.")

    add("algo",
        "Implement a function that validates whether a string has balanced brackets.",
        "```python\ndef balanced(s: str) -> bool:\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack: list[str] = []\n    for ch in s:\n        if ch in '([{':\n            stack.append(ch)\n        elif ch in ')]}':\n            if not stack or stack.pop() != pairs[ch]:\n                return False\n    return not stack\n```\nStack-based; the canonical interview problem.")

    add("algo",
        "Write a function that finds the index of a peak element in a list.",
        "```python\ndef find_peak(xs: list[int]) -> int:\n    lo, hi = 0, len(xs) - 1\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if xs[mid] < xs[mid + 1]:\n            lo = mid + 1\n        else:\n            hi = mid\n    return lo\n```\nBinary search in O(log n); 'peak' here means greater than its neighbours.")

    add("algo",
        "Implement a function that returns the n-th row of Pascal's triangle.",
        "```python\ndef pascal_row(n: int) -> list[int]:\n    row = [1]\n    for k in range(n):\n        row.append(row[-1] * (n - k) // (k + 1))\n    return row\n```\nUses the recurrence C(n, k+1) = C(n, k) * (n-k) / (k+1); avoids big factorials.")

    add("algo",
        "Write a function that finds the minimum element in a rotated sorted array.",
        "```python\ndef rotated_min(xs: list[int]) -> int:\n    if not xs:\n        raise ValueError('empty')\n    lo, hi = 0, len(xs) - 1\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if xs[mid] > xs[hi]:\n            lo = mid + 1\n        else:\n            hi = mid\n    return xs[lo]\n```\nBinary search variant; O(log n) for distinct values.")

    add("algo",
        "Implement a function that returns whether a number is a power of two.",
        "```python\ndef is_power_of_two(n: int) -> bool:\n    return n > 0 and n & (n - 1) == 0\n```\nThe bit-trick: powers of two have exactly one bit set, so `n & (n-1)` clears it.")

    add("algo",
        "Write a function that returns the longest palindromic substring.",
        "```python\ndef longest_palindrome(s: str) -> str:\n    if not s:\n        return ''\n    def expand(l, r):\n        while l >= 0 and r < len(s) and s[l] == s[r]:\n            l -= 1; r += 1\n        return s[l+1:r]\n    best = ''\n    for i in range(len(s)):\n        for cand in (expand(i, i), expand(i, i+1)):\n            if len(cand) > len(best):\n                best = cand\n    return best\n```\nExpand-around-center; O(n^2). Manacher is O(n) but rarely needed in practice.")

    add("algo",
        "Implement a function that performs run-length encoding on a string.",
        "```python\nfrom itertools import groupby\n\ndef rle(s: str) -> list[tuple[str, int]]:\n    return [(ch, sum(1 for _ in g)) for ch, g in groupby(s)]\n```\n`groupby` makes this a one-liner; works on any iterable, not just strings.")

    add("algo",
        "Write a function that decodes a run-length-encoded list back to a string.",
        "```python\ndef rle_decode(pairs: list[tuple[str, int]]) -> str:\n    return ''.join(ch * n for ch, n in pairs)\n```\n`str.__mul__` repeats characters cheaply; `''.join` of a generator avoids intermediate lists.")

    add("algo",
        "Implement a function that computes the dot product of two lists.",
        "```python\ndef dot(a: list[float], b: list[float]) -> float:\n    if len(a) != len(b):\n        raise ValueError('length mismatch')\n    return sum(x * y for x, y in zip(a, b))\n```\nFor numeric work prefer `numpy.dot`; this is the pure-Python equivalent.")

    add("algo",
        "Write a function that returns the transpose of a 2D list.",
        "```python\ndef transpose(m: list[list]) -> list[list]:\n    return [list(row) for row in zip(*m)]\n```\n`zip(*m)` is the standard transpose idiom; wrap in `list` because `zip` returns tuples.")

    add("algo",
        "Implement a function that multiplies two matrices represented as lists of lists.",
        "```python\ndef matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:\n    if not a or not b or len(a[0]) != len(b):\n        raise ValueError('shape mismatch')\n    bt = list(zip(*b))\n    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]\n```\nTransposing `b` once lets the inner loop iterate sequentially. Use numpy for real workloads.")

    add("algo",
        "Write a function that returns whether a Sudoku row is valid.",
        "```python\ndef valid_row(row: list[int]) -> bool:\n    seen: set = set()\n    for x in row:\n        if x == 0:\n            continue\n        if x in seen or not 1 <= x <= 9:\n            return False\n        seen.add(x)\n    return True\n```\nTreats 0 as empty; any other policy can be wired in by tweaking the guard.")

    add("algo",
        "Implement a function that converts CamelCase to snake_case.",
        "```python\nimport re\n\n_PATTERN = re.compile(r'(?<!^)(?=[A-Z])')\n\ndef camel_to_snake(s: str) -> str:\n    return _PATTERN.sub('_', s).lower()\n```\nLookbehind `(?<!^)` avoids inserting a leading underscore on PascalCase input.")

    add("algo",
        "Write a function that converts snake_case to camelCase.",
        "```python\ndef snake_to_camel(s: str) -> str:\n    head, *tail = s.split('_')\n    return head + ''.join(t.title() for t in tail)\n```\nFirst segment stays lowercase; subsequent segments are title-cased. Use `t[:1].upper() + t[1:]` if you must preserve mid-segment casing.")

    add("algo",
        "Implement a function that returns whether two lists have the same elements regardless of order.",
        "```python\nfrom collections import Counter\n\ndef same_multiset(a: list, b: list) -> bool:\n    return Counter(a) == Counter(b)\n```\nWorks for hashable elements; treats duplicates correctly (`set` would not).")

    add("algo",
        "Write a function that returns the index of the first repeating element.",
        "```python\ndef first_repeat_index(xs: list) -> int:\n    seen: set = set()\n    for i, x in enumerate(xs):\n        if x in seen:\n            return i\n        seen.add(x)\n    return -1\n```\nO(n) time, O(n) space; returns -1 if no duplicates exist.")

    add("algo",
        "Implement a function that returns the longest word in a sentence.",
        "```python\ndef longest_word(s: str) -> str:\n    words = s.split()\n    return max(words, key=len) if words else ''\n```\n`max` with a `key` is the cleanest pattern for 'argmax' style problems.")

    add("algo",
        "Write a function that returns the sum of digits of an integer.",
        "```python\ndef digit_sum(n: int) -> int:\n    return sum(int(c) for c in str(abs(n)))\n```\nString-conversion is fast in CPython; avoid the divmod loop unless you're avoiding str.")

    add("algo",
        "Implement a function that returns whether an integer is an Armstrong number.",
        "```python\ndef is_armstrong(n: int) -> bool:\n    s = str(n)\n    p = len(s)\n    return n == sum(int(c) ** p for c in s)\n```\n153 = 1^3 + 5^3 + 3^3, etc.")

    add("algo",
        "Write a function that converts a list to a frequency dict, sorted by frequency descending.",
        "```python\nfrom collections import Counter\n\ndef freq_sorted(xs: list) -> dict:\n    return dict(Counter(xs).most_common())\n```\n`most_common()` already gives highest-first ordering.")

    add("algo",
        "Implement a function that returns whether two intervals overlap.",
        "```python\ndef overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:\n    return a[0] <= b[1] and b[0] <= a[1]\n```\nThe two-condition form -- 'a starts before b ends, and b starts before a ends' -- is the cleanest.")

    add("algo",
        "Write a function that merges overlapping intervals.",
        "```python\ndef merge_intervals(ivs: list[tuple[int, int]]) -> list[tuple[int, int]]:\n    if not ivs:\n        return []\n    ivs = sorted(ivs)\n    out = [ivs[0]]\n    for s, e in ivs[1:]:\n        ls, le = out[-1]\n        if s <= le:\n            out[-1] = (ls, max(le, e))\n        else:\n            out.append((s, e))\n    return out\n```\nSort by start, then sweep -- O(n log n).")

    add("algo",
        "Implement a function that returns the n-th letter of a column header (Excel style).",
        "```python\ndef excel_col(n: int) -> str:\n    if n < 1:\n        raise ValueError('1-indexed')\n    out = []\n    while n:\n        n, r = divmod(n - 1, 26)\n        out.append(chr(ord('A') + r))\n    return ''.join(reversed(out))\n```\nBijective base-26: the `n - 1` shift handles the 'no zero digit' case.")

    add("algo",
        "Write a function that converts Excel column headers like 'AA' to integers.",
        "```python\ndef excel_col_to_int(s: str) -> int:\n    n = 0\n    for ch in s.upper():\n        if not 'A' <= ch <= 'Z':\n            raise ValueError(f'invalid char: {ch}')\n        n = n * 26 + (ord(ch) - ord('A') + 1)\n    return n\n```\nInverse of the previous function.")

    add("algo",
        "Implement a function that returns the day of the week for a given date.",
        "```python\nfrom datetime import date\n\ndef day_of_week(year: int, month: int, day: int) -> str:\n    return date(year, month, day).strftime('%A')\n```\nDelegate to `datetime`; date arithmetic is one of the easiest places to introduce subtle bugs.")

    add("algo",
        "Write a function that returns the number of days between two ISO date strings.",
        "```python\nfrom datetime import date\n\ndef days_between(a: str, b: str) -> int:\n    return abs((date.fromisoformat(b) - date.fromisoformat(a)).days)\n```\n`fromisoformat` is strict about format; use `dateutil.parser.parse` if your input is messy.")

    add("algo",
        "Implement a function that returns whether a year is a leap year.",
        "```python\ndef is_leap(year: int) -> bool:\n    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)\n```\nGregorian rule. For pre-1582 dates the calendar itself changes; the rule no longer applies cleanly.")

    add("algo",
        "Write a function that returns the title-case version of a string.",
        "```python\ndef title_case(s: str) -> str:\n    return ' '.join(w.capitalize() for w in s.split())\n```\nPython's built-in `str.title()` capitalises after every non-alpha char (e.g. apostrophes), which is rarely what you want.")

    add("algo",
        "Implement a function that returns the most common word in a paragraph.",
        "```python\nimport re\nfrom collections import Counter\n\ndef most_common_word(text: str) -> str | None:\n    words = re.findall(r\"[a-zA-Z']+\", text.lower())\n    if not words:\n        return None\n    return Counter(words).most_common(1)[0][0]\n```\nRegex is more robust than `split` for separating words from punctuation.")

    add("algo",
        "Write a function that returns the cube root of a non-negative integer using binary search.",
        "```python\ndef cube_root(n: int) -> int:\n    if n < 0:\n        raise ValueError('non-negative only')\n    lo, hi = 0, max(n, 1)\n    while lo < hi:\n        mid = (lo + hi + 1) // 2\n        if mid ** 3 <= n:\n            lo = mid\n        else:\n            hi = mid - 1\n    return lo\n```\nReturns floor of the real cube root; the `+1` in the midpoint avoids an infinite loop when `lo == hi - 1`.")

    add("algo",
        "Implement a function that returns the integer square root.",
        "```python\nfrom math import isqrt\n\ndef integer_sqrt(n: int) -> int:\n    if n < 0:\n        raise ValueError('non-negative only')\n    return isqrt(n)\n```\n`math.isqrt` is exact and fast; using `int(math.sqrt(n))` is wrong for large integers due to float rounding.")

    add("algo",
        "Write a function that performs insertion sort in place.",
        "```python\ndef insertion_sort(xs: list) -> None:\n    for i in range(1, len(xs)):\n        x = xs[i]\n        j = i - 1\n        while j >= 0 and xs[j] > x:\n            xs[j+1] = xs[j]\n            j -= 1\n        xs[j+1] = x\n```\nO(n^2) worst case but blistering fast for small or nearly-sorted inputs; CPython uses it inside Timsort.")

    add("algo",
        "Implement quicksort on a list (not in place).",
        "```python\ndef quicksort(xs: list) -> list:\n    if len(xs) <= 1:\n        return xs[:]\n    pivot = xs[len(xs) // 2]\n    less = [x for x in xs if x < pivot]\n    equal = [x for x in xs if x == pivot]\n    greater = [x for x in xs if x > pivot]\n    return quicksort(less) + equal + quicksort(greater)\n```\nClear and O(n log n) average -- but allocates heavily. For real code use `sorted(xs)`.")

    add("algo",
        "Write a function that performs merge sort.",
        "```python\ndef merge_sort(xs: list) -> list:\n    if len(xs) <= 1:\n        return xs[:]\n    m = len(xs) // 2\n    left, right = merge_sort(xs[:m]), merge_sort(xs[m:])\n    out, i, j = [], 0, 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            out.append(left[i]); i += 1\n        else:\n            out.append(right[j]); j += 1\n    out.extend(left[i:]); out.extend(right[j:])\n    return out\n```\nStable, O(n log n) guaranteed; written for clarity not speed.")

    add("algo",
        "Implement a function that does iterative depth-first search on a graph.",
        "```python\ndef dfs(graph: dict, start) -> list:\n    visited: set = set()\n    order = []\n    stack = [start]\n    while stack:\n        node = stack.pop()\n        if node in visited:\n            continue\n        visited.add(node); order.append(node)\n        stack.extend(graph.get(node, []))\n    return order\n```\nIterative avoids recursion limits; ordering depends on neighbour order.")

    add("algo",
        "Write a function that does breadth-first search on a graph.",
        "```python\nfrom collections import deque\n\ndef bfs(graph: dict, start) -> list:\n    visited: set = {start}\n    order = []\n    q = deque([start])\n    while q:\n        node = q.popleft()\n        order.append(node)\n        for n in graph.get(node, []):\n            if n not in visited:\n                visited.add(n); q.append(n)\n    return order\n```\n`deque.popleft` is O(1); use a list and you get O(n^2).")

    add("algo",
        "Implement a function that finds the shortest path in an unweighted graph.",
        "```python\nfrom collections import deque\n\ndef shortest_path(graph: dict, start, goal) -> list | None:\n    if start == goal:\n        return [start]\n    parents = {start: None}\n    q = deque([start])\n    while q:\n        node = q.popleft()\n        for n in graph.get(node, []):\n            if n in parents:\n                continue\n            parents[n] = node\n            if n == goal:\n                path = [n]\n                while parents[path[-1]] is not None:\n                    path.append(parents[path[-1]])\n                return list(reversed(path))\n            q.append(n)\n    return None\n```\nBFS guarantees fewest edges; reconstruct via parent pointers.")

    add("algo",
        "Write a function that returns whether a graph has a cycle (undirected).",
        "```python\ndef has_cycle(graph: dict) -> bool:\n    visited: set = set()\n    def dfs(node, parent):\n        visited.add(node)\n        for n in graph.get(node, []):\n            if n not in visited:\n                if dfs(n, node):\n                    return True\n            elif n != parent:\n                return True\n        return False\n    return any(dfs(v, None) for v in graph if v not in visited)\n```\nO(V + E); 'parent' avoids treating the immediate back-edge as a cycle.")

    add("algo",
        "Implement a function that returns the topological order of a DAG.",
        "```python\nfrom collections import deque\n\ndef topo_sort(graph: dict[int, list[int]]) -> list[int]:\n    indeg: dict[int, int] = {v: 0 for v in graph}\n    for v in graph:\n        for n in graph[v]:\n            indeg[n] = indeg.get(n, 0) + 1\n            indeg.setdefault(v, 0)\n    q = deque([v for v, d in indeg.items() if d == 0])\n    out = []\n    while q:\n        v = q.popleft(); out.append(v)\n        for n in graph.get(v, []):\n            indeg[n] -= 1\n            if indeg[n] == 0:\n                q.append(n)\n    if len(out) != len(indeg):\n        raise ValueError('graph has a cycle')\n    return out\n```\nKahn's algorithm; raises if the input is cyclic.")

    add("algo",
        "Write a function that returns the n-th Catalan number.",
        "```python\nfrom math import comb\n\ndef catalan(n: int) -> int:\n    if n < 0:\n        raise ValueError('non-negative only')\n    return comb(2 * n, n) // (n + 1)\n```\n`math.comb` does exact arbitrary-precision binomials.")

    add("algo",
        "Implement a function that returns the number of trailing zeros in n!.",
        "```python\ndef trailing_zeros(n: int) -> int:\n    if n < 0:\n        raise ValueError('non-negative only')\n    count = 0\n    p = 5\n    while p <= n:\n        count += n // p\n        p *= 5\n    return count\n```\nCount factors of 5 in n!; factors of 2 are always more abundant.")

    add("algo",
        "Write a function that swaps two variables without a temporary.",
        "```python\ndef swap_demo(a, b):\n    a, b = b, a\n    return a, b\n```\nTuple unpacking is the Pythonic answer; the XOR trick from C is slower and harder to read.")

    add("algo",
        "Implement a function that returns whether a number is a perfect square.",
        "```python\nfrom math import isqrt\n\ndef is_perfect_square(n: int) -> bool:\n    if n < 0:\n        return False\n    r = isqrt(n)\n    return r * r == n\n```\nUse `math.isqrt`; `int(math.sqrt(n))` fails for large n due to float precision.")

    add("algo",
        "Write a function that returns the prime factorization of an integer.",
        "```python\ndef factorize(n: int) -> dict[int, int]:\n    if n < 2:\n        return {}\n    out: dict[int, int] = {}\n    d = 2\n    while d * d <= n:\n        while n % d == 0:\n            out[d] = out.get(d, 0) + 1\n            n //= d\n        d += 1\n    if n > 1:\n        out[n] = out.get(n, 0) + 1\n    return out\n```\nTrial division is fine up to ~10^12; sympy.factorint scales further.")

    add("algo",
        "Implement a function that returns Pascal's triangle as a 2D list.",
        "```python\ndef pascal(n: int) -> list[list[int]]:\n    rows: list[list[int]] = []\n    for i in range(n):\n        row = [1] * (i + 1)\n        for j in range(1, i):\n            row[j] = rows[i-1][j-1] + rows[i-1][j]\n        rows.append(row)\n    return rows\n```\nEach row is built from the previous row's adjacent sums.")

    add("algo",
        "Write a function that compresses a string by collapsing repeats.",
        "```python\nfrom itertools import groupby\n\ndef compress(s: str) -> str:\n    out = []\n    for ch, g in groupby(s):\n        out.append(ch + str(sum(1 for _ in g)))\n    result = ''.join(out)\n    return result if len(result) < len(s) else s\n```\nReturn the original if compression doesn't shrink it -- the standard interview twist.")

    add("algo",
        "Implement a function that returns the Hamming distance between two equal-length strings.",
        "```python\ndef hamming(a: str, b: str) -> int:\n    if len(a) != len(b):\n        raise ValueError('length mismatch')\n    return sum(x != y for x, y in zip(a, b))\n```\nGenerator-of-bools summed: True is 1.")

    add("algo",
        "Write a function that decodes a list of (item, count) into a flat list.",
        "```python\ndef rle_expand(pairs: list[tuple]) -> list:\n    return [item for item, count in pairs for _ in range(count)]\n```\nNested comprehension; reads left-to-right same as nested for loops.")

    add("algo",
        "Implement a function that returns the n-th tetranacci number.",
        "```python\ndef tetranacci(n: int) -> int:\n    if n < 0:\n        raise ValueError('non-negative only')\n    a, b, c, d = 0, 0, 0, 1\n    for _ in range(n):\n        a, b, c, d = b, c, d, a + b + c + d\n    return a\n```\nLike Fibonacci but with four-term recurrence. O(n) time, O(1) space.")

    add("algo",
        "Write a function that returns whether a list is a permutation of 1..n.",
        "```python\ndef is_permutation_1n(xs: list[int]) -> bool:\n    n = len(xs)\n    seen = [False] * n\n    for x in xs:\n        if not 1 <= x <= n or seen[x-1]:\n            return False\n        seen[x-1] = True\n    return True\n```\nO(n) time, O(n) space; rejects duplicates and out-of-range values in one pass.")

    add("algo",
        "Implement a function that finds the k-th smallest element in a list.",
        "```python\nimport heapq\n\ndef kth_smallest(xs: list[int], k: int) -> int:\n    if not 1 <= k <= len(xs):\n        raise ValueError('k out of range')\n    return heapq.nsmallest(k, xs)[-1]\n```\nO(n log k); for huge n with small k, this beats `sorted(xs)[k-1]`.")

    add("algo",
        "Write a function that returns the distinct sub-strings of length k.",
        "```python\ndef distinct_substrings(s: str, k: int) -> set[str]:\n    if k <= 0 or k > len(s):\n        return set()\n    return {s[i:i+k] for i in range(len(s) - k + 1)}\n```\nSet comprehension over rolling slices; O(n*k) memory.")

    add("algo",
        "Implement a function that returns whether a sequence is strictly monotonic.",
        "```python\ndef is_strict_monotonic(xs: list) -> bool:\n    if len(xs) < 2:\n        return True\n    inc = all(a < b for a, b in zip(xs, xs[1:]))\n    dec = all(a > b for a, b in zip(xs, xs[1:]))\n    return inc or dec\n```\nTwo passes but O(n); `all` short-circuits on first violation.")

    add("algo",
        "Write a function that returns the index of the closest value in a sorted list.",
        "```python\nfrom bisect import bisect_left\n\ndef closest_index(xs: list[float], target: float) -> int:\n    if not xs:\n        raise ValueError('empty')\n    i = bisect_left(xs, target)\n    if i == 0:\n        return 0\n    if i == len(xs):\n        return len(xs) - 1\n    return i if abs(xs[i] - target) < abs(xs[i-1] - target) else i - 1\n```\nO(log n); the boundary checks avoid IndexError at the ends.")

    add("algo",
        "Implement a function that returns whether a string follows a given pattern (word-pattern).",
        "```python\ndef word_pattern(pattern: str, s: str) -> bool:\n    words = s.split()\n    if len(words) != len(pattern):\n        return False\n    return len(set(zip(pattern, words))) == len(set(pattern)) == len(set(words))\n```\nThe two-way bijection check: equal cardinalities of the pairs and each side.")

    add("algo",
        "Write a function that returns the longest common subsequence of two strings.",
        "```python\ndef lcs(a: str, b: str) -> str:\n    n, m = len(a), len(b)\n    dp = [[''] * (m + 1) for _ in range(n + 1)]\n    for i in range(n):\n        for j in range(m):\n            if a[i] == b[j]:\n                dp[i+1][j+1] = dp[i][j] + a[i]\n            else:\n                dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1], key=len)\n    return dp[n][m]\n```\nO(n*m) time and memory. For long strings, switch to length-only DP and reconstruct only when needed.")

    add("algo",
        "Implement a function that returns the median of two sorted lists.",
        "```python\ndef median_two_sorted(a: list[int], b: list[int]) -> float:\n    merged = sorted(a + b)\n    n = len(merged)\n    if n == 0:\n        raise ValueError('both empty')\n    mid = n // 2\n    return merged[mid] if n % 2 else (merged[mid-1] + merged[mid]) / 2\n```\nO((n+m) log(n+m)) but plain. The O(log min(n,m)) version exists -- only worth it under interview pressure.")
