# SPDX-License-Identifier: MIT
"""Algorithms / data structures / utilities entries (100). Wave 4 part 1."""


def register_all(add):
    add("algo",
        "Write a function that returns the second-largest element of a list.",
        "```python\ndef second_largest(xs: list[int]) -> int:\n    if len(xs) < 2:\n        raise ValueError('need at least two elements')\n    a = b = float('-inf')\n    for x in xs:\n        if x > a:\n            a, b = x, a\n        elif x > b and x != a:\n            b = x\n    if b == float('-inf'):\n        raise ValueError('all elements equal')\n    return b\n```\nSingle-pass O(n); avoids the `sorted(set(xs))[-2]` allocation.")

    add("algo",
        "Write a function that rotates a list to the right by k positions.",
        "```python\nfrom collections import deque\n\ndef rotate(xs: list, k: int) -> list:\n    if not xs:\n        return xs\n    d = deque(xs)\n    d.rotate(k)\n    return list(d)\n```\n`deque.rotate` is C-implemented and handles negative k naturally.")

    add("algo",
        "Implement a function that returns the running average of a stream.",
        "```python\ndef running_average():\n    total, n = 0.0, 0\n    while True:\n        x = yield (total / n if n else 0.0)\n        total += x\n        n += 1\n```\nGenerator-based: `g = running_average(); next(g); g.send(5)`. Constant memory.")

    add("algo",
        "Write a function that returns True if a list is sorted.",
        "```python\ndef is_sorted(xs: list) -> bool:\n    return all(a <= b for a, b in zip(xs, xs[1:]))\n```\nUses `all` short-circuit so it returns at the first inversion.")

    add("algo",
        "Write a function that finds the missing number in a list of 1..n with one missing.",
        "```python\ndef missing(xs: list[int]) -> int:\n    n = len(xs) + 1\n    return n * (n + 1) // 2 - sum(xs)\n```\nGauss's formula, O(n) time, O(1) extra space.")

    add("algo",
        "Implement a function that returns the majority element if one exists (>n/2).",
        "```python\ndef majority(xs: list[int]) -> int | None:\n    candidate, count = None, 0\n    for x in xs:\n        if count == 0:\n            candidate = x\n        count += 1 if x == candidate else -1\n    if xs.count(candidate) * 2 > len(xs):\n        return candidate\n    return None\n```\nBoyer-Moore vote in O(n) time / O(1) space, with a verification pass.")

    add("algo",
        "Write a function that returns all permutations of a list.",
        "```python\nfrom itertools import permutations\n\ndef perms(xs: list) -> list[tuple]:\n    return list(permutations(xs))\n```\nDelegate to `itertools`; rolling your own recursion is slower and bug-prone.")

    add("algo",
        "Implement integer exponentiation by squaring.",
        "```python\ndef power(base: int, exp: int) -> int:\n    if exp < 0:\n        raise ValueError('non-negative only')\n    result = 1\n    while exp:\n        if exp & 1:\n            result *= base\n        base *= base\n        exp >>= 1\n    return result\n```\nO(log exp). For built-in use, `pow(base, exp)` already does this.")

    add("algo",
        "Write a function that returns the most frequent k elements.",
        "```python\nfrom collections import Counter\n\ndef top_k(xs: list, k: int) -> list:\n    return [x for x, _ in Counter(xs).most_common(k)]\n```\n`Counter.most_common` uses a heap internally for partial ordering.")

    add("algo",
        "Implement a function that reverses a singly linked list.",
        "```python\nclass Node:\n    def __init__(self, val, nxt=None):\n        self.val, self.next = val, nxt\n\ndef reverse(head: Node | None) -> Node | None:\n    prev = None\n    while head:\n        head.next, prev, head = prev, head, head.next\n    return prev\n```\nIterative tuple-swap is the cleanest in Python.")

    add("algo",
        "Write a function that returns the depth of a binary tree.",
        "```python\nclass TreeNode:\n    def __init__(self, val, left=None, right=None):\n        self.val, self.left, self.right = val, left, right\n\ndef depth(root: TreeNode | None) -> int:\n    if root is None:\n        return 0\n    return 1 + max(depth(root.left), depth(root.right))\n```\nFor very deep trees use an explicit stack to avoid recursion limits.")

    add("algo",
        "Implement a function that checks if two binary trees are identical.",
        "```python\ndef same(a, b) -> bool:\n    if a is None and b is None:\n        return True\n    if a is None or b is None:\n        return False\n    return a.val == b.val and same(a.left, b.left) and same(a.right, b.right)\n```\nShort-circuit on the first mismatch.")

    add("algo",
        "Write a function that returns the index pair summing to a target.",
        "```python\ndef two_sum(xs: list[int], target: int) -> tuple[int, int] | None:\n    seen: dict[int, int] = {}\n    for i, x in enumerate(xs):\n        if target - x in seen:\n            return seen[target - x], i\n        seen[x] = i\n    return None\n```\nO(n) hash-map approach beats the O(n\u00b2) double loop.")

    add("algo",
        "Implement Kadane's algorithm for maximum subarray sum.",
        "```python\ndef max_subarray(xs: list[int]) -> int:\n    if not xs:\n        raise ValueError('empty input')\n    best = cur = xs[0]\n    for x in xs[1:]:\n        cur = max(x, cur + x)\n        best = max(best, cur)\n    return best\n```\nO(n) -- the textbook example of dynamic programming.")

    add("algo",
        "Write a function that returns True if a string has all unique characters.",
        "```python\ndef all_unique(s: str) -> bool:\n    return len(set(s)) == len(s)\n```\nCleaner than a manual loop; the set construction is C-implemented.")

    add("algo",
        "Implement a function that converts Roman numerals to integers.",
        "```python\ndef roman_to_int(s: str) -> int:\n    vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n    total, prev = 0, 0\n    for c in reversed(s):\n        v = vals[c]\n        total += -v if v < prev else v\n        prev = v\n    return total\n```\nRight-to-left scan: subtract when a smaller numeral precedes a larger one.")

    add("algo",
        "Write a function that returns the diagonals of a square matrix.",
        "```python\ndef diagonals(m: list[list]) -> tuple[list, list]:\n    n = len(m)\n    main = [m[i][i] for i in range(n)]\n    anti = [m[i][n - 1 - i] for i in range(n)]\n    return main, anti\n```\nFor numpy use `np.diag(m)` and `np.diag(np.fliplr(m))`.")

    add("algo",
        "Implement a function that returns the Cartesian product of two iterables.",
        "```python\nfrom itertools import product\n\ndef cartesian(a, b) -> list[tuple]:\n    return list(product(a, b))\n```\n`itertools.product` extends to any number of iterables and supports `repeat=`.")

    add("algo",
        "Write a function that pads a list to length n with a fill value.",
        "```python\ndef pad(xs: list, n: int, fill=0) -> list:\n    if len(xs) >= n:\n        return xs[:n]\n    return xs + [fill] * (n - len(xs))\n```\nNote it returns a new list rather than mutating the input.")

    add("algo",
        "Implement a function that converts a snake_case string to camelCase.",
        "```python\ndef to_camel(s: str) -> str:\n    parts = s.split('_')\n    return parts[0] + ''.join(p.title() for p in parts[1:])\n```\nFirst part stays lowercase; the rest get title-cased.")

    add("algo",
        "Write a function that converts camelCase to snake_case.",
        "```python\nimport re\n\ndef to_snake(s: str) -> str:\n    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()\n```\nLookahead inserts an underscore before each uppercase letter (except at the start), then lowercase.")

    add("algo",
        "Implement a function that returns a defaultdict-of-list grouping by a key function.",
        "```python\nfrom collections import defaultdict\nfrom typing import Callable, Iterable\n\ndef group_by(xs: Iterable, key: Callable) -> dict[object, list]:\n    out: dict[object, list] = defaultdict(list)\n    for x in xs:\n        out[key(x)].append(x)\n    return dict(out)\n```\nReturning `dict(out)` gives the caller a normal dict; `defaultdict` is an implementation detail.")

    add("algo",
        "Write a function that returns the difference of two lists preserving order.",
        "```python\ndef diff(a: list, b: list) -> list:\n    bs = set(b)\n    return [x for x in a if x not in bs]\n```\nBuilding `bs` once gives O(1) lookups for each element of `a`.")

    add("algo",
        "Implement a function that returns the intersection of two lists preserving order from the first.",
        "```python\ndef intersect(a: list, b: list) -> list:\n    bs = set(b)\n    seen: set = set()\n    out: list = []\n    for x in a:\n        if x in bs and x not in seen:\n            out.append(x); seen.add(x)\n    return out\n```\nDeduplicates while preserving order.")

    add("algo",
        "Write a function that returns the symmetric difference of two lists.",
        "```python\ndef sym_diff(a: list, b: list) -> list:\n    return list(set(a) ^ set(b))\n```\nSet `^` operator is the symmetric difference.")

    add("algo",
        "Implement a function that returns the n-th Catalan number.",
        "```python\nimport math\n\ndef catalan(n: int) -> int:\n    return math.comb(2 * n, n) // (n + 1)\n```\nUses the closed-form binomial; `math.comb` is C-fast and exact.")

    add("algo",
        "Write a function that computes the dot product of two vectors.",
        "```python\ndef dot(a: list[float], b: list[float]) -> float:\n    if len(a) != len(b):\n        raise ValueError('length mismatch')\n    return sum(x * y for x, y in zip(a, b))\n```\nFor large vectors use `numpy.dot` -- a couple orders of magnitude faster.")

    add("algo",
        "Implement a function that transposes a matrix without numpy.",
        "```python\ndef transpose(m: list[list]) -> list[list]:\n    return [list(row) for row in zip(*m)]\n```\n`zip(*m)` is the canonical Python transpose.")

    add("algo",
        "Write a function that returns whether one string is a rotation of another.",
        "```python\ndef is_rotation(a: str, b: str) -> bool:\n    return len(a) == len(b) and b in a + a\n```\nA rotation appears as a substring of `a + a`.")

    add("algo",
        "Implement a function that finds all primes up to n with the sieve of Eratosthenes.",
        "```python\ndef primes_up_to(n: int) -> list[int]:\n    if n < 2:\n        return []\n    sieve = [True] * (n + 1)\n    sieve[0] = sieve[1] = False\n    for i in range(2, int(n**0.5) + 1):\n        if sieve[i]:\n            sieve[i*i::i] = [False] * len(sieve[i*i::i])\n    return [i for i, p in enumerate(sieve) if p]\n```\nSlice assignment marks composites in C-speed.")

    add("algo",
        "Write a function that decodes a run-length-encoded string like '3a2b1c' to 'aaabbc'.",
        "```python\nimport re\n\ndef rle_decode(s: str) -> str:\n    return ''.join(c * int(n) for n, c in re.findall(r'(\\d+)(.)', s))\n```\nRegex pulls (count, char) pairs; a generator expression rebuilds the string.")

    add("algo",
        "Implement a function that encodes a string with run-length encoding.",
        "```python\nfrom itertools import groupby\n\ndef rle_encode(s: str) -> str:\n    return ''.join(f'{sum(1 for _ in g)}{c}' for c, g in groupby(s))\n```\n`groupby` clusters consecutive equal characters.")

    add("algo",
        "Write a function that returns the n-th row of Pascal's triangle.",
        "```python\nimport math\n\ndef pascal_row(n: int) -> list[int]:\n    return [math.comb(n, k) for k in range(n + 1)]\n```\n`math.comb` is exact and faster than building the triangle row by row.")

    add("algo",
        "Implement a function that returns indices of all occurrences of a substring.",
        "```python\ndef find_all(s: str, sub: str) -> list[int]:\n    if not sub:\n        return []\n    out, i = [], 0\n    while True:\n        i = s.find(sub, i)\n        if i == -1:\n            return out\n        out.append(i)\n        i += 1\n```\nNote `i += 1` (not `len(sub)`) finds overlapping matches.")

    add("algo",
        "Write a function that returns the running maximum of a list.",
        "```python\nimport itertools\n\ndef running_max(xs: list[int]) -> list[int]:\n    return list(itertools.accumulate(xs, max))\n```\n`accumulate` with a binary function is the swiss-army knife of running aggregates.")

    add("algo",
        "Implement a function that flattens an arbitrarily nested list.",
        "```python\ndef flatten(xs):\n    for x in xs:\n        if isinstance(x, list):\n            yield from flatten(x)\n        else:\n            yield x\n```\nGenerator-based; caller wraps with `list(...)` if they want a list. `yield from` handles the recursion cleanly.")

    add("algo",
        "Write a function that pairs adjacent elements in a list.",
        "```python\ndef pairs(xs: list) -> list[tuple]:\n    return list(zip(xs, xs[1:]))\n```\nFor an iterator-friendly version use `itertools.pairwise(xs)` (Python 3.10+).")

    add("algo",
        "Implement a function that splits a list into chunks of size n.",
        "```python\ndef chunks(xs: list, n: int) -> list[list]:\n    if n <= 0:\n        raise ValueError('n must be positive')\n    return [xs[i:i + n] for i in range(0, len(xs), n)]\n```\nFor iterators, use `itertools.batched(xs, n)` (Python 3.12+).")

    add("algo",
        "Write a function that finds the closest pair of values to a target.",
        "```python\ndef closest_pair(xs: list[float], target: float) -> tuple[float, float]:\n    xs = sorted(xs)\n    i, j = 0, len(xs) - 1\n    best, best_diff = (xs[i], xs[j]), float('inf')\n    while i < j:\n        s = xs[i] + xs[j]\n        if abs(s - target) < best_diff:\n            best, best_diff = (xs[i], xs[j]), abs(s - target)\n        if s < target:\n            i += 1\n        else:\n            j -= 1\n    return best\n```\nTwo-pointer in O(n log n) (the sort dominates).")

    add("algo",
        "Implement a function that converts an integer to a base-n string.",
        "```python\ndef to_base(n: int, base: int) -> str:\n    if not 2 <= base <= 36:\n        raise ValueError('base must be 2..36')\n    if n == 0:\n        return '0'\n    digits = '0123456789abcdefghijklmnopqrstuvwxyz'\n    sign = '-' if n < 0 else ''\n    n = abs(n)\n    out = []\n    while n:\n        n, r = divmod(n, base)\n        out.append(digits[r])\n    return sign + ''.join(reversed(out))\n```\nFor bases 2/8/16, the built-in `bin/oct/hex` is faster.")

    add("algo",
        "Write a function that finds the longest run of consecutive integers in a list.",
        "```python\ndef longest_run(xs: list[int]) -> int:\n    s = set(xs)\n    best = 0\n    for x in s:\n        if x - 1 not in s:\n            length = 1\n            while x + length in s:\n                length += 1\n            best = max(best, length)\n    return best\n```\nO(n) by only starting a count at run-beginnings.")

    add("algo",
        "Implement a function that rotates a square matrix 90 degrees clockwise.",
        "```python\ndef rotate90(m: list[list]) -> list[list]:\n    return [list(row) for row in zip(*m[::-1])]\n```\nReverse rows then transpose -- the standard trick.")

    add("algo",
        "Write a function that returns whether a list is a palindrome.",
        "```python\ndef is_palindrome(xs: list) -> bool:\n    return xs == xs[::-1]\n```\nThe slice copy costs O(n) memory; for huge lists use a two-pointer loop.")

    add("algo",
        "Implement a function that computes the harmonic mean of a list.",
        "```python\nimport statistics\n\ndef harmonic(xs: list[float]) -> float:\n    return statistics.harmonic_mean(xs)\n```\n`statistics` covers most of what you'd hand-roll. It also handles empty / zero inputs with a clear error.")

    add("algo",
        "Write a function that returns the median of a list.",
        "```python\nimport statistics\n\ndef median(xs: list[float]) -> float:\n    return statistics.median(xs)\n```\nFor very large data use `numpy.median` (sorts in C).")

    add("algo",
        "Implement a function that returns the mode of a list.",
        "```python\nimport statistics\n\ndef mode(xs: list) -> object:\n    return statistics.mode(xs)\n```\n`statistics.mode` returns the first encountered mode on ties; use `multimode` if you want all of them.")

    add("algo",
        "Write a function that returns the variance of a sample.",
        "```python\nimport statistics\n\ndef variance(xs: list[float]) -> float:\n    return statistics.variance(xs)\n```\n`statistics.variance` uses Bessel's correction; use `pvariance` for the population variance.")

    add("algo",
        "Implement a function that flattens a dictionary with nested dicts using dot notation.",
        "```python\ndef flatten_dict(d: dict, prefix: str = '') -> dict:\n    out: dict = {}\n    for k, v in d.items():\n        key = f'{prefix}.{k}' if prefix else k\n        if isinstance(v, dict):\n            out.update(flatten_dict(v, key))\n        else:\n            out[key] = v\n    return out\n```\nUseful for flattening config files for Datadog or similar tools.")

    add("algo",
        "Write a function that unflattens a dotted-key dict into nested dicts.",
        "```python\ndef unflatten(d: dict) -> dict:\n    out: dict = {}\n    for key, v in d.items():\n        cur = out\n        parts = key.split('.')\n        for p in parts[:-1]:\n            cur = cur.setdefault(p, {})\n        cur[parts[-1]] = v\n    return out\n```\nInverse of the previous; uses `setdefault` to walk-and-create.")

    add("algo",
        "Implement a function that merges overlapping intervals.",
        "```python\ndef merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:\n    if not intervals:\n        return []\n    intervals = sorted(intervals)\n    out = [intervals[0]]\n    for start, end in intervals[1:]:\n        last_start, last_end = out[-1]\n        if start <= last_end:\n            out[-1] = (last_start, max(last_end, end))\n        else:\n            out.append((start, end))\n    return out\n```\nSort then sweep -- O(n log n).")

    add("algo",
        "Write a function that returns the longest common subsequence length.",
        "```python\ndef lcs(a: str, b: str) -> int:\n    if len(a) < len(b):\n        a, b = b, a\n    prev = [0] * (len(b) + 1)\n    for ca in a:\n        cur = [0] * (len(b) + 1)\n        for j, cb in enumerate(b, 1):\n            cur[j] = prev[j-1] + 1 if ca == cb else max(prev[j], cur[j-1])\n        prev = cur\n    return prev[-1]\n```\nO(n*m) time, O(min(n,m)) space.")

    add("algo",
        "Implement a function that converts an IP address string to an integer.",
        "```python\nimport ipaddress\n\ndef ip_to_int(ip: str) -> int:\n    return int(ipaddress.IPv4Address(ip))\n```\n`ipaddress` validates and converts; manual `.split('.')` parsing skips validation.")

    add("algo",
        "Write a function that converts an integer to an IP address.",
        "```python\nimport ipaddress\n\ndef int_to_ip(n: int) -> str:\n    return str(ipaddress.IPv4Address(n))\n```\nLet stdlib do the formatting -- correctness for free.")

    add("algo",
        "Implement a function that returns whether two intervals overlap.",
        "```python\ndef overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:\n    return a[0] < b[1] and b[0] < a[1]\n```\nHalf-open intervals (`[start, end)`) make this clean and avoid off-by-one mistakes.")

    add("algo",
        "Write a function that returns the symmetric difference of two sets.",
        "```python\ndef sym_diff(a: set, b: set) -> set:\n    return a ^ b\n```\nTrivial, but worth a wrapper if your codebase prefers verb-named helpers over operators.")

    add("algo",
        "Implement a function that compresses a string by removing consecutive duplicates.",
        "```python\nfrom itertools import groupby\n\ndef dedupe_consecutive(s: str) -> str:\n    return ''.join(c for c, _ in groupby(s))\n```\n`groupby` is the right tool for run-length problems.")

    add("algo",
        "Write a function that returns the n-th triangular number.",
        "```python\ndef triangular(n: int) -> int:\n    return n * (n + 1) // 2\n```\nClosed form beats summing in a loop.")

    add("algo",
        "Implement a function that returns the digit sum of an integer.",
        "```python\ndef digit_sum(n: int) -> int:\n    return sum(int(d) for d in str(abs(n)))\n```\nString conversion is short and clear; if performance matters, divmod by 10 in a loop.")

    add("algo",
        "Write a function that returns the digital root of an integer.",
        "```python\ndef digital_root(n: int) -> int:\n    return 0 if n == 0 else 1 + (abs(n) - 1) % 9\n```\nClosed form; avoids the iterative summing loop entirely.")

    add("algo",
        "Implement a function that converts seconds to a HH:MM:SS string.",
        "```python\ndef format_seconds(s: int) -> str:\n    h, rem = divmod(s, 3600)\n    m, sec = divmod(rem, 60)\n    return f'{h:02d}:{m:02d}:{sec:02d}'\n```\nFor durations that may exceed 24h, this is more correct than `time.strftime`.")

    add("algo",
        "Write a function that decides whether a year is a leap year.",
        "```python\ndef is_leap(y: int) -> bool:\n    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)\n```\nOr just `calendar.isleap(y)` from the stdlib.")

    add("algo",
        "Implement a function that returns the Hamming distance of two equal-length strings.",
        "```python\ndef hamming(a: str, b: str) -> int:\n    if len(a) != len(b):\n        raise ValueError('length mismatch')\n    return sum(x != y for x, y in zip(a, b))\n```\nBooleans sum as 0/1 -- idiomatic and concise.")

    add("algo",
        "Write a function that returns the Jaccard similarity of two sets.",
        "```python\ndef jaccard(a: set, b: set) -> float:\n    if not a and not b:\n        return 1.0\n    return len(a & b) / len(a | b)\n```\nSimple set arithmetic; the empty/empty case is a convention -- pick one and document it.")

    add("algo",
        "Implement a function that returns the dot product of two sparse vectors as dicts.",
        "```python\ndef sparse_dot(a: dict, b: dict) -> float:\n    if len(a) > len(b):\n        a, b = b, a\n    return sum(v * b.get(k, 0.0) for k, v in a.items())\n```\nIterate over the smaller dict; saves time on highly sparse data.")

    add("algo",
        "Write a function that returns the Cartesian distance between two 2D points.",
        "```python\nimport math\n\ndef distance(a: tuple[float, float], b: tuple[float, float]) -> float:\n    return math.dist(a, b)\n```\n`math.dist` (3.8+) handles arbitrary-dimension points and avoids manual squaring.")

    add("algo",
        "Implement a function that returns whether three points are collinear.",
        "```python\ndef collinear(a, b, c) -> bool:\n    return (b[0] - a[0]) * (c[1] - a[1]) == (b[1] - a[1]) * (c[0] - a[0])\n```\nCross-product test using integers avoids floating-point drift.")

    add("algo",
        "Write a function that returns the convex hull of 2D points (Graham scan).",
        "```python\ndef convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:\n    points = sorted(set(points))\n    if len(points) <= 1:\n        return points\n    def cross(o, a, b):\n        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])\n    lower = []\n    for p in points:\n        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:\n            lower.pop()\n        lower.append(p)\n    upper = []\n    for p in reversed(points):\n        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:\n            upper.pop()\n        upper.append(p)\n    return lower[:-1] + upper[:-1]\n```\nAndrew's monotone chain in O(n log n).")

    add("algo",
        "Implement a function that returns the n-th term of a recurrence with memoization.",
        "```python\nfrom functools import lru_cache\n\n@lru_cache(maxsize=None)\ndef tribonacci(n: int) -> int:\n    if n < 3:\n        return [0, 0, 1][n]\n    return tribonacci(n - 1) + tribonacci(n - 2) + tribonacci(n - 3)\n```\n`lru_cache` makes recursive DP a one-line change.")

    add("algo",
        "Write a function that pretty-prints a binary tree.",
        "```python\ndef pretty(node, prefix: str = '', is_left: bool = True) -> None:\n    if node is None:\n        return\n    print(prefix + ('|-- ' if is_left else '`-- ') + str(node.val))\n    new_prefix = prefix + ('|   ' if is_left else '    ')\n    pretty(node.left, new_prefix, True)\n    pretty(node.right, new_prefix, False)\n```\nReturns nothing -- prints as a side effect, like `tree(1)` shell command.")

    add("algo",
        "Implement a function that returns the BFS order of a graph.",
        "```python\nfrom collections import deque\n\ndef bfs(graph: dict, start) -> list:\n    seen, order, q = {start}, [], deque([start])\n    while q:\n        n = q.popleft()\n        order.append(n)\n        for nb in graph.get(n, []):\n            if nb not in seen:\n                seen.add(nb); q.append(nb)\n    return order\n```\n`deque` gives O(1) popleft; using `list.pop(0)` is O(n).")

    add("algo",
        "Write a function that returns the DFS order of a graph iteratively.",
        "```python\ndef dfs(graph: dict, start) -> list:\n    seen, order, stack = set(), [], [start]\n    while stack:\n        n = stack.pop()\n        if n in seen:\n            continue\n        seen.add(n); order.append(n)\n        stack.extend(graph.get(n, []))\n    return order\n```\nIterative avoids Python's recursion limit on deep graphs.")

    add("algo",
        "Implement a function that detects a cycle in an undirected graph.",
        "```python\ndef has_cycle(graph: dict) -> bool:\n    seen: set = set()\n    def visit(node, parent) -> bool:\n        seen.add(node)\n        for nb in graph.get(node, []):\n            if nb not in seen:\n                if visit(nb, node):\n                    return True\n            elif nb != parent:\n                return True\n        return False\n    return any(visit(n, None) for n in graph if n not in seen)\n```\nSkip the immediate parent to avoid false positives on the back-edge.")

    add("algo",
        "Write a function that performs topological sort using Kahn's algorithm.",
        "```python\nfrom collections import deque\n\ndef topo_sort(graph: dict) -> list:\n    in_deg = {n: 0 for n in graph}\n    for nbrs in graph.values():\n        for nb in nbrs:\n            in_deg[nb] = in_deg.get(nb, 0) + 1\n    q = deque(n for n, d in in_deg.items() if d == 0)\n    out = []\n    while q:\n        n = q.popleft(); out.append(n)\n        for nb in graph.get(n, []):\n            in_deg[nb] -= 1\n            if in_deg[nb] == 0:\n                q.append(nb)\n    if len(out) != len(in_deg):\n        raise ValueError('graph has a cycle')\n    return out\n```\nKahn's variant uses indegrees; Tarjan's recursive DFS is the alternative.")

    add("algo",
        "Implement Dijkstra's shortest path.",
        "```python\nimport heapq\n\ndef dijkstra(graph: dict, start) -> dict:\n    dist = {start: 0}\n    pq = [(0, start)]\n    while pq:\n        d, n = heapq.heappop(pq)\n        if d > dist.get(n, float('inf')):\n            continue\n        for nb, w in graph.get(n, []):\n            nd = d + w\n            if nd < dist.get(nb, float('inf')):\n                dist[nb] = nd\n                heapq.heappush(pq, (nd, nb))\n    return dist\n```\n`heapq` supports tuple ordering; the lazy-delete pattern avoids decrease-key complications.")

    add("algo",
        "Write a function that returns whether a graph is bipartite.",
        "```python\nfrom collections import deque\n\ndef is_bipartite(graph: dict) -> bool:\n    color: dict = {}\n    for start in graph:\n        if start in color:\n            continue\n        color[start] = 0\n        q = deque([start])\n        while q:\n            n = q.popleft()\n            for nb in graph.get(n, []):\n                if nb not in color:\n                    color[nb] = 1 - color[n]; q.append(nb)\n                elif color[nb] == color[n]:\n                    return False\n    return True\n```\nTwo-color BFS over each connected component.")

    add("algo",
        "Implement a function that returns connected components of an undirected graph.",
        "```python\ndef components(graph: dict) -> list[list]:\n    seen: set = set()\n    comps: list[list] = []\n    for start in graph:\n        if start in seen:\n            continue\n        stack = [start]; comp = []\n        while stack:\n            n = stack.pop()\n            if n in seen: continue\n            seen.add(n); comp.append(n)\n            stack.extend(graph.get(n, []))\n        comps.append(comp)\n    return comps\n```\nIterative DFS per unseen start node.")

    add("algo",
        "Write a function that returns the n-th element of a 2D matrix in spiral order.",
        "```python\ndef spiral(m: list[list]) -> list:\n    out = []\n    while m:\n        out += m.pop(0)\n        if m and m[0]:\n            for row in m:\n                out.append(row.pop())\n        if m:\n            out += m.pop()[::-1]\n        if m and m[0]:\n            for row in reversed(m):\n                out.append(row.pop(0))\n    return out\n```\nMutates a copy; for a non-mutating version track index bounds.")

    add("algo",
        "Implement a function that returns whether a number is prime.",
        "```python\ndef is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    if n < 4:\n        return True\n    if n % 2 == 0:\n        return False\n    for i in range(3, int(n**0.5) + 1, 2):\n        if n % i == 0:\n            return False\n    return True\n```\nFor very large n use `sympy.isprime` (Miller-Rabin).")

    add("algo",
        "Write a function that returns the prime factorization of n.",
        "```python\ndef factorize(n: int) -> list[int]:\n    factors = []\n    d = 2\n    while d * d <= n:\n        while n % d == 0:\n            factors.append(d); n //= d\n        d += 1\n    if n > 1:\n        factors.append(n)\n    return factors\n```\nTrial division up to sqrt(n); fine for inputs into the billions.")

    add("algo",
        "Implement a function that returns the n-th Fibonacci using matrix exponentiation.",
        "```python\ndef fib(n: int) -> int:\n    def mul(a, b):\n        return ((a[0]*b[0] + a[1]*b[2], a[0]*b[1] + a[1]*b[3]),\n                (a[2]*b[0] + a[3]*b[2], a[2]*b[1] + a[3]*b[3]))\n    def power(m, p):\n        result = (1, 0, 0, 1)\n        while p:\n            if p & 1: result = mul(result, m)\n            m = mul(m, m); p >>= 1\n        return result\n    return power((1, 1, 1, 0), n)[1]\n```\nO(log n) -- fast for very large n.")

    add("algo",
        "Write a function that returns the kth permutation of [1..n] without enumerating all.",
        "```python\nimport math\n\ndef kth_perm(n: int, k: int) -> list[int]:\n    nums = list(range(1, n + 1))\n    k -= 1\n    out = []\n    for i in range(n, 0, -1):\n        f = math.factorial(i - 1)\n        idx, k = divmod(k, f)\n        out.append(nums.pop(idx))\n    return out\n```\nFactorial number system -- O(n\u00b2) due to `pop`.")

    add("algo",
        "Implement a function that returns the longest increasing subsequence length.",
        "```python\nfrom bisect import bisect_left\n\ndef lis_length(xs: list[int]) -> int:\n    tails: list[int] = []\n    for x in xs:\n        i = bisect_left(tails, x)\n        if i == len(tails):\n            tails.append(x)\n        else:\n            tails[i] = x\n    return len(tails)\n```\nO(n log n) patience-sorting approach.")

    add("algo",
        "Write a function that returns whether a string contains balanced HTML-like tags.",
        "```python\nimport re\n\ndef balanced_tags(s: str) -> bool:\n    stack: list[str] = []\n    for tag in re.findall(r'</?(\\w+)>', s):\n        if not stack or not s[s.index(f'<{tag}>'):]:\n            pass\n        if s.find(f'</{tag}>') != -1 and stack and stack[-1] == tag:\n            stack.pop()\n        else:\n            stack.append(tag)\n    return not stack\n```\nFor real HTML use a parser like `html.parser` or `lxml`; regex is only suitable for the textbook version.")

    add("algo",
        "Implement a function that finds the median of a stream using two heaps.",
        "```python\nimport heapq\n\nclass RunningMedian:\n    def __init__(self) -> None:\n        self.lo: list = []  # max-heap (negated)\n        self.hi: list = []  # min-heap\n    def add(self, x: float) -> None:\n        heapq.heappush(self.lo, -heapq.heappushpop(self.hi, x))\n        if len(self.lo) > len(self.hi):\n            heapq.heappush(self.hi, -heapq.heappop(self.lo))\n    def median(self) -> float:\n        if len(self.hi) > len(self.lo):\n            return self.hi[0]\n        return (self.hi[0] - self.lo[0]) / 2\n```\nClassic two-heap design; O(log n) per insert.")

    add("algo",
        "Write a function that returns the indices of the next greater element for each item.",
        "```python\ndef next_greater(xs: list[int]) -> list[int]:\n    out = [-1] * len(xs)\n    stack: list[int] = []\n    for i, x in enumerate(xs):\n        while stack and xs[stack[-1]] < x:\n            out[stack.pop()] = i\n        stack.append(i)\n    return out\n```\nMonotonic stack in O(n).")

    add("algo",
        "Implement a function that counts set bits in an integer.",
        "```python\ndef popcount(n: int) -> int:\n    return n.bit_count()\n```\n`int.bit_count()` (Python 3.10+) is C-implemented. For earlier versions, `bin(n).count('1')` is the fallback.")

    add("algo",
        "Write a function that returns the n-th ugly number (factors only 2, 3, 5).",
        "```python\ndef nth_ugly(n: int) -> int:\n    ugly = [1]\n    i2 = i3 = i5 = 0\n    while len(ugly) < n:\n        nxt = min(ugly[i2] * 2, ugly[i3] * 3, ugly[i5] * 5)\n        ugly.append(nxt)\n        if nxt == ugly[i2] * 2: i2 += 1\n        if nxt == ugly[i3] * 3: i3 += 1\n        if nxt == ugly[i5] * 5: i5 += 1\n    return ugly[-1]\n```\nThree-pointer DP -- O(n).")

    add("algo",
        "Implement a function that returns whether a linked list has a cycle (Floyd's).",
        "```python\ndef has_cycle(head) -> bool:\n    slow = fast = head\n    while fast and fast.next:\n        slow, fast = slow.next, fast.next.next\n        if slow is fast:\n            return True\n    return False\n```\nO(n) time, O(1) space -- the textbook tortoise-and-hare.")

    add("algo",
        "Write a function that returns the length of the longest run of zeros in the binary representation.",
        "```python\ndef longest_zero_run(n: int) -> int:\n    s = bin(n)[2:].rstrip('0')\n    if not s:\n        return 0\n    return max((len(part) for part in s.split('1') if part), default=0)\n```\nStrip trailing zeros so we don't count the open run at the end.")

    add("algo",
        "Implement a function that simulates a single dice roll deterministically.",
        "```python\nimport random\n\ndef roll(rng: random.Random | None = None) -> int:\n    return (rng or random).randint(1, 6)\n```\nAccept an injected RNG so tests can pass `random.Random(seed)` for determinism.")

    add("algo",
        "Write a function that returns whether a Sudoku row/column/box is valid.",
        "```python\ndef is_valid_unit(unit: list[str]) -> bool:\n    seen = [c for c in unit if c != '.']\n    return len(seen) == len(set(seen))\n```\nIgnore blanks, then check no duplicates among the digits seen.")

    add("algo",
        "Implement a function that returns whether a 9x9 Sudoku board is currently valid.",
        "```python\ndef valid_sudoku(board: list[list[str]]) -> bool:\n    def ok(unit):\n        seen = [c for c in unit if c != '.']\n        return len(seen) == len(set(seen))\n    for i in range(9):\n        if not ok(board[i]):\n            return False\n        if not ok([board[r][i] for r in range(9)]):\n            return False\n    for br in range(0, 9, 3):\n        for bc in range(0, 9, 3):\n            box = [board[r][c] for r in range(br, br+3) for c in range(bc, bc+3)]\n            if not ok(box):\n                return False\n    return True\n```\nValidate rows, columns, and 3x3 boxes -- 27 unit checks total.")

    add("algo",
        "Write a function that returns the longest palindromic substring.",
        "```python\ndef longest_palindrome(s: str) -> str:\n    if not s:\n        return ''\n    def expand(l, r):\n        while l >= 0 and r < len(s) and s[l] == s[r]:\n            l -= 1; r += 1\n        return s[l+1:r]\n    best = ''\n    for i in range(len(s)):\n        for cand in (expand(i, i), expand(i, i+1)):\n            if len(cand) > len(best):\n                best = cand\n    return best\n```\nExpand-around-center -- O(n\u00b2). Manacher's is O(n) but rarely worth the complexity.")

    add("algo",
        "Implement a function that simulates `range` for floats.",
        "```python\ndef frange(start: float, stop: float, step: float = 1.0):\n    if step == 0:\n        raise ValueError('step must not be zero')\n    n = int(round((stop - start) / step))\n    for i in range(n):\n        yield start + i * step\n```\nMultiply-from-start avoids cumulative floating-point drift.")

    add("algo",
        "Write a function that returns the longest word in a sentence.",
        "```python\ndef longest_word(s: str) -> str:\n    return max(s.split(), key=len, default='')\n```\n`max` with `key=` and `default=` is the cleanest pattern.")

    add("algo",
        "Implement a function that returns the index of the smallest letter greater than a target.",
        "```python\nfrom bisect import bisect_right\n\ndef next_letter(letters: list[str], target: str) -> str:\n    i = bisect_right(letters, target)\n    return letters[i % len(letters)]\n```\nWraps around -- a common variant of the bisect problem.")

    add("algo",
        "Write a function that finds the duplicate in a list of n+1 ints in [1..n].",
        "```python\ndef find_duplicate(xs: list[int]) -> int:\n    slow = fast = xs[0]\n    while True:\n        slow = xs[slow]\n        fast = xs[xs[fast]]\n        if slow == fast:\n            break\n    slow = xs[0]\n    while slow != fast:\n        slow, fast = xs[slow], xs[fast]\n    return slow\n```\nFloyd's cycle detection -- O(n) time, O(1) space, no input mutation.")

    add("algo",
        "Implement a function that returns the maximum profit from one stock buy/sell.",
        "```python\ndef max_profit(prices: list[float]) -> float:\n    best, lo = 0.0, float('inf')\n    for p in prices:\n        lo = min(lo, p)\n        best = max(best, p - lo)\n    return best\n```\nTrack the running min and best gain in one pass.")

    add("algo",
        "Write a function that returns the maximum profit with unlimited transactions.",
        "```python\ndef max_profit_multi(prices: list[float]) -> float:\n    return sum(max(0, b - a) for a, b in zip(prices, prices[1:]))\n```\nSum every positive day-to-day delta -- equivalent to optimal multi-transaction.")

    add("algo",
        "Implement a function that decodes a URL-safe base64 string.",
        "```python\nimport base64\n\ndef decode_b64(s: str) -> bytes:\n    pad = '=' * (-len(s) % 4)\n    return base64.urlsafe_b64decode(s + pad)\n```\nAuto-pads so the input doesn't need to come pre-padded.")

    add("algo",
        "Write a function that hashes a string with SHA-256 and returns hex.",
        "```python\nimport hashlib\n\ndef sha256_hex(s: str) -> str:\n    return hashlib.sha256(s.encode()).hexdigest()\n```\nAlways encode explicitly; Python 3 hashes operate on bytes.")
