# SPDX-License-Identifier: MIT
"""Algorithms / data structures entries part 1 (75). Wave 6 part 1."""


def register_all(add):
    add("algo",
        "Write a function that returns the maximum subarray sum (Kadane's algorithm).",
        "```python\ndef max_subarray(xs: list[int]) -> int:\n    if not xs:\n        raise ValueError('empty input')\n    best = cur = xs[0]\n    for x in xs[1:]:\n        cur = max(x, cur + x)\n        best = max(best, cur)\n    return best\n```\nO(n) time, O(1) space. The trick: at each index, either extend the prior subarray or start fresh.")

    add("algo",
        "Implement a function that rotates a list to the right by k positions.",
        "```python\ndef rotate(xs: list, k: int) -> list:\n    if not xs:\n        return []\n    k %= len(xs)\n    return xs[-k:] + xs[:-k]\n```\nSlicing makes this O(n) and easy to read. The modulus handles `k > len(xs)` gracefully.")

    add("algo",
        "Write a function that merges two dictionaries, summing values for shared keys.",
        "```python\nfrom collections import Counter\n\ndef merge_sum(a: dict, b: dict) -> dict:\n    return dict(Counter(a) + Counter(b))\n```\n`Counter`'s `+` discards zero/negative entries; if you need them preserved, use a manual loop with `dict.get(k, 0)`.")

    add("algo",
        "Implement a function that returns whether two strings are anagrams.",
        "```python\nfrom collections import Counter\n\ndef anagrams(a: str, b: str) -> bool:\n    return Counter(a) == Counter(b)\n```\nO(n) and clear. Sorting both strings also works but is O(n log n).")

    add("algo",
        "Write a function that returns the most common element in a list.",
        "```python\nfrom collections import Counter\n\ndef most_common(xs: list):\n    if not xs:\n        return None\n    return Counter(xs).most_common(1)[0][0]\n```\n`Counter.most_common` is implemented with a heap, so it's efficient even for large inputs when you only want the top few.")

    add("algo",
        "Implement a function that removes duplicates from a list while preserving order.",
        "```python\ndef dedupe(xs: list) -> list:\n    seen: set = set()\n    out = []\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\n```\nFor hashable items only; if you need to dedupe dicts, key on a canonical tuple form.")

    add("algo",
        "Write a function that returns the Cartesian product of two lists.",
        "```python\nfrom itertools import product\n\ndef cartesian(a: list, b: list) -> list[tuple]:\n    return list(product(a, b))\n```\n`itertools.product` generalizes to any number of inputs and supports `repeat=`.")

    add("algo",
        "Implement a function that groups items by a key.",
        "```python\nfrom collections import defaultdict\n\ndef group_by(xs, key):\n    out = defaultdict(list)\n    for x in xs:\n        out[key(x)].append(x)\n    return dict(out)\n```\n`itertools.groupby` only groups consecutive items, so for the general case `defaultdict` is the right tool.")

    add("algo",
        "Write a function that returns the n smallest elements of an iterable.",
        "```python\nimport heapq\n\ndef n_smallest(xs, n: int) -> list:\n    return heapq.nsmallest(n, xs)\n```\nO(n log k) where k=n; faster than sorting when k << len(xs).")

    add("algo",
        "Implement a function that computes a running average.",
        "```python\nfrom typing import Iterator\n\ndef running_avg(xs) -> Iterator[float]:\n    total = 0.0\n    for i, x in enumerate(xs, 1):\n        total += x\n        yield total / i\n```\nGenerator avoids storing the whole sequence; emits one average per input element.")

    add("algo",
        "Write a function that flattens a dictionary of dictionaries with dotted keys.",
        "```python\ndef flatten_dict(d: dict, prefix: str = '') -> dict:\n    out = {}\n    for k, v in d.items():\n        key = f'{prefix}.{k}' if prefix else k\n        if isinstance(v, dict):\n            out.update(flatten_dict(v, key))\n        else:\n            out[key] = v\n    return out\n```\nUseful for turning nested config into env-style flat keys.")

    add("algo",
        "Implement a function that swaps keys and values in a dict.",
        "```python\ndef invert(d: dict) -> dict:\n    return {v: k for k, v in d.items()}\n```\nIf values aren't unique, later keys overwrite earlier ones. Build a `defaultdict(list)` instead if you need many-to-one inversion.")

    add("algo",
        "Write a function that finds two numbers in a list that sum to a target.",
        "```python\ndef two_sum(xs: list[int], target: int) -> tuple[int, int] | None:\n    seen: dict = {}\n    for i, x in enumerate(xs):\n        if target - x in seen:\n            return seen[target - x], i\n        seen[x] = i\n    return None\n```\nO(n) with a hash map. The classic interview problem; the trick is storing complement -> index.")

    add("algo",
        "Implement a function that converts a list of pairs into a dict, keeping the last value per key.",
        "```python\ndef pairs_to_dict(pairs: list[tuple]) -> dict:\n    return dict(pairs)\n```\n`dict()` of an iterable of pairs already keeps the last value per key. If you need the first, reverse first or build manually.")

    add("algo",
        "Write a function that returns the longest run of consecutive equal items.",
        "```python\nfrom itertools import groupby\n\ndef longest_run(xs: list) -> int:\n    return max((sum(1 for _ in g) for _, g in groupby(xs)), default=0)\n```\n`groupby` collapses consecutive equal items into groups; we just count each.")

    add("algo",
        "Implement a function that pads a list to length n with a fill value.",
        "```python\ndef pad(xs: list, n: int, fill=None) -> list:\n    return xs + [fill] * max(0, n - len(xs))\n```\nReturns a copy. If you need to pad in place, `xs.extend([fill] * (n - len(xs)))`.")

    add("algo",
        "Write a function that checks whether a string contains balanced parentheses.",
        "```python\ndef balanced(s: str) -> bool:\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack = []\n    for c in s:\n        if c in '([{':\n            stack.append(c)\n        elif c in pairs:\n            if not stack or stack.pop() != pairs[c]:\n                return False\n    return not stack\n```\nClassic stack problem; works for any bracket types listed in `pairs`.")

    add("algo",
        "Implement a Caesar cipher encode function.",
        "```python\ndef caesar(s: str, shift: int) -> str:\n    out = []\n    for c in s:\n        if c.isupper():\n            out.append(chr((ord(c) - 65 + shift) % 26 + 65))\n        elif c.islower():\n            out.append(chr((ord(c) - 97 + shift) % 26 + 97))\n        else:\n            out.append(c)\n    return ''.join(out)\n```\nNon-letters pass through unchanged. Decode by passing `-shift`.")

    add("algo",
        "Write a function that returns the cumulative sum of a list.",
        "```python\nfrom itertools import accumulate\n\ndef cumsum(xs: list) -> list:\n    return list(accumulate(xs))\n```\n`itertools.accumulate` accepts a custom binary op via `func=`, so you can also do running max, running product, etc.")

    add("algo",
        "Implement a function that zips two lists, padding the shorter one.",
        "```python\nfrom itertools import zip_longest\n\ndef zip_padded(a: list, b: list, fill=None) -> list[tuple]:\n    return list(zip_longest(a, b, fillvalue=fill))\n```\nUse this when you need every element of both inputs; default `zip` truncates silently which is a common bug source.")

    add("algo",
        "Write a function that returns whether a list is sorted.",
        "```python\ndef is_sorted(xs: list) -> bool:\n    return all(a <= b for a, b in zip(xs, xs[1:]))\n```\nWorks for any orderable items. For huge lists, the `zip` of slices avoids building intermediate copies because zip is lazy.")

    add("algo",
        "Implement a function that returns the difference between consecutive elements.",
        "```python\ndef diffs(xs: list) -> list:\n    return [b - a for a, b in zip(xs, xs[1:])]\n```\nOutput length is `len(xs) - 1`. For numpy arrays use `np.diff`.")

    add("algo",
        "Write a function that removes a key from a nested dict by dotted path.",
        "```python\ndef remove_path(d: dict, path: str) -> None:\n    parts = path.split('.')\n    cur = d\n    for p in parts[:-1]:\n        if not isinstance(cur, dict) or p not in cur:\n            return\n        cur = cur[p]\n    if isinstance(cur, dict):\n        cur.pop(parts[-1], None)\n```\nMutates in place; missing intermediate keys are silently skipped.")

    add("algo",
        "Implement a function that returns the chunk-of-n iterator for a list.",
        "```python\ndef chunks(xs: list, n: int):\n    for i in range(0, len(xs), n):\n        yield xs[i:i+n]\n```\nGenerator avoids materializing all chunks at once. Useful for batched API calls.")

    add("algo",
        "Write a function that computes the dot product of two vectors.",
        "```python\ndef dot(a: list[float], b: list[float]) -> float:\n    if len(a) != len(b):\n        raise ValueError('length mismatch')\n    return sum(x*y for x, y in zip(a, b))\n```\nFor numeric work, prefer `numpy.dot` -- it's vectorized and ~50x faster for non-trivial sizes.")

    add("algo",
        "Implement a function that converts an integer to a Roman numeral.",
        "```python\ndef to_roman(n: int) -> str:\n    pairs = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]\n    out = []\n    for v, s in pairs:\n        while n >= v:\n            out.append(s); n -= v\n    return ''.join(out)\n```\nGreedy; the ordered pair list encodes the subtractive forms (CM, CD, XC...).")

    add("algo",
        "Write a function that returns the digit sum of an integer.",
        "```python\ndef digit_sum(n: int) -> int:\n    return sum(int(c) for c in str(abs(n)))\n```\nString conversion is the simplest correct approach in Python; arithmetic via `divmod` is faster but less readable.")

    add("algo",
        "Implement a function that reverses an integer.",
        "```python\ndef reverse_int(n: int) -> int:\n    sign = -1 if n < 0 else 1\n    return sign * int(str(abs(n))[::-1])\n```\nPreserves sign. Watch for trailing zeros: 120 reversed becomes 21, not 021.")

    add("algo",
        "Write a function that returns whether a number is a perfect square.",
        "```python\nimport math\n\ndef is_square(n: int) -> bool:\n    if n < 0:\n        return False\n    r = math.isqrt(n)\n    return r * r == n\n```\n`math.isqrt` is exact (no float rounding). `int(math.sqrt(n))` fails for very large integers.")

    add("algo",
        "Implement a function that converts seconds into HH:MM:SS.",
        "```python\ndef hms(secs: int) -> str:\n    h, rem = divmod(secs, 3600)\n    m, s = divmod(rem, 60)\n    return f'{h:02d}:{m:02d}:{s:02d}'\n```\n`divmod` returns quotient and remainder in one call -- cleaner than two divisions.")

    add("algo",
        "Write a function that finds the missing number in a list of 0..n.",
        "```python\ndef missing(xs: list[int]) -> int:\n    n = len(xs)\n    return n * (n + 1) // 2 - sum(xs)\n```\nClosed-form Gauss sum minus the actual sum. O(n) time, O(1) space; beats hash sets here.")

    add("algo",
        "Implement a function that returns the union-find / disjoint-set parent helpers.",
        "```python\nclass DSU:\n    def __init__(self, n: int):\n        self.p = list(range(n))\n        self.r = [0] * n\n    def find(self, x: int) -> int:\n        while self.p[x] != x:\n            self.p[x] = self.p[self.p[x]]\n            x = self.p[x]\n        return x\n    def union(self, a: int, b: int) -> bool:\n        ra, rb = self.find(a), self.find(b)\n        if ra == rb: return False\n        if self.r[ra] < self.r[rb]: ra, rb = rb, ra\n        self.p[rb] = ra\n        if self.r[ra] == self.r[rb]: self.r[ra] += 1\n        return True\n```\nPath compression + union by rank gives near-O(1) amortized per op.")

    add("algo",
        "Write a function that returns BFS order from a starting node in a graph.",
        "```python\nfrom collections import deque\n\ndef bfs(graph: dict, start) -> list:\n    seen = {start}\n    out = []\n    q = deque([start])\n    while q:\n        node = q.popleft()\n        out.append(node)\n        for n in graph.get(node, []):\n            if n not in seen:\n                seen.add(n)\n                q.append(n)\n    return out\n```\n`deque` for O(1) popleft; `set` for O(1) membership.")

    add("algo",
        "Implement DFS for a graph.",
        "```python\ndef dfs(graph: dict, start) -> list:\n    seen = set()\n    out = []\n    def visit(n):\n        if n in seen: return\n        seen.add(n); out.append(n)\n        for m in graph.get(n, []):\n            visit(m)\n    visit(start)\n    return out\n```\nRecursive form; for very deep graphs use an explicit stack to avoid `RecursionError`.")

    add("algo",
        "Write a function that detects whether a directed graph has a cycle.",
        "```python\ndef has_cycle(graph: dict) -> bool:\n    WHITE, GRAY, BLACK = 0, 1, 2\n    color = {n: WHITE for n in graph}\n    def dfs(n):\n        color[n] = GRAY\n        for m in graph.get(n, []):\n            if color.get(m, WHITE) == GRAY: return True\n            if color.get(m, WHITE) == WHITE and dfs(m): return True\n        color[n] = BLACK\n        return False\n    return any(dfs(n) for n in graph if color[n] == WHITE)\n```\nThree-color DFS: gray means on the current path, so revisiting one is a back-edge.")

    add("algo",
        "Implement a topological sort.",
        "```python\nfrom collections import deque\n\ndef toposort(graph: dict) -> list:\n    indeg = {n: 0 for n in graph}\n    for n in graph:\n        for m in graph[n]:\n            indeg[m] = indeg.get(m, 0) + 1\n    q = deque([n for n, d in indeg.items() if d == 0])\n    out = []\n    while q:\n        n = q.popleft(); out.append(n)\n        for m in graph.get(n, []):\n            indeg[m] -= 1\n            if indeg[m] == 0: q.append(m)\n    if len(out) != len(indeg): raise ValueError('cycle')\n    return out\n```\nKahn's algorithm; if the output is shorter than the node count, there's a cycle.")

    add("algo",
        "Write a function that computes Levenshtein edit distance.",
        "```python\ndef edit_distance(a: str, b: str) -> int:\n    if len(a) < len(b): a, b = b, a\n    prev = list(range(len(b) + 1))\n    for i, ca in enumerate(a, 1):\n        cur = [i]\n        for j, cb in enumerate(b, 1):\n            cost = 0 if ca == cb else 1\n            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j-1] + cost))\n        prev = cur\n    return prev[-1]\n```\nO(n*m) time, O(min(n,m)) space using two rolling rows.")

    add("algo",
        "Implement a function that returns whether two ranges overlap.",
        "```python\ndef overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:\n    return a[0] < b[1] and b[0] < a[1]\n```\nHalf-open intervals `[lo, hi)`. The two strict inequalities are the elegant test -- shorter than `not (a[1] <= b[0] or b[1] <= a[0])`.")

    add("algo",
        "Write a function that merges overlapping intervals.",
        "```python\ndef merge_intervals(ivs: list[tuple[int,int]]) -> list[tuple[int,int]]:\n    if not ivs: return []\n    ivs = sorted(ivs)\n    out = [ivs[0]]\n    for s, e in ivs[1:]:\n        if s <= out[-1][1]:\n            out[-1] = (out[-1][0], max(out[-1][1], e))\n        else:\n            out.append((s, e))\n    return out\n```\nSort, then sweep. O(n log n) dominated by the sort.")

    add("algo",
        "Implement a function that returns whether a list is a permutation of 1..n.",
        "```python\ndef is_perm(xs: list[int]) -> bool:\n    n = len(xs)\n    return sorted(xs) == list(range(1, n + 1))\n```\nSimple. For O(n), check `set(xs) == set(range(1, n+1))` -- equivalent for hashable items.")

    add("algo",
        "Write a function that returns the n-th term of an arithmetic progression.",
        "```python\ndef ap(a0: float, d: float, n: int) -> float:\n    return a0 + (n - 1) * d\n```\nClosed form is O(1); only iterate if you need every intermediate term.")

    add("algo",
        "Implement a sliding-window maximum.",
        "```python\nfrom collections import deque\n\ndef window_max(xs: list[int], k: int) -> list[int]:\n    q: deque = deque()\n    out = []\n    for i, x in enumerate(xs):\n        while q and xs[q[-1]] <= x: q.pop()\n        q.append(i)\n        if q[0] <= i - k: q.popleft()\n        if i >= k - 1: out.append(xs[q[0]])\n    return out\n```\nMonotonic deque keeps indices of candidates in decreasing order. O(n) total.")

    add("algo",
        "Write a function that returns the trailing zeros of n!.",
        "```python\ndef trailing_zeros(n: int) -> int:\n    count = 0\n    while n:\n        n //= 5\n        count += n\n    return count\n```\nEvery factor of 10 in n! comes from a 2 and a 5; 5s are the bottleneck. Sum of `n // 5^k` gives the answer.")

    add("algo",
        "Implement a function that converts a column number to Excel-style letters.",
        "```python\ndef to_excel(n: int) -> str:\n    out = []\n    while n > 0:\n        n, r = divmod(n - 1, 26)\n        out.append(chr(65 + r))\n    return ''.join(reversed(out))\n```\nThe `n - 1` accounts for 1-based indexing without a zero digit (A is 1, Z is 26, AA is 27).")

    add("algo",
        "Write a function that finds the majority element (>n/2 occurrences).",
        "```python\ndef majority(xs: list) -> object | None:\n    cand = None\n    count = 0\n    for x in xs:\n        if count == 0: cand = x\n        count += 1 if x == cand else -1\n    return cand if xs.count(cand) > len(xs) // 2 else None\n```\nBoyer-Moore voting: O(n) time, O(1) space. Verify the candidate at the end since the algorithm assumes a majority exists.")

    add("algo",
        "Implement a function that returns the nth row of Pascal's triangle.",
        "```python\nfrom math import comb\n\ndef pascal_row(n: int) -> list[int]:\n    return [comb(n, k) for k in range(n + 1)]\n```\n`math.comb` (Python 3.8+) avoids float intermediates. For very large n, build iteratively with `row[k] = row[k-1] * (n-k+1) // k`.")

    add("algo",
        "Write a function that returns the smallest k elements using a max-heap.",
        "```python\nimport heapq\n\ndef k_smallest(xs: list[int], k: int) -> list[int]:\n    heap: list = []\n    for x in xs:\n        heapq.heappush(heap, -x)\n        if len(heap) > k:\n            heapq.heappop(heap)\n    return sorted(-x for x in heap)\n```\nNegate values to simulate a max-heap. O(n log k) -- beats sorting when k is small.")

    add("algo",
        "Implement a function that decodes a run-length-encoded string.",
        "```python\nimport re\n\ndef rle_decode(s: str) -> str:\n    return ''.join(c * int(n) for n, c in re.findall(r'(\\d+)(.)', s))\n```\nRegex `(\\d+)(.)` captures (count, char) pairs; multiplication via `c * n` produces the run.")

    add("algo",
        "Write a function that encodes a string with run-length encoding.",
        "```python\nfrom itertools import groupby\n\ndef rle_encode(s: str) -> str:\n    return ''.join(f'{sum(1 for _ in g)}{ch}' for ch, g in groupby(s))\n```\n`groupby` groups consecutive identical chars; format is `<count><char>`.")

    add("algo",
        "Implement a function that converts a snake_case string to camelCase.",
        "```python\ndef to_camel(s: str) -> str:\n    parts = s.split('_')\n    return parts[0] + ''.join(p.capitalize() for p in parts[1:])\n```\nFirst part stays lowercase; the rest are title-cased and concatenated.")

    add("algo",
        "Write a function that converts camelCase to snake_case.",
        "```python\nimport re\n\ndef to_snake(s: str) -> str:\n    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()\n```\nThe lookbehind `(?<!^)` prevents an underscore at index 0 for already-PascalCase input.")

    add("algo",
        "Implement a function that returns whether a year is a leap year.",
        "```python\ndef is_leap(y: int) -> bool:\n    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)\n```\nDivisible by 4 unless divisible by 100, except divisible by 400. Python's `calendar.isleap` does the same.")

    add("algo",
        "Write a function that returns whether a number is an Armstrong number.",
        "```python\ndef armstrong(n: int) -> bool:\n    digits = str(n)\n    p = len(digits)\n    return sum(int(d)**p for d in digits) == n\n```\n153 = 1^3 + 5^3 + 3^3. Common puzzle problem; the closed form needs no special tricks.")

    add("algo",
        "Implement a function that converts a binary string to an integer.",
        "```python\ndef bin_to_int(s: str) -> int:\n    return int(s, 2)\n```\n`int(s, base)` accepts bases 2-36. For the inverse, `bin(n)[2:]` strips the `0b` prefix.")

    add("algo",
        "Write a function that returns the XOR of all elements in a list.",
        "```python\nfrom functools import reduce\nimport operator\n\ndef xor_all(xs: list[int]) -> int:\n    return reduce(operator.xor, xs, 0)\n```\nXOR is associative and commutative; identity is 0. Useful for finding the unique element when all others appear in pairs.")

    add("algo",
        "Implement a function that finds the unique element when all others appear twice.",
        "```python\nfrom functools import reduce\nimport operator\n\ndef single(xs: list[int]) -> int:\n    return reduce(operator.xor, xs)\n```\nXOR cancels duplicates: `x ^ x = 0`, `x ^ 0 = x`. Result is the unique element. O(n) time, O(1) space.")

    add("algo",
        "Write a function that compresses spaces: collapses runs of whitespace into one space.",
        "```python\nimport re\n\ndef collapse_ws(s: str) -> str:\n    return re.sub(r'\\s+', ' ', s).strip()\n```\n`\\s+` matches any run of whitespace including tabs and newlines. `.strip()` removes leading/trailing.")

    add("algo",
        "Implement a function that returns the median of a list.",
        "```python\nimport statistics\n\ndef median(xs: list[float]) -> float:\n    if not xs:\n        raise ValueError('empty')\n    return statistics.median(xs)\n```\n`statistics.median` handles odd/even lengths. For numeric arrays at scale, `numpy.median` is faster.")

    add("algo",
        "Write a function that returns the mode of a list, ties broken by first occurrence.",
        "```python\nfrom collections import Counter\n\ndef mode(xs: list):\n    if not xs:\n        return None\n    counts = Counter(xs)\n    best = max(counts.values())\n    for x in xs:\n        if counts[x] == best:\n            return x\n```\nIterating original order resolves ties to first-seen.")

    add("algo",
        "Implement a function that returns whether two strings are rotations of each other.",
        "```python\ndef is_rotation(a: str, b: str) -> bool:\n    return len(a) == len(b) and b in a + a\n```\nClever one-liner: any rotation of `a` is a substring of `a + a`. O(n) using `in` (string search).")

    add("algo",
        "Write a function that returns the longest substring without repeating characters.",
        "```python\ndef longest_unique(s: str) -> int:\n    last: dict = {}\n    start = best = 0\n    for i, c in enumerate(s):\n        if c in last and last[c] >= start:\n            start = last[c] + 1\n        last[c] = i\n        best = max(best, i - start + 1)\n    return best\n```\nSliding window with last-seen index. O(n) time, O(k) space where k is alphabet size.")

    add("algo",
        "Implement a function that compresses repeated characters: 'aabcccd' -> 'a2bc3d'.",
        "```python\nfrom itertools import groupby\n\ndef compress(s: str) -> str:\n    out = []\n    for ch, g in groupby(s):\n        n = sum(1 for _ in g)\n        out.append(ch + (str(n) if n > 1 else ''))\n    return ''.join(out)\n```\nOmits the count for runs of length 1 -- the more common spec.")

    add("algo",
        "Write a function that returns whether one string is a permutation of another.",
        "```python\nfrom collections import Counter\n\ndef is_perm_str(a: str, b: str) -> bool:\n    return Counter(a) == Counter(b)\n```\nO(n). Same as anagram check; the names tend to overlap in the wild.")

    add("algo",
        "Implement a function that finds the kth-largest element.",
        "```python\nimport heapq\n\ndef kth_largest(xs: list[int], k: int) -> int:\n    return heapq.nlargest(k, xs)[-1]\n```\nO(n log k); for repeated queries on the same list, pre-sort once and index.")

    add("algo",
        "Write a function that produces a memoized Fibonacci.",
        "```python\nfrom functools import lru_cache\n\n@lru_cache(maxsize=None)\ndef fib(n: int) -> int:\n    if n < 2:\n        return n\n    return fib(n-1) + fib(n-2)\n```\n`lru_cache` makes the recursive form efficient. For very large n, iterate -- recursion depth becomes the bottleneck.")

    add("algo",
        "Implement a function that returns the longest increasing subsequence length.",
        "```python\nfrom bisect import bisect_left\n\ndef lis(xs: list[int]) -> int:\n    tails: list = []\n    for x in xs:\n        i = bisect_left(tails, x)\n        if i == len(tails):\n            tails.append(x)\n        else:\n            tails[i] = x\n    return len(tails)\n```\nO(n log n) via patience-sort tail trick. `tails[i]` is the smallest tail of an increasing subseq of length i+1.")

    add("algo",
        "Write a function that returns whether a string is a valid number (int or float).",
        "```python\ndef is_number(s: str) -> bool:\n    try:\n        float(s)\n        return True\n    except ValueError:\n        return False\n```\nLeaning on `float` is shorter and handles edge cases (scientific notation, signs) without bugs.")

    add("algo",
        "Implement a function that returns the n-th Catalan number.",
        "```python\nfrom math import comb\n\ndef catalan(n: int) -> int:\n    return comb(2*n, n) // (n + 1)\n```\nClosed form via the central binomial coefficient. O(n) for the comb computation in arbitrary precision.")

    add("algo",
        "Write a function that returns the median of two sorted arrays merged.",
        "```python\nimport heapq, statistics\n\ndef merged_median(a: list[int], b: list[int]) -> float:\n    return statistics.median(list(heapq.merge(a, b)))\n```\n`heapq.merge` is lazy and O(n+m); building the merged list dominates. The O(log min(n,m)) algorithm is faster but rarely needed.")

    add("algo",
        "Implement a function that returns the count of inversions in a list.",
        "```python\ndef inversions(xs: list[int]) -> int:\n    def merge(a, b):\n        out, i, j, inv = [], 0, 0, 0\n        while i < len(a) and j < len(b):\n            if a[i] <= b[j]:\n                out.append(a[i]); i += 1\n            else:\n                out.append(b[j]); j += 1; inv += len(a) - i\n        out.extend(a[i:]); out.extend(b[j:])\n        return out, inv\n    def go(xs):\n        if len(xs) < 2: return xs, 0\n        m = len(xs)//2\n        a, ia = go(xs[:m]); b, ib = go(xs[m:])\n        c, ic = merge(a, b)\n        return c, ia + ib + ic\n    return go(xs)[1]\n```\nMerge-sort variant; O(n log n).")

    add("algo",
        "Write a function that returns whether a sequence is monotonic.",
        "```python\ndef monotonic(xs: list) -> bool:\n    inc = all(a <= b for a, b in zip(xs, xs[1:]))\n    dec = all(a >= b for a, b in zip(xs, xs[1:]))\n    return inc or dec\n```\nTwo passes is fine; a single pass with state-tracking flags would be faster but harder to read.")
