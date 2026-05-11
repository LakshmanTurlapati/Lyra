# SPDX-License-Identifier: MIT
"""Algorithms / data structures entries part 1 (75). Wave 7 part 1."""


def register_all(add):
    add("algo",
        "Write a function that returns the GCD of two positive integers using Euclid's algorithm.",
        "```python\ndef gcd(a: int, b: int) -> int:\n    while b:\n        a, b = b, a % b\n    return a\n```\nO(log min(a,b)). The standard library has `math.gcd` -- use that in real code; this version is for when you need to teach the algorithm.")

    add("algo",
        "Implement a function that computes the LCM of a list of integers.",
        "```python\nfrom math import gcd\nfrom functools import reduce\n\ndef lcm_all(xs: list[int]) -> int:\n    return reduce(lambda a, b: a * b // gcd(a, b), xs)\n```\n`math.lcm` exists in 3.9+ and accepts variadic args; reach for it first. The reduce form is portable to older versions.")

    add("algo",
        "Write a function that returns the n-th Fibonacci number iteratively.",
        "```python\ndef fib(n: int) -> int:\n    if n < 0:\n        raise ValueError('n must be non-negative')\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```\nO(n) time, O(1) space. The recursive version without memoization is exponential -- avoid it.")

    add("algo",
        "Implement binary search on a sorted list.",
        "```python\ndef bsearch(xs: list[int], target: int) -> int:\n    lo, hi = 0, len(xs) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if xs[mid] == target:\n            return mid\n        if xs[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n```\nThe stdlib `bisect.bisect_left` is what you actually want in production; this is the reference implementation.")

    add("algo",
        "Write a function that returns whether a number is prime.",
        "```python\ndef is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    if n % 2 == 0:\n        return n == 2\n    i = 3\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 2\n    return True\n```\nO(sqrt(n)). For very large numbers use Miller-Rabin (`sympy.isprime`).")

    add("algo",
        "Implement the Sieve of Eratosthenes for primes up to n.",
        "```python\ndef sieve(n: int) -> list[int]:\n    is_p = [True] * (n + 1)\n    is_p[0:2] = [False, False]\n    for i in range(2, int(n ** 0.5) + 1):\n        if is_p[i]:\n            for j in range(i * i, n + 1, i):\n                is_p[j] = False\n    return [i for i, p in enumerate(is_p) if p]\n```\nO(n log log n). Start the inner loop at `i*i` because smaller multiples were already crossed off.")

    add("algo",
        "Write a function that reverses a string without using slicing.",
        "```python\ndef reverse(s: str) -> str:\n    chars = list(s)\n    i, j = 0, len(chars) - 1\n    while i < j:\n        chars[i], chars[j] = chars[j], chars[i]\n        i += 1\n        j -= 1\n    return ''.join(chars)\n```\nIn real code just do `s[::-1]` -- it's faster and obvious. This form is for interview/teaching contexts.")

    add("algo",
        "Implement a function that returns the first non-repeating character in a string.",
        "```python\nfrom collections import Counter\n\ndef first_unique(s: str) -> str | None:\n    counts = Counter(s)\n    for c in s:\n        if counts[c] == 1:\n            return c\n    return None\n```\nTwo passes, O(n). The single-pass alternative needs an OrderedDict trick and is harder to read.")

    add("algo",
        "Write a function that checks if a string is a palindrome (ignoring case and spaces).",
        "```python\ndef is_palindrome(s: str) -> bool:\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]\n```\nNormalize first, compare second. The two-pointer variant saves memory but isn't faster in practice.")

    add("algo",
        "Implement a function that counts the number of words in a string.",
        "```python\ndef word_count(s: str) -> int:\n    return len(s.split())\n```\n`str.split()` with no args splits on any whitespace and discards empty strings. Use a regex (`re.findall(r'\\w+', s)`) if you need to exclude punctuation.")

    add("algo",
        "Write a function that returns the longest common prefix of a list of strings.",
        "```python\ndef common_prefix(strs: list[str]) -> str:\n    if not strs:\n        return ''\n    out = []\n    for chars in zip(*strs):\n        if len(set(chars)) == 1:\n            out.append(chars[0])\n        else:\n            break\n    return ''.join(out)\n```\n`zip(*strs)` walks the strings column-by-column; stops at the shortest, which is exactly what we want.")

    add("algo",
        "Implement a function that returns all permutations of a list.",
        "```python\nfrom itertools import permutations\n\ndef perms(xs: list) -> list[tuple]:\n    return list(permutations(xs))\n```\n`itertools.permutations` is C-optimized; rolling your own is slower and more error-prone. There are n! permutations -- callers should be careful with large n.")

    add("algo",
        "Write a function that returns all subsets (powerset) of a list.",
        "```python\nfrom itertools import chain, combinations\n\ndef powerset(xs: list) -> list[tuple]:\n    return list(chain.from_iterable(combinations(xs, r) for r in range(len(xs) + 1)))\n```\nStraight from the `itertools` recipes. 2^n subsets, so feasible only for small n.")

    add("algo",
        "Implement a stack class with push, pop, and peek.",
        "```python\nclass Stack:\n    def __init__(self) -> None:\n        self._items: list = []\n\n    def push(self, x) -> None:\n        self._items.append(x)\n\n    def pop(self):\n        if not self._items:\n            raise IndexError('pop from empty stack')\n        return self._items.pop()\n\n    def peek(self):\n        if not self._items:\n            raise IndexError('peek from empty stack')\n        return self._items[-1]\n\n    def __len__(self) -> int:\n        return len(self._items)\n```\nA list is already a perfect stack -- this wrapper buys you a clearer name and explicit empty-check errors.")

    add("algo",
        "Write a queue class using collections.deque.",
        "```python\nfrom collections import deque\n\nclass Queue:\n    def __init__(self) -> None:\n        self._dq: deque = deque()\n\n    def enqueue(self, x) -> None:\n        self._dq.append(x)\n\n    def dequeue(self):\n        if not self._dq:\n            raise IndexError('dequeue from empty queue')\n        return self._dq.popleft()\n\n    def __len__(self) -> int:\n        return len(self._dq)\n```\n`deque.popleft` is O(1); `list.pop(0)` is O(n). Always reach for `deque` when FIFO.")

    add("algo",
        "Implement a linked list with append and __iter__.",
        "```python\nfrom dataclasses import dataclass\nfrom typing import Optional, Iterator\n\n@dataclass\nclass Node:\n    val: int\n    next: Optional['Node'] = None\n\nclass LinkedList:\n    def __init__(self) -> None:\n        self.head: Optional[Node] = None\n\n    def append(self, v: int) -> None:\n        node = Node(v)\n        if not self.head:\n            self.head = node\n            return\n        cur = self.head\n        while cur.next:\n            cur = cur.next\n        cur.next = node\n\n    def __iter__(self) -> Iterator[int]:\n        cur = self.head\n        while cur:\n            yield cur.val\n            cur = cur.next\n```\nFor production use a list -- linked lists are pedagogical in Python.")

    add("algo",
        "Write a function that reverses a linked list in place.",
        "```python\ndef reverse_list(head):\n    prev = None\n    cur = head\n    while cur:\n        cur.next, prev, cur = prev, cur, cur.next\n    return prev\n```\nThe triple-assignment trick is the cleanest in-place reversal. Returns the new head.")

    add("algo",
        "Implement a function that detects a cycle in a linked list (Floyd's tortoise and hare).",
        "```python\ndef has_cycle(head) -> bool:\n    slow = fast = head\n    while fast and fast.next:\n        slow = slow.next\n        fast = fast.next.next\n        if slow is fast:\n            return True\n    return False\n```\nO(n) time, O(1) space. The hare moves twice as fast; if they ever meet, there's a cycle.")

    add("algo",
        "Write a function that performs a depth-first traversal of a graph (adjacency dict).",
        "```python\ndef dfs(graph: dict, start) -> list:\n    seen, order, stack = set(), [], [start]\n    while stack:\n        node = stack.pop()\n        if node in seen:\n            continue\n        seen.add(node)\n        order.append(node)\n        stack.extend(reversed(graph.get(node, [])))\n    return order\n```\nIterative DFS with an explicit stack avoids Python's recursion-depth limit. `reversed` keeps neighbour order intuitive.")

    add("algo",
        "Implement a breadth-first traversal of a graph.",
        "```python\nfrom collections import deque\n\ndef bfs(graph: dict, start) -> list:\n    seen, order, q = {start}, [], deque([start])\n    while q:\n        node = q.popleft()\n        order.append(node)\n        for n in graph.get(node, []):\n            if n not in seen:\n                seen.add(n)\n                q.append(n)\n    return order\n```\nAdd to `seen` when enqueueing, not dequeueing -- otherwise you can enqueue the same node many times.")

    add("algo",
        "Write Dijkstra's shortest-path algorithm.",
        "```python\nimport heapq\n\ndef dijkstra(graph: dict, start) -> dict:\n    dist = {start: 0}\n    pq = [(0, start)]\n    while pq:\n        d, u = heapq.heappop(pq)\n        if d > dist[u]:\n            continue\n        for v, w in graph.get(u, []):\n            nd = d + w\n            if nd < dist.get(v, float('inf')):\n                dist[v] = nd\n                heapq.heappush(pq, (nd, v))\n    return dist\n```\nO((V+E) log V). Skip stale entries with the `d > dist[u]` check; that's cheaper than `decrease-key`.")

    add("algo",
        "Implement quicksort.",
        "```python\ndef quicksort(xs: list) -> list:\n    if len(xs) <= 1:\n        return xs\n    pivot = xs[len(xs) // 2]\n    lo = [x for x in xs if x < pivot]\n    eq = [x for x in xs if x == pivot]\n    hi = [x for x in xs if x > pivot]\n    return quicksort(lo) + eq + quicksort(hi)\n```\nClassic three-way partition; O(n log n) average. Python's built-in `sorted` (Timsort) is faster -- use it in real code.")

    add("algo",
        "Write a merge function for two sorted lists.",
        "```python\ndef merge(a: list, b: list) -> list:\n    out, i, j = [], 0, 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    out.extend(a[i:])\n    out.extend(b[j:])\n    return out\n```\n`heapq.merge` does the same thing lazily and accepts any number of inputs.")

    add("algo",
        "Implement insertion sort.",
        "```python\ndef insertion_sort(xs: list) -> list:\n    a = list(xs)\n    for i in range(1, len(a)):\n        key, j = a[i], i - 1\n        while j >= 0 and a[j] > key:\n            a[j + 1] = a[j]\n            j -= 1\n        a[j + 1] = key\n    return a\n```\nO(n^2) worst case but very fast for small or nearly-sorted inputs. Timsort uses it as a sub-routine.")

    add("algo",
        "Write a function that returns the k-th largest element using a heap.",
        "```python\nimport heapq\n\ndef kth_largest(xs: list[int], k: int) -> int:\n    if k <= 0 or k > len(xs):\n        raise ValueError('k out of range')\n    return heapq.nlargest(k, xs)[-1]\n```\nO(n log k). Quickselect is O(n) average but `heapq.nlargest` is the practical default.")

    add("algo",
        "Implement a function that returns the nth Catalan number.",
        "```python\nfrom math import comb\n\ndef catalan(n: int) -> int:\n    return comb(2 * n, n) // (n + 1)\n```\nClosed form via binomial coefficient. The recursive definition exists but is exponential without memoization.")

    add("algo",
        "Write a function that computes n choose k.",
        "```python\nfrom math import comb\n\ndef choose(n: int, k: int) -> int:\n    return comb(n, k)\n```\n`math.comb` (3.8+) is exact for arbitrarily large integers. Don't roll your own with factorials -- it overflows/floats badly.")

    add("algo",
        "Implement a function that computes Pascal's triangle up to n rows.",
        "```python\ndef pascal(n: int) -> list[list[int]]:\n    rows: list[list[int]] = []\n    for i in range(n):\n        row = [1] * (i + 1)\n        for j in range(1, i):\n            row[j] = rows[i - 1][j - 1] + rows[i - 1][j]\n        rows.append(row)\n    return rows\n```\nO(n^2). Each row is built from the previous; the math.comb form is also fine for small n.")

    add("algo",
        "Write a function that returns the longest increasing subsequence length.",
        "```python\nfrom bisect import bisect_left\n\ndef lis(xs: list[int]) -> int:\n    tails: list[int] = []\n    for x in xs:\n        i = bisect_left(tails, x)\n        if i == len(tails):\n            tails.append(x)\n        else:\n            tails[i] = x\n    return len(tails)\n```\nO(n log n). `tails[i]` is the smallest possible tail of an increasing subseq of length i+1.")

    add("algo",
        "Implement Levenshtein edit distance.",
        "```python\ndef edit_distance(a: str, b: str) -> int:\n    prev = list(range(len(b) + 1))\n    for i, ca in enumerate(a, 1):\n        cur = [i] + [0] * len(b)\n        for j, cb in enumerate(b, 1):\n            cur[j] = prev[j - 1] if ca == cb else 1 + min(prev[j - 1], prev[j], cur[j - 1])\n        prev = cur\n    return prev[-1]\n```\nO(n*m) time, O(min(n,m)) space with the rolling-row trick.")

    add("algo",
        "Write a function that compresses a string by run-length encoding.",
        "```python\nfrom itertools import groupby\n\ndef rle(s: str) -> str:\n    return ''.join(f'{c}{sum(1 for _ in g)}' for c, g in groupby(s))\n```\n`groupby` collapses runs; we just count each. For binary protocols use `struct` instead.")

    add("algo",
        "Implement a trie with insert and search.",
        "```python\nclass Trie:\n    def __init__(self) -> None:\n        self.children: dict[str, 'Trie'] = {}\n        self.end: bool = False\n\n    def insert(self, word: str) -> None:\n        node = self\n        for c in word:\n            node = node.children.setdefault(c, Trie())\n        node.end = True\n\n    def search(self, word: str) -> bool:\n        node = self\n        for c in word:\n            if c not in node.children:\n                return False\n            node = node.children[c]\n        return node.end\n```\nGreat for prefix queries and autocomplete; for plain set membership a `set` is faster.")

    add("algo",
        "Write a function that finds all anagrams of a pattern in a string.",
        "```python\nfrom collections import Counter\n\ndef find_anagrams(s: str, p: str) -> list[int]:\n    if len(p) > len(s):\n        return []\n    need = Counter(p)\n    have = Counter(s[:len(p)])\n    out = [0] if have == need else []\n    for i in range(len(p), len(s)):\n        have[s[i]] += 1\n        have[s[i - len(p)]] -= 1\n        if have[s[i - len(p)]] == 0:\n            del have[s[i - len(p)]]\n        if have == need:\n            out.append(i - len(p) + 1)\n    return out\n```\nSliding-window Counter; O(n) time.")

    add("algo",
        "Implement a function that returns the majority element (Boyer-Moore vote).",
        "```python\ndef majority(xs: list[int]) -> int | None:\n    cand, cnt = None, 0\n    for x in xs:\n        if cnt == 0:\n            cand = x\n        cnt += 1 if x == cand else -1\n    return cand if xs.count(cand) > len(xs) // 2 else None\n```\nO(n) time, O(1) space. The verification pass is required if a majority isn't guaranteed.")

    add("algo",
        "Write a function that returns the intersection of two lists preserving order.",
        "```python\ndef intersect(a: list, b: list) -> list:\n    bset = set(b)\n    return [x for x in a if x in bset]\n```\n`set` membership is O(1); doing `x in b` with a list is O(n) per element.")

    add("algo",
        "Implement a function that splits a list into chunks of size n.",
        "```python\ndef chunks(xs: list, n: int) -> list[list]:\n    if n <= 0:\n        raise ValueError('n must be positive')\n    return [xs[i:i + n] for i in range(0, len(xs), n)]\n```\nFor lazy iteration, use `itertools.batched(xs, n)` (Python 3.12+).")

    add("algo",
        "Write a function that interleaves two lists.",
        "```python\nfrom itertools import zip_longest\n\ndef interleave(a: list, b: list, fill=None) -> list:\n    out = []\n    for x, y in zip_longest(a, b, fillvalue=fill):\n        if x is not fill or fill is None:\n            out.append(x)\n        if y is not fill or fill is None:\n            out.append(y)\n    return out\n```\nUse `zip_longest` so the longer input isn't truncated.")

    add("algo",
        "Implement a function that returns the symmetric difference of two sets as a sorted list.",
        "```python\ndef sym_diff(a: set, b: set) -> list:\n    return sorted(a ^ b)\n```\nThe `^` operator on sets is symmetric difference; sort for deterministic output.")

    add("algo",
        "Write a function that converts an integer to its binary string (without bin()).",
        "```python\ndef to_binary(n: int) -> str:\n    if n == 0:\n        return '0'\n    sign = '-' if n < 0 else ''\n    n = abs(n)\n    bits = []\n    while n:\n        bits.append(str(n & 1))\n        n >>= 1\n    return sign + ''.join(reversed(bits))\n```\nIn real code use `bin(n)` or `format(n, 'b')`.")

    add("algo",
        "Implement a function that counts set bits in an integer.",
        "```python\ndef popcount(n: int) -> int:\n    return bin(n).count('1') if n >= 0 else bin(n & 0xFFFFFFFF).count('1')\n```\nPython 3.10+ has `int.bit_count()` which is faster. The mask handles negative ints under two's-complement assumptions.")

    add("algo",
        "Write a function that returns the median of a list.",
        "```python\ndef median(xs: list[float]) -> float:\n    if not xs:\n        raise ValueError('empty input')\n    s = sorted(xs)\n    n = len(s)\n    mid = n // 2\n    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2\n```\nThe stdlib `statistics.median` does this and handles edge cases; prefer it.")

    add("algo",
        "Implement a function that returns the mode of a list.",
        "```python\nfrom collections import Counter\n\ndef mode(xs: list) -> list:\n    if not xs:\n        return []\n    c = Counter(xs)\n    top = max(c.values())\n    return [k for k, v in c.items() if v == top]\n```\nReturns all modes when there's a tie; the stdlib `statistics.mode` raises on ties (`multimode` is the multi-value version).")

    add("algo",
        "Write a function that compares two version strings (e.g. '1.10.2' vs '1.9.5').",
        "```python\ndef cmp_version(a: str, b: str) -> int:\n    pa = [int(x) for x in a.split('.')]\n    pb = [int(x) for x in b.split('.')]\n    if pa < pb:\n        return -1\n    if pa > pb:\n        return 1\n    return 0\n```\nFor real semver use `packaging.version.Version` -- it handles pre-release tags correctly.")

    add("algo",
        "Implement a memoization decorator using functools.lru_cache.",
        "```python\nfrom functools import lru_cache\n\n@lru_cache(maxsize=None)\ndef fib(n: int) -> int:\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n```\n`lru_cache(maxsize=None)` is equivalent to `cache` (3.9+). Cached args must be hashable.")

    add("algo",
        "Write a function that returns prime factorization of an integer.",
        "```python\ndef factorize(n: int) -> dict[int, int]:\n    if n < 2:\n        return {}\n    out: dict[int, int] = {}\n    d = 2\n    while d * d <= n:\n        while n % d == 0:\n            out[d] = out.get(d, 0) + 1\n            n //= d\n        d += 1\n    if n > 1:\n        out[n] = out.get(n, 0) + 1\n    return out\n```\nO(sqrt(n)). For very large numbers use Pollard's rho; for typical inputs this is fine.")

    add("algo",
        "Implement a function that reverses words in a sentence.",
        "```python\ndef reverse_words(s: str) -> str:\n    return ' '.join(s.split()[::-1])\n```\n`split()` collapses whitespace runs; if you need to preserve exact spacing, use `re.split` with capturing groups.")

    add("algo",
        "Write a function that returns whether two binary trees are identical.",
        "```python\ndef same_tree(a, b) -> bool:\n    if a is None and b is None:\n        return True\n    if a is None or b is None:\n        return False\n    return a.val == b.val and same_tree(a.left, b.left) and same_tree(a.right, b.right)\n```\nRecursion mirrors the structure; depth bounded by tree height.")

    add("algo",
        "Implement an in-order traversal of a binary tree.",
        "```python\ndef inorder(root) -> list:\n    out, stack, cur = [], [], root\n    while cur or stack:\n        while cur:\n            stack.append(cur)\n            cur = cur.left\n        cur = stack.pop()\n        out.append(cur.val)\n        cur = cur.right\n    return out\n```\nIterative form avoids recursion-depth issues on tall trees.")

    add("algo",
        "Write a function that returns the maximum depth of a binary tree.",
        "```python\ndef max_depth(root) -> int:\n    if root is None:\n        return 0\n    return 1 + max(max_depth(root.left), max_depth(root.right))\n```\nFor very deep trees use BFS with `collections.deque` and a level counter to avoid recursion limits.")

    add("algo",
        "Implement a function that determines if a binary tree is balanced.",
        "```python\ndef is_balanced(root) -> bool:\n    def height(node) -> int:\n        if not node:\n            return 0\n        lh = height(node.left)\n        if lh == -1:\n            return -1\n        rh = height(node.right)\n        if rh == -1 or abs(lh - rh) > 1:\n            return -1\n        return 1 + max(lh, rh)\n    return height(root) != -1\n```\nReturning -1 as a sentinel collapses the height computation and balance check into one pass.")

    add("algo",
        "Write a function that computes the sum of digits of an integer.",
        "```python\ndef digit_sum(n: int) -> int:\n    return sum(int(c) for c in str(abs(n)))\n```\nStringify-then-sum is the most readable form; the arithmetic version (`while n: total += n % 10; n //= 10`) is a hair faster.")

    add("algo",
        "Implement a function that returns whether a number is a power of two.",
        "```python\ndef is_pow2(n: int) -> bool:\n    return n > 0 and (n & (n - 1)) == 0\n```\nClassic bit trick: powers of two have exactly one bit set, so `n & (n-1)` clears it to zero.")

    add("algo",
        "Write a function that converts a snake_case string to camelCase.",
        "```python\ndef snake_to_camel(s: str) -> str:\n    parts = s.split('_')\n    return parts[0] + ''.join(p.title() for p in parts[1:])\n```\nUse `title()` rather than manual indexing so `_url` -> `Url` works correctly.")

    add("algo",
        "Implement camelCase to snake_case conversion.",
        "```python\nimport re\n\ndef camel_to_snake(s: str) -> str:\n    s = re.sub(r'(.)([A-Z][a-z]+)', r'\\1_\\2', s)\n    return re.sub(r'([a-z0-9])([A-Z])', r'\\1_\\2', s).lower()\n```\nTwo passes handle acronyms like `HTTPServer` -> `http_server` correctly.")

    add("algo",
        "Write a function that returns the longest substring without repeating characters.",
        "```python\ndef longest_unique(s: str) -> int:\n    last: dict = {}\n    start = best = 0\n    for i, c in enumerate(s):\n        if c in last and last[c] >= start:\n            start = last[c] + 1\n        last[c] = i\n        best = max(best, i - start + 1)\n    return best\n```\nSliding window with last-seen index; O(n).")

    add("algo",
        "Implement a function that rotates a 2D matrix 90 degrees clockwise.",
        "```python\ndef rotate90(m: list[list[int]]) -> list[list[int]]:\n    return [list(row) for row in zip(*m[::-1])]\n```\nReverse rows, then transpose. The `zip` trick is the cleanest transpose in Python.")

    add("algo",
        "Write a function that returns transposes a matrix.",
        "```python\ndef transpose(m: list[list]) -> list[list]:\n    return [list(row) for row in zip(*m)]\n```\nIf you're doing real linear algebra reach for numpy and `arr.T` -- it's a view, not a copy.")

    add("algo",
        "Implement a function that returns the spiral order of a matrix.",
        "```python\ndef spiral(m: list[list[int]]) -> list[int]:\n    out: list[int] = []\n    while m:\n        out.extend(m.pop(0))\n        m = [list(row) for row in zip(*m)][::-1]\n    return out\n```\nPeel the top row, then rotate the remainder 90 degrees counter-clockwise; repeat until empty.")

    add("algo",
        "Write a function that finds the missing number in 0..n.",
        "```python\ndef missing(xs: list[int]) -> int:\n    n = len(xs)\n    return n * (n + 1) // 2 - sum(xs)\n```\nO(n), O(1). Gauss's sum formula avoids a hash set.")

    add("algo",
        "Implement a function that finds the duplicate number in a list of n+1 ints (each 1..n).",
        "```python\ndef find_dup(xs: list[int]) -> int:\n    slow = fast = xs[0]\n    while True:\n        slow = xs[slow]\n        fast = xs[xs[fast]]\n        if slow == fast:\n            break\n    slow = xs[0]\n    while slow != fast:\n        slow = xs[slow]\n        fast = xs[fast]\n    return slow\n```\nFloyd cycle detection treating the array as a function; O(n) time, O(1) space.")

    add("algo",
        "Write a function that returns whether a list is sorted ascending.",
        "```python\ndef is_sorted(xs: list) -> bool:\n    return all(a <= b for a, b in zip(xs, xs[1:]))\n```\n`zip(xs, xs[1:])` yields adjacent pairs; `all` short-circuits on the first violation.")

    add("algo",
        "Implement a function that returns the maximum profit from a single stock buy/sell.",
        "```python\ndef max_profit(prices: list[int]) -> int:\n    best = 0\n    lo = float('inf')\n    for p in prices:\n        lo = min(lo, p)\n        best = max(best, p - lo)\n    return best\n```\nOne pass; track the lowest price seen so far.")

    add("algo",
        "Write a function that returns the n-th row of Pascal's triangle.",
        "```python\nfrom math import comb\n\ndef pascal_row(n: int) -> list[int]:\n    return [comb(n, k) for k in range(n + 1)]\n```\n`math.comb` is exact and fast; building the row from the previous row also works but is more code.")

    add("algo",
        "Implement a function that converts Roman numerals to integers.",
        "```python\ndef roman_to_int(s: str) -> int:\n    vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n    total = 0\n    prev = 0\n    for c in reversed(s):\n        v = vals[c]\n        total += -v if v < prev else v\n        prev = v\n    return total\n```\nWalking right-to-left makes the subtractive rule (IV, IX, XL...) trivial.")

    add("algo",
        "Write a function that returns the longest palindromic substring.",
        "```python\ndef longest_palindrome(s: str) -> str:\n    if not s:\n        return ''\n    start, end = 0, 0\n    def expand(l: int, r: int) -> tuple[int, int]:\n        while l >= 0 and r < len(s) and s[l] == s[r]:\n            l -= 1\n            r += 1\n        return l + 1, r\n    for i in range(len(s)):\n        for l, r in (expand(i, i), expand(i, i + 1)):\n            if r - l > end - start:\n                start, end = l, r\n    return s[start:end]\n```\nExpand-around-center is O(n^2) time, O(1) space; Manacher's is O(n) but rarely needed.")

    add("algo",
        "Implement a function that converts an integer to a Roman numeral.",
        "```python\ndef int_to_roman(n: int) -> str:\n    pairs = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'),\n             (90, 'XC'), (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'),\n             (5, 'V'), (4, 'IV'), (1, 'I')]\n    out = []\n    for v, sym in pairs:\n        while n >= v:\n            out.append(sym)\n            n -= v\n    return ''.join(out)\n```\nGreedy works because the table includes the subtractive forms.")

    add("algo",
        "Write a function that returns whether a string contains only unique characters.",
        "```python\ndef all_unique(s: str) -> bool:\n    return len(set(s)) == len(s)\n```\nO(n) and obvious. The bit-vector trick is faster only for ASCII and rarely worth the obscurity.")

    add("algo",
        "Implement a function that returns the first n primes.",
        "```python\ndef first_n_primes(n: int) -> list[int]:\n    primes: list[int] = []\n    cand = 2\n    while len(primes) < n:\n        if all(cand % p != 0 for p in primes if p * p <= cand):\n            primes.append(cand)\n        cand += 1\n    return primes\n```\nTrial division against known primes up to sqrt; faster than trying all integers.")

    add("algo",
        "Write a function that solves Tower of Hanoi and returns the move sequence.",
        "```python\ndef hanoi(n: int, src: str = 'A', dst: str = 'C', via: str = 'B') -> list[tuple[str, str]]:\n    if n == 0:\n        return []\n    return hanoi(n - 1, src, via, dst) + [(src, dst)] + hanoi(n - 1, via, dst, src)\n```\n2^n - 1 moves. Beautifully recursive; iterative versions exist but obscure the structure.")

    add("algo",
        "Implement a function that returns whether parentheses in an expression are properly nested.",
        "```python\ndef nested(s: str) -> bool:\n    depth = 0\n    for c in s:\n        if c == '(':\n            depth += 1\n        elif c == ')':\n            depth -= 1\n            if depth < 0:\n                return False\n    return depth == 0\n```\nSingle-bracket case doesn't need a stack; just a counter.")

    add("algo",
        "Write a function that converts a list of dicts to a dict of lists (column store).",
        "```python\ndef to_columnar(rows: list[dict]) -> dict[str, list]:\n    if not rows:\n        return {}\n    keys = rows[0].keys()\n    return {k: [r.get(k) for r in rows] for k in keys}\n```\nUseful for feeding a row-oriented JSON dump into pandas/polars.")

    add("algo",
        "Implement a function that finds peaks (local maxima) in a list.",
        "```python\ndef peaks(xs: list[float]) -> list[int]:\n    out = []\n    for i in range(1, len(xs) - 1):\n        if xs[i - 1] < xs[i] > xs[i + 1]:\n            out.append(i)\n    return out\n```\nFor noisy real signals use `scipy.signal.find_peaks` -- it has prominence and distance filters.")

    add("algo",
        "Write a function that decodes a string like '3[ab]2[c]' into 'abababcc'.",
        "```python\ndef decode(s: str) -> str:\n    stack: list = []\n    cur = ''\n    k = 0\n    for c in s:\n        if c.isdigit():\n            k = k * 10 + int(c)\n        elif c == '[':\n            stack.append((cur, k))\n            cur, k = '', 0\n        elif c == ']':\n            prev, mult = stack.pop()\n            cur = prev + cur * mult\n        else:\n            cur += c\n    return cur\n```\nStack of (prefix, multiplier) handles arbitrary nesting like `2[a3[b]]`.")

    add("algo",
        "Implement Boyer-Moore-style fast majority vote returning the candidate without verification.",
        "```python\ndef vote(xs: list[int]) -> int | None:\n    if not xs:\n        return None\n    cand, cnt = xs[0], 0\n    for x in xs:\n        if cnt == 0:\n            cand = x\n        cnt += 1 if x == cand else -1\n    return cand\n```\nReturns the strict majority if one exists; verify with a second pass if it isn't guaranteed.")

    add("algo",
        "Write a function that simulates a simple LRU cache via OrderedDict.",
        "```python\nfrom collections import OrderedDict\n\nclass LRU:\n    def __init__(self, cap: int) -> None:\n        self.cap = cap\n        self.d: OrderedDict = OrderedDict()\n\n    def get(self, k):\n        if k not in self.d:\n            return None\n        self.d.move_to_end(k)\n        return self.d[k]\n\n    def put(self, k, v) -> None:\n        if k in self.d:\n            self.d.move_to_end(k)\n        self.d[k] = v\n        if len(self.d) > self.cap:\n            self.d.popitem(last=False)\n```\n`functools.lru_cache` does this for function results; this class is for caching arbitrary key-value pairs.")
