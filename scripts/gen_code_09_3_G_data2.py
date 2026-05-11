# SPDX-License-Identifier: MIT
"""Algorithms / data structures entries part 2 (75). Wave 7 part 2."""


def register_all(add):
    add("algo",
        "Write a function that returns the union of two lists preserving first-seen order.",
        "```python\ndef union(a: list, b: list) -> list:\n    seen: set = set()\n    out = []\n    for x in a + b:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\n```\nLooks like dedup over a concatenation -- it is. For unhashable items use a list-based `if x not in seen`.")

    add("algo",
        "Implement a function that finds the longest common subsequence length.",
        "```python\ndef lcs(a: str, b: str) -> int:\n    if len(a) < len(b):\n        a, b = b, a\n    prev = [0] * (len(b) + 1)\n    for ca in a:\n        cur = [0] * (len(b) + 1)\n        for j, cb in enumerate(b, 1):\n            cur[j] = prev[j - 1] + 1 if ca == cb else max(prev[j], cur[j - 1])\n        prev = cur\n    return prev[-1]\n```\nO(n*m) time, O(min(n,m)) space.")

    add("algo",
        "Write a function that performs topological sort on a DAG.",
        "```python\nfrom collections import deque, defaultdict\n\ndef topo_sort(graph: dict) -> list:\n    indeg: dict = defaultdict(int)\n    for u, vs in graph.items():\n        indeg.setdefault(u, 0)\n        for v in vs:\n            indeg[v] += 1\n    q = deque(u for u, d in indeg.items() if d == 0)\n    out = []\n    while q:\n        u = q.popleft()\n        out.append(u)\n        for v in graph.get(u, []):\n            indeg[v] -= 1\n            if indeg[v] == 0:\n                q.append(v)\n    if len(out) != len(indeg):\n        raise ValueError('graph has a cycle')\n    return out\n```\nKahn's algorithm; O(V+E). Detects cycles automatically when output length differs.")

    add("algo",
        "Implement a function that finds connected components of an undirected graph.",
        "```python\ndef components(graph: dict) -> list[list]:\n    seen: set = set()\n    comps: list[list] = []\n    for start in graph:\n        if start in seen:\n            continue\n        stack = [start]\n        comp = []\n        while stack:\n            u = stack.pop()\n            if u in seen:\n                continue\n            seen.add(u)\n            comp.append(u)\n            stack.extend(graph.get(u, []))\n        comps.append(comp)\n    return comps\n```\nIterative DFS per unvisited start.")

    add("algo",
        "Write a function that implements a binary search tree insert.",
        "```python\nfrom dataclasses import dataclass, field\nfrom typing import Optional\n\n@dataclass\nclass BST:\n    val: int\n    left: Optional['BST'] = None\n    right: Optional['BST'] = None\n\n    def insert(self, v: int) -> None:\n        if v < self.val:\n            if self.left is None:\n                self.left = BST(v)\n            else:\n                self.left.insert(v)\n        else:\n            if self.right is None:\n                self.right = BST(v)\n            else:\n                self.right.insert(v)\n```\nFor production use `sortedcontainers.SortedList` -- it's a B-tree under the hood and far more cache-friendly.")

    add("algo",
        "Implement union-find (disjoint set union).",
        "```python\nclass DSU:\n    def __init__(self, n: int) -> None:\n        self.p = list(range(n))\n        self.r = [0] * n\n\n    def find(self, x: int) -> int:\n        while self.p[x] != x:\n            self.p[x] = self.p[self.p[x]]\n            x = self.p[x]\n        return x\n\n    def union(self, a: int, b: int) -> bool:\n        ra, rb = self.find(a), self.find(b)\n        if ra == rb:\n            return False\n        if self.r[ra] < self.r[rb]:\n            ra, rb = rb, ra\n        self.p[rb] = ra\n        if self.r[ra] == self.r[rb]:\n            self.r[ra] += 1\n        return True\n```\nPath compression + union by rank gives near-O(1) amortized per op.")

    add("algo",
        "Write a function that finds the k closest points to the origin.",
        "```python\nimport heapq\n\ndef k_closest(points: list[tuple[float, float]], k: int) -> list[tuple[float, float]]:\n    return heapq.nsmallest(k, points, key=lambda p: p[0] ** 2 + p[1] ** 2)\n```\nSquared distance avoids the sqrt; ordering is preserved.")

    add("algo",
        "Implement a function that merges k sorted lists.",
        "```python\nimport heapq\n\ndef merge_k(lists: list[list]) -> list:\n    return list(heapq.merge(*lists))\n```\n`heapq.merge` is lazy, so it scales to large inputs without buffering. O(N log k).")

    add("algo",
        "Write a function that partitions a list around a pivot.",
        "```python\ndef partition(xs: list[int], pivot: int) -> tuple[list[int], list[int], list[int]]:\n    lo = [x for x in xs if x < pivot]\n    eq = [x for x in xs if x == pivot]\n    hi = [x for x in xs if x > pivot]\n    return lo, eq, hi\n```\nThree-way partition; the foundation of three-way quicksort which handles many duplicates well.")

    add("algo",
        "Implement a function that counts inversions in a list.",
        "```python\ndef count_inversions(xs: list[int]) -> int:\n    def merge_count(a: list[int]) -> tuple[list[int], int]:\n        if len(a) <= 1:\n            return a, 0\n        m = len(a) // 2\n        left, lc = merge_count(a[:m])\n        right, rc = merge_count(a[m:])\n        merged, mc = [], 0\n        i = j = 0\n        while i < len(left) and j < len(right):\n            if left[i] <= right[j]:\n                merged.append(left[i]); i += 1\n            else:\n                merged.append(right[j]); j += 1\n                mc += len(left) - i\n        merged += left[i:] + right[j:]\n        return merged, lc + rc + mc\n    return merge_count(list(xs))[1]\n```\nMerge-sort variant; O(n log n).")

    add("algo",
        "Write a function that returns the smallest range covering at least one element from each list.",
        "```python\nimport heapq\n\ndef smallest_range(lists: list[list[int]]) -> tuple[int, int]:\n    pq = [(xs[0], i, 0) for i, xs in enumerate(lists)]\n    heapq.heapify(pq)\n    cur_max = max(xs[0] for xs in lists)\n    best = (pq[0][0], cur_max)\n    while True:\n        v, i, j = heapq.heappop(pq)\n        if cur_max - v < best[1] - best[0]:\n            best = (v, cur_max)\n        if j + 1 == len(lists[i]):\n            return best\n        nv = lists[i][j + 1]\n        cur_max = max(cur_max, nv)\n        heapq.heappush(pq, (nv, i, j + 1))\n```\nClassic k-way pointer + heap; advance the smallest element until one list is exhausted.")

    add("algo",
        "Implement a function that returns the median of a sliding window.",
        "```python\nimport heapq\n\ndef median_window(xs: list[float], k: int) -> list[float]:\n    if k <= 0 or k > len(xs):\n        return []\n    out: list[float] = []\n    for i in range(len(xs) - k + 1):\n        window = sorted(xs[i:i + k])\n        if k % 2:\n            out.append(window[k // 2])\n        else:\n            out.append((window[k // 2 - 1] + window[k // 2]) / 2)\n    return out\n```\nO(n*k log k); for true O(n log k) use a balanced BST or two heaps with lazy deletion.")

    add("algo",
        "Write a function that returns whether you can partition a list into two equal-sum subsets.",
        "```python\ndef can_partition(xs: list[int]) -> bool:\n    s = sum(xs)\n    if s % 2:\n        return False\n    target = s // 2\n    dp = {0}\n    for x in xs:\n        dp |= {v + x for v in dp if v + x <= target}\n    return target in dp\n```\nSubset-sum via a set of reachable totals; O(n*target).")

    add("algo",
        "Implement a coin-change min-coins function.",
        "```python\ndef min_coins(coins: list[int], amount: int) -> int:\n    INF = amount + 1\n    dp = [0] + [INF] * amount\n    for i in range(1, amount + 1):\n        dp[i] = min((dp[i - c] + 1 for c in coins if i - c >= 0), default=INF)\n    return -1 if dp[amount] >= INF else dp[amount]\n```\nClassic 1-D DP; O(amount * len(coins)).")

    add("algo",
        "Write a function that returns the number of ways to climb n stairs taking 1 or 2 steps.",
        "```python\ndef climb(n: int) -> int:\n    a, b = 1, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```\nFibonacci in disguise; O(n) time, O(1) space.")

    add("algo",
        "Implement the maximum rectangle area in a histogram.",
        "```python\ndef max_rect(heights: list[int]) -> int:\n    stack: list[int] = []\n    best = 0\n    for i, h in enumerate(heights + [0]):\n        while stack and heights[stack[-1]] >= h:\n            top = stack.pop()\n            left = stack[-1] if stack else -1\n            best = max(best, heights[top] * (i - left - 1))\n        stack.append(i)\n    return best\n```\nMonotonic stack; the appended sentinel `0` flushes remaining bars.")

    add("algo",
        "Write a function that returns whether s2 contains a permutation of s1.",
        "```python\nfrom collections import Counter\n\ndef contains_perm(s1: str, s2: str) -> bool:\n    if len(s1) > len(s2):\n        return False\n    need = Counter(s1)\n    have = Counter(s2[:len(s1)])\n    if have == need:\n        return True\n    for i in range(len(s1), len(s2)):\n        have[s2[i]] += 1\n        have[s2[i - len(s1)]] -= 1\n        if have[s2[i - len(s1)]] == 0:\n            del have[s2[i - len(s1)]]\n        if have == need:\n            return True\n    return False\n```\nSliding window of fixed size; O(n).")

    add("algo",
        "Implement a function that returns the longest valid parentheses substring length.",
        "```python\ndef longest_valid(s: str) -> int:\n    stack = [-1]\n    best = 0\n    for i, c in enumerate(s):\n        if c == '(':\n            stack.append(i)\n        else:\n            stack.pop()\n            if not stack:\n                stack.append(i)\n            else:\n                best = max(best, i - stack[-1])\n    return best\n```\nStack of indices; the bottom marks the start of the current valid run.")

    add("algo",
        "Write a function that detects whether two rectangles overlap.",
        "```python\ndef overlap(r1: tuple, r2: tuple) -> bool:\n    x1, y1, x2, y2 = r1\n    a1, b1, a2, b2 = r2\n    return not (x2 <= a1 or a2 <= x1 or y2 <= b1 or b2 <= y1)\n```\nNegate the four 'completely-to-one-side' cases. Edges-touching counts as no overlap.")

    add("algo",
        "Implement a function that returns whether a sudoku board is valid.",
        "```python\ndef valid_sudoku(board: list[list[str]]) -> bool:\n    seen: set = set()\n    for i in range(9):\n        for j in range(9):\n            v = board[i][j]\n            if v == '.':\n                continue\n            keys = (f'r{i}{v}', f'c{j}{v}', f'b{i // 3}{j // 3}{v}')\n            if any(k in seen for k in keys):\n                return False\n            seen.update(keys)\n    return True\n```\nOne pass; encode (row, value), (col, value), (box, value) as strings in a single set.")

    add("algo",
        "Write a function that finds the longest word in a list that can be built one letter at a time from other words.",
        "```python\ndef longest_buildable(words: list[str]) -> str:\n    words.sort(key=lambda w: (-len(w), w))\n    word_set = {''}\n    best = ''\n    for w in sorted(words):\n        if w[:-1] in word_set:\n            word_set.add(w)\n            if len(w) > len(best) or (len(w) == len(best) and w < best):\n                best = w\n    return best\n```\nSort lexicographically so prefix-availability is built up correctly.")

    add("algo",
        "Implement a function that returns the smallest substring of s containing all characters of t.",
        "```python\nfrom collections import Counter\n\ndef min_window(s: str, t: str) -> str:\n    if not t or not s:\n        return ''\n    need = Counter(t)\n    missing = len(t)\n    i = start = end = 0\n    for j, c in enumerate(s, 1):\n        if need[c] > 0:\n            missing -= 1\n        need[c] -= 1\n        if missing == 0:\n            while i < j and need[s[i]] < 0:\n                need[s[i]] += 1\n                i += 1\n            if not end or j - i < end - start:\n                start, end = i, j\n            need[s[i]] += 1\n            missing += 1\n            i += 1\n    return s[start:end]\n```\nClassic two-pointer minimum window; O(n).")

    add("algo",
        "Write a function that simulates rolling a fair n-sided die k times.",
        "```python\nimport random\n\ndef roll(n: int, k: int, seed: int | None = None) -> list[int]:\n    rng = random.Random(seed)\n    return [rng.randint(1, n) for _ in range(k)]\n```\nA local `Random` instance keeps results reproducible without leaking into the global PRNG.")

    add("algo",
        "Implement reservoir sampling for k items from a stream.",
        "```python\nimport random\nfrom typing import Iterable, TypeVar\n\nT = TypeVar('T')\n\ndef reservoir(stream: Iterable[T], k: int) -> list[T]:\n    rng = random.Random()\n    out: list[T] = []\n    for i, x in enumerate(stream):\n        if i < k:\n            out.append(x)\n        else:\n            j = rng.randint(0, i)\n            if j < k:\n                out[j] = x\n    return out\n```\nUniform sample of size k without knowing the stream length up front.")

    add("algo",
        "Write a function that computes the convex hull of 2D points (Andrew's monotone chain).",
        "```python\ndef convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:\n    pts = sorted(set(points))\n    if len(pts) <= 1:\n        return pts\n    def cross(o, a, b):\n        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])\n    lower: list = []\n    for p in pts:\n        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:\n            lower.pop()\n        lower.append(p)\n    upper: list = []\n    for p in reversed(pts):\n        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:\n            upper.pop()\n        upper.append(p)\n    return lower[:-1] + upper[:-1]\n```\nO(n log n). For real CG work use `scipy.spatial.ConvexHull`.")

    add("algo",
        "Implement a function that returns whether a string can be segmented into a dictionary of words.",
        "```python\ndef word_break(s: str, words: list[str]) -> bool:\n    word_set = set(words)\n    dp = [True] + [False] * len(s)\n    for i in range(1, len(s) + 1):\n        for j in range(i):\n            if dp[j] and s[j:i] in word_set:\n                dp[i] = True\n                break\n    return dp[-1]\n```\nO(n^2) DP; cap the inner loop by max word length for a constant-factor win.")

    add("algo",
        "Write a function that decodes a string of digits into all possible letter mappings (1->A, 26->Z).",
        "```python\ndef num_decodings(s: str) -> int:\n    if not s or s[0] == '0':\n        return 0\n    n = len(s)\n    dp = [0] * (n + 1)\n    dp[0] = dp[1] = 1\n    for i in range(2, n + 1):\n        if s[i - 1] != '0':\n            dp[i] += dp[i - 1]\n        if 10 <= int(s[i - 2:i]) <= 26:\n            dp[i] += dp[i - 2]\n    return dp[n]\n```\nFibonacci-shaped DP with leading-zero guards.")

    add("algo",
        "Implement a function that counts the number of distinct islands in a grid.",
        "```python\ndef num_islands(grid: list[list[str]]) -> int:\n    rows, cols = len(grid), len(grid[0])\n    visited: set = set()\n    count = 0\n    def dfs(r: int, c: int) -> None:\n        if (r, c) in visited or not (0 <= r < rows and 0 <= c < cols) or grid[r][c] != '1':\n            return\n        visited.add((r, c))\n        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):\n            dfs(r + dr, c + dc)\n    for r in range(rows):\n        for c in range(cols):\n            if grid[r][c] == '1' and (r, c) not in visited:\n                dfs(r, c)\n                count += 1\n    return count\n```\nFlood-fill via DFS; for very tall grids prefer the iterative-stack form to avoid recursion limits.")

    add("algo",
        "Write a function that returns the maximum sum path in a triangle.",
        "```python\ndef max_path_sum(triangle: list[list[int]]) -> int:\n    if not triangle:\n        return 0\n    dp = list(triangle[-1])\n    for row in reversed(triangle[:-1]):\n        for i, v in enumerate(row):\n            dp[i] = v + max(dp[i], dp[i + 1])\n    return dp[0]\n```\nBottom-up DP collapses to O(n) extra space.")

    add("algo",
        "Implement a function that returns whether any subset sums to a target.",
        "```python\ndef subset_sum(xs: list[int], target: int) -> bool:\n    reachable = {0}\n    for x in xs:\n        reachable |= {r + x for r in reachable}\n        if target in reachable:\n            return True\n    return target in reachable\n```\nEarly exit when target is reached. Worst-case O(2^n) but with deduplication usually much faster.")

    add("algo",
        "Write a function that returns the longest zigzag subsequence length.",
        "```python\ndef zigzag(xs: list[int]) -> int:\n    if not xs:\n        return 0\n    up = down = 1\n    for i in range(1, len(xs)):\n        if xs[i] > xs[i - 1]:\n            up = down + 1\n        elif xs[i] < xs[i - 1]:\n            down = up + 1\n    return max(up, down)\n```\nO(n) via two interleaved counters.")

    add("algo",
        "Implement a function that finds the maximum product of a contiguous subarray.",
        "```python\ndef max_product(xs: list[int]) -> int:\n    if not xs:\n        raise ValueError('empty')\n    best = lo = hi = xs[0]\n    for x in xs[1:]:\n        if x < 0:\n            lo, hi = hi, lo\n        lo = min(x, lo * x)\n        hi = max(x, hi * x)\n        best = max(best, hi)\n    return best\n```\nTrack both the min and max because a negative times a negative can become the new max.")

    add("algo",
        "Write a function that returns the minimum path sum in a grid.",
        "```python\ndef min_path_sum(grid: list[list[int]]) -> int:\n    rows, cols = len(grid), len(grid[0])\n    dp = list(grid[0])\n    for c in range(1, cols):\n        dp[c] += dp[c - 1]\n    for r in range(1, rows):\n        dp[0] += grid[r][0]\n        for c in range(1, cols):\n            dp[c] = grid[r][c] + min(dp[c], dp[c - 1])\n    return dp[-1]\n```\nIn-place DP across rows; O(cols) extra space.")

    add("algo",
        "Implement a function that finds all unique triplets summing to zero.",
        "```python\ndef three_sum(xs: list[int]) -> list[list[int]]:\n    xs = sorted(xs)\n    out: list[list[int]] = []\n    for i, a in enumerate(xs):\n        if i > 0 and a == xs[i - 1]:\n            continue\n        l, r = i + 1, len(xs) - 1\n        while l < r:\n            s = a + xs[l] + xs[r]\n            if s == 0:\n                out.append([a, xs[l], xs[r]])\n                while l < r and xs[l] == xs[l + 1]:\n                    l += 1\n                while l < r and xs[r] == xs[r - 1]:\n                    r -= 1\n                l += 1\n                r -= 1\n            elif s < 0:\n                l += 1\n            else:\n                r -= 1\n    return out\n```\nSort + two-pointer; the dedup steps avoid emitting the same triplet twice.")

    add("algo",
        "Write a function that returns whether a binary tree is symmetric.",
        "```python\ndef is_symmetric(root) -> bool:\n    def mirror(a, b) -> bool:\n        if a is None and b is None:\n            return True\n        if a is None or b is None:\n            return False\n        return a.val == b.val and mirror(a.left, b.right) and mirror(a.right, b.left)\n    return mirror(root, root) if root else True\n```\nRecurse on the two halves with mirrored child order.")

    add("algo",
        "Implement a function that flattens a nested list arbitrary depth.",
        "```python\nfrom collections.abc import Iterable\n\ndef flatten(xs):\n    for x in xs:\n        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):\n            yield from flatten(x)\n        else:\n            yield x\n```\nGenerator-based; the str/bytes exclusion avoids infinite recursion since every char is itself iterable.")

    add("algo",
        "Write a function that returns the most frequent words in text.",
        "```python\nimport re\nfrom collections import Counter\n\ndef top_words(text: str, n: int = 10) -> list[tuple[str, int]]:\n    words = re.findall(r\"\\b\\w+\\b\", text.lower())\n    return Counter(words).most_common(n)\n```\nRegex normalizes word boundaries so punctuation doesn't pollute keys.")

    add("algo",
        "Implement a function that converts a string to an integer (atoi).",
        "```python\ndef my_atoi(s: str) -> int:\n    s = s.lstrip()\n    if not s:\n        return 0\n    sign = 1\n    i = 0\n    if s[0] in '+-':\n        sign = -1 if s[0] == '-' else 1\n        i = 1\n    n = 0\n    while i < len(s) and s[i].isdigit():\n        n = n * 10 + int(s[i])\n        i += 1\n    return max(-2 ** 31, min(2 ** 31 - 1, sign * n))\n```\nClamps to 32-bit range as the classic problem requires; Python ints are unbounded so the clamp is the spec, not a need.")

    add("algo",
        "Write a function that returns whether a number is a perfect square without sqrt.",
        "```python\ndef perfect_square(n: int) -> bool:\n    if n < 0:\n        return False\n    lo, hi = 0, n\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        sq = mid * mid\n        if sq == n:\n            return True\n        if sq < n:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return False\n```\nBinary search avoids floating-point precision pitfalls of `math.sqrt`.")

    add("algo",
        "Implement a function that returns the largest number from concatenating a list of ints.",
        "```python\nfrom functools import cmp_to_key\n\ndef largest_number(xs: list[int]) -> str:\n    if not any(xs):\n        return '0'\n    strs = [str(x) for x in xs]\n    strs.sort(key=cmp_to_key(lambda a, b: (b + a > a + b) - (b + a < a + b)))\n    return ''.join(strs)\n```\nCompare by concatenation order; `'9'` should come before `'34'` because `'934' > '349'`.")

    add("algo",
        "Write a function that returns the kth smallest element in a BST (in-order traversal).",
        "```python\ndef kth_smallest(root, k: int):\n    stack: list = []\n    cur = root\n    while cur or stack:\n        while cur:\n            stack.append(cur)\n            cur = cur.left\n        cur = stack.pop()\n        k -= 1\n        if k == 0:\n            return cur.val\n        cur = cur.right\n    raise ValueError('k out of range')\n```\nIterative in-order; stops as soon as the kth value is popped.")

    add("algo",
        "Implement a function that simulates a Fenwick (BIT) tree.",
        "```python\nclass BIT:\n    def __init__(self, n: int) -> None:\n        self.n = n\n        self.t = [0] * (n + 1)\n\n    def update(self, i: int, delta: int) -> None:\n        i += 1\n        while i <= self.n:\n            self.t[i] += delta\n            i += i & -i\n\n    def query(self, i: int) -> int:\n        i += 1\n        s = 0\n        while i > 0:\n            s += self.t[i]\n            i -= i & -i\n        return s\n```\nO(log n) point update and prefix sum; the bit trick `i & -i` isolates the lowest set bit.")

    add("algo",
        "Write a function that returns the number of paths in a grid from top-left to bottom-right.",
        "```python\nfrom math import comb\n\ndef grid_paths(m: int, n: int) -> int:\n    return comb(m + n - 2, m - 1)\n```\nClosed-form via binomial coefficient. The DP version is also fine but unnecessary here.")

    add("algo",
        "Implement a function that returns the longest consecutive sequence length in an unsorted list.",
        "```python\ndef longest_consecutive(xs: list[int]) -> int:\n    s = set(xs)\n    best = 0\n    for x in s:\n        if x - 1 not in s:\n            cur = x\n            length = 1\n            while cur + 1 in s:\n                cur += 1\n                length += 1\n            best = max(best, length)\n    return best\n```\nO(n) by only starting walks from sequence-start elements (those without a predecessor).")

    add("algo",
        "Write a function that finds the kth permutation sequence of 1..n.",
        "```python\nfrom math import factorial\n\ndef kth_permutation(n: int, k: int) -> str:\n    nums = list(range(1, n + 1))\n    k -= 1\n    out = []\n    for i in range(n, 0, -1):\n        f = factorial(i - 1)\n        idx, k = divmod(k, f)\n        out.append(str(nums.pop(idx)))\n    return ''.join(out)\n```\nFactorial-base decomposition selects the digit at each position directly.")

    add("algo",
        "Implement a function that returns whether two strings are one-edit apart.",
        "```python\ndef one_edit(a: str, b: str) -> bool:\n    if abs(len(a) - len(b)) > 1:\n        return False\n    if len(a) > len(b):\n        a, b = b, a\n    i = 0\n    while i < len(a) and a[i] == b[i]:\n        i += 1\n    if i == len(a):\n        return len(a) != len(b)\n    if len(a) == len(b):\n        return a[i + 1:] == b[i + 1:]\n    return a[i:] == b[i + 1:]\n```\nO(n). Categorize by length difference, then walk to the first mismatch.")

    add("algo",
        "Write a function that returns the maximum sum of any path in a binary tree.",
        "```python\ndef max_path(root) -> int:\n    best = float('-inf')\n    def gain(node) -> int:\n        nonlocal best\n        if not node:\n            return 0\n        l = max(0, gain(node.left))\n        r = max(0, gain(node.right))\n        best = max(best, node.val + l + r)\n        return node.val + max(l, r)\n    gain(root)\n    return int(best)\n```\nGain returns single-branch contribution; the through-node sum is checked separately.")

    add("algo",
        "Implement a function that finds the lowest common ancestor in a binary tree.",
        "```python\ndef lca(root, p, q):\n    if root is None or root is p or root is q:\n        return root\n    left = lca(root.left, p, q)\n    right = lca(root.right, p, q)\n    if left and right:\n        return root\n    return left or right\n```\nO(n). The first node where descendants of both p and q meet is the LCA.")

    add("algo",
        "Write a function that returns whether you can finish all courses given prerequisites (cycle detection).",
        "```python\nfrom collections import defaultdict, deque\n\ndef can_finish(num: int, prereqs: list[list[int]]) -> bool:\n    graph: dict = defaultdict(list)\n    indeg = [0] * num\n    for a, b in prereqs:\n        graph[b].append(a)\n        indeg[a] += 1\n    q = deque(i for i, d in enumerate(indeg) if d == 0)\n    seen = 0\n    while q:\n        u = q.popleft()\n        seen += 1\n        for v in graph[u]:\n            indeg[v] -= 1\n            if indeg[v] == 0:\n                q.append(v)\n    return seen == num\n```\nKahn's topo sort; if you can't visit every node there's a cycle.")

    add("algo",
        "Implement a function that returns the minimum height trees in an undirected graph.",
        "```python\nfrom collections import defaultdict, deque\n\ndef min_height_trees(n: int, edges: list[list[int]]) -> list[int]:\n    if n == 1:\n        return [0]\n    graph: dict = defaultdict(set)\n    for a, b in edges:\n        graph[a].add(b)\n        graph[b].add(a)\n    leaves = deque(i for i in range(n) if len(graph[i]) == 1)\n    remaining = n\n    while remaining > 2:\n        size = len(leaves)\n        remaining -= size\n        for _ in range(size):\n            leaf = leaves.popleft()\n            nb = graph[leaf].pop()\n            graph[nb].discard(leaf)\n            if len(graph[nb]) == 1:\n                leaves.append(nb)\n    return list(leaves)\n```\nPeel layers of leaves until at most 2 centroids remain.")

    add("algo",
        "Write a function that returns whether a string is a valid number.",
        "```python\ndef is_number(s: str) -> bool:\n    s = s.strip()\n    if not s:\n        return False\n    try:\n        float(s)\n        return True\n    except ValueError:\n        return False\n```\nDelegate to `float`; it already knows the grammar (signs, decimals, exponent).")

    add("algo",
        "Implement a function that returns the maximum XOR of any two numbers in a list.",
        "```python\ndef max_xor(xs: list[int]) -> int:\n    best = 0\n    mask = 0\n    for i in range(31, -1, -1):\n        mask |= 1 << i\n        prefixes = {x & mask for x in xs}\n        cand = best | (1 << i)\n        if any(cand ^ p in prefixes for p in prefixes):\n            best = cand\n    return best\n```\nGreedy bit-by-bit using a hash set; O(32n).")

    add("algo",
        "Write a function that simulates Conway's Game of Life one step.",
        "```python\ndef life_step(board: list[list[int]]) -> list[list[int]]:\n    rows, cols = len(board), len(board[0])\n    out = [[0] * cols for _ in range(rows)]\n    for r in range(rows):\n        for c in range(cols):\n            n = sum(board[rr][cc] for rr in range(max(0, r - 1), min(rows, r + 2))\n                                 for cc in range(max(0, c - 1), min(cols, c + 2))\n                                 if (rr, cc) != (r, c))\n            if board[r][c]:\n                out[r][c] = 1 if n in (2, 3) else 0\n            else:\n                out[r][c] = 1 if n == 3 else 0\n    return out\n```\nReturn a fresh grid; mutating in place breaks neighbour counting.")

    add("algo",
        "Implement a function that returns the largest divisible subset.",
        "```python\ndef largest_divisible(xs: list[int]) -> list[int]:\n    if not xs:\n        return []\n    xs = sorted(xs)\n    n = len(xs)\n    sz = [1] * n\n    prev = [-1] * n\n    best = 0\n    for i in range(n):\n        for j in range(i):\n            if xs[i] % xs[j] == 0 and sz[j] + 1 > sz[i]:\n                sz[i] = sz[j] + 1\n                prev[i] = j\n        if sz[i] > sz[best]:\n            best = i\n    out: list[int] = []\n    while best != -1:\n        out.append(xs[best])\n        best = prev[best]\n    return out[::-1]\n```\nLIS-style DP with the divisibility relation as the order.")

    add("algo",
        "Write a function that returns the next greater element for each item in a list.",
        "```python\ndef next_greater(xs: list[int]) -> list[int]:\n    out = [-1] * len(xs)\n    stack: list[int] = []\n    for i, x in enumerate(xs):\n        while stack and xs[stack[-1]] < x:\n            out[stack.pop()] = x\n        stack.append(i)\n    return out\n```\nMonotonic decreasing stack of indices; O(n).")

    add("algo",
        "Implement a function that returns whether a directed graph has a cycle.",
        "```python\ndef has_cycle_dg(graph: dict) -> bool:\n    WHITE, GRAY, BLACK = 0, 1, 2\n    color: dict = {u: WHITE for u in graph}\n    def dfs(u) -> bool:\n        color[u] = GRAY\n        for v in graph.get(u, []):\n            if color.get(v, WHITE) == GRAY:\n                return True\n            if color.get(v, WHITE) == WHITE and dfs(v):\n                return True\n        color[u] = BLACK\n        return False\n    return any(color[u] == WHITE and dfs(u) for u in list(color))\n```\nThree-color DFS; a back-edge to a GRAY node signals a cycle.")

    add("algo",
        "Write a function that returns the most water that can be trapped between two vertical lines.",
        "```python\ndef max_area(heights: list[int]) -> int:\n    l, r = 0, len(heights) - 1\n    best = 0\n    while l < r:\n        h = min(heights[l], heights[r])\n        best = max(best, h * (r - l))\n        if heights[l] < heights[r]:\n            l += 1\n        else:\n            r -= 1\n    return best\n```\nTwo pointers; always move the shorter wall because moving the taller can never improve the bound.")

    add("algo",
        "Implement a function that returns total water trapped after rain on a histogram.",
        "```python\ndef trap(heights: list[int]) -> int:\n    if not heights:\n        return 0\n    l, r = 0, len(heights) - 1\n    lmax = rmax = total = 0\n    while l < r:\n        if heights[l] < heights[r]:\n            lmax = max(lmax, heights[l])\n            total += lmax - heights[l]\n            l += 1\n        else:\n            rmax = max(rmax, heights[r])\n            total += rmax - heights[r]\n            r -= 1\n    return total\n```\nTwo pointers + running maxes; O(n) time and O(1) space.")

    add("algo",
        "Write a function that finds the maximum sliding window value.",
        "```python\nfrom collections import deque\n\ndef max_sliding(xs: list[int], k: int) -> list[int]:\n    if k <= 0 or k > len(xs):\n        return []\n    dq: deque = deque()\n    out: list[int] = []\n    for i, x in enumerate(xs):\n        while dq and dq[0] <= i - k:\n            dq.popleft()\n        while dq and xs[dq[-1]] < x:\n            dq.pop()\n        dq.append(i)\n        if i >= k - 1:\n            out.append(xs[dq[0]])\n    return out\n```\nMonotonic deque keeps candidates ordered; amortized O(n).")

    add("algo",
        "Implement a function that returns the longest path in a DAG.",
        "```python\nfrom collections import defaultdict\n\ndef longest_path(graph: dict) -> int:\n    memo: dict = {}\n    def dfs(u: int) -> int:\n        if u in memo:\n            return memo[u]\n        best = 0\n        for v in graph.get(u, []):\n            best = max(best, 1 + dfs(v))\n        memo[u] = best\n        return best\n    return max((dfs(u) for u in graph), default=0)\n```\nMemoized DFS works because the graph is acyclic; for general graphs the problem is NP-hard.")

    add("algo",
        "Write a function that returns the running median of a stream using two heaps.",
        "```python\nimport heapq\n\nclass MedianStream:\n    def __init__(self) -> None:\n        self.lo: list = []  # max-heap (negate)\n        self.hi: list = []  # min-heap\n\n    def add(self, x: float) -> None:\n        heapq.heappush(self.lo, -heapq.heappushpop(self.hi, x))\n        if len(self.lo) > len(self.hi):\n            heapq.heappush(self.hi, -heapq.heappop(self.lo))\n\n    def median(self) -> float:\n        if not self.hi:\n            raise ValueError('empty')\n        if len(self.hi) > len(self.lo):\n            return self.hi[0]\n        return (self.hi[0] + -self.lo[0]) / 2\n```\nKeep |hi| in {|lo|, |lo|+1}; median is `hi[0]` (odd) or the average of tops (even).")

    add("algo",
        "Implement a function that returns whether a 9x9 sudoku puzzle can be solved (backtracking).",
        "```python\ndef solve_sudoku(board: list[list[str]]) -> bool:\n    def find_empty():\n        for i in range(9):\n            for j in range(9):\n                if board[i][j] == '.':\n                    return i, j\n        return None\n    def valid(i: int, j: int, v: str) -> bool:\n        for k in range(9):\n            if board[i][k] == v or board[k][j] == v:\n                return False\n            if board[3 * (i // 3) + k // 3][3 * (j // 3) + k % 3] == v:\n                return False\n        return True\n    spot = find_empty()\n    if not spot:\n        return True\n    i, j = spot\n    for v in '123456789':\n        if valid(i, j, v):\n            board[i][j] = v\n            if solve_sudoku(board):\n                return True\n            board[i][j] = '.'\n    return False\n```\nBacktracking; mutates the board in place.")

    add("algo",
        "Write a function that finds all combinations of k numbers from 1..n.",
        "```python\nfrom itertools import combinations\n\ndef combos(n: int, k: int) -> list[tuple[int, ...]]:\n    return list(combinations(range(1, n + 1), k))\n```\nC-implemented in itertools; recursion is unnecessary.")

    add("algo",
        "Implement a function that returns the maximum sum of two non-overlapping subarrays of given lengths.",
        "```python\ndef max_two_subarrays(xs: list[int], a: int, b: int) -> int:\n    def helper(la: int, lb: int) -> int:\n        prefix = [0]\n        for x in xs:\n            prefix.append(prefix[-1] + x)\n        best = 0\n        max_a = 0\n        for i in range(la + lb, len(prefix)):\n            max_a = max(max_a, prefix[i - lb] - prefix[i - lb - la])\n            best = max(best, max_a + prefix[i] - prefix[i - lb])\n        return best\n    return max(helper(a, b), helper(b, a))\n```\nPrefix sums + sliding window over both orders.")

    add("algo",
        "Write a function that returns whether a target value exists in a sorted matrix.",
        "```python\ndef search_matrix(m: list[list[int]], target: int) -> bool:\n    if not m or not m[0]:\n        return False\n    r, c = 0, len(m[0]) - 1\n    while r < len(m) and c >= 0:\n        if m[r][c] == target:\n            return True\n        if m[r][c] > target:\n            c -= 1\n        else:\n            r += 1\n    return False\n```\nWalk from top-right corner; each step rules out a row or column. O(m+n).")

    add("algo",
        "Implement a function that returns whether a number's digits sum to a multiple of n.",
        "```python\ndef divisible_by_digit_sum(num: int, n: int) -> bool:\n    return sum(int(c) for c in str(abs(num))) % n == 0\n```\nString conversion is the most readable form. The arithmetic loop is fractionally faster but uglier.")

    add("algo",
        "Write a function that computes the entropy of a probability distribution.",
        "```python\nimport math\n\ndef entropy(probs: list[float]) -> float:\n    return -sum(p * math.log2(p) for p in probs if p > 0)\n```\nIn bits; skip zeros to avoid `log2(0)`. Validate that the distribution sums to 1 in calling code.")

    add("algo",
        "Implement a function that returns the angle between two 2D vectors in degrees.",
        "```python\nimport math\n\ndef angle_deg(a: tuple[float, float], b: tuple[float, float]) -> float:\n    dot = a[0] * b[0] + a[1] * b[1]\n    na = math.hypot(*a)\n    nb = math.hypot(*b)\n    if na == 0 or nb == 0:\n        raise ValueError('zero vector')\n    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (na * nb)))))\n```\nClamp the cosine before `acos` to avoid floating-point domain errors.")

    add("algo",
        "Write a function that returns whether three points are collinear.",
        "```python\ndef collinear(p1, p2, p3, eps: float = 1e-9) -> bool:\n    cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])\n    return abs(cross) < eps\n```\nCross product of two edge vectors; zero means parallel meaning collinear.")
