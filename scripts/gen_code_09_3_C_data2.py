# SPDX-License-Identifier: MIT
"""More algorithms / data structures (77). Wave 3 part 2."""


def register_all(add):
    add("algo",
        "Write a function that returns the longest substring with at most k distinct characters.",
        "```python\nfrom collections import defaultdict\n\ndef longest_k_distinct(s: str, k: int) -> int:\n    counts: dict = defaultdict(int)\n    l = best = 0\n    for r, c in enumerate(s):\n        counts[c] += 1\n        while len(counts) > k:\n            counts[s[l]] -= 1\n            if counts[s[l]] == 0:\n                del counts[s[l]]\n            l += 1\n        best = max(best, r - l + 1)\n    return best\n```\nSliding window keeping a count map of the active characters.")

    add("algo",
        "Implement a function that returns whether two strings are one edit apart.",
        "```python\ndef one_edit(a: str, b: str) -> bool:\n    if abs(len(a) - len(b)) > 1:\n        return False\n    if len(a) > len(b):\n        a, b = b, a\n    i = j = diffs = 0\n    while i < len(a) and j < len(b):\n        if a[i] != b[j]:\n            if diffs:\n                return False\n            diffs += 1\n            if len(a) == len(b):\n                i += 1\n        else:\n            i += 1\n        j += 1\n    return True\n```\nLinear scan with a single-edit budget.")

    add("algo",
        "Write a function that returns the missing number from 0..n given a list of size n.",
        "```python\ndef missing_number(xs: list[int]) -> int:\n    n = len(xs)\n    return n * (n + 1) // 2 - sum(xs)\n```\nGauss's formula minus the actual sum is O(n) time and O(1) extra space.")

    add("algo",
        "Implement a function that performs zip-like rotation across an unknown number of iterables.",
        "```python\ndef rotate_zip(*iters):\n    return list(zip(*iters))\n```\nUnpacking with `*iters` makes the call site clean. Use `itertools.zip_longest` if iterables differ in length.")

    add("algo",
        "Write a function that returns whether a number is happy (eventually reaches 1 by squaring digits).",
        "```python\ndef is_happy(n: int) -> bool:\n    seen: set = set()\n    while n != 1 and n not in seen:\n        seen.add(n)\n        n = sum(int(d) ** 2 for d in str(n))\n    return n == 1\n```\nDetect cycles via a `seen` set; alternative is Floyd's tortoise/hare for O(1) memory.")

    add("algo",
        "Implement a function that returns a deep copy of a nested dict/list.",
        "```python\nimport copy\n\ndef deep_copy(x):\n    return copy.deepcopy(x)\n```\nUse stdlib unless profiling proves it's a bottleneck. Hand-rolled `dict()`/`list()` copies miss nested mutables.")

    add("algo",
        "Write a function that returns whether a path forms a closed loop in 2D given moves NSEW.",
        "```python\ndef is_closed_loop(path: str) -> bool:\n    x = y = 0\n    for c in path:\n        dx, dy = {'N': (0, 1), 'S': (0, -1), 'E': (1, 0), 'W': (-1, 0)}[c]\n        x += dx; y += dy\n    return x == 0 and y == 0\n```\nNet displacement check; O(n).")

    add("algo",
        "Implement a function that returns the next permutation of a list in place.",
        "```python\ndef next_permutation(xs: list[int]) -> None:\n    i = len(xs) - 2\n    while i >= 0 and xs[i] >= xs[i + 1]:\n        i -= 1\n    if i >= 0:\n        j = len(xs) - 1\n        while xs[j] <= xs[i]:\n            j -= 1\n        xs[i], xs[j] = xs[j], xs[i]\n    xs[i+1:] = reversed(xs[i+1:])\n```\nStandard 'find pivot, swap successor, reverse suffix' lex-order routine.")

    add("algo",
        "Write a function that returns the GCD of a list of integers.",
        "```python\nimport math\nfrom functools import reduce\n\ndef gcd_list(xs: list[int]) -> int:\n    return reduce(math.gcd, xs)\n```\nGCD is associative; `reduce` makes the n-ary case a one-liner.")

    add("algo",
        "Implement a function that returns the maximum profit from buy-once-sell-once stock prices.",
        "```python\ndef max_profit(prices: list[int]) -> int:\n    best, low = 0, float('inf')\n    for p in prices:\n        low = min(low, p)\n        best = max(best, p - low)\n    return best\n```\nO(n) one-pass tracking min-so-far.")

    add("algo",
        "Write a function that returns the number of distinct subsequences equal to t in s.",
        "```python\ndef num_distinct(s: str, t: str) -> int:\n    dp = [0] * (len(t) + 1)\n    dp[0] = 1\n    for c in s:\n        for j in range(len(t), 0, -1):\n            if c == t[j - 1]:\n                dp[j] += dp[j - 1]\n    return dp[-1]\n```\nReverse iteration over `t` so we read previous-row values before overwriting.")

    add("algo",
        "Implement a function that returns whether a string matches a wildcard pattern with `*` and `?`.",
        "```python\ndef wildcard_match(s: str, pattern: str) -> bool:\n    si = pi = 0\n    star = -1\n    match = 0\n    while si < len(s):\n        if pi < len(pattern) and pattern[pi] in (s[si], '?'):\n            si += 1; pi += 1\n        elif pi < len(pattern) and pattern[pi] == '*':\n            star = pi; match = si; pi += 1\n        elif star != -1:\n            pi = star + 1; match += 1; si = match\n        else:\n            return False\n    while pi < len(pattern) and pattern[pi] == '*':\n        pi += 1\n    return pi == len(pattern)\n```\nIterative O(n*m) worst-case; the star-backtrack is the tricky part.")

    add("algo",
        "Write a function that returns the in-order traversal of a binary tree iteratively.",
        "```python\ndef inorder(root) -> list:\n    out, stack = [], []\n    node = root\n    while node or stack:\n        while node:\n            stack.append(node)\n            node = node.left\n        node = stack.pop()\n        out.append(node.val)\n        node = node.right\n    return out\n```\nIterative version avoids recursion-depth issues on skewed trees.")

    add("algo",
        "Implement a function that returns whether two intervals overlap.",
        "```python\ndef overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:\n    return a[0] < b[1] and b[0] < a[1]\n```\nHalf-open intervals; flip strict to non-strict if your domain treats endpoints as overlapping.")

    add("algo",
        "Write a function that returns the longest zig-zag subsequence length.",
        "```python\ndef zigzag(xs: list[int]) -> int:\n    if len(xs) < 2:\n        return len(xs)\n    up = down = 1\n    for i in range(1, len(xs)):\n        if xs[i] > xs[i-1]:\n            up = down + 1\n        elif xs[i] < xs[i-1]:\n            down = up + 1\n    return max(up, down)\n```\nO(n) DP tracking last-direction sequence lengths.")

    add("algo",
        "Implement a function that returns the n-th ugly number (factors only 2, 3, 5).",
        "```python\ndef nth_ugly(n: int) -> int:\n    ugly = [1]\n    i2 = i3 = i5 = 0\n    while len(ugly) < n:\n        nxt = min(ugly[i2] * 2, ugly[i3] * 3, ugly[i5] * 5)\n        ugly.append(nxt)\n        if nxt == ugly[i2] * 2: i2 += 1\n        if nxt == ugly[i3] * 3: i3 += 1\n        if nxt == ugly[i5] * 5: i5 += 1\n    return ugly[-1]\n```\nThree-pointer merge avoids the heap-with-duplicates trap.")

    add("algo",
        "Write a function that returns the sum of two large numbers represented as strings.",
        "```python\ndef add_strings(a: str, b: str) -> str:\n    return str(int(a) + int(b))\n```\nPython integers are arbitrary precision -- no need for the schoolbook digit-by-digit version unless the input might not be a valid int.")

    add("algo",
        "Implement a function that returns whether a number is an Armstrong number.",
        "```python\ndef is_armstrong(n: int) -> bool:\n    digits = str(n)\n    p = len(digits)\n    return sum(int(d) ** p for d in digits) == n\n```\nAlso called narcissistic numbers; the sum of digits raised to the digit count equals n.")

    add("algo",
        "Write a function that returns the array shifted such that all zeros are at the end.",
        "```python\ndef move_zeros(xs: list[int]) -> None:\n    write = 0\n    for x in xs:\n        if x != 0:\n            xs[write] = x\n            write += 1\n    for i in range(write, len(xs)):\n        xs[i] = 0\n```\nIn-place with a write pointer; O(n) time and O(1) extra space.")

    add("algo",
        "Implement a function that returns the smallest letter greater than target in a sorted list.",
        "```python\nfrom bisect import bisect_right\n\ndef next_greater_letter(letters: list[str], target: str) -> str:\n    i = bisect_right(letters, target)\n    return letters[i % len(letters)]\n```\nWrap-around on the modulo handles the 'no greater letter' case.")

    add("algo",
        "Write a function that returns the result of multiplying two strings of digits.",
        "```python\ndef multiply_strings(a: str, b: str) -> str:\n    return str(int(a) * int(b))\n```\nAgain, Python's bigints make the schoolbook algorithm unnecessary outside of interview settings.")

    add("algo",
        "Implement a function that returns the maximum sum of k consecutive elements.",
        "```python\ndef max_window_sum(xs: list[int], k: int) -> int:\n    if k <= 0 or k > len(xs):\n        raise ValueError('invalid k')\n    s = sum(xs[:k])\n    best = s\n    for i in range(k, len(xs)):\n        s += xs[i] - xs[i - k]\n        best = max(best, s)\n    return best\n```\nO(n) sliding-window sum; recomputing each window from scratch would be O(n*k).")

    add("algo",
        "Write a function that returns whether a binary tree is height-balanced.",
        "```python\ndef is_balanced(root) -> bool:\n    def height(n):\n        if n is None:\n            return 0\n        lh = height(n.left)\n        if lh == -1: return -1\n        rh = height(n.right)\n        if rh == -1 or abs(lh - rh) > 1:\n            return -1\n        return max(lh, rh) + 1\n    return height(root) != -1\n```\nReturn -1 to short-circuit once an imbalance is found; O(n) total.")

    add("algo",
        "Implement a function that returns the spiral order of a 2D matrix.",
        "```python\ndef spiral(m: list[list[int]]) -> list[int]:\n    out: list = []\n    while m:\n        out.extend(m.pop(0))\n        m = list(zip(*m))[::-1]\n        m = [list(r) for r in m]\n    return out\n```\nPeel the top row, rotate counter-clockwise, repeat. Cute but allocates; an index-based loop is faster.")

    add("algo",
        "Write a function that returns the number of valid parenthesis strings of length 2n.",
        "```python\nimport math\n\ndef num_valid_parens(n: int) -> int:\n    return math.comb(2 * n, n) // (n + 1)\n```\nThe Catalan number formula -- same as for many counting problems.")

    add("algo",
        "Implement a function that returns the Pascal's Triangle rows up to n.",
        "```python\ndef pascal(n: int) -> list[list[int]]:\n    rows: list[list[int]] = []\n    for i in range(n):\n        row = [1] * (i + 1)\n        for j in range(1, i):\n            row[j] = rows[i-1][j-1] + rows[i-1][j]\n        rows.append(row)\n    return rows\n```\nEach row computed from the previous; clean O(n\u00b2).")

    add("algo",
        "Write a function that returns whether a target sum can be made from any subset of a list.",
        "```python\ndef subset_sum(xs: list[int], target: int) -> bool:\n    sums = {0}\n    for x in xs:\n        sums |= {s + x for s in sums}\n        if target in sums:\n            return True\n    return target in sums\n```\nSet-based DP; O(n * sum) in the worst case.")

    add("algo",
        "Implement a function that returns the level-order traversal of a binary tree.",
        "```python\nfrom collections import deque\n\ndef level_order(root) -> list[list[int]]:\n    if not root:\n        return []\n    out, q = [], deque([root])\n    while q:\n        level = []\n        for _ in range(len(q)):\n            n = q.popleft()\n            level.append(n.val)\n            if n.left:\n                q.append(n.left)\n            if n.right:\n                q.append(n.right)\n        out.append(level)\n    return out\n```\nBFS with a per-level loop using `len(q)` to bound the iteration.")

    add("algo",
        "Write a function that converts an integer to its English words representation up to a million.",
        "```python\nones = ['', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten',\n        'Eleven', 'Twelve', 'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen', 'Seventeen', 'Eighteen', 'Nineteen']\ntens = ['', '', 'Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty', 'Seventy', 'Eighty', 'Ninety']\n\ndef under_hundred(n: int) -> str:\n    if n < 20:\n        return ones[n]\n    return tens[n // 10] + ('' if n % 10 == 0 else ' ' + ones[n % 10])\n\ndef under_thousand(n: int) -> str:\n    if n < 100:\n        return under_hundred(n)\n    rest = under_hundred(n % 100)\n    return ones[n // 100] + ' Hundred' + (' ' + rest if rest else '')\n\ndef num_to_words(n: int) -> str:\n    if n == 0:\n        return 'Zero'\n    parts = []\n    if n >= 1_000_000:\n        parts.append(under_thousand(n // 1_000_000) + ' Million'); n %= 1_000_000\n    if n >= 1_000:\n        parts.append(under_thousand(n // 1_000) + ' Thousand'); n %= 1_000\n    if n:\n        parts.append(under_thousand(n))\n    return ' '.join(parts)\n```\nDecompose by power-of-1000 chunks; each chunk's words are independent.")

    add("algo",
        "Implement a function that returns the median of two sorted arrays in O(log(min(m,n))).",
        "```python\ndef median_sorted(a: list[int], b: list[int]) -> float:\n    if len(a) > len(b):\n        a, b = b, a\n    m, n = len(a), len(b)\n    lo, hi = 0, m\n    half = (m + n + 1) // 2\n    while lo <= hi:\n        i = (lo + hi) // 2\n        j = half - i\n        a_left = a[i-1] if i > 0 else float('-inf')\n        a_right = a[i] if i < m else float('inf')\n        b_left = b[j-1] if j > 0 else float('-inf')\n        b_right = b[j] if j < n else float('inf')\n        if a_left <= b_right and b_left <= a_right:\n            if (m + n) % 2:\n                return max(a_left, b_left)\n            return (max(a_left, b_left) + min(a_right, b_right)) / 2\n        if a_left > b_right:\n            hi = i - 1\n        else:\n            lo = i + 1\n    raise ValueError('inputs not sorted')\n```\nBinary search the partition point on the shorter array.")

    add("algo",
        "Write a function that returns whether a string can be segmented into dictionary words.",
        "```python\ndef word_break(s: str, words: set[str]) -> bool:\n    dp = [False] * (len(s) + 1)\n    dp[0] = True\n    for i in range(1, len(s) + 1):\n        for j in range(i):\n            if dp[j] and s[j:i] in words:\n                dp[i] = True\n                break\n    return dp[-1]\n```\nClassic DP; passing `words` as a set keeps lookup O(1).")

    add("algo",
        "Implement a function that returns the number of distinct islands shapes in a grid.",
        "```python\ndef distinct_islands(grid: list[list[int]]) -> int:\n    R, C = len(grid), len(grid[0])\n    seen: set = set()\n    shapes: set = set()\n    def dfs(r: int, c: int, base_r: int, base_c: int, shape: list) -> None:\n        if not (0 <= r < R and 0 <= c < C) or grid[r][c] == 0 or (r, c) in seen:\n            return\n        seen.add((r, c))\n        shape.append((r - base_r, c - base_c))\n        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):\n            dfs(r+dr, c+dc, base_r, base_c, shape)\n    for r in range(R):\n        for c in range(C):\n            if grid[r][c] == 1 and (r, c) not in seen:\n                shape: list = []\n                dfs(r, c, r, c, shape)\n                shapes.add(tuple(shape))\n    return len(shapes)\n```\nNormalize each island's coordinates by anchoring at its starting cell.")

    add("algo",
        "Write a function that returns the longest valid parenthesis substring length.",
        "```python\ndef longest_valid_parens(s: str) -> int:\n    stack = [-1]\n    best = 0\n    for i, c in enumerate(s):\n        if c == '(':\n            stack.append(i)\n        else:\n            stack.pop()\n            if not stack:\n                stack.append(i)\n            else:\n                best = max(best, i - stack[-1])\n    return best\n```\nStack stores indices of unmatched `(` plus a sentinel for the last invalid position.")

    add("algo",
        "Implement a function that returns the rotation count in a sorted-rotated array.",
        "```python\ndef rotation_count(xs: list[int]) -> int:\n    lo, hi = 0, len(xs) - 1\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if xs[mid] > xs[hi]:\n            lo = mid + 1\n        else:\n            hi = mid\n    return lo\n```\nBinary search for the index of the smallest element. Assumes distinct values.")

    add("algo",
        "Write a function that returns whether you can complete a circular gas-station tour.",
        "```python\ndef can_complete_circuit(gas: list[int], cost: list[int]) -> int:\n    if sum(gas) < sum(cost):\n        return -1\n    tank = start = 0\n    for i, (g, c) in enumerate(zip(gas, cost)):\n        tank += g - c\n        if tank < 0:\n            start = i + 1\n            tank = 0\n    return start\n```\nGreedy O(n); when tank dips below zero, no station up to here can be the start.")

    add("algo",
        "Implement a function that returns the result of `pow(base, exp)` using fast exponentiation.",
        "```python\ndef fast_pow(base: float, exp: int) -> float:\n    if exp < 0:\n        return 1 / fast_pow(base, -exp)\n    result = 1.0\n    while exp:\n        if exp & 1:\n            result *= base\n        base *= base\n        exp >>= 1\n    return result\n```\nO(log exp). For pure-int math with mod, use `pow(base, exp, mod)` -- it's built in.")

    add("algo",
        "Write a function that returns the longest path in an acyclic directed graph.",
        "```python\nfrom collections import defaultdict\n\ndef longest_path_dag(graph: dict[int, list[int]]) -> int:\n    memo: dict = {}\n    def dp(u: int) -> int:\n        if u in memo:\n            return memo[u]\n        memo[u] = 1 + max((dp(v) for v in graph.get(u, [])), default=0)\n        return memo[u]\n    return max((dp(u) for u in graph), default=0)\n```\nMemoized DFS; caller must guarantee acyclicity (or detect via topological sort).")

    add("algo",
        "Implement a function that returns the smallest range that includes at least one element from each of k sorted lists.",
        "```python\nimport heapq\n\ndef smallest_range(lists: list[list[int]]) -> tuple[int, int]:\n    heap = [(row[0], i, 0) for i, row in enumerate(lists)]\n    heapq.heapify(heap)\n    cur_max = max(row[0] for row in lists)\n    best = (heap[0][0], cur_max)\n    while True:\n        val, i, j = heapq.heappop(heap)\n        if cur_max - val < best[1] - best[0]:\n            best = (val, cur_max)\n        if j + 1 == len(lists[i]):\n            return best\n        nxt = lists[i][j + 1]\n        cur_max = max(cur_max, nxt)\n        heapq.heappush(heap, (nxt, i, j + 1))\n```\nMin-heap across the k lists; advance the smallest pointer each step.")

    add("algo",
        "Write a function that returns the number of bits to flip to convert a to b.",
        "```python\ndef bit_flip(a: int, b: int) -> int:\n    return (a ^ b).bit_count()\n```\nXOR + popcount; one of the cleanest bit tricks.")

    add("algo",
        "Implement a function that finds the median of a stream of integers.",
        "```python\nimport heapq\n\nclass MedianFinder:\n    def __init__(self) -> None:\n        self._lo: list = []  # max-heap (negated)\n        self._hi: list = []  # min-heap\n    def add(self, x: int) -> None:\n        heapq.heappush(self._lo, -heapq.heappushpop(self._hi, x))\n        if len(self._lo) > len(self._hi) + 1:\n            heapq.heappush(self._hi, -heapq.heappop(self._lo))\n    def median(self) -> float:\n        if len(self._lo) > len(self._hi):\n            return -self._lo[0]\n        return (-self._lo[0] + self._hi[0]) / 2\n```\nTwo heaps balanced so the median is always at the top of one or both.")

    add("algo",
        "Write a function that returns whether a graph (edge list) has a cycle using union-find.",
        "```python\ndef has_cycle_uf(n: int, edges: list[tuple[int, int]]) -> bool:\n    parent = list(range(n))\n    def find(x: int) -> int:\n        while parent[x] != x:\n            parent[x] = parent[parent[x]]\n            x = parent[x]\n        return x\n    for u, v in edges:\n        ru, rv = find(u), find(v)\n        if ru == rv:\n            return True\n        parent[ru] = rv\n    return False\n```\nPath compression via 'parent halving' keeps it nearly O(\u03b1(n)) per op.")

    add("algo",
        "Implement a function that returns the number of subarrays with sum equal to k.",
        "```python\nfrom collections import defaultdict\n\ndef subarray_sum_count(xs: list[int], k: int) -> int:\n    counts: dict = defaultdict(int)\n    counts[0] = 1\n    s = result = 0\n    for x in xs:\n        s += x\n        result += counts[s - k]\n        counts[s] += 1\n    return result\n```\nPrefix-sum + hash map; O(n).")

    add("algo",
        "Write a function that returns whether a Sudoku board is solvable (return one solution).",
        "```python\ndef solve_sudoku(board: list[list[str]]) -> bool:\n    rows = [set() for _ in range(9)]\n    cols = [set() for _ in range(9)]\n    boxes = [set() for _ in range(9)]\n    empties = []\n    for r in range(9):\n        for c in range(9):\n            v = board[r][c]\n            if v == '.':\n                empties.append((r, c))\n            else:\n                rows[r].add(v); cols[c].add(v); boxes[r//3*3 + c//3].add(v)\n    def backtrack(i: int) -> bool:\n        if i == len(empties):\n            return True\n        r, c = empties[i]\n        b = r // 3 * 3 + c // 3\n        for d in '123456789':\n            if d not in rows[r] and d not in cols[c] and d not in boxes[b]:\n                board[r][c] = d\n                rows[r].add(d); cols[c].add(d); boxes[b].add(d)\n                if backtrack(i + 1):\n                    return True\n                board[r][c] = '.'\n                rows[r].remove(d); cols[c].remove(d); boxes[b].remove(d)\n        return False\n    return backtrack(0)\n```\nBacktracking with constraint sets; mutates the board in place.")

    add("algo",
        "Implement a function that returns whether two rectangles overlap (axis-aligned).",
        "```python\ndef rects_overlap(a: tuple[int,int,int,int], b: tuple[int,int,int,int]) -> bool:\n    ax1, ay1, ax2, ay2 = a\n    bx1, by1, bx2, by2 = b\n    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2\n```\nAxis projection: rectangles overlap iff they overlap on both axes.")

    add("algo",
        "Write a function that converts an Excel column number to its letter representation.",
        "```python\ndef col_to_letter(n: int) -> str:\n    letters = []\n    while n:\n        n, r = divmod(n - 1, 26)\n        letters.append(chr(ord('A') + r))\n    return ''.join(reversed(letters))\n```\nBijective base-26: subtract 1 before each divmod to handle the 'A=1' offset.")

    add("algo",
        "Implement a function that returns the longest consecutive sequence length in an unsorted list.",
        "```python\ndef longest_consecutive(xs: list[int]) -> int:\n    s = set(xs)\n    best = 0\n    for x in s:\n        if x - 1 not in s:\n            length = 1\n            while x + length in s:\n                length += 1\n            best = max(best, length)\n    return best\n```\nO(n) thanks to skipping non-starts of runs.")

    add("algo",
        "Write a function that returns the kth smallest element in a BST.",
        "```python\ndef kth_smallest_bst(root, k: int):\n    stack: list = []\n    node = root\n    while node or stack:\n        while node:\n            stack.append(node)\n            node = node.left\n        node = stack.pop()\n        k -= 1\n        if k == 0:\n            return node.val\n        node = node.right\n    raise ValueError('k out of range')\n```\nIterative in-order; stops as soon as we've visited k nodes.")

    add("algo",
        "Implement a function that returns the maximum profit from at most two stock transactions.",
        "```python\ndef max_profit_two(prices: list[int]) -> int:\n    if not prices:\n        return 0\n    buy1 = buy2 = float('-inf')\n    sell1 = sell2 = 0\n    for p in prices:\n        buy1 = max(buy1, -p)\n        sell1 = max(sell1, buy1 + p)\n        buy2 = max(buy2, sell1 - p)\n        sell2 = max(sell2, buy2 + p)\n    return sell2\n```\nFour state variables tracking the optimal value at each phase.")

    add("algo",
        "Write a function that returns the next greater element for each item in a list.",
        "```python\ndef next_greater(xs: list[int]) -> list[int]:\n    result = [-1] * len(xs)\n    stack: list[int] = []\n    for i, x in enumerate(xs):\n        while stack and xs[stack[-1]] < x:\n            result[stack.pop()] = x\n        stack.append(i)\n    return result\n```\nMonotonic decreasing stack; each element pushed and popped at most once.")

    add("algo",
        "Implement a function that returns the maximum XOR of two numbers in a list.",
        "```python\ndef max_xor(xs: list[int]) -> int:\n    best = 0\n    mask = 0\n    for i in range(31, -1, -1):\n        mask |= 1 << i\n        prefixes = {x & mask for x in xs}\n        candidate = best | (1 << i)\n        if any(p ^ candidate in prefixes for p in prefixes):\n            best = candidate\n    return best\n```\nBit-by-bit greedy with a set of prefixes; O(32 * n).")

    add("algo",
        "Write a function that returns whether parentheses with `(`, `)`, and `*` (wildcard) are balanced.",
        "```python\ndef check_valid_string(s: str) -> bool:\n    lo = hi = 0\n    for c in s:\n        if c == '(':\n            lo += 1; hi += 1\n        elif c == ')':\n            lo -= 1; hi -= 1\n        else:\n            lo -= 1; hi += 1\n        if hi < 0:\n            return False\n        lo = max(lo, 0)\n    return lo == 0\n```\nTrack a range of possible open-paren counts; valid iff the range straddles 0 at the end.")

    add("algo",
        "Implement a function that detects a duplicate in a list of size n+1 with values 1..n.",
        "```python\ndef find_duplicate(xs: list[int]) -> int:\n    slow = fast = xs[0]\n    while True:\n        slow = xs[slow]\n        fast = xs[xs[fast]]\n        if slow == fast:\n            break\n    slow = xs[0]\n    while slow != fast:\n        slow = xs[slow]\n        fast = xs[fast]\n    return slow\n```\nFloyd's cycle detection on the indices-as-pointers graph; O(1) extra space.")

    add("algo",
        "Write a function that returns the length of the longest arithmetic progression in a list.",
        "```python\ndef longest_arith_seq(xs: list[int]) -> int:\n    if len(xs) < 2:\n        return len(xs)\n    dp: list[dict] = [{} for _ in xs]\n    best = 2\n    for i in range(len(xs)):\n        for j in range(i):\n            d = xs[i] - xs[j]\n            dp[i][d] = dp[j].get(d, 1) + 1\n            best = max(best, dp[i][d])\n    return best\n```\nO(n\u00b2) DP keyed by (index, common difference).")

    add("algo",
        "Implement a function that returns whether you can win a Nim game with n stones.",
        "```python\ndef can_win_nim(n: int) -> bool:\n    return n % 4 != 0\n```\nThe Sprague-Grundy classic: lose iff n is a multiple of 4.")

    add("algo",
        "Write a function that returns the number of distinct ways to climb stairs taking 1 or 2 steps.",
        "```python\ndef climb_stairs(n: int) -> int:\n    a, b = 1, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```\nFibonacci in disguise; iterative two-variable form.")

    add("algo",
        "Implement a function that returns the longest substring of repeating characters after at most k replacements.",
        "```python\nfrom collections import defaultdict\n\ndef character_replacement(s: str, k: int) -> int:\n    counts: dict = defaultdict(int)\n    l = best = max_freq = 0\n    for r, c in enumerate(s):\n        counts[c] += 1\n        max_freq = max(max_freq, counts[c])\n        if r - l + 1 - max_freq > k:\n            counts[s[l]] -= 1\n            l += 1\n        best = max(best, r - l + 1)\n    return best\n```\nSliding window where the constraint is `(window_length - max_freq) <= k`.")

    add("algo",
        "Write a function that returns whether a binary tree is a valid BST.",
        "```python\ndef is_valid_bst(root) -> bool:\n    def check(n, lo, hi) -> bool:\n        if n is None:\n            return True\n        if not (lo < n.val < hi):\n            return False\n        return check(n.left, lo, n.val) and check(n.right, n.val, hi)\n    return check(root, float('-inf'), float('inf'))\n```\nPropagate tight (lo, hi) bounds into recursion; comparing only with parent isn't enough.")

    add("algo",
        "Implement a function that finds two numbers in a sorted list summing to target.",
        "```python\ndef two_sum_sorted(xs: list[int], target: int) -> tuple[int, int] | None:\n    l, r = 0, len(xs) - 1\n    while l < r:\n        s = xs[l] + xs[r]\n        if s == target:\n            return (l, r)\n        if s < target:\n            l += 1\n        else:\n            r -= 1\n    return None\n```\nTwo-pointer; O(n) on sorted input. For unsorted, use a dict for O(n) total.")

    add("algo",
        "Write a function that returns the number of subsets summing to a target.",
        "```python\ndef count_subsets(xs: list[int], target: int) -> int:\n    dp = [0] * (target + 1)\n    dp[0] = 1\n    for x in xs:\n        for s in range(target, x - 1, -1):\n            dp[s] += dp[s - x]\n    return dp[target]\n```\n0/1-knapsack count variant; iterate sums backward to avoid reusing an item.")

    add("algo",
        "Implement a function that decodes a string like '3[a2[c]]' into 'accaccacc'.",
        "```python\ndef decode_string(s: str) -> str:\n    stack: list = []\n    cur_str = ''\n    cur_num = 0\n    for c in s:\n        if c.isdigit():\n            cur_num = cur_num * 10 + int(c)\n        elif c == '[':\n            stack.append((cur_str, cur_num))\n            cur_str, cur_num = '', 0\n        elif c == ']':\n            prev_str, num = stack.pop()\n            cur_str = prev_str + cur_str * num\n        else:\n            cur_str += c\n    return cur_str\n```\nStack of (string, multiplier) pairs; resolve on closing bracket.")

    add("algo",
        "Write a function that finds the k-closest points to the origin.",
        "```python\nimport heapq\n\ndef k_closest(points: list[tuple[float, float]], k: int) -> list[tuple[float, float]]:\n    return heapq.nsmallest(k, points, key=lambda p: p[0] ** 2 + p[1] ** 2)\n```\n`heapq.nsmallest` with a key beats sort-then-slice when k << n.")

    add("algo",
        "Implement a function that returns whether a number is the sum of two squares.",
        "```python\nimport math\n\ndef sum_of_two_squares(n: int) -> bool:\n    if n < 0:\n        return False\n    a = 0\n    while a * a * 2 <= n:\n        b = math.isqrt(n - a * a)\n        if a * a + b * b == n:\n            return True\n        a += 1\n    return False\n```\nLoop one variable, derive the other via integer square root.")

    add("algo",
        "Write a function that performs DFS on a graph represented as an adjacency dict.",
        "```python\ndef dfs(graph: dict, start) -> list:\n    seen: set = set()\n    order: list = []\n    stack = [start]\n    while stack:\n        node = stack.pop()\n        if node in seen:\n            continue\n        seen.add(node)\n        order.append(node)\n        stack.extend(graph.get(node, []))\n    return order\n```\nIterative DFS to avoid Python's recursion limit.")

    add("algo",
        "Implement a function that returns whether all pairs (a, b) in a list have a + b == k for some other pair.",
        "```python\ndef has_complementary_pair_sum(xs: list[int], k: int) -> bool:\n    seen: set = set()\n    for x in xs:\n        if k - x in seen:\n            return True\n        seen.add(x)\n    return False\n```\nThe standard 'two-sum exists' template using a running set.")

    add("algo",
        "Write a function that returns whether a string is a valid number.",
        "```python\ndef is_number(s: str) -> bool:\n    s = s.strip()\n    try:\n        float(s)\n        return True\n    except ValueError:\n        return False\n```\n`float` already handles signs, decimals, and scientific notation. Beware that it also accepts `'inf'`/`'nan'`.")

    add("algo",
        "Implement a function that returns the result of integer division without using `/` or `*`.",
        "```python\ndef divide(a: int, b: int) -> int:\n    if b == 0:\n        raise ZeroDivisionError('division by zero')\n    sign = -1 if (a < 0) ^ (b < 0) else 1\n    a, b = abs(a), abs(b)\n    quotient = 0\n    while a >= b:\n        temp, multiple = b, 1\n        while a >= (temp << 1):\n            temp <<= 1\n            multiple <<= 1\n        a -= temp\n        quotient += multiple\n    return sign * quotient\n```\nDouble the divisor until it exceeds the dividend; classic shift-and-subtract algorithm.")

    add("algo",
        "Write a function that converts a sorted list to a height-balanced BST.",
        "```python\ndef sorted_to_bst(xs: list[int]):\n    if not xs:\n        return None\n    class Node:\n        def __init__(self, v):\n            self.val, self.left, self.right = v, None, None\n    def build(lo: int, hi: int):\n        if lo > hi:\n            return None\n        mid = (lo + hi) // 2\n        node = Node(xs[mid])\n        node.left = build(lo, mid - 1)\n        node.right = build(mid + 1, hi)\n        return node\n    return build(0, len(xs) - 1)\n```\nRecursive midpoint-as-root keeps the tree balanced.")

    add("algo",
        "Implement a function that returns the length of the shortest unsorted subarray.",
        "```python\ndef unsorted_subarray_len(xs: list[int]) -> int:\n    n = len(xs)\n    lo, hi = -1, -1\n    cur_max, cur_min = float('-inf'), float('inf')\n    for i in range(n):\n        cur_max = max(cur_max, xs[i])\n        if xs[i] < cur_max:\n            hi = i\n        cur_min = min(cur_min, xs[n - 1 - i])\n        if xs[n - 1 - i] > cur_min:\n            lo = n - 1 - i\n    return hi - lo + 1 if hi != -1 else 0\n```\nTwo passes: rightmost out-of-order from left, leftmost out-of-order from right.")

    add("algo",
        "Write a function that returns the maximum number of non-overlapping intervals.",
        "```python\ndef max_non_overlapping(intervals: list[tuple[int, int]]) -> int:\n    intervals = sorted(intervals, key=lambda x: x[1])\n    count = end = 0\n    for s, e in intervals:\n        if s >= end:\n            count += 1\n            end = e\n    return count\n```\nGreedy by earliest end-time -- the classic activity-selection.")

    add("algo",
        "Implement a function that returns whether a course schedule with prerequisites is feasible.",
        "```python\nfrom collections import defaultdict, deque\n\ndef can_finish(num_courses: int, prerequisites: list[tuple[int, int]]) -> bool:\n    graph: dict = defaultdict(list)\n    indeg = [0] * num_courses\n    for a, b in prerequisites:\n        graph[b].append(a)\n        indeg[a] += 1\n    q = deque(i for i in range(num_courses) if indeg[i] == 0)\n    visited = 0\n    while q:\n        u = q.popleft()\n        visited += 1\n        for v in graph[u]:\n            indeg[v] -= 1\n            if indeg[v] == 0:\n                q.append(v)\n    return visited == num_courses\n```\nKahn's topological-sort cycle detection.")

    add("algo",
        "Write a function that returns the n-th smallest sum from two sorted arrays.",
        "```python\nimport heapq\n\ndef kth_pair_sum(a: list[int], b: list[int], k: int) -> int:\n    heap = [(a[0] + b[j], 0, j) for j in range(min(k, len(b)))]\n    heapq.heapify(heap)\n    for _ in range(k - 1):\n        s, i, j = heapq.heappop(heap)\n        if i + 1 < len(a):\n            heapq.heappush(heap, (a[i+1] + b[j], i+1, j))\n    return heap[0][0]\n```\nMin-heap traversal of a sorted-pair grid; O((n + k) log k).")
