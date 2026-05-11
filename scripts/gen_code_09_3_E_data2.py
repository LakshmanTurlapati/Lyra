# SPDX-License-Identifier: MIT
"""Algorithms (50) + web/API (50). Wave 5 part 2."""


def register_all(add):
    # ----- algorithms / data structures (50) -----
    add("algo",
        "Write a function that finds duplicates in a list and returns them with counts.",
        "```python\nfrom collections import Counter\n\ndef duplicates(xs: list) -> dict:\n    return {k: v for k, v in Counter(xs).items() if v > 1}\n```\nDict comprehension over `Counter.items()` is the cleanest filter.")

    add("algo",
        "Implement a function that returns the smallest window in s containing all chars of t.",
        "```python\nfrom collections import Counter\n\ndef min_window(s: str, t: str) -> str:\n    if not t or not s:\n        return ''\n    need = Counter(t)\n    missing = len(t)\n    left = best = 0\n    best_len = float('inf')\n    for right, ch in enumerate(s, 1):\n        if need[ch] > 0:\n            missing -= 1\n        need[ch] -= 1\n        if missing == 0:\n            while need[s[left]] < 0:\n                need[s[left]] += 1\n                left += 1\n            if right - left < best_len:\n                best_len = right - left\n                best = left\n    return '' if best_len == float('inf') else s[best:best+best_len]\n```\nClassic sliding window; O(n + m).")

    add("algo",
        "Write a function that returns the n-th ugly number (factors only 2, 3, 5).",
        "```python\ndef nth_ugly(n: int) -> int:\n    if n < 1:\n        raise ValueError('1-indexed')\n    ugly = [1]\n    i2 = i3 = i5 = 0\n    while len(ugly) < n:\n        nxt = min(ugly[i2] * 2, ugly[i3] * 3, ugly[i5] * 5)\n        ugly.append(nxt)\n        if nxt == ugly[i2] * 2: i2 += 1\n        if nxt == ugly[i3] * 3: i3 += 1\n        if nxt == ugly[i5] * 5: i5 += 1\n    return ugly[-1]\n```\nThree-pointer DP; O(n) time and space.")

    add("algo",
        "Implement a function that returns the maximum depth of nested parentheses.",
        "```python\ndef max_paren_depth(s: str) -> int:\n    depth = best = 0\n    for ch in s:\n        if ch == '(':\n            depth += 1\n            best = max(best, depth)\n        elif ch == ')':\n            depth -= 1\n    return best\n```\nIf the input is untrusted, validate balance first or raise mid-loop.")

    add("algo",
        "Write a function that returns whether a number is a happy number.",
        "```python\ndef is_happy(n: int) -> bool:\n    seen: set = set()\n    while n != 1 and n not in seen:\n        seen.add(n)\n        n = sum(int(d) ** 2 for d in str(n))\n    return n == 1\n```\nFloyd's cycle check would be O(1) memory; the set version is clearer.")

    add("algo",
        "Implement a function that returns the count of numbers with unique digits up to 10^n.",
        "```python\ndef unique_digit_numbers(n: int) -> int:\n    if n == 0:\n        return 1\n    total, unique, available = 10, 9, 9\n    for _ in range(n - 1):\n        unique *= available\n        total += unique\n        available -= 1\n        if available == 0:\n            break\n    return total\n```\nCombinatorial: 9 choices for the first digit, then decreasing pool.")

    add("algo",
        "Write a function that flattens a list of (key, list-of-values) into pairs.",
        "```python\ndef flatten_kv(items: list[tuple[str, list]]) -> list[tuple]:\n    return [(k, v) for k, vs in items for v in vs]\n```\nNested comprehension reads top-to-bottom like nested loops.")

    add("algo",
        "Implement a function that returns the longest sequence of consecutive integers in a list.",
        "```python\ndef longest_consecutive(xs: list[int]) -> int:\n    s = set(xs)\n    best = 0\n    for x in s:\n        if x - 1 not in s:\n            length = 1\n            while x + length in s:\n                length += 1\n            best = max(best, length)\n    return best\n```\nO(n) by only starting runs at sequence-starts.")

    add("algo",
        "Write a function that returns whether a list is a subsequence of another.",
        "```python\ndef is_subsequence(sub: list, full: list) -> bool:\n    it = iter(full)\n    return all(any(x == y for y in it) for x in sub)\n```\nThe `iter(full)` is shared across both `any` calls, advancing positionally.")

    add("algo",
        "Implement a function that returns the smallest missing positive integer in a list.",
        "```python\ndef first_missing_positive(xs: list[int]) -> int:\n    s = set(xs)\n    i = 1\n    while i in s:\n        i += 1\n    return i\n```\nO(n) with O(n) extra space; the in-place O(1)-space version is cute but error-prone.")

    add("algo",
        "Write a function that returns the n-th harmonic number.",
        "```python\nfrom fractions import Fraction\n\ndef harmonic(n: int) -> Fraction:\n    if n < 1:\n        raise ValueError('positive only')\n    total = Fraction(0)\n    for i in range(1, n + 1):\n        total += Fraction(1, i)\n    return total\n```\n`Fraction` keeps the result exact; for floats use `sum(1/i for i in range(1, n+1))`.")

    add("algo",
        "Implement a function that returns the most efficient single-buy single-sell stock profit.",
        "```python\ndef max_profit(prices: list[int]) -> int:\n    best = 0\n    lo = float('inf')\n    for p in prices:\n        if p < lo:\n            lo = p\n        elif p - lo > best:\n            best = p - lo\n    return best\n```\nTrack the running minimum and the best gap; O(n) one pass.")

    add("algo",
        "Write a function that returns whether two strings are one edit apart.",
        "```python\ndef one_edit(a: str, b: str) -> bool:\n    if abs(len(a) - len(b)) > 1:\n        return False\n    if len(a) > len(b):\n        a, b = b, a\n    i = j = diffs = 0\n    while i < len(a) and j < len(b):\n        if a[i] != b[j]:\n            diffs += 1\n            if diffs > 1:\n                return False\n            if len(a) == len(b):\n                i += 1\n        else:\n            i += 1\n        j += 1\n    return True\n```\nLinear scan with one allowed mismatch.")

    add("algo",
        "Implement a function that returns the count of each unique character class (digit/letter/space/other) in a string.",
        "```python\ndef class_counts(s: str) -> dict[str, int]:\n    out = {'digit': 0, 'letter': 0, 'space': 0, 'other': 0}\n    for ch in s:\n        if ch.isdigit():\n            out['digit'] += 1\n        elif ch.isalpha():\n            out['letter'] += 1\n        elif ch.isspace():\n            out['space'] += 1\n        else:\n            out['other'] += 1\n    return out\n```\nUses `str` predicates so it's Unicode-aware.")

    add("algo",
        "Write a function that returns whether a 2D matrix is symmetric.",
        "```python\ndef is_symmetric(m: list[list[int]]) -> bool:\n    n = len(m)\n    return all(len(row) == n for row in m) and all(\n        m[i][j] == m[j][i] for i in range(n) for j in range(i+1, n)\n    )\n```\nOnly check the upper triangle; equality is symmetric.")

    add("algo",
        "Implement a function that returns the number of islands in a grid.",
        "```python\ndef num_islands(grid: list[list[str]]) -> int:\n    if not grid:\n        return 0\n    rows, cols = len(grid), len(grid[0])\n    seen: set = set()\n    def dfs(r, c):\n        stack = [(r, c)]\n        while stack:\n            r, c = stack.pop()\n            if (r, c) in seen:\n                continue\n            if not (0 <= r < rows and 0 <= c < cols) or grid[r][c] != '1':\n                continue\n            seen.add((r, c))\n            stack.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])\n    count = 0\n    for r in range(rows):\n        for c in range(cols):\n            if grid[r][c] == '1' and (r, c) not in seen:\n                count += 1\n                dfs(r, c)\n    return count\n```\nIterative DFS with a stack; safer than recursion on large grids.")

    add("algo",
        "Write a function that returns the count of pairs with absolute difference k.",
        "```python\nfrom collections import Counter\n\ndef diff_pairs(xs: list[int], k: int) -> int:\n    if k < 0:\n        return 0\n    counts = Counter(xs)\n    if k == 0:\n        return sum(c * (c - 1) // 2 for c in counts.values())\n    return sum(counts[x] * counts[x + k] for x in counts)\n```\nO(n) using a frequency table.")

    add("algo",
        "Implement a function that returns whether a number is an Armstrong / narcissistic number for a given base.",
        "```python\ndef narcissistic(n: int, base: int = 10) -> bool:\n    if base < 2:\n        raise ValueError('base must be >= 2')\n    digits = []\n    x = abs(n)\n    while x:\n        digits.append(x % base)\n        x //= base\n    if not digits:\n        digits = [0]\n    p = len(digits)\n    return n == sum(d ** p for d in digits)\n```\nGeneralisation of base-10 Armstrong numbers.")

    add("algo",
        "Write a function that returns the n-th lucky number.",
        "```python\ndef lucky_numbers(limit: int) -> list[int]:\n    nums = list(range(1, limit + 1, 2))\n    i = 1\n    while i < len(nums) and nums[i] <= len(nums):\n        step = nums[i]\n        nums = [x for j, x in enumerate(nums, 1) if j % step != 0]\n        i += 1\n    return nums\n```\nClassic sieve construction; returns all lucky numbers up to `limit`.")

    add("algo",
        "Implement a function that converts an IPv4 string to an integer.",
        "```python\ndef ipv4_to_int(ip: str) -> int:\n    parts = ip.split('.')\n    if len(parts) != 4:\n        raise ValueError('expected four octets')\n    n = 0\n    for p in parts:\n        v = int(p)\n        if not 0 <= v <= 255:\n            raise ValueError(f'octet out of range: {v}')\n        n = (n << 8) | v\n    return n\n```\nFor production use the stdlib `ipaddress` module -- it handles edge cases and IPv6.")

    add("algo",
        "Write a function that returns the longest sequence of distinct elements in a list.",
        "```python\ndef longest_distinct(xs: list) -> int:\n    last: dict = {}\n    start = best = 0\n    for i, x in enumerate(xs):\n        if x in last and last[x] >= start:\n            start = last[x] + 1\n        last[x] = i\n        best = max(best, i - start + 1)\n    return best\n```\nSliding window with last-seen index; O(n).")

    add("algo",
        "Implement a function that returns whether a list can be partitioned into two equal-sum halves.",
        "```python\ndef can_partition(xs: list[int]) -> bool:\n    total = sum(xs)\n    if total % 2:\n        return False\n    target = total // 2\n    reachable = {0}\n    for x in xs:\n        reachable |= {r + x for r in reachable if r + x <= target}\n        if target in reachable:\n            return True\n    return False\n```\nSubset-sum DP done with sets; O(n * target).")

    add("algo",
        "Write a function that returns whether a directed graph has a path between two nodes.",
        "```python\ndef has_path(graph: dict, src, dst) -> bool:\n    if src == dst:\n        return True\n    seen: set = {src}\n    stack = [src]\n    while stack:\n        node = stack.pop()\n        for n in graph.get(node, []):\n            if n == dst:\n                return True\n            if n not in seen:\n                seen.add(n)\n                stack.append(n)\n    return False\n```\nDFS short-circuits on the first hit.")

    add("algo",
        "Implement a function that returns the longest contiguous subarray sum equal to k.",
        "```python\ndef longest_subarray_sum_k(xs: list[int], k: int) -> int:\n    prefix = 0\n    first: dict[int, int] = {0: -1}\n    best = 0\n    for i, x in enumerate(xs):\n        prefix += x\n        if prefix - k in first:\n            best = max(best, i - first[prefix - k])\n        first.setdefault(prefix, i)\n    return best\n```\nPrefix-sum + earliest-index map; O(n).")

    add("algo",
        "Write a function that returns the closest pair of points by brute force.",
        "```python\nfrom math import hypot\n\ndef closest_pair(points: list[tuple[float, float]]):\n    if len(points) < 2:\n        return None\n    best = None\n    best_d = float('inf')\n    for i in range(len(points)):\n        for j in range(i + 1, len(points)):\n            d = hypot(points[i][0] - points[j][0], points[i][1] - points[j][1])\n            if d < best_d:\n                best_d = d\n                best = (points[i], points[j])\n    return best\n```\nO(n^2); the divide-and-conquer O(n log n) version is overkill for typical inputs.")

    add("algo",
        "Implement a function that simulates Conway's Game of Life next-state for a grid.",
        "```python\ndef step(grid: list[list[int]]) -> list[list[int]]:\n    rows, cols = len(grid), len(grid[0])\n    out = [[0] * cols for _ in range(rows)]\n    for r in range(rows):\n        for c in range(cols):\n            n = sum(grid[rr][cc]\n                    for rr in range(max(0, r-1), min(rows, r+2))\n                    for cc in range(max(0, c-1), min(cols, c+2))\n                    if (rr, cc) != (r, c))\n            if grid[r][c] and n in (2, 3):\n                out[r][c] = 1\n            elif not grid[r][c] and n == 3:\n                out[r][c] = 1\n    return out\n```\nDoes not wrap; clamp neighbour offsets at the borders.")

    add("algo",
        "Write a function that returns the longest substring with at most k distinct characters.",
        "```python\nfrom collections import defaultdict\n\ndef longest_at_most_k_distinct(s: str, k: int) -> int:\n    if k <= 0:\n        return 0\n    counts: dict = defaultdict(int)\n    left = best = 0\n    for right, ch in enumerate(s):\n        counts[ch] += 1\n        while len(counts) > k:\n            counts[s[left]] -= 1\n            if counts[s[left]] == 0:\n                del counts[s[left]]\n            left += 1\n        best = max(best, right - left + 1)\n    return best\n```\nClassic sliding-window template.")

    add("algo",
        "Implement a function that detects whether a linked list has a cycle.",
        "```python\ndef has_cycle(head) -> bool:\n    slow = fast = head\n    while fast and fast.next:\n        slow = slow.next\n        fast = fast.next.next\n        if slow is fast:\n            return True\n    return False\n```\nFloyd's tortoise and hare; O(1) space.")

    add("algo",
        "Write a function that returns the n-th term of the Look-and-Say sequence.",
        "```python\nfrom itertools import groupby\n\ndef look_and_say(n: int) -> str:\n    if n < 1:\n        raise ValueError('1-indexed')\n    s = '1'\n    for _ in range(n - 1):\n        s = ''.join(f'{sum(1 for _ in g)}{ch}' for ch, g in groupby(s))\n    return s\n```\n`groupby` makes each step a one-liner.")

    add("algo",
        "Implement a function that returns the depth of the deepest dictionary value.",
        "```python\ndef dict_depth(d) -> int:\n    if isinstance(d, dict) and d:\n        return 1 + max(dict_depth(v) for v in d.values())\n    if isinstance(d, list) and d:\n        return max((dict_depth(v) for v in d), default=0)\n    return 0\n```\nHandles list values too; empty containers contribute 0.")

    add("algo",
        "Write a function that returns the number of ways to climb n stairs with 1 or 2 steps.",
        "```python\ndef stairs(n: int) -> int:\n    if n < 0:\n        raise ValueError('non-negative only')\n    a, b = 1, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```\nFibonacci in disguise; O(n) time, O(1) space.")

    add("algo",
        "Implement a function that returns the minimum number of coins to make change.",
        "```python\ndef min_coins(coins: list[int], target: int) -> int:\n    if target < 0:\n        raise ValueError('non-negative only')\n    dp = [0] + [float('inf')] * target\n    for x in range(1, target + 1):\n        for c in coins:\n            if c <= x and dp[x-c] + 1 < dp[x]:\n                dp[x] = dp[x-c] + 1\n    return dp[target] if dp[target] != float('inf') else -1\n```\nCoin-change DP; O(target * len(coins)).")

    add("algo",
        "Write a function that returns the count of distinct paths in an m x n grid (right/down only).",
        "```python\nfrom math import comb\n\ndef unique_paths(m: int, n: int) -> int:\n    if m < 1 or n < 1:\n        return 0\n    return comb(m + n - 2, m - 1)\n```\nClosed-form binomial; faster than DP and exact.")

    add("algo",
        "Implement a function that returns the maximum sum of a non-adjacent subset.",
        "```python\ndef rob(xs: list[int]) -> int:\n    prev = curr = 0\n    for x in xs:\n        prev, curr = curr, max(curr, prev + x)\n    return curr\n```\nClassic 'house robber' DP; two rolling variables.")

    add("algo",
        "Write a function that returns the minimum window subarray with sum >= k.",
        "```python\ndef min_subarray_sum_at_least(xs: list[int], k: int) -> int:\n    left = total = 0\n    best = float('inf')\n    for right, x in enumerate(xs):\n        total += x\n        while total >= k:\n            best = min(best, right - left + 1)\n            total -= xs[left]\n            left += 1\n    return 0 if best == float('inf') else best\n```\nValid only for non-negative `xs`; otherwise use prefix sums + monotonic deque.")

    add("algo",
        "Implement a function that returns whether a string is a valid number (int or float).",
        "```python\ndef is_number(s: str) -> bool:\n    s = s.strip()\n    if not s:\n        return False\n    try:\n        float(s)\n        return True\n    except ValueError:\n        return False\n```\nDelegate to `float()`. Rolling a regex for this is a famously bug-prone exercise.")

    add("algo",
        "Write a function that returns whether two trees are mirror images.",
        "```python\ndef is_mirror(a, b) -> bool:\n    if a is None and b is None:\n        return True\n    if a is None or b is None:\n        return False\n    return a.val == b.val and is_mirror(a.left, b.right) and is_mirror(a.right, b.left)\n```\nRecursion is cleanest; for large trees use an explicit stack.")

    add("algo",
        "Implement a function that finds the kth largest element using quickselect.",
        "```python\nimport random\n\ndef kth_largest(xs: list[int], k: int) -> int:\n    if not 1 <= k <= len(xs):\n        raise ValueError('k out of range')\n    xs = list(xs)\n    target = len(xs) - k\n    lo, hi = 0, len(xs) - 1\n    while lo < hi:\n        pivot = xs[random.randint(lo, hi)]\n        i, j = lo, hi\n        while i <= j:\n            while xs[i] < pivot: i += 1\n            while xs[j] > pivot: j -= 1\n            if i <= j:\n                xs[i], xs[j] = xs[j], xs[i]\n                i += 1; j -= 1\n        if target <= j: hi = j\n        elif target >= i: lo = i\n        else: return xs[target]\n    return xs[lo]\n```\nO(n) average. For correctness without speed, `sorted(xs)[-k]` works.")

    add("algo",
        "Write a function that returns the next greater element for each item in a list.",
        "```python\ndef next_greater(xs: list[int]) -> list[int]:\n    out = [-1] * len(xs)\n    stack: list[int] = []\n    for i, x in enumerate(xs):\n        while stack and xs[stack[-1]] < x:\n            out[stack.pop()] = x\n        stack.append(i)\n    return out\n```\nMonotonic stack; O(n) total.")

    add("algo",
        "Implement a function that returns the trapped rainwater volume between bars.",
        "```python\ndef trap_water(h: list[int]) -> int:\n    if not h:\n        return 0\n    left, right = 0, len(h) - 1\n    lmax = rmax = total = 0\n    while left < right:\n        if h[left] < h[right]:\n            lmax = max(lmax, h[left])\n            total += lmax - h[left]\n            left += 1\n        else:\n            rmax = max(rmax, h[right])\n            total += rmax - h[right]\n            right -= 1\n    return total\n```\nTwo-pointer sweep; O(n) time, O(1) space.")

    add("algo",
        "Write a function that returns whether a board is a valid Sudoku state.",
        "```python\ndef valid_sudoku(board: list[list[str]]) -> bool:\n    rows = [set() for _ in range(9)]\n    cols = [set() for _ in range(9)]\n    boxes = [set() for _ in range(9)]\n    for r in range(9):\n        for c in range(9):\n            v = board[r][c]\n            if v == '.':\n                continue\n            b = (r // 3) * 3 + c // 3\n            if v in rows[r] or v in cols[c] or v in boxes[b]:\n                return False\n            rows[r].add(v); cols[c].add(v); boxes[b].add(v)\n    return True\n```\nThree sets per index; one pass.")

    add("algo",
        "Implement a function that produces all valid combinations of n pairs of parens.",
        "```python\ndef gen_parens(n: int) -> list[str]:\n    out: list[str] = []\n    def rec(s: str, opens: int, closes: int):\n        if len(s) == 2 * n:\n            out.append(s); return\n        if opens < n:\n            rec(s + '(', opens + 1, closes)\n        if closes < opens:\n            rec(s + ')', opens, closes + 1)\n    rec('', 0, 0)\n    return out\n```\nBacktracking with a balance invariant.")

    add("algo",
        "Write a function that returns the minimum edit distance between two strings (insert/delete/replace).",
        "```python\ndef edit_distance(a: str, b: str) -> int:\n    n, m = len(a), len(b)\n    if n < m:\n        a, b = b, a\n        n, m = m, n\n    prev = list(range(m + 1))\n    for i in range(1, n + 1):\n        curr = [i] + [0] * m\n        for j in range(1, m + 1):\n            cost = 0 if a[i-1] == b[j-1] else 1\n            curr[j] = min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + cost)\n        prev = curr\n    return prev[m]\n```\nSame Wagner-Fischer DP as Levenshtein, single-row state.")

    add("algo",
        "Implement a function that returns the count of distinct ways to decode a digit string ('A'=1..'Z'=26).",
        "```python\ndef decode_count(s: str) -> int:\n    if not s or s[0] == '0':\n        return 0\n    a = b = 1\n    for i in range(1, len(s)):\n        cur = 0\n        if s[i] != '0':\n            cur += b\n        two = int(s[i-1:i+1])\n        if 10 <= two <= 26:\n            cur += a\n        a, b = b, cur\n    return b\n```\nFibonacci-style DP with two rolling variables.")

    add("algo",
        "Write a function that returns whether a string is a valid IPv4 address.",
        "```python\ndef is_ipv4(s: str) -> bool:\n    parts = s.split('.')\n    if len(parts) != 4:\n        return False\n    for p in parts:\n        if not p.isdigit() or (len(p) > 1 and p[0] == '0'):\n            return False\n        if not 0 <= int(p) <= 255:\n            return False\n    return True\n```\nReject leading zeros to match strict dotted-quad. For everything-IP-related, reach for `ipaddress`.")

    add("algo",
        "Implement a function that returns the longest zigzag subsequence length.",
        "```python\ndef zigzag_length(xs: list[int]) -> int:\n    if not xs:\n        return 0\n    up = down = 1\n    for i in range(1, len(xs)):\n        if xs[i] > xs[i-1]:\n            up = down + 1\n        elif xs[i] < xs[i-1]:\n            down = up + 1\n    return max(up, down)\n```\nO(n) two-state DP.")

    add("algo",
        "Write a function that returns the GCD of a list of integers.",
        "```python\nfrom functools import reduce\nfrom math import gcd\n\ndef list_gcd(xs: list[int]) -> int:\n    if not xs:\n        return 0\n    return reduce(gcd, xs)\n```\n`reduce` with `math.gcd`; works because GCD is associative.")

    add("algo",
        "Implement a function that returns the LCM of a list of integers.",
        "```python\nfrom functools import reduce\nfrom math import gcd\n\ndef list_lcm(xs: list[int]) -> int:\n    if not xs:\n        return 0\n    def lcm(a: int, b: int) -> int:\n        return abs(a * b) // gcd(a, b) if a and b else 0\n    return reduce(lcm, xs)\n```\nPython 3.9+ exposes `math.lcm` directly.")

    add("algo",
        "Implement a function that rotates a 2D matrix 90 degrees clockwise in place.",
        "```python\ndef rotate_90(m: list[list[int]]) -> None:\n    n = len(m)\n    for i in range(n):\n        for j in range(i + 1, n):\n            m[i][j], m[j][i] = m[j][i], m[i][j]\n    for row in m:\n        row.reverse()\n```\nTranspose then reverse each row -- the cleanest in-place 90 degree rotation.")

    add("algo",
        "Write a function that returns whether a string is a valid shuffle of two strings.",
        "```python\ndef is_shuffle(a: str, b: str, c: str) -> bool:\n    if len(a) + len(b) != len(c):\n        return False\n    n, m = len(a), len(b)\n    dp = [[False] * (m + 1) for _ in range(n + 1)]\n    dp[0][0] = True\n    for i in range(n + 1):\n        for j in range(m + 1):\n            k = i + j\n            if i and a[i-1] == c[k-1]:\n                dp[i][j] = dp[i][j] or dp[i-1][j]\n            if j and b[j-1] == c[k-1]:\n                dp[i][j] = dp[i][j] or dp[i][j-1]\n    return dp[n][m]\n```\n2D DP; O(n*m) time and space.")

    # ----- web / API (50) -----
    add("web",
        "Write a minimal FastAPI app that returns the current UTC time.",
        "```python\nfrom datetime import datetime, timezone\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get('/now')\ndef now() -> dict:\n    return {'utc': datetime.now(timezone.utc).isoformat()}\n```\nRun with `uvicorn main:app --reload`. Always use timezone-aware datetimes.")

    add("web",
        "Write a Flask endpoint that returns JSON with a status code.",
        "```python\nfrom flask import Flask, jsonify\n\napp = Flask(__name__)\n\n@app.get('/health')\ndef health():\n    return jsonify(status='ok'), 200\n```\n`jsonify` sets the right Content-Type; the `(body, status)` tuple is the cleanest way to pick the code.")

    add("web",
        "How do I parse JSON request bodies in FastAPI with validation?",
        "Define a Pydantic model and accept it as a parameter:\n```python\nfrom pydantic import BaseModel\nfrom fastapi import FastAPI\n\nclass Item(BaseModel):\n    name: str\n    qty: int\n\napp = FastAPI()\n\n@app.post('/items')\ndef create(item: Item) -> dict:\n    return {'received': item.model_dump()}\n```\nFastAPI returns 422 with the validation errors automatically.")

    add("web",
        "Write a FastAPI endpoint that streams a large file to the client.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.responses import StreamingResponse\n\napp = FastAPI()\n\ndef iter_file(path: str, chunk: int = 64 * 1024):\n    with open(path, 'rb') as f:\n        while data := f.read(chunk):\n            yield data\n\n@app.get('/download')\ndef download():\n    return StreamingResponse(iter_file('big.bin'), media_type='application/octet-stream')\n```\nStreaming avoids loading the whole file in memory.")

    add("web",
        "Write a FastAPI endpoint that uploads a file.",
        "```python\nfrom fastapi import FastAPI, UploadFile\n\napp = FastAPI()\n\n@app.post('/upload')\nasync def upload(file: UploadFile) -> dict:\n    data = await file.read()\n    return {'filename': file.filename, 'size': len(data)}\n```\nFor large files iterate `file.file` in chunks rather than calling `.read()`.")

    add("web",
        "How do I read a query parameter with a default in FastAPI?",
        "```python\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get('/search')\ndef search(q: str = '', limit: int = 10) -> dict:\n    return {'q': q, 'limit': limit}\n```\nThe defaults make them optional; declared types drive automatic validation and the OpenAPI schema.")

    add("web",
        "Write a FastAPI dependency that extracts a bearer token.",
        "```python\nfrom fastapi import Depends, HTTPException, Request\n\nasync def bearer_token(request: Request) -> str:\n    auth = request.headers.get('authorization', '')\n    if not auth.startswith('Bearer '):\n        raise HTTPException(status_code=401, detail='missing bearer token')\n    return auth.removeprefix('Bearer ')\n\n@app.get('/me')\ndef me(token: str = Depends(bearer_token)) -> dict:\n    return {'token': token[:8] + '...'}\n```\nKeep token validation in the dependency; never repeat it in handlers.")

    add("web",
        "How do I add CORS middleware to a FastAPI app?",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.middleware.cors import CORSMiddleware\n\napp = FastAPI()\napp.add_middleware(\n    CORSMiddleware,\n    allow_origins=['https://example.com'],\n    allow_methods=['GET', 'POST'],\n    allow_headers=['*'],\n    allow_credentials=True,\n)\n```\nUse an explicit allowlist in production -- `'*'` plus credentialed requests is rejected by browsers.")

    add("web",
        "Write a FastAPI background task that logs after the response is sent.",
        "```python\nimport logging\nfrom fastapi import BackgroundTasks, FastAPI\n\nlog = logging.getLogger(__name__)\napp = FastAPI()\n\ndef record_event(name: str):\n    log.info('event: %s', name)\n\n@app.post('/events/{name}')\ndef event(name: str, tasks: BackgroundTasks) -> dict:\n    tasks.add_task(record_event, name)\n    return {'queued': name}\n```\n`BackgroundTasks` runs in the same process; for heavier work use Celery / RQ.")

    add("web",
        "Write a Flask blueprint for user routes.",
        "```python\nfrom flask import Blueprint, jsonify\n\nusers_bp = Blueprint('users', __name__, url_prefix='/users')\n\n@users_bp.get('/')\ndef list_users():\n    return jsonify(users=[])\n\n@users_bp.get('/<int:user_id>')\ndef get_user(user_id: int):\n    return jsonify(id=user_id)\n```\nRegister with `app.register_blueprint(users_bp)`. Blueprints keep large apps maintainable.")

    add("web",
        "How do I make HTTP requests with retries in Python?",
        "```python\nimport requests\nfrom requests.adapters import HTTPAdapter\nfrom urllib3.util.retry import Retry\n\nretry = Retry(\n    total=3, backoff_factor=0.5,\n    status_forcelist=(500, 502, 503, 504),\n    allowed_methods=frozenset(['GET', 'POST']),\n)\nsession = requests.Session()\nsession.mount('https://', HTTPAdapter(max_retries=retry))\nsession.mount('http://', HTTPAdapter(max_retries=retry))\n\nresp = session.get('https://api.example.com/v1/things', timeout=5)\nresp.raise_for_status()\n```\nAlways set a timeout. The default is no timeout, which means hangs forever.")

    add("web",
        "Write an async HTTP client with httpx.",
        "```python\nimport asyncio\nimport httpx\n\nasync def fetch_all(urls: list[str]) -> list[str]:\n    async with httpx.AsyncClient(timeout=10.0) as client:\n        results = await asyncio.gather(*(client.get(u) for u in urls))\n    return [r.text for r in results]\n\nif __name__ == '__main__':\n    print(asyncio.run(fetch_all(['https://example.com'])))\n```\n`httpx` is a drop-in replacement for `requests` with async support.")

    add("web",
        "How do I implement pagination on a FastAPI list endpoint?",
        "```python\nfrom fastapi import FastAPI, Query\n\napp = FastAPI()\nITEMS = list(range(1000))\n\n@app.get('/items')\ndef list_items(\n    page: int = Query(1, ge=1),\n    size: int = Query(20, ge=1, le=100),\n) -> dict:\n    start = (page - 1) * size\n    return {\n        'items': ITEMS[start:start + size],\n        'page': page,\n        'size': size,\n        'total': len(ITEMS),\n    }\n```\n`Query` validates the bounds so you don't have to.")

    add("web",
        "Write a Flask error handler for HTTPException returning JSON.",
        "```python\nfrom flask import Flask, jsonify\nfrom werkzeug.exceptions import HTTPException\n\napp = Flask(__name__)\n\n@app.errorhandler(HTTPException)\ndef handle_http(exc: HTTPException):\n    return jsonify(error={'code': exc.code, 'message': exc.description}), exc.code\n```\nProduces stable JSON-shaped errors instead of Flask's default HTML page.")

    add("web",
        "How do I add rate limiting to a FastAPI app?",
        "Use `slowapi`:\n```python\nfrom fastapi import FastAPI, Request\nfrom slowapi import Limiter, _rate_limit_exceeded_handler\nfrom slowapi.errors import RateLimitExceeded\nfrom slowapi.util import get_remote_address\n\nlimiter = Limiter(key_func=get_remote_address)\napp = FastAPI()\napp.state.limiter = limiter\napp.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)\n\n@app.get('/items')\n@limiter.limit('10/minute')\ndef items(request: Request):\n    return {'ok': True}\n```\nUse Redis as the storage backend in production -- in-process counters break behind a load balancer.")

    add("web",
        "Write a SQLAlchemy 2.0 declarative model.",
        "```python\nfrom datetime import datetime\nfrom sqlalchemy import String, DateTime, func\nfrom sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column\n\nclass Base(DeclarativeBase):\n    pass\n\nclass User(Base):\n    __tablename__ = 'users'\n    id: Mapped[int] = mapped_column(primary_key=True)\n    email: Mapped[str] = mapped_column(String(255), unique=True)\n    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())\n```\n`Mapped` annotations give full type-checker support, unlike the v1 style.")

    add("web",
        "How do I serve static files from FastAPI?",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.staticfiles import StaticFiles\n\napp = FastAPI()\napp.mount('/static', StaticFiles(directory='static'), name='static')\n```\nIn production a reverse proxy (nginx, Caddy) should serve static assets directly -- the Python app shouldn't be in the hot path.")

    add("web",
        "Write a FastAPI app with startup and shutdown hooks.",
        "```python\nfrom contextlib import asynccontextmanager\nfrom fastapi import FastAPI\n\n@asynccontextmanager\nasync def lifespan(app: FastAPI):\n    # startup\n    app.state.cache = {}\n    yield\n    # shutdown\n    app.state.cache.clear()\n\napp = FastAPI(lifespan=lifespan)\n```\nThe `lifespan` context manager replaces the deprecated `@app.on_event` decorators.")

    add("web",
        "How do I validate an email field with Pydantic?",
        "Install `email-validator` and use `EmailStr`:\n```python\nfrom pydantic import BaseModel, EmailStr\n\nclass Signup(BaseModel):\n    email: EmailStr\n    password: str\n```\nPydantic returns 422 with field-level errors automatically when used with FastAPI.")

    add("web",
        "Write a small WebSocket echo server in FastAPI.",
        "```python\nfrom fastapi import FastAPI, WebSocket, WebSocketDisconnect\n\napp = FastAPI()\n\n@app.websocket('/ws')\nasync def echo(ws: WebSocket):\n    await ws.accept()\n    try:\n        while True:\n            msg = await ws.receive_text()\n            await ws.send_text(f'echo: {msg}')\n    except WebSocketDisconnect:\n        return\n```\n`WebSocketDisconnect` is the normal-close path; treat it like an exit, not an error.")

    add("web",
        "How do I add a request ID to every log line in a Flask app?",
        "```python\nimport uuid, logging\nfrom flask import Flask, g, request\n\napp = Flask(__name__)\nlogger = logging.getLogger(__name__)\n\n@app.before_request\ndef set_request_id():\n    g.request_id = request.headers.get('X-Request-ID') or uuid.uuid4().hex\n\nclass RequestIdFilter(logging.Filter):\n    def filter(self, record):\n        record.request_id = getattr(g, 'request_id', '-')\n        return True\n\nfor h in logging.getLogger().handlers:\n    h.addFilter(RequestIdFilter())\n```\nUse a JSON formatter so consumers can index on `request_id`.")

    add("web",
        "Write a FastAPI endpoint that returns a Pydantic response model.",
        "```python\nfrom fastapi import FastAPI\nfrom pydantic import BaseModel\n\napp = FastAPI()\n\nclass User(BaseModel):\n    id: int\n    email: str\n\n@app.get('/users/{user_id}', response_model=User)\ndef get_user(user_id: int) -> User:\n    return User(id=user_id, email=f'u{user_id}@example.com')\n```\n`response_model` filters extra fields and drives the OpenAPI docs.")

    add("web",
        "How do I send a POST request with JSON in requests?",
        "```python\nimport requests\n\nresp = requests.post(\n    'https://api.example.com/items',\n    json={'name': 'widget', 'qty': 3},\n    timeout=5,\n)\nresp.raise_for_status()\nprint(resp.json())\n```\nUse `json=` (auto-serialises and sets Content-Type), not `data=json.dumps(...)`.")

    add("web",
        "Write an async FastAPI route that calls a downstream API.",
        "```python\nimport httpx\nfrom fastapi import FastAPI, HTTPException\n\napp = FastAPI()\n\n@app.get('/proxy/{name}')\nasync def proxy(name: str):\n    async with httpx.AsyncClient(timeout=5.0) as client:\n        try:\n            r = await client.get(f'https://api.example.com/items/{name}')\n            r.raise_for_status()\n        except httpx.HTTPError as exc:\n            raise HTTPException(503, str(exc))\n    return r.json()\n```\nKeep the timeout aggressive -- async tasks held by a slow downstream pile up fast.")

    add("web",
        "How do I include FastAPI routes from sub-modules?",
        "```python\n# users.py\nfrom fastapi import APIRouter\nrouter = APIRouter(prefix='/users', tags=['users'])\n\n@router.get('/')\ndef list_users():\n    return []\n\n# main.py\nfrom fastapi import FastAPI\nfrom users import router as users_router\n\napp = FastAPI()\napp.include_router(users_router)\n```\n`APIRouter` is the FastAPI equivalent of Flask blueprints.")

    add("web",
        "Write a Flask route protected by an API-key header.",
        "```python\nimport os\nfrom functools import wraps\nfrom flask import Flask, request, jsonify\n\napp = Flask(__name__)\nAPI_KEY = os.environ['API_KEY']\n\ndef require_key(f):\n    @wraps(f)\n    def wrapper(*args, **kwargs):\n        if request.headers.get('X-API-Key') != API_KEY:\n            return jsonify(error='unauthorized'), 401\n        return f(*args, **kwargs)\n    return wrapper\n\n@app.get('/secret')\n@require_key\ndef secret():\n    return jsonify(msg='hello')\n```\n`hmac.compare_digest` would be safer than `==` against timing attacks.")

    add("web",
        "How do I read environment variables with type validation?",
        "Use `pydantic-settings`:\n```python\nfrom pydantic import Field\nfrom pydantic_settings import BaseSettings\n\nclass Settings(BaseSettings):\n    database_url: str\n    debug: bool = False\n    port: int = Field(8000, ge=1, le=65535)\n    class Config:\n        env_file = '.env'\n\nsettings = Settings()\n```\nValidates types at boot; a missing required var crashes early rather than blowing up at first use.")

    add("web",
        "Write a FastAPI route that returns a CSV download.",
        "```python\nimport csv, io\nfrom fastapi import FastAPI\nfrom fastapi.responses import StreamingResponse\n\napp = FastAPI()\n\n@app.get('/export.csv')\ndef export():\n    buf = io.StringIO()\n    w = csv.writer(buf)\n    w.writerow(['id', 'name'])\n    for i in range(100):\n        w.writerow([i, f'item-{i}'])\n    buf.seek(0)\n    return StreamingResponse(\n        buf, media_type='text/csv',\n        headers={'Content-Disposition': 'attachment; filename=export.csv'},\n    )\n```\nFor large exports stream rows directly with a generator.")

    add("web",
        "How do I add gzip compression to FastAPI responses?",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.middleware.gzip import GZipMiddleware\n\napp = FastAPI()\napp.add_middleware(GZipMiddleware, minimum_size=1024)\n```\nThe `minimum_size` skips small payloads where compression overhead exceeds savings.")

    add("web",
        "Write a SQLAlchemy session-per-request dependency for FastAPI.",
        "```python\nfrom fastapi import Depends, FastAPI\nfrom sqlalchemy import create_engine\nfrom sqlalchemy.orm import Session, sessionmaker\n\nengine = create_engine('sqlite:///app.db', future=True)\nSessionLocal = sessionmaker(bind=engine, expire_on_commit=False)\n\ndef get_db():\n    db = SessionLocal()\n    try:\n        yield db\n    finally:\n        db.close()\n\napp = FastAPI()\n\n@app.get('/users')\ndef users(db: Session = Depends(get_db)):\n    return db.execute(...).all()\n```\nThe `try/finally` guarantees the session is closed even if the handler raises.")

    add("web",
        "How do I parse form data in FastAPI?",
        "```python\nfrom fastapi import FastAPI, Form\n\napp = FastAPI()\n\n@app.post('/login')\ndef login(username: str = Form(...), password: str = Form(...)) -> dict:\n    return {'user': username}\n```\n`Form(...)` declares the field as required (no default).")

    add("web",
        "Write a FastAPI middleware that times every request.",
        "```python\nimport time\nfrom fastapi import FastAPI, Request\n\napp = FastAPI()\n\n@app.middleware('http')\nasync def add_timing(request: Request, call_next):\n    start = time.perf_counter()\n    resp = await call_next(request)\n    resp.headers['X-Response-Time-ms'] = f'{(time.perf_counter() - start) * 1000:.1f}'\n    return resp\n```\nSurfaces latency as a header and can be scraped by your monitoring agent.")

    add("web",
        "How do I configure structured JSON logging for a Flask app?",
        "Install `python-json-logger`:\n```python\nimport logging\nfrom pythonjsonlogger import jsonlogger\n\nhandler = logging.StreamHandler()\nhandler.setFormatter(jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s'))\nroot = logging.getLogger()\nroot.addHandler(handler)\nroot.setLevel(logging.INFO)\n```\nOne JSON object per line is the format every log aggregator parses out of the box.")

    add("web",
        "Write a FastAPI app that uses dependency injection for a config object.",
        "```python\nfrom functools import lru_cache\nfrom fastapi import Depends, FastAPI\nfrom pydantic_settings import BaseSettings\n\nclass Settings(BaseSettings):\n    api_key: str\n    debug: bool = False\n\n@lru_cache\ndef get_settings() -> Settings:\n    return Settings()\n\napp = FastAPI()\n\n@app.get('/config')\ndef config(settings: Settings = Depends(get_settings)) -> dict:\n    return {'debug': settings.debug}\n```\n`lru_cache` makes settings effectively a singleton with easy override in tests.")

    add("web",
        "How do I implement basic auth with FastAPI?",
        "```python\nimport secrets\nfrom fastapi import Depends, FastAPI, HTTPException, status\nfrom fastapi.security import HTTPBasic, HTTPBasicCredentials\n\nsecurity = HTTPBasic()\napp = FastAPI()\n\ndef check_creds(creds: HTTPBasicCredentials = Depends(security)):\n    user_ok = secrets.compare_digest(creds.username, 'admin')\n    pass_ok = secrets.compare_digest(creds.password, 'hunter2')\n    if not (user_ok and pass_ok):\n        raise HTTPException(\n            status_code=status.HTTP_401_UNAUTHORIZED,\n            detail='invalid credentials',\n            headers={'WWW-Authenticate': 'Basic'},\n        )\n    return creds.username\n\n@app.get('/admin')\ndef admin(user: str = Depends(check_creds)):\n    return {'user': user}\n```\n`secrets.compare_digest` blocks timing-attack vectors that `==` opens.")

    add("web",
        "Write a FastAPI endpoint that returns paginated SQL results.",
        "```python\nfrom fastapi import FastAPI, Depends, Query\nfrom sqlalchemy import select\nfrom sqlalchemy.orm import Session\n\napp = FastAPI()\n\n@app.get('/users')\ndef list_users(\n    page: int = Query(1, ge=1),\n    size: int = Query(20, ge=1, le=100),\n    db: Session = Depends(get_db),\n):\n    stmt = select(User).limit(size).offset((page - 1) * size)\n    return db.execute(stmt).scalars().all()\n```\nFor large tables, keyset pagination on an indexed cursor outperforms `OFFSET`.")

    add("web",
        "How do I add OpenTelemetry tracing to a Flask app?",
        "```python\nfrom flask import Flask\nfrom opentelemetry import trace\nfrom opentelemetry.instrumentation.flask import FlaskInstrumentor\nfrom opentelemetry.sdk.trace import TracerProvider\nfrom opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter\n\ntrace.set_tracer_provider(TracerProvider())\ntrace.get_tracer_provider().add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))\n\napp = Flask(__name__)\nFlaskInstrumentor().instrument_app(app)\n```\nSwap `ConsoleSpanExporter` for an OTLP exporter pointing at your collector.")

    add("web",
        "Write a Flask route that returns a paginated SQLAlchemy query.",
        "```python\nfrom flask import Flask, jsonify, request\n\napp = Flask(__name__)\n\n@app.get('/users')\ndef list_users():\n    page = max(int(request.args.get('page', 1)), 1)\n    size = min(max(int(request.args.get('size', 20)), 1), 100)\n    q = User.query.order_by(User.id).paginate(page=page, per_page=size)\n    return jsonify(\n        items=[{'id': u.id, 'email': u.email} for u in q.items],\n        page=q.page, size=q.per_page, total=q.total,\n    )\n```\n`order_by` matters: pagination without it returns inconsistent slices.")

    add("web",
        "How do I serialize datetimes consistently in a FastAPI response?",
        "Pydantic serialises `datetime` to ISO-8601 by default. To force UTC suffix:\n```python\nfrom datetime import datetime, timezone\nfrom pydantic import BaseModel, field_serializer\n\nclass Event(BaseModel):\n    when: datetime\n    @field_serializer('when')\n    def serialize(self, v: datetime) -> str:\n        return v.astimezone(timezone.utc).isoformat()\n```\nAlways store and emit timezone-aware datetimes; naive datetimes silently corrupt across DST boundaries.")

    add("web",
        "Write a httpx-based fake-friendly API client.",
        "```python\nimport httpx\n\nclass ApiClient:\n    def __init__(self, base: str, client: httpx.Client | None = None):\n        self.base = base\n        self._c = client or httpx.Client(base_url=base, timeout=5.0)\n    def get_user(self, user_id: int) -> dict:\n        r = self._c.get(f'/users/{user_id}')\n        r.raise_for_status()\n        return r.json()\n    def close(self):\n        self._c.close()\n```\nInjecting the client lets tests pass `httpx.Client(transport=httpx.MockTransport(handler))`.")

    add("web",
        "How do I add a graceful shutdown signal handler for an asyncio app?",
        "```python\nimport asyncio, signal\n\nasync def main():\n    stop = asyncio.Event()\n    loop = asyncio.get_running_loop()\n    for sig in (signal.SIGINT, signal.SIGTERM):\n        loop.add_signal_handler(sig, stop.set)\n    print('serving...')\n    await stop.wait()\n    print('shutting down')\n\nasyncio.run(main())\n```\nLet long-running tasks observe `stop` and finish their unit of work before exiting.")

    add("web",
        "Write a FastAPI route that returns a streaming response of JSON lines.",
        "```python\nimport json\nfrom fastapi import FastAPI\nfrom fastapi.responses import StreamingResponse\n\napp = FastAPI()\n\n@app.get('/events')\ndef events():\n    def gen():\n        for i in range(1000):\n            yield json.dumps({'i': i}) + '\\n'\n    return StreamingResponse(gen(), media_type='application/x-ndjson')\n```\nNDJSON is the easiest streaming format -- consumers parse line by line.")

    add("web",
        "How do I handle file uploads larger than memory in FastAPI?",
        "Stream the upload to disk:\n```python\nfrom fastapi import FastAPI, UploadFile\nfrom pathlib import Path\n\napp = FastAPI()\n\n@app.post('/upload')\nasync def upload(file: UploadFile) -> dict:\n    dest = Path('/tmp') / file.filename\n    with dest.open('wb') as out:\n        while chunk := await file.read(1 << 20):\n            out.write(chunk)\n    return {'path': str(dest), 'size': dest.stat().st_size}\n```\nNever call `file.read()` without an argument on user uploads; that loads the entire body into RAM.")

    add("web",
        "Write a Flask request hook that enforces a rate limit per IP.",
        "Use `flask-limiter`:\n```python\nfrom flask import Flask\nfrom flask_limiter import Limiter\nfrom flask_limiter.util import get_remote_address\n\napp = Flask(__name__)\nlimiter = Limiter(get_remote_address, app=app, default_limits=['200/hour'])\n\n@app.get('/api/items')\n@limiter.limit('10/minute')\ndef items():\n    return {'items': []}\n```\nConfigure `storage_uri='redis://...'` for multi-process correctness.")

    add("web",
        "How do I issue and verify a JWT in Python?",
        "```python\nimport jwt, time\n\nSECRET = 'change-me'\n\ndef issue(sub: str, ttl: int = 3600) -> str:\n    payload = {'sub': sub, 'iat': int(time.time()), 'exp': int(time.time()) + ttl}\n    return jwt.encode(payload, SECRET, algorithm='HS256')\n\ndef verify(token: str) -> dict:\n    return jwt.decode(token, SECRET, algorithms=['HS256'])\n```\n`PyJWT` validates `exp`/`nbf` automatically. Use asymmetric algorithms (RS256/EdDSA) once you cross trust boundaries.")

    add("web",
        "Write a small async producer-consumer with asyncio.Queue.",
        "```python\nimport asyncio, random\n\nasync def produce(q: asyncio.Queue, n: int):\n    for i in range(n):\n        await asyncio.sleep(random.random() * 0.1)\n        await q.put(i)\n    await q.put(None)\n\nasync def consume(q: asyncio.Queue):\n    while True:\n        item = await q.get()\n        if item is None:\n            return\n        print('got', item)\n\nasync def main():\n    q: asyncio.Queue = asyncio.Queue()\n    await asyncio.gather(produce(q, 10), consume(q))\n\nasyncio.run(main())\n```\n`None` is the sentinel here; for multiple consumers use `q.task_done()` and `await q.join()`.")

    add("web",
        "How do I run a FastAPI app behind gunicorn with uvicorn workers?",
        "```bash\ngunicorn main:app \\\n    --workers 4 \\\n    --worker-class uvicorn.workers.UvicornWorker \\\n    --bind 0.0.0.0:8000 \\\n    --access-logfile -\n```\nWorker count: usually `2 * CPU + 1` for IO-bound apps, `CPU` for CPU-bound. Put nginx or a managed LB in front for TLS.")

    add("web",
        "Write a small URL signer using HMAC-SHA256.",
        "```python\nimport hmac, hashlib, time\nfrom urllib.parse import urlencode\n\ndef sign(url: str, secret: str, ttl: int = 300) -> str:\n    expires = int(time.time()) + ttl\n    msg = f'{url}|{expires}'.encode()\n    sig = hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()\n    return f'{url}?{urlencode({\"expires\": expires, \"sig\": sig})}'\n\ndef verify(url: str, expected: str, secret: str, expires: int) -> bool:\n    if expires < time.time():\n        return False\n    msg = f'{url}|{expires}'.encode()\n    sig = hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()\n    return hmac.compare_digest(sig, expected)\n```\nUse `hmac.compare_digest` for the comparison -- the equality check matters here.")

    add("web",
        "How do I send streaming server-sent events from FastAPI?",
        "```python\nimport asyncio\nfrom fastapi import FastAPI\nfrom fastapi.responses import StreamingResponse\n\napp = FastAPI()\n\nasync def event_stream():\n    for i in range(10):\n        yield f'data: tick-{i}\\n\\n'\n        await asyncio.sleep(1)\n\n@app.get('/sse')\ndef sse():\n    return StreamingResponse(event_stream(), media_type='text/event-stream')\n```\nKeep the connection alive with periodic comments (`: ping\\n\\n`) so proxies don't time out.")

    add("web",
        "Write a context manager that opens a database connection and rolls back on error.",
        "```python\nfrom contextlib import contextmanager\nimport sqlite3\n\n@contextmanager\ndef tx(db_path: str):\n    conn = sqlite3.connect(db_path)\n    try:\n        yield conn\n        conn.commit()\n    except Exception:\n        conn.rollback()\n        raise\n    finally:\n        conn.close()\n```\nUse the same shape for any DB-API driver; SQLAlchemy's `Session.begin()` does this for you.")
