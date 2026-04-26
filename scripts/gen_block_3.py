#!/usr/bin/env python3
"""Block 3: Hypothesis property-based testing (items 41-60)."""
import json
from pathlib import Path

samples = []

def add(system, user, assistant, followup=None):
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    if followup:
        msgs.append({"role": "user", "content": followup[0]})
        msgs.append({"role": "assistant", "content": followup[1]})
    samples.append({"messages": msgs, "tools": [], "domain": "code"})


add(
    "You are a property-based testing expert.",
    "Introduce Hypothesis with a simple property test for a reverse function.",
    "Hypothesis generates many inputs and checks invariants instead of hardcoded examples.\n\n```python\nfrom hypothesis import given, strategies as st\n\ndef reverse(xs):\n    return xs[::-1]\n\n@given(st.lists(st.integers()))\ndef test_reverse_twice_is_identity(xs):\n    assert reverse(reverse(xs)) == xs\n\n@given(st.lists(st.integers()))\ndef test_reverse_preserves_length(xs):\n    assert len(reverse(xs)) == len(xs)\n\n@given(st.lists(st.integers(), min_size=1))\ndef test_reverse_swaps_ends(xs):\n    r = reverse(xs)\n    assert r[0] == xs[-1]\n    assert r[-1] == xs[0]\n```\n\nEach test runs ~100 randomized inputs by default. When one fails, Hypothesis *shrinks* it to the smallest counterexample: e.g. `[0]` rather than `[-837293, 4, 100]`. That shrink is the magic: your bug reports are minimal, not noisy. Save found failures to a local database so the next run replays them first.",
)

add(
    "You are a property-based testing expert.",
    "Show me Hypothesis strategies for common types.",
    "The `strategies` module is your palette.\n\n```python\nfrom hypothesis import given, strategies as st\n\n@given(st.integers(min_value=0, max_value=1000))\ndef test_nat(n):\n    assert n >= 0\n\n@given(st.floats(allow_nan=False, allow_infinity=False))\ndef test_finite_float(x):\n    assert x == x  # NaN would break this\n\n@given(st.text(alphabet=\"abc\", min_size=1, max_size=10))\ndef test_small_text(s):\n    assert set(s) <= set(\"abc\")\n\n@given(st.lists(st.integers(), min_size=1, max_size=20, unique=True))\ndef test_unique_list(xs):\n    assert len(set(xs)) == len(xs)\n\n@given(st.dictionaries(\n    keys=st.text(min_size=1, max_size=5),\n    values=st.integers(),\n    max_size=10,\n))\ndef test_dict(d):\n    assert all(isinstance(k, str) for k in d)\n\n@given(st.one_of(st.none(), st.integers(), st.text()))\ndef test_union(x):\n    assert x is None or isinstance(x, (int, str))\n```\n\nKey builders: `integers`, `floats`, `text`, `binary`, `booleans`, `none`, `lists`, `tuples`, `dictionaries`, `sets`, `datetimes`, `uuids`, `emails`, `ip_addresses`, `from_regex`, `sampled_from`, `one_of`, `just`, and `builds` for composing dataclasses.",
)

add(
    "You are a property-based testing expert.",
    "Write a Hypothesis test using composite strategies.",
    "`@st.composite` lets one strategy depend on another's drawn value.\n\n```python\nfrom hypothesis import given, strategies as st\n\n@st.composite\ndef sorted_lists(draw, min_size=0, max_size=50):\n    xs = draw(st.lists(st.integers(), min_size=min_size, max_size=max_size))\n    return sorted(xs)\n\n@st.composite\ndef index_and_list(draw):\n    xs = draw(st.lists(st.integers(), min_size=1, max_size=20))\n    i = draw(st.integers(min_value=0, max_value=len(xs) - 1))\n    return xs, i\n\n@given(sorted_lists())\ndef test_sorted_is_sorted(xs):\n    assert xs == sorted(xs)\n\n@given(index_and_list())\ndef test_index_in_bounds(pair):\n    xs, i = pair\n    assert 0 <= i < len(xs)\n    xs[i]  # does not raise\n```\n\nComposite strategies are essential when you need correlated inputs: a list plus a valid index into it, a date plus a time in the same timezone, a graph plus a node in it. Call `draw(other_strategy)` to pull a value; return whatever shape you need. They compose cleanly with `@given`.",
)

add(
    "You are a property-based testing expert.",
    "How do I use Hypothesis assume() to discard bad inputs?",
    "`assume(condition)` tells Hypothesis this example is uninteresting; generate another.\n\n```python\nfrom hypothesis import given, assume, strategies as st\n\ndef divide(a, b):\n    return a / b\n\n@given(st.integers(), st.integers())\ndef test_divide(a, b):\n    assume(b != 0)\n    result = divide(a, b)\n    assert isinstance(result, float)\n\n@given(st.text())\ndef test_ascii_only_roundtrip(s):\n    assume(s.isascii())\n    assert s.encode(\"ascii\").decode(\"ascii\") == s\n```\n\n`assume` is useful but has a cost: if you discard too many inputs, Hypothesis raises `Unsatisfiable`. The better fix is to narrow the strategy so you do not generate bad inputs in the first place:\n\n```python\n@given(st.integers(), st.integers().filter(lambda b: b != 0))\ndef test_divide_narrow(a, b):\n    divide(a, b)\n\n# or best:\n@given(st.integers(), st.integers(min_value=1))  # strictly positive\ndef test_divide_pos(a, b):\n    divide(a, b)\n```\n\nRule of thumb: use `assume` for rare-edge filtering, custom strategies for common constraints.",
)

add(
    "You are a property-based testing expert.",
    "Show a stateful Hypothesis test using RuleBasedStateMachine.",
    "Model an object as a state machine; Hypothesis drives it with random action sequences.\n\n```python\nfrom hypothesis.stateful import RuleBasedStateMachine, rule, invariant\nfrom hypothesis import strategies as st\n\nclass Stack:\n    def __init__(self):\n        self._data = []\n    def push(self, x):\n        self._data.append(x)\n    def pop(self):\n        return self._data.pop()\n    def peek(self):\n        return self._data[-1]\n    def __len__(self):\n        return len(self._data)\n\nclass StackMachine(RuleBasedStateMachine):\n    def __init__(self):\n        super().__init__()\n        self.stack = Stack()\n        self.model = []  # reference implementation\n\n    @rule(v=st.integers())\n    def push(self, v):\n        self.stack.push(v)\n        self.model.append(v)\n\n    @rule()\n    def pop(self):\n        if self.model:\n            assert self.stack.pop() == self.model.pop()\n\n    @invariant()\n    def same_length(self):\n        assert len(self.stack) == len(self.model)\n\nTestStack = StackMachine.TestCase\n```\n\nHypothesis fuzzes orderings of push/pop, ensuring the real stack behaves like the simple list model. Rules are actions, invariants run after each rule. When a discrepancy surfaces, the shrinking process finds the shortest sequence that exposes the bug.",
)

add(
    "You are a property-based testing expert.",
    "How do I test serialization round-trips with Hypothesis?",
    "Round-trip is the canonical property-based test: `decode(encode(x)) == x`.\n\n```python\nimport json\nfrom hypothesis import given, strategies as st\n\njson_safe = st.recursive(\n    st.one_of(\n        st.none(),\n        st.booleans(),\n        st.integers(min_value=-2**53, max_value=2**53),\n        st.floats(allow_nan=False, allow_infinity=False),\n        st.text(),\n    ),\n    lambda children: st.one_of(\n        st.lists(children, max_size=5),\n        st.dictionaries(st.text(min_size=1, max_size=5), children, max_size=5),\n    ),\n    max_leaves=10,\n)\n\n@given(json_safe)\ndef test_json_roundtrip(x):\n    assert json.loads(json.dumps(x)) == x\n```\n\n`st.recursive` builds arbitrarily nested data while capping total size with `max_leaves` to keep runs fast. Excluding NaN is important because `NaN != NaN`, which would break the equality assertion. Int bounds stay within JSON's safe integer range to match JavaScript consumers. Extend with custom types by registering strategies or adding a `builds(...)` arm.",
)

add(
    "You are a property-based testing expert.",
    "What is Hypothesis shrinking and how do I interpret failure output?",
    "Shrinking reduces a failing example to a minimal form so the bug is easier to diagnose.\n\n```python\nfrom hypothesis import given, strategies as st\n\ndef buggy_sum(xs):\n    # bug: returns wrong answer when list contains -1\n    total = 0\n    for x in xs:\n        if x == -1:\n            continue\n        total += x\n    return total\n\n@given(st.lists(st.integers()))\ndef test_sum(xs):\n    assert buggy_sum(xs) == sum(xs)\n```\n\nWhen this fails, output looks like:\n\n```\nFalsifying example: test_sum(\n    xs=[-1],\n)\n```\n\nNot `[-1, 837, -29304]`. Hypothesis tries removing elements, moving values toward 0, shortening strings, and so on. The final example is *the* smallest trigger. Add `@example(xs=[-1])` to pin it as a regression test:\n\n```python\nfrom hypothesis import given, example, strategies as st\n\n@given(st.lists(st.integers()))\n@example(xs=[-1])\ndef test_sum(xs):\n    ...\n```\n\nHypothesis also saves failures to `.hypothesis/examples/` and replays them first, so once seen a bug is never lost.",
)

add(
    "You are a property-based testing expert.",
    "How do I set Hypothesis settings per-test and globally?",
    "Use `@settings(...)` and `settings.register_profile(...)`.\n\n```python\nfrom hypothesis import given, settings, HealthCheck, strategies as st\n\n@given(st.integers())\n@settings(max_examples=500, deadline=None,\n          suppress_health_check=[HealthCheck.too_slow])\ndef test_more_thorough(n):\n    assert n == n\n```\n\nGlobal profiles for different contexts:\n\n```python\n# conftest.py\nfrom hypothesis import settings, Verbosity\n\nsettings.register_profile(\"ci\", max_examples=1000, deadline=None)\nsettings.register_profile(\"dev\", max_examples=20, verbosity=Verbosity.verbose)\nsettings.register_profile(\"debug\", max_examples=10, verbosity=Verbosity.verbose,\n                          print_blob=True)\n\nimport os\nsettings.load_profile(os.environ.get(\"HYPOTHESIS_PROFILE\", \"dev\"))\n```\n\nThen run `HYPOTHESIS_PROFILE=ci pytest` for thorough CI runs and `HYPOTHESIS_PROFILE=dev pytest` for fast iteration. `deadline=None` disables the per-example timeout, useful for slow properties. `print_blob=True` makes failures copy-pasteable into `@example`.",
)

add(
    "You are a property-based testing expert.",
    "Show me strategies.builds() for generating dataclass instances.",
    "`builds` wraps a callable and uses strategies for each argument.\n\n```python\nfrom dataclasses import dataclass\nfrom hypothesis import given, strategies as st\n\n@dataclass\nclass User:\n    id: int\n    email: str\n    age: int\n\n@dataclass\nclass Order:\n    id: int\n    user: User\n    items: list\n    total_cents: int\n\nuser_st = st.builds(\n    User,\n    id=st.integers(min_value=1),\n    email=st.emails(),\n    age=st.integers(min_value=13, max_value=120),\n)\n\norder_st = st.builds(\n    Order,\n    id=st.integers(min_value=1),\n    user=user_st,\n    items=st.lists(st.text(min_size=1, max_size=20), max_size=10),\n    total_cents=st.integers(min_value=0, max_value=10_000_000),\n)\n\n@given(order_st)\ndef test_order_total_nonneg(order):\n    assert order.total_cents >= 0\n    assert order.user.age >= 13\n```\n\nFor Pydantic models or attrs classes, `hypothesis.extra.pydantic` and `hypothesis.strategies.from_type` can infer strategies from type hints, removing most of the boilerplate. `from_type(User)` works when annotations are complete.",
)

add(
    "You are a property-based testing expert.",
    "Test that a sort function is idempotent and preserves elements using Hypothesis.",
    "Three independent properties pin down correctness without reimplementing sort.\n\n```python\nfrom hypothesis import given, strategies as st\nfrom collections import Counter\n\ndef my_sort(xs):\n    return sorted(xs)\n\n@given(st.lists(st.integers()))\ndef test_sort_is_sorted(xs):\n    r = my_sort(xs)\n    assert all(r[i] <= r[i+1] for i in range(len(r) - 1))\n\n@given(st.lists(st.integers()))\ndef test_sort_preserves_multiset(xs):\n    assert Counter(my_sort(xs)) == Counter(xs)\n\n@given(st.lists(st.integers()))\ndef test_sort_idempotent(xs):\n    once = my_sort(xs)\n    assert my_sort(once) == once\n```\n\nThree properties cover: (1) output is ordered, (2) no elements added or dropped, (3) sorting twice equals sorting once. Together these nearly pin down sort. An impostor like \"sort by discarding odd numbers\" fails property 2; \"reverse instead of sort\" fails 1; \"shuffle\" fails 1 and 3. This is the shape of a good property-based test suite: small properties that together constrain behavior tightly.",
)

add(
    "You are a property-based testing expert.",
    "How do I combine Hypothesis with pytest fixtures?",
    "Request fixtures on the test function; `@given` provides extra args.\n\n```python\nimport pytest\nfrom hypothesis import given, strategies as st\n\n@pytest.fixture\ndef store():\n    return {}\n\n@given(key=st.text(min_size=1, max_size=20), value=st.integers())\ndef test_store_set_get(store, key, value):\n    store[key] = value\n    assert store[key] == value\n```\n\nBut be careful: `store` is constructed *once* per pytest test function, but `@given` runs the body ~100 times. State leaks across iterations. If you need fresh state each Hypothesis iteration, construct it inside:\n\n```python\n@given(key=st.text(min_size=1), value=st.integers())\ndef test_store_clean(key, value):\n    store = {}  # fresh every iteration\n    store[key] = value\n    assert store[key] == value\n```\n\nOr use a function-scoped fixture that yields a factory:\n\n```python\n@pytest.fixture\ndef store_factory():\n    return lambda: {}\n\n@given(key=st.text(min_size=1), value=st.integers())\ndef test_store_factory(store_factory, key, value):\n    store = store_factory()\n    store[key] = value\n```\n\nHypothesis warns about fixture reuse since 6.x because the issue is so common.",
)

add(
    "You are a property-based testing expert.",
    "Write a Hypothesis test for a function that parses an integer from bytes.",
    "Round-trip test: encoded bytes should decode back to the original.\n\n```python\nimport struct\nfrom hypothesis import given, strategies as st\n\ndef encode_u32(n):\n    if n < 0 or n > 2**32 - 1:\n        raise ValueError(\"out of range\")\n    return struct.pack(\">I\", n)\n\ndef decode_u32(b):\n    if len(b) != 4:\n        raise ValueError(\"need 4 bytes\")\n    return struct.unpack(\">I\", b)[0]\n\n@given(st.integers(min_value=0, max_value=2**32 - 1))\ndef test_roundtrip(n):\n    assert decode_u32(encode_u32(n)) == n\n\n@given(st.binary(min_size=4, max_size=4))\ndef test_decode_of_random_bytes(b):\n    n = decode_u32(b)\n    assert 0 <= n < 2**32\n    assert encode_u32(n) == b\n\n@given(st.integers())\ndef test_bounds(n):\n    import pytest\n    if not (0 <= n < 2**32):\n        with pytest.raises(ValueError):\n            encode_u32(n)\n```\n\nThree complementary properties: encode->decode roundtrip, decode->encode roundtrip on random bytes, and out-of-range rejection. Together they catch off-by-one boundary bugs, endianness mistakes, and silent truncation.",
)

add(
    "You are a property-based testing expert.",
    "How do I use sampled_from and just strategies?",
    "`sampled_from` picks one of a fixed set; `just` always returns the same value.\n\n```python\nfrom hypothesis import given, strategies as st\n\nSTATUS = [\"pending\", \"active\", \"closed\", \"archived\"]\n\n@given(st.sampled_from(STATUS))\ndef test_status_is_valid(status):\n    assert status in STATUS\n\n@given(st.lists(st.sampled_from(STATUS), min_size=1, max_size=10))\ndef test_many_statuses(statuses):\n    assert all(s in STATUS for s in statuses)\n\n# just: a strategy that always yields one value; useful in one_of\n@given(st.one_of(st.just(None), st.integers()))\ndef test_maybe_int(x):\n    assert x is None or isinstance(x, int)\n\n# sampled_from for enums\nfrom enum import Enum\nclass Color(Enum):\n    RED = 1; GREEN = 2; BLUE = 3\n\n@given(st.sampled_from(list(Color)))\ndef test_color(c):\n    assert isinstance(c, Color)\n```\n\n`sampled_from` is ideal for closed categorical domains: enum values, HTTP methods, valid country codes. Hypothesis can also shrink toward earlier items in the list, so put \"simpler\" values first (e.g. `\"\"` before long strings) to get cleaner counterexamples.",
)

add(
    "You are a property-based testing expert.",
    "How do I generate valid email addresses, URLs, and UUIDs?",
    "Hypothesis ships high-quality strategies for common web formats.\n\n```python\nfrom hypothesis import given, strategies as st\nimport re\nimport uuid\n\n@given(st.emails())\ndef test_email_has_at(e):\n    assert \"@\" in e\n    assert e.count(\"@\") == 1\n\n@given(st.uuids())\ndef test_uuid_version(u):\n    assert isinstance(u, uuid.UUID)\n\n@given(st.ip_addresses(v=4))\ndef test_ipv4(addr):\n    parts = str(addr).split(\".\")\n    assert len(parts) == 4\n    assert all(0 <= int(p) <= 255 for p in parts)\n\n# URLs: from_regex is the reliable way\nURL = st.from_regex(\n    r\"^https?://[a-z]{3,10}\\.(com|net|org)(/[a-z]{0,10}){0,3}$\",\n    fullmatch=True,\n)\n\n@given(URL)\ndef test_url_scheme(u):\n    assert u.startswith((\"http://\", \"https://\"))\n```\n\n`emails()` produces RFC-compliant addresses; `uuids()` produces valid UUIDs; `ip_addresses(v=4|6)` gives `IPv4Address` / `IPv6Address` objects. For URL generation use `from_regex` with `fullmatch=True` so the entire string must match, or compose path/host manually with `builds(urlparse.urlunparse, ...)`.",
)

add(
    "You are a property-based testing expert.",
    "Find a bug in my_add using Hypothesis.",
    "Write a property any addition must satisfy, then run it.\n\n```python\nfrom hypothesis import given, strategies as st\n\ndef my_add(a, b):\n    # bug: special-cases 0 incorrectly\n    if b == 0:\n        return a + 1\n    return a + b\n\n@given(st.integers(), st.integers())\ndef test_commutative(a, b):\n    assert my_add(a, b) == my_add(b, a)\n\n@given(st.integers())\ndef test_identity(a):\n    assert my_add(a, 0) == a  # this fails, shrunk to a=0\n\n@given(st.integers(), st.integers(), st.integers())\ndef test_associative(a, b, c):\n    assert my_add(my_add(a, b), c) == my_add(a, my_add(b, c))\n```\n\nHypothesis reports `Falsifying example: test_identity(a=0)` because `my_add(0, 0) == 1`, not 0. The identity property caught it on the smallest possible input. The broader lesson: for math-ish functions, write the algebraic laws (identity, commutativity, associativity, distributivity) as properties. They constrain behavior better than any hand-picked examples because they must hold for *every* input, not just the cases a human thought to try.",
)

add(
    "You are a property-based testing expert.",
    "Generate valid JSON schemas for API tests with Hypothesis.",
    "Compose strategies that mirror your schema's structure.\n\n```python\nfrom hypothesis import given, strategies as st\n\n# API contract: POST /users expects {name: str, age: int>=0, tags: list[str]}\nuser_payload = st.fixed_dictionaries({\n    \"name\": st.text(min_size=1, max_size=50),\n    \"age\": st.integers(min_value=0, max_value=150),\n    \"tags\": st.lists(st.text(min_size=1, max_size=20),\n                     max_size=10, unique=True),\n})\n\n# optional fields via dictionaries with optional keys\nuser_with_optional = st.fixed_dictionaries(\n    mapping={\n        \"name\": st.text(min_size=1),\n        \"age\": st.integers(min_value=0),\n    },\n    optional={\n        \"email\": st.emails(),\n        \"bio\": st.text(max_size=500),\n    },\n)\n\n@given(user_payload)\ndef test_accepts_valid_payload(payload):\n    from myapp.validator import validate_user\n    assert validate_user(payload) is True\n\n@given(user_with_optional)\ndef test_accepts_with_optional(payload):\n    from myapp.validator import validate_user\n    assert validate_user(payload) is True\n    for k in payload:\n        assert k in {\"name\", \"age\", \"email\", \"bio\"}\n```\n\n`fixed_dictionaries` forces all keys; `optional=` lets Hypothesis sometimes include them. This mirrors OpenAPI `required` vs `optional` behavior exactly.",
)

add(
    "You are a property-based testing expert.",
    "How do I test numerical invariants with Hypothesis floats?",
    "Use careful float strategies and tolerance-based assertions.\n\n```python\nimport math\nfrom hypothesis import given, strategies as st, assume\n\ndef normalize(x, y):\n    n = math.hypot(x, y)\n    if n == 0:\n        return (0.0, 0.0)\n    return (x / n, y / n)\n\n@given(\n    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),\n    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),\n)\ndef test_unit_length(x, y):\n    assume(math.hypot(x, y) > 1e-9)  # skip degenerate zero vectors\n    nx, ny = normalize(x, y)\n    assert math.isclose(math.hypot(nx, ny), 1.0, abs_tol=1e-9)\n\n@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100))\ndef test_doubling(x):\n    assert math.isclose(x + x, 2 * x)\n```\n\nTips: always exclude NaN/Inf unless the function must handle them. Use `math.isclose` (or `numpy.testing.assert_allclose`) for float equality, never `==`. Bound magnitudes to avoid overflow or catastrophic cancellation when the property would not actually be meaningful at the extremes.",
)

add(
    "You are a property-based testing expert.",
    "How do I target specific edge cases with Hypothesis example decorator?",
    "Use `@example(...)` for regression tests and specific tricky inputs.\n\n```python\nfrom hypothesis import given, example, strategies as st\n\ndef reverse_words(s):\n    return \" \".join(s.split()[::-1])\n\n@given(st.text())\n@example(s=\"\")\n@example(s=\"   \")\n@example(s=\"hello world\")\n@example(s=\"\\tnewline\\nmixed  spaces\")\ndef test_reverse_words_roundtrip(s):\n    once = reverse_words(s)\n    twice = reverse_words(once)\n    # split() collapses whitespace, so we compare on the normalized form\n    assert reverse_words(twice) == once\n```\n\n`@example` decorators run *before* any random examples and are guaranteed to execute every time. Use them for:\n- Bugs you found in production (pin them to prevent regression)\n- Known edge cases (empty string, zero, None, single-element list)\n- Performance boundary cases (max allowed size)\n\nWhen Hypothesis shrinks a failure, it prints the counterexample in `@example`-ready form so you can paste it straight in.",
)

add(
    "You are a property-based testing expert.",
    "Use Hypothesis to test a cache implementation for correctness.",
    "Model it as a state machine and compare against a dict.\n\n```python\nfrom collections import OrderedDict\nfrom hypothesis import strategies as st\nfrom hypothesis.stateful import RuleBasedStateMachine, rule, invariant, precondition\n\nclass LRUCache:\n    def __init__(self, capacity):\n        self.cap = capacity\n        self._data = OrderedDict()\n    def get(self, k):\n        if k not in self._data: return None\n        self._data.move_to_end(k)\n        return self._data[k]\n    def put(self, k, v):\n        if k in self._data: self._data.move_to_end(k)\n        self._data[k] = v\n        if len(self._data) > self.cap:\n            self._data.popitem(last=False)\n\nclass LRUMachine(RuleBasedStateMachine):\n    def __init__(self):\n        super().__init__()\n        self.cap = 3\n        self.cache = LRUCache(self.cap)\n\n    @rule(k=st.integers(min_value=0, max_value=10),\n          v=st.integers())\n    def put(self, k, v):\n        self.cache.put(k, v)\n\n    @rule(k=st.integers(min_value=0, max_value=10))\n    def get(self, k):\n        self.cache.get(k)\n\n    @invariant()\n    def capacity(self):\n        assert len(self.cache._data) <= self.cap\n\nTestLRU = LRUMachine.TestCase\n```\n\nThe capacity invariant is checked after every rule. To test correctness, add a parallel reference (dict + a list tracking order) and assert equivalence on every `get`.",
)

add(
    "You are a property-based testing expert.",
    "Integrate Hypothesis into pytest CI with a tight time budget.",
    "Use profiles to balance thoroughness and CI wall-clock.\n\n```python\n# conftest.py\nimport os\nfrom hypothesis import settings, HealthCheck\n\nsettings.register_profile(\n    \"ci\",\n    max_examples=200,\n    deadline=1000,  # ms per example\n    suppress_health_check=[HealthCheck.too_slow],\n    derandomize=True,  # deterministic runs on CI\n)\nsettings.register_profile(\n    \"nightly\",\n    max_examples=5000,\n    deadline=None,\n)\nsettings.register_profile(\n    \"dev\",\n    max_examples=50,\n    deadline=None,\n)\n\nsettings.load_profile(os.environ.get(\"HYPOTHESIS_PROFILE\", \"dev\"))\n```\n\n```yaml\n# .github/workflows/test.yml excerpt\n- name: Run tests\n  env:\n    HYPOTHESIS_PROFILE: ci\n  run: pytest --hypothesis-show-statistics\n\n- name: Nightly thorough run\n  if: github.event.schedule != ''\n  env:\n    HYPOTHESIS_PROFILE: nightly\n  run: pytest\n```\n\n`derandomize=True` uses a fixed seed so CI failures reproduce locally. `--hypothesis-show-statistics` prints per-test example counts and shrink stats, helpful for spotting slow strategies. Commit the `.hypothesis/` database to cache seen failures across runs.",
)


Path("/tmp/gen_block_3.json").write_text(json.dumps(samples))
print(f"Block 3: {len(samples)}")
