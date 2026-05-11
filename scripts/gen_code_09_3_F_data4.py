# SPDX-License-Identifier: MIT
"""Data science part 2 (50) + Debug part 1 (50). Wave 6 part 4."""


def register_all(add):
    # ---- 50 more data science entries ----
    add("data",
        "Compute the mean and std along a numpy axis.",
        "```python\nimport numpy as np\n\nA = np.arange(20).reshape(4, 5).astype(float)\nprint(A.mean(axis=0), A.std(axis=0, ddof=1))\n```\nUse `ddof=1` for sample (unbiased) standard deviation; default is population.")

    add("data",
        "Sort a DataFrame by index.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'v': [1,2,3]}, index=[3, 1, 2])\nprint(df.sort_index())\n```\nMonotonic indexes enable fast `.loc` slicing -- pandas warns when you slice an unsorted index.")

    add("data",
        "Forward-fill missing values.",
        "```python\nimport pandas as pd\n\ns = pd.Series([1, None, None, 4])\nprint(s.ffill())\n```\nUseful for time series where last-known-value is the right imputation. Use `bfill` for backward-fill.")

    add("data",
        "Compute the mode of a column.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'cat': ['a', 'b', 'a', 'c']})\nprint(df['cat'].mode().iloc[0])\n```\n`.mode()` returns a Series because there can be ties; take `.iloc[0]` for the first.")

    add("data",
        "Calculate quantile bins for a numeric column.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'val': range(20)})\ndf['quartile'] = pd.qcut(df['val'], q=4, labels=['Q1','Q2','Q3','Q4'])\nprint(df.head())\n```\n`qcut` produces equal-frequency bins; `cut` produces equal-width bins. Choose based on whether the distribution is skewed.")

    add("data",
        "Convert a column to datetime.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'ts': ['2024-01-01', '2024-02-15']})\ndf['ts'] = pd.to_datetime(df['ts'])\nprint(df.dtypes)\n```\nPass `format=` for known formats -- 10-100x faster than format inference.")

    add("data",
        "Extract year and month from a datetime column.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'ts': pd.date_range('2024-01-01', periods=3, freq='ME')})\ndf['year'] = df['ts'].dt.year\ndf['month'] = df['ts'].dt.month\nprint(df)\n```\n`.dt` accessor exposes datetime fields without manual `apply`.")

    add("data",
        "Compute year-over-year change.",
        "```python\nimport pandas as pd\n\ns = pd.Series([100, 110, 120, 130, 140, 150], index=pd.date_range('2024-01-01', periods=6, freq='ME'))\nprint(s.pct_change(periods=12))  # YoY for monthly data\n```\n`pct_change(periods=N)` compares each value to N-back. Useful for seasonality-aware deltas.")

    add("data",
        "Build a frequency table.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'cat': ['a','b','a','c','a','b']})\nprint(df['cat'].value_counts())\nprint(df['cat'].value_counts(normalize=True))  # proportions\n```\n`normalize=True` gives proportions instead of counts -- usually what you want for reporting.")

    add("data",
        "Cross-tabulate two categorical columns.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'a': ['x','y','x','y'], 'b': ['p','p','q','q']})\nprint(pd.crosstab(df['a'], df['b']))\n```\nUse `normalize='index'` for row proportions, `'columns'` for column proportions, `'all'` for joint.")

    add("data",
        "Compute Spearman correlation.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'a': [1,2,3,4,5], 'b': [1,4,9,16,25]})\nprint(df.corr(method='spearman'))\n```\nSpearman is rank-based, robust to monotonic non-linearity. For these inputs, Pearson would be 0.99, Spearman is 1.0.")

    add("data",
        "Build a time-indexed DataFrame from raw timestamps.",
        "```python\nimport pandas as pd\n\ntimestamps = ['2024-01-01 10:00', '2024-01-02 11:00', '2024-01-03 12:00']\nvalues = [10, 20, 30]\ndf = pd.DataFrame({'val': values}, index=pd.to_datetime(timestamps))\ndf.index.name = 'ts'\nprint(df)\n```\nNamed index makes `df.reset_index()` produce a usable column.")

    add("data",
        "Find the max value's index.",
        "```python\nimport pandas as pd\n\ns = pd.Series([3, 1, 4, 1, 5, 9, 2, 6])\nprint(s.idxmax())  # 5\n```\n`idxmax`/`idxmin` return the label, not the position. Use `argmax`/`argmin` for positional.")

    add("data",
        "Compute moving average with custom weights.",
        "```python\nimport numpy as np\nimport pandas as pd\n\ns = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])\nweights = np.array([0.1, 0.3, 0.6])\nprint(s.rolling(3).apply(lambda x: np.dot(x, weights), raw=True))\n```\n`raw=True` passes a numpy array to the function -- much faster than the default Series.")

    add("data",
        "Concat multiple DataFrames vertically.",
        "```python\nimport pandas as pd\n\na = pd.DataFrame({'x': [1,2]})\nb = pd.DataFrame({'x': [3,4]})\nprint(pd.concat([a, b], ignore_index=True))\n```\n`ignore_index=True` resets the index. Without it, you get duplicate index values which break later operations.")

    add("data",
        "Compute pairwise correlation between specific columns.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6], 'c': [7,8,9]})\nprint(df[['a','b']].corr())\n```\nSelect first to avoid computing the entire matrix when you only need one pair.")

    add("data",
        "Plot a boxplot per category.",
        "```python\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nrng = np.random.default_rng(0)\ndf = pd.DataFrame({'val': rng.normal(size=300), 'cat': rng.choice(['a','b','c'], 300)})\nfig, ax = plt.subplots()\ndf.boxplot(by='cat', column='val', ax=ax)\nfig.savefig('box.png')\n```\nUse seaborn's `boxplot` for prettier output and built-in hue support.")

    add("data",
        "Compute a value's z-score against a reference distribution.",
        "```python\nimport numpy as np\n\nref = np.array([10, 12, 14, 11, 13, 15, 12, 13])\nval = 20\nz = (val - ref.mean()) / ref.std(ddof=1)\nprint(f'z={z:.2f}')\n```\n|z| > 3 is roughly the conventional outlier threshold for normal data.")

    add("data",
        "Generate a date range with business-day frequency.",
        "```python\nimport pandas as pd\n\ndates = pd.bdate_range('2024-01-01', '2024-01-31')\nprint(len(dates))\n```\n`bdate_range` skips weekends. For exchange holidays, use `pandas.tseries.offsets.CustomBusinessDay` with a holiday calendar.")

    add("data",
        "Compute weighted average with numpy.",
        "```python\nimport numpy as np\n\nvalues = np.array([85, 90, 95])\nweights = np.array([0.3, 0.5, 0.2])\nprint(np.average(values, weights=weights))\n```\n`np.average` accepts weights; `np.mean` does not. Different functions, easy to confuse.")

    add("data",
        "Read multiple CSVs and concat.",
        "```python\nfrom pathlib import Path\nimport pandas as pd\n\nfiles = sorted(Path('data/').glob('*.csv'))\ndfs = [pd.read_csv(f) for f in files]\ndf = pd.concat(dfs, ignore_index=True)\n```\nFor very large file sets, switch to `pyarrow.dataset` -- it scans without materializing everything in memory.")

    add("data",
        "Drop duplicate rows.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'a': [1,1,2], 'b': [10, 10, 20]})\nprint(df.drop_duplicates())\nprint(df.drop_duplicates(subset=['a'], keep='last'))\n```\n`subset=` restricts comparison columns; `keep='last'` retains the last occurrence (default 'first').")

    add("data",
        "Use isin for filtering.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'cat': ['a','b','c','d']})\nselected = df[df['cat'].isin(['a','c'])]\nprint(selected)\n```\nO(1) hash-based membership; far faster than chained `or` for many values.")

    add("data",
        "Update DataFrame values conditionally.",
        "```python\nimport pandas as pd\nimport numpy as np\n\ndf = pd.DataFrame({'score': [50, 75, 90, 30]})\ndf['grade'] = np.where(df['score'] >= 60, 'pass', 'fail')\nprint(df)\n```\n`np.where` is the vectorized ternary. For more than two cases, use `np.select`.")

    add("data",
        "Use np.select for multi-branch column logic.",
        "```python\nimport pandas as pd\nimport numpy as np\n\ndf = pd.DataFrame({'score': [95, 85, 70, 50]})\nconds = [df['score'] >= 90, df['score'] >= 80, df['score'] >= 70]\nchoices = ['A', 'B', 'C']\ndf['grade'] = np.select(conds, choices, default='F')\nprint(df)\n```\n`np.select` evaluates each condition in order; first match wins.")

    add("data",
        "Save a matplotlib figure as SVG.",
        "```python\nimport matplotlib.pyplot as plt\n\nfig, ax = plt.subplots()\nax.plot([1, 2, 3], [1, 4, 9])\nfig.savefig('plot.svg', format='svg', bbox_inches='tight')\n```\nSVG scales infinitely without losing quality -- ideal for documentation, papers, websites.")

    add("data",
        "Compute the rank of values.",
        "```python\nimport pandas as pd\n\ns = pd.Series([10, 20, 20, 30, 40])\nprint(s.rank(method='dense'))  # ties get the same rank, no gap\nprint(s.rank(method='min'))    # ties get the minimum rank, gap after\n```\n`method='average'` (default) splits ties; `'dense'` and `'min'` are usually more useful for reporting.")

    add("data",
        "Compute moving sum with explicit window.",
        "```python\nimport pandas as pd\n\ns = pd.Series(range(10))\nprint(s.rolling(window=3, min_periods=1).sum())\n```\n`min_periods=1` returns partial-window values rather than NaNs at the start.")

    add("data",
        "Compute time difference between consecutive rows.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'ts': pd.to_datetime(['2024-01-01 10:00', '2024-01-01 10:05', '2024-01-01 10:15'])})\ndf['delta'] = df['ts'].diff().dt.total_seconds()\nprint(df)\n```\n`.dt.total_seconds()` gives float seconds. For minutes, divide by 60.")

    add("data",
        "Apply a row-level function to a DataFrame.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})\ndf['ratio'] = df.apply(lambda r: r['a'] / r['b'], axis=1)\nprint(df)\n```\nAvoid `apply` when vectorization is possible -- it's 10-100x slower. Here `df['a']/df['b']` is the right answer.")

    add("data",
        "Sort then take top N rows.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'val': [3, 1, 4, 1, 5, 9, 2, 6]})\nprint(df.nlargest(3, 'val'))\n```\n`nlargest`/`nsmallest` are O(n log k) -- faster than sorting the entire DataFrame.")

    add("data",
        "Compute cumulative sum per group.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'cat': ['a','a','b','b'], 'val': [1,2,3,4]})\ndf['cum'] = df.groupby('cat')['val'].cumsum()\nprint(df)\n```\nGroup-wise transforms produce a Series aligned with the original index -- assignable directly.")

    add("data",
        "Use .loc with boolean masks.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'a': [1,2,3,4]})\ndf.loc[df['a'] > 2, 'a'] = 99\nprint(df)\n```\n`.loc[mask, col] = value` modifies in place. Avoids the SettingWithCopyWarning that arises from chained indexing.")

    add("data",
        "Profile a numpy operation with timeit.",
        "```python\nimport timeit, numpy as np\n\nx = np.arange(1_000_000)\nt = timeit.timeit('x.sum()', globals={'x': x}, number=100)\nprint(f'{t/100*1000:.2f}ms per call')\n```\nUse `timeit` not `time.time` for short ops -- it warms up and averages, smoothing out OS jitter.")

    add("data",
        "Read SQL into pandas.",
        "```python\nimport pandas as pd\nfrom sqlalchemy import create_engine\n\nengine = create_engine('sqlite:///data.db')\ndf = pd.read_sql('SELECT * FROM users WHERE active=1', engine)\n```\nPass the engine, not a raw connection -- pandas handles cleanup correctly. For huge results, use `chunksize=`.")

    add("data",
        "Aggregate with multiple metrics and rename.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'cat': ['a','a','b','b'], 'val': [1,2,3,4]})\nout = df.groupby('cat').agg(\n    count=('val','size'),\n    total=('val','sum'),\n    avg=('val','mean'),\n)\nprint(out)\n```\nNamed aggregation avoids the awkward MultiIndex from passing a dict.")

    add("data",
        "Visualize a confusion matrix.",
        "```python\nfrom sklearn.metrics import ConfusionMatrixDisplay\nimport matplotlib.pyplot as plt\n\ny_true = [0,1,1,0,1,0]\ny_pred = [0,1,0,0,1,1]\nfig, ax = plt.subplots()\nConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)\nfig.savefig('cm.png')\n```\nFor imbalanced classes pass `normalize='true'` to see proportions instead of raw counts.")

    add("data",
        "Calculate IQR and identify outliers.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'val': [10, 12, 14, 11, 13, 15, 200]})\nq1, q3 = df['val'].quantile([0.25, 0.75])\niqr = q3 - q1\nlo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr\noutliers = df[(df['val'] < lo) | (df['val'] > hi)]\nprint(outliers)\n```\nClassic Tukey fences; robust to heavy tails.")

    add("data",
        "Compute log-transform on skewed data.",
        "```python\nimport numpy as np\nimport pandas as pd\n\ndf = pd.DataFrame({'income': [10000, 50000, 200000, 1_000_000]})\ndf['log_income'] = np.log1p(df['income'])\nprint(df)\n```\n`log1p` is `log(1+x)`, which handles zero values without `-inf`. Standard preprocessing for income/price.")

    add("data",
        "Plot a stacked bar chart.",
        "```python\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\ndf = pd.DataFrame({'a': [3, 5, 7], 'b': [2, 4, 6]}, index=['x','y','z'])\nax = df.plot(kind='bar', stacked=True)\nax.figure.savefig('stacked.png')\n```\nFor visualizations with negative values, stacked bars get confusing -- use grouped bars instead.")

    add("data",
        "Compute the dot product as a percentage of total.",
        "```python\nimport numpy as np\n\nweights = np.array([0.5, 0.3, 0.2])\nvalues = np.array([100, 80, 50])\nweighted_avg = np.dot(weights, values)\nprint(f'{weighted_avg:.2f}')  # 86.0\n```\nWhen weights sum to 1, the dot product IS the weighted average.")

    add("data",
        "Build a sparse matrix.",
        "```python\nfrom scipy.sparse import csr_matrix\nimport numpy as np\n\nrows = [0, 0, 1, 2]\ncols = [0, 2, 1, 0]\ndata = [1, 2, 3, 4]\nm = csr_matrix((data, (rows, cols)), shape=(3, 3))\nprint(m.toarray())\n```\nCSR for fast row slicing and matrix-vector products. CSC for column ops.")

    add("data",
        "Use seaborn's pairplot.",
        "```python\nimport seaborn as sns\nimport matplotlib.pyplot as plt\nfrom sklearn.datasets import load_iris\nimport pandas as pd\n\niris = load_iris(as_frame=True)\ndf = iris.data.assign(target=iris.target)\ng = sns.pairplot(df, hue='target')\ng.savefig('pair.png')\n```\nFantastic for first-look EDA; expensive on >20 columns -- restrict the column list.")

    add("data",
        "Generate a sample DataFrame for testing.",
        "```python\nimport pandas as pd\nimport numpy as np\n\nrng = np.random.default_rng(0)\ndf = pd.DataFrame({\n    'id': range(100),\n    'group': rng.choice(['A','B','C'], 100),\n    'value': rng.normal(0, 1, 100),\n    'date': pd.date_range('2024-01-01', periods=100, freq='D'),\n})\nprint(df.head())\n```\nFixed seed via `default_rng(0)` makes the output reproducible across machines.")

    add("data",
        "Pivot to compute monthly totals per category.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({\n    'date': pd.date_range('2024-01-01', periods=12, freq='ME'),\n    'cat': ['a','b']*6,\n    'val': range(12),\n})\nout = df.pivot_table(index=df['date'].dt.month, columns='cat', values='val', aggfunc='sum')\nprint(out)\n```\n`pivot_table` handles duplicate (index, column) pairs via `aggfunc=`; use it when you can't guarantee uniqueness.")

    add("data",
        "Compute moving median.",
        "```python\nimport pandas as pd\n\ns = pd.Series([1, 5, 2, 8, 3, 7, 4, 6])\nprint(s.rolling(3).median())\n```\nMedian is robust to outliers -- preferable to mean for noisy series with occasional spikes.")

    add("data",
        "Compute percentage change with handling for zeros.",
        "```python\nimport numpy as np\nimport pandas as pd\n\ns = pd.Series([100, 0, 50, 0, 75])\npct = s.pct_change()\npct[s.shift() == 0] = np.nan  # avoid inf\nprint(pct)\n```\nDefault `pct_change` returns `inf` when the previous value is zero -- mask those explicitly.")

    add("data",
        "Plot multiple time series on shared axis.",
        "```python\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\ndf = pd.DataFrame({\n    'a': [1, 2, 3, 4],\n    'b': [4, 3, 2, 1],\n}, index=pd.date_range('2024-01-01', periods=4))\nfig, ax = plt.subplots()\ndf.plot(ax=ax)\nax.set_ylabel('value')\nfig.savefig('lines.png')\n```\n`df.plot` with multiple columns auto-legends; pass `subplots=True` for separate panels per column.")

    add("data",
        "Compute simple linear trend with numpy polyfit.",
        "```python\nimport numpy as np\nimport pandas as pd\n\ny = np.array([1.1, 2.0, 2.9, 4.1, 5.0, 5.9])\nx = np.arange(len(y))\nslope, intercept = np.polyfit(x, y, deg=1)\nprint(f'slope={slope:.2f}, intercept={intercept:.2f}')\n```\nFor proper inference (R^2, p-values, confidence intervals), use `statsmodels.OLS` instead.")

    add("data",
        "Use pandas categorical dtype for memory efficiency.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'country': ['US','UK','US','FR']*1000})\nprint(df['country'].memory_usage(deep=True))\ndf['country'] = df['country'].astype('category')\nprint(df['country'].memory_usage(deep=True))\n```\nCategorical stores codes + categories, dramatically smaller for repeated string values. Read CSVs with low-cardinality string columns and convert to category for free memory savings.")

    # ---- 50 debug entries ----
    add("debug",
        "Why does this code raise IndexError when the list is empty?",
        "Likely you're indexing `xs[0]` (or `xs[-1]`) without checking `if xs`. Either guard with `if not xs: return ...` or use `next(iter(xs), default)` for the first element. The empty case is common for filtered results; always handle it explicitly.")

    add("debug",
        "Help me debug why this regex doesn't match what I expect.",
        "Three frequent culprits: (1) you forgot to escape a metacharacter (`.`, `(`, `[`); (2) the regex is greedy when you wanted lazy (`.*` vs `.*?`); (3) you're matching against a multiline string without `re.MULTILINE` and `^`/`$` aren't behaving as you expect. Test on regex101.com -- it explains every token.")

    add("debug",
        "Why does this list comprehension produce duplicates?",
        "Probably the source iterable contains duplicates. Comprehensions don't dedupe -- they're a 1:1 transformation. Wrap with `set(...)` if order doesn't matter, or use `dict.fromkeys(...)` to dedupe while preserving insertion order.")

    add("debug",
        "I get TypeError: unhashable type: 'list'. What's wrong?",
        "You're using a list as a dict key or set element. Lists are mutable and not hashable -- convert to a tuple: `dict[(1,2,3)]`. Same applies to dicts and sets as keys.")

    add("debug",
        "Why does my function modify the default argument across calls?",
        "Mutable default arguments (`def f(x=[])`) are evaluated once at function definition and shared across calls. Use `def f(x=None): x = x if x is not None else []` instead. This is the most-cited Python footgun for a reason.")

    add("debug",
        "My pandas operation triggers SettingWithCopyWarning. How do I fix it?",
        "You're chaining indexers: `df[df.x > 0]['y'] = ...`. The intermediate slice may be a view or a copy -- pandas can't tell, so it warns. Use `.loc`: `df.loc[df.x > 0, 'y'] = ...`. The single `.loc` is unambiguous and is the canonical fix.")

    add("debug",
        "I'm getting RecursionError. What should I do?",
        "Either (1) the recursion is unbounded -- check the base case fires; (2) the input is too deep -- rewrite iteratively with an explicit stack; or (3) you legitimately need depth -- `sys.setrecursionlimit(10000)`, but the iterative form is almost always better.")

    add("debug",
        "Why does requests.get hang forever?",
        "No timeout set; the default is 'forever'. Always pass `timeout=10` (or whatever's reasonable). For finer control: `timeout=(connect, read)` separates connection from read timeouts. Add a global default via a Session if you make many calls.")

    add("debug",
        "Help me fix this datetime comparison that fails with TypeError.",
        "Mixing naive and aware datetimes: `datetime.now()` is naive, `datetime.now(timezone.utc)` is aware. Comparing them raises. Pick one (almost always aware UTC) and stick with it across the codebase. Audit every `datetime` call to remove naive ones.")

    add("debug",
        "Why does this multiprocessing code never finish?",
        "Common pinning point: the worker raised an unpickled exception that the main process never sees. Wrap workers in try/except that returns the exception, then re-raise in the main. Also: ensure the entry point is under `if __name__ == '__main__':` on Windows/macOS spawn.")

    add("debug",
        "My async function returns None instead of the expected result.",
        "You probably forgot `await` -- calling `coro()` returns a coroutine object, not its result. The lint rule `RUF006` (or `B008`-adjacent) catches some of these. Type checking with strict mode flags it consistently.")

    add("debug",
        "Why does my SQLAlchemy query return stale data?",
        "You're inside a long-lived session and another process committed changes. Call `session.expire_all()` or `session.refresh(obj)` to force a reload. Or close the session and open a fresh one for each unit of work -- the recommended pattern.")

    add("debug",
        "I get UnicodeDecodeError when reading a file. What's happening?",
        "The file isn't UTF-8 (probably latin-1 or Windows-1252). Open with the explicit encoding: `open(path, encoding='cp1252')`. If you must accept arbitrary encodings, use `chardet` to guess, then re-open.")

    add("debug",
        "My Flask app's session keeps getting reset. Why?",
        "Most likely `SECRET_KEY` is unset or different across processes. Sessions are signed with this key; restart with a different one and old sessions become invalid. Set it via env var with a long random value and persist it across deploys.")

    add("debug",
        "Why does my generator stop after one iteration?",
        "Generators are exhausted after a single pass. If you need multiple iterations, materialize once with `list(gen)` and iterate the list, or call the generator factory each time. There's no rewind.")

    add("debug",
        "Why is my numpy result a different dtype than I expected?",
        "Mixed-dtype operations follow promotion rules: int+float gives float, int8+int32 gives int32. `np.result_type(a, b)` shows what the result will be. Cast early with `.astype(np.float32)` if you want a specific type.")

    add("debug",
        "Help debug this slow pandas apply.",
        "`apply(axis=1)` runs Python per row -- often 100x slower than the equivalent vectorized operation. Try expressing the logic with column-wise operators (`+`, `*`, `np.where`, `np.select`). If absolutely necessary, `pyjanitor` and numba's `@vectorize` are faster than `apply`.")

    add("debug",
        "Why does my Click CLI not see my new command?",
        "You added a function but didn't register it: `cli.add_command(myfunc)` or use the `@cli.command()` decorator. Click does no magic discovery. Run `python -m mypkg --help` to see what's registered.")

    add("debug",
        "My pytest fixture is being called multiple times unexpectedly.",
        "Fixture scope mismatch. Default is `function` -- one per test. Use `scope='module'` or `'session'` for expensive setup. Beware: session-scoped fixtures must not mutate shared state, or tests leak.")

    add("debug",
        "Why does my regex catastrophically backtrack and hang?",
        "Patterns like `(a+)+b` on `aaaaaaa` cause exponential blowup. Replace nested quantifiers with a single one (`a+b`), or switch to a regex engine that supports atomic groups / possessive quantifiers (Python's re doesn't, but `regex` package does). Validate user-supplied patterns or set a timeout.")

    add("debug",
        "Help me figure out why my asyncio task isn't running.",
        "You probably created a coroutine without scheduling it. `loop.create_task(coro)` or `asyncio.create_task(coro)` schedules it. Just calling `coro()` makes a coroutine object that does nothing until awaited or scheduled.")

    add("debug",
        "Why does my unit test pass alone but fail in the suite?",
        "Test isolation problem: a previous test mutates global state (a module-level cache, env var, monkeypatched function, sys.path entry, DB row). Run with `pytest --random-order` to expose ordering bugs. Use fixtures with proper teardown to fix.")

    add("debug",
        "Help me fix a memory leak in my long-running script.",
        "Profile with `tracemalloc.start(); ... ; tracemalloc.take_snapshot()`. Top allocations point at the leak. Common culprits: caches without bounds (use `lru_cache(maxsize=N)` not `None`), captured large closures, unclosed file handles. `objgraph` visualizes reference chains.")

    add("debug",
        "Why does my pandas merge return more rows than expected?",
        "One side has duplicate keys, multiplying matches. Pre-validate: `df.duplicated('key').any()`. Use `validate='one_to_many'` or `'one_to_one'` to make pandas raise if expectations are violated. Catches the bug at merge time, not three steps downstream.")

    add("debug",
        "My script gets KeyError on a dict that should have the key.",
        "Three usual suspects: (1) typo in the key (case sensitivity!); (2) the key was deleted earlier; (3) keys are different types (`'1'` vs `1`). Print the actual `dict.keys()` and `repr(target)` to compare. `.get(key)` returns None instead of raising for safer access.")

    add("debug",
        "Why is my JSON serialization failing on a datetime?",
        "Standard `json` doesn't know how to serialize datetimes. Either pre-convert with `obj.isoformat()`, pass `default=str` to `json.dumps`, or use a library like `orjson` that handles datetimes natively. For Pydantic models, use `model.model_dump_json()`.")

    add("debug",
        "Help me debug a 500 error with no traceback.",
        "Check stderr/log capture -- the framework may have swallowed it. Add `logging.basicConfig(level=logging.DEBUG)` and a global exception handler that logs full tracebacks. In production, route to Sentry or Honeybadger so you don't ssh-grep logs.")

    add("debug",
        "Why does my SQLAlchemy column comparison return no rows?",
        "You may have SQL-string-formatted: `where(User.email == f\"%{x}%\")`. Use `User.email.like(f'%{x}%')` for LIKE, or `func.lower(User.email) == x.lower()` for case-insensitive equality. Print the compiled SQL with `print(stmt.compile(compile_kwargs={'literal_binds': True}))` to see what's actually executed.")

    add("debug",
        "My type hints fail mypy with 'incompatible types in assignment'.",
        "Most often you've narrowed a type in one branch but mypy can't follow the flow. Use `assert isinstance(x, T)` to widen mypy's understanding, or refactor to a single assignment. `cast(T, x)` is the escape hatch when you know better than the checker.")

    add("debug",
        "Help me fix this off-by-one error in a loop.",
        "Print the index alongside the value to see exactly which iteration is wrong. Common shapes: `range(len(xs)-1)` when you want all of them; `xs[i+1]` without bounds-checking the last iteration; sliding window of size k starting at `range(len(xs)-k+1)`. Use slicing or `zip` to avoid manual indexing where possible.")

    add("debug",
        "Why does my requests.post send empty body?",
        "Probably you used `data=payload` with a dict thinking it would be JSON. `data=` form-encodes; `json=` JSON-encodes and sets the right header. The most-confused parameter pair in the requests API.")

    add("debug",
        "My logging.info() doesn't print anything.",
        "Default level is WARNING. Call `logging.basicConfig(level=logging.INFO)` once at startup. If you've created a named logger (`logging.getLogger('myapp')`), set its level too -- child loggers inherit but can be overridden.")

    add("debug",
        "Why does my CI pipeline pass locally but fail on the server?",
        "Environment drift: different Python version, different deps versions, different OS, missing env var, no terminal (so `input()` hangs), different timezone. Pin versions in `pyproject.toml`/`requirements.txt`, run CI image locally with Docker to reproduce, and never read interactively in test code.")

    add("debug",
        "I get 'maximum recursion depth exceeded' on JSON deserialization.",
        "Cycle in the data, or extremely deep nesting. Python's default limit is 1000. For deep-but-finite data, `sys.setrecursionlimit(10000)` works. For cycles, walk iteratively with a `seen` set to detect repeated references.")

    add("debug",
        "Why does my flask-sqlalchemy query not commit?",
        "You forgot `db.session.commit()`. Just calling `db.session.add(obj)` schedules but doesn't persist. Wrap multi-step writes in try/except with rollback to keep the session clean on error.")

    add("debug",
        "My pytest marks aren't being respected.",
        "Did you register them in `pyproject.toml` under `[tool.pytest.ini_options] markers = [\"slow\"]`? Strict mode (`--strict-markers`) errors on unregistered marks; that's the right setting. Run `pytest --markers` to see what's known.")

    add("debug",
        "Why does my FastAPI request return 422 for what looks like valid input?",
        "Pydantic validation failed somewhere. The 422 body contains a precise list of validation errors -- read it. Most often: missing field, wrong type (string '1' vs int 1), or constraint violation (length, pattern). Echo the body with curl to inspect.")

    add("debug",
        "Help me debug a stuck thread in my application.",
        "`py-spy dump --pid <pid>` shows the live stack trace of every thread. Look for sockets waiting without timeout, locks held by the wrong thread, or queue.get() with no producer. Adding a per-call timeout almost always replaces a hang with a useful error.")

    add("debug",
        "Why does my dictionary lose ordering after deepcopy?",
        "It shouldn't -- since Python 3.7 dicts preserve insertion order through copy and deepcopy. If you're seeing unordered output, you're probably iterating a `set`, or printing a dict via something that reorders (some YAML dumpers). Check the actual data type with `print(type(x))`.")

    add("debug",
        "I get 'No module named X' on import. What do I check?",
        "Verify (1) it's installed: `pip show X`; (2) you're in the right venv: `which python`; (3) module name matches package name (it often doesn't -- `pip install python-dateutil` provides `dateutil`); (4) `sys.path` is sane (a stray `__init__.py` can shadow).")

    add("debug",
        "Why does my unit test print output not show in the console?",
        "pytest captures stdout by default. Run with `-s` (or `--capture=no`) to disable. Better: use `caplog` for log assertions and `capsys` to inspect captured output.")

    add("debug",
        "Help me find what's holding open a file handle.",
        "On Linux, `lsof -p <pid> | grep <filename>`. In Python, the warnings module helps -- `python -W error::ResourceWarning` raises on unclosed files. Always use `with open(...)` rather than bare `open(...)` to guarantee close.")

    add("debug",
        "Why does my pickle file fail to load on another machine?",
        "Pickle loads import the original module path. If the class moved or the version of the library changed, unpickling fails. Use a portable format (JSON, Parquet, MsgPack) for anything you'll deserialize elsewhere. Pickle is for ephemeral interprocess transport, not durable storage.")

    add("debug",
        "My assertion error has no message. How do I make pytest more helpful?",
        "Just use `assert x == y, f'expected {y}, got {x}'`. Pytest's `--tb=long` and assert rewriting already show both sides for plain `assert x == y`, but custom messages add semantic context. For complex objects, `pytest --tb=auto` plus `pytest-clarity` for diffs.")

    add("debug",
        "Why does my numpy comparison return ambiguous truth?",
        "`if arr == 0:` raises 'truth value ambiguous'. Numpy comparisons return arrays; you need `(arr == 0).all()`, `.any()`, or boolean indexing. Most idiomatic: `if (arr == 0).all():`.")

    add("debug",
        "Help me fix this off-by-one in date range.",
        "`pd.date_range(start, end)` is **inclusive** on both sides; numpy's `arange` is exclusive on the right. If you wanted N days starting from `start`, use `periods=N` instead of computing the end by hand. Avoids the manual ±1 bookkeeping.")

    add("debug",
        "Why does my regex return overlapping matches?",
        "It doesn't -- by default Python's `re.findall` returns non-overlapping. To get overlapping matches, use a lookahead trick: `re.findall(r'(?=(pattern))', s)`. Captures the position; consume nothing, so the engine moves one char and tries again.")

    add("debug",
        "My script's exit code is always 0 even when it errored.",
        "You probably caught `Exception` somewhere and didn't re-raise or `sys.exit(1)`. CI relies on exit codes; if your code says 'success' on failure, every CI run reports green. Either let the exception propagate, or call `sys.exit(1)` explicitly in the catch.")

    add("debug",
        "Why does this ValueError reference a different line than the actual error?",
        "Most likely you're catching the exception and re-raising at a different point. `raise from e` preserves the chain; `raise e` discards the original traceback. Use `raise` (no expression) inside an `except` to re-raise unmodified.")

    add("debug",
        "Help me debug this slow startup time of my Python script.",
        "`python -X importtime myscript.py 2> import.log` logs the time each import takes. Sort by self-time to find the slow ones. Common offenders: pandas (1-2s), tensorflow (3-10s), heavy plugin systems. Move slow imports inside functions if not always used.")
