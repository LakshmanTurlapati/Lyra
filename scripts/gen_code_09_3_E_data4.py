# SPDX-License-Identifier: MIT
"""Data science (75) + debugging (25). Wave 5 part 4."""


def register_all(add):
    # ----- data science (75) -----
    add("data",
        "How do I drop duplicate rows in a pandas DataFrame keeping the latest by date?",
        "```python\nimport pandas as pd\n\ndedup = (\n    df.sort_values('date')\n      .drop_duplicates(subset=['user_id'], keep='last')\n      .reset_index(drop=True)\n)\n```\nSort first, then `drop_duplicates(keep='last')` -- the result is deterministic and uses pandas' fast C path.")

    add("data",
        "Write code that computes weekly active users from an events table.",
        "```python\nimport pandas as pd\n\nevents = pd.read_csv('events.csv', parse_dates=['ts'])\nweekly = (\n    events.assign(week=events['ts'].dt.to_period('W-SUN'))\n          .groupby('week')['user_id'].nunique()\n          .rename('wau')\n)\n```\n`Period('W-SUN')` aligns weeks to a Sunday close; pick whichever Monday/Sunday convention your team agreed on.")

    add("data",
        "How do I do a left anti-join in pandas?",
        "```python\nimport pandas as pd\n\nmerged = a.merge(b, on='id', how='left', indicator=True)\nleft_only = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])\n```\nThe `indicator=True` column tells you which side of the join each row came from; perfect for set-difference style operations.")

    add("data",
        "Write a numpy snippet that normalises a vector to unit length.",
        "```python\nimport numpy as np\n\ndef normalize(v: np.ndarray) -> np.ndarray:\n    norm = np.linalg.norm(v)\n    if norm == 0:\n        return v\n    return v / norm\n```\nGuard the zero vector explicitly; otherwise you get a NaN array and a hard-to-trace bug downstream.")

    add("data",
        "How do I plot a heatmap of a correlation matrix?",
        "```python\nimport seaborn as sns\nimport matplotlib.pyplot as plt\n\nfig, ax = plt.subplots(figsize=(8, 6))\nsns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)\nfig.tight_layout()\n```\n`center=0` puts the diverging palette around uncorrelated; `annot=True` prints the values in cells.")

    add("data",
        "Write a function that bins a continuous column into quartiles.",
        "```python\nimport pandas as pd\n\ndef quartile_bin(s: pd.Series) -> pd.Series:\n    return pd.qcut(s, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])\n```\n`qcut` makes equal-frequency bins; use `cut` with explicit edges if you need fixed thresholds.")

    add("data",
        "How do I save a sklearn model to disk?",
        "```python\nimport joblib\nfrom sklearn.ensemble import RandomForestClassifier\n\nmodel = RandomForestClassifier().fit(X, y)\njoblib.dump(model, 'model.joblib')\nloaded = joblib.load('model.joblib')\n```\nVersion-control the training code, not the binary; pickled models are fragile across library versions.")

    add("data",
        "Write a pandas snippet that fills forward then back-fills remaining NaNs.",
        "```python\nimport pandas as pd\n\ndf = df.ffill().bfill()\n```\nGood for time-series gaps; if the column starts with NaN, `ffill` alone leaves it -- the chained `bfill` handles that.")

    add("data",
        "How do I plot a stacked bar chart in matplotlib?",
        "```python\nimport matplotlib.pyplot as plt\nimport numpy as np\n\ncats = ['A', 'B', 'C']\nseries = {'a': [3, 4, 5], 'b': [1, 2, 1]}\nbottom = np.zeros(len(cats))\nfig, ax = plt.subplots()\nfor name, vals in series.items():\n    ax.bar(cats, vals, label=name, bottom=bottom)\n    bottom += np.array(vals)\nax.legend()\nfig.tight_layout()\n```\nKeep a running `bottom` and add to it after each layer.")

    add("data",
        "Write a numpy function that reshapes a 1D array into rolling windows.",
        "```python\nimport numpy as np\nfrom numpy.lib.stride_tricks import sliding_window_view\n\ndef windows(a: np.ndarray, w: int) -> np.ndarray:\n    return sliding_window_view(a, window_shape=w)\n```\nReturns a view, not a copy -- O(1) and ideal for rolling features. Be careful: writes propagate to the original.")

    add("data",
        "How do I efficiently compute a histogram for a huge dataset?",
        "```python\nimport numpy as np\n\nedges = np.linspace(0, 100, 51)\ncounts = np.zeros(50, dtype=np.int64)\nfor chunk in pd.read_csv('big.csv', chunksize=1_000_000)['value']:\n    h, _ = np.histogram(chunk, bins=edges)\n    counts += h\n```\nProcess in chunks, accumulate the counts; never load the entire column.")

    add("data",
        "Write a snippet that computes the F1 score from precision and recall.",
        "```python\ndef f1(precision: float, recall: float) -> float:\n    if precision + recall == 0:\n        return 0.0\n    return 2 * precision * recall / (precision + recall)\n```\nThe guard avoids a ZeroDivisionError when both metrics are zero.")

    add("data",
        "How do I plot a time series with weekly tick labels?",
        "```python\nimport matplotlib.pyplot as plt\nimport matplotlib.dates as mdates\n\nfig, ax = plt.subplots()\nax.plot(df.index, df['value'])\nax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))\nax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))\nfig.autofmt_xdate()\n```\nUse `matplotlib.dates` for date-aware ticks; otherwise the locators stride over float pixel values.")

    add("data",
        "Write a pandas snippet that computes the Pareto top 80% by revenue.",
        "```python\nimport pandas as pd\n\nrev = df.groupby('customer')['revenue'].sum().sort_values(ascending=False)\ncum = rev.cumsum() / rev.sum()\ntop_pareto = rev[cum <= 0.8]\n```\nTrack the cumulative share; the boolean mask gives you the customers that account for the first 80%.")

    add("data",
        "How do I read JSON Lines into a DataFrame?",
        "```python\nimport pandas as pd\n\ndf = pd.read_json('events.jsonl', lines=True)\n```\nFor very large files use `pd.read_json(..., lines=True, chunksize=100_000)` and concatenate the chunks.")

    add("data",
        "Write a function that computes a moving median in pandas.",
        "```python\nimport pandas as pd\n\ndef moving_median(s: pd.Series, window: int = 7) -> pd.Series:\n    return s.rolling(window=window, min_periods=1).median()\n```\nMoving median is more robust to outliers than moving mean; use it when the series is noisy.")

    add("data",
        "How do I convert a date column from string to datetime?",
        "```python\nimport pandas as pd\n\ndf['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='raise')\n```\nPass `format=` if you know it -- pandas's auto-detect is slow on millions of rows.")

    add("data",
        "Write a numpy snippet that finds local maxima in a 1D array.",
        "```python\nimport numpy as np\n\ndef local_maxima(a: np.ndarray) -> np.ndarray:\n    return np.where((a[1:-1] > a[:-2]) & (a[1:-1] > a[2:]))[0] + 1\n```\nReturns indices into the original array; for noisy data use `scipy.signal.find_peaks` with prominence/distance filters.")

    add("data",
        "How do I plot multiple subplots that share a y-axis?",
        "```python\nimport matplotlib.pyplot as plt\n\nfig, axes = plt.subplots(1, 3, sharey=True, figsize=(12, 4))\nfor ax, col in zip(axes, ['a', 'b', 'c']):\n    ax.hist(df[col], bins=30)\n    ax.set_title(col)\nfig.tight_layout()\n```\n`sharey=True` makes visual comparison meaningful; otherwise each subplot autoscales independently.")

    add("data",
        "Write a function that computes the entropy of a discrete distribution.",
        "```python\nimport numpy as np\n\ndef entropy(p: np.ndarray) -> float:\n    p = p[p > 0]\n    return -float(np.sum(p * np.log2(p)))\n```\nDrop zeros before the log; otherwise you get NaN. Returns bits.")

    add("data",
        "How do I encode a date column as cyclical features?",
        "```python\nimport numpy as np\nimport pandas as pd\n\ndef cyclical(values: pd.Series, period: int) -> pd.DataFrame:\n    rad = 2 * np.pi * values / period\n    return pd.DataFrame({'sin': np.sin(rad), 'cos': np.cos(rad)})\n\nhour_features = cyclical(df['date'].dt.hour, 24)\n```\nTrigonometric encodings let models learn that hour 23 is close to hour 0 -- ordinary integer encoding fails on this.")

    add("data",
        "Write a pandas snippet that joins two large tables in chunks.",
        "```python\nimport pandas as pd\n\nleft = pd.read_csv('orders.csv', chunksize=200_000)\nright = pd.read_csv('users.csv').set_index('user_id')\n\nfor chunk in left:\n    merged = chunk.join(right, on='user_id', how='left')\n    merged.to_parquet(f'out/orders_{chunk.index[0]}.parquet')\n```\nHash the right side once; stream the left side; write each chunk independently.")

    add("data",
        "How do I compute weekday distribution from timestamps?",
        "```python\nimport pandas as pd\n\nweekdays = df['timestamp'].dt.day_name().value_counts(normalize=True).sort_index()\n```\n`value_counts(normalize=True)` returns proportions; `sort_index()` orders Monday..Sunday alphabetically -- pass a `Categorical` if you need calendar order.")

    add("data",
        "Write a numpy snippet that scales features to the [0, 1] range.",
        "```python\nimport numpy as np\n\ndef minmax(a: np.ndarray) -> np.ndarray:\n    lo = a.min(axis=0)\n    hi = a.max(axis=0)\n    span = np.where(hi - lo == 0, 1, hi - lo)\n    return (a - lo) / span\n```\nThe `np.where` guard handles constant columns -- otherwise you get NaN.")

    add("data",
        "How do I detect drift between two distributions with the KS test?",
        "```python\nfrom scipy.stats import ks_2samp\n\nstat, pvalue = ks_2samp(reference, current)\nif pvalue < 0.05:\n    print(f'distribution drift detected: stat={stat:.3f}, p={pvalue:.4f}')\n```\nKS works for univariate continuous data; for high-dimensional drift consider PSI or Wasserstein.")

    add("data",
        "Write a pandas pipeline using method chaining.",
        "```python\nimport pandas as pd\n\nresult = (\n    df.assign(amount=lambda d: d['amount'].fillna(0))\n      .query('status == \"completed\"')\n      .groupby('region', as_index=False)\n      .agg(total=('amount', 'sum'))\n      .sort_values('total', ascending=False)\n)\n```\nParens-with-newlines keep each step on one line and let you comment them inline.")

    add("data",
        "How do I plot error bars on a bar chart?",
        "```python\nimport matplotlib.pyplot as plt\n\nlabels = ['A', 'B', 'C']\nmeans = [2.5, 3.1, 4.0]\nstds = [0.4, 0.2, 0.5]\nfig, ax = plt.subplots()\nax.bar(labels, means, yerr=stds, capsize=5)\nax.set_ylabel('Mean')\nfig.tight_layout()\n```\nDocument what the bars represent (1-sigma? 95% CI?) -- the chart alone is ambiguous.")

    add("data",
        "Write a numpy function that performs k-means on a 2D dataset.",
        "```python\nimport numpy as np\n\ndef kmeans(x: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> np.ndarray:\n    rng = np.random.default_rng(seed)\n    centers = x[rng.choice(len(x), k, replace=False)]\n    for _ in range(iters):\n        d = np.linalg.norm(x[:, None] - centers[None], axis=-1)\n        labels = d.argmin(axis=1)\n        new_centers = np.array([x[labels == j].mean(axis=0) if (labels == j).any() else centers[j] for j in range(k)])\n        if np.allclose(centers, new_centers):\n            break\n        centers = new_centers\n    return labels\n```\nFor real workloads use `sklearn.cluster.KMeans` -- it has k-means++ init and is much faster.")

    add("data",
        "How do I melt a wide DataFrame to long format?",
        "```python\nimport pandas as pd\n\nlong = df.melt(id_vars=['date'], var_name='product', value_name='units')\n```\nThe inverse is `pivot_table`. Long-format DataFrames make `groupby` and `seaborn` plotting trivial.")

    add("data",
        "Write a function that downsamples a high-frequency series to daily means.",
        "```python\nimport pandas as pd\n\ndef daily_mean(df: pd.DataFrame, time_col: str = 'ts') -> pd.DataFrame:\n    return df.set_index(time_col).resample('D').mean(numeric_only=True).reset_index()\n```\n`resample('D')` aligns to midnight; pass `closed='left', label='left'` if your data convention starts at hour 00:00.")

    add("data",
        "How do I plot a calendar heatmap with matplotlib?",
        "```python\nimport calplot\nimport pandas as pd\n\nseries = df.set_index('date')['value']\ncalplot.calplot(series, cmap='YlGn', figsize=(12, 4))\n```\n`calplot` is a thin wrapper -- if you can't add a dep, use `seaborn.heatmap` on a year-by-week pivot.")

    add("data",
        "Write a numpy snippet that one-hot encodes integer labels.",
        "```python\nimport numpy as np\n\ndef one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:\n    out = np.zeros((labels.size, n_classes), dtype=np.float32)\n    out[np.arange(labels.size), labels] = 1.0\n    return out\n```\nFancy indexing with `np.arange` is fast; for sklearn use `OneHotEncoder` for full preprocessing pipelines.")

    add("data",
        "How do I compute a weighted average per group?",
        "```python\nimport pandas as pd\n\ndef weighted_mean(g: pd.DataFrame, val: str, weight: str) -> float:\n    return (g[val] * g[weight]).sum() / g[weight].sum()\n\nresult = df.groupby('region').apply(weighted_mean, val='price', weight='qty')\n```\n`apply` is slower than vectorized math; for performance precompute the weighted columns and do two `sum`s.")

    add("data",
        "Write a pandas snippet that adds a rank column within groups.",
        "```python\nimport pandas as pd\n\ndf['rank'] = df.groupby('category')['score'].rank(method='dense', ascending=False)\n```\n`method='dense'` makes ties share the same rank without leaving gaps; `'min'` and `'first'` are also useful.")

    add("data",
        "How do I plot a violin plot grouped by category?",
        "```python\nimport seaborn as sns\nimport matplotlib.pyplot as plt\n\nfig, ax = plt.subplots()\nsns.violinplot(data=df, x='group', y='value', inner='quartile', ax=ax)\nfig.tight_layout()\n```\nViolins hide sample size; add `inner='box'` or annotate counts when sample sizes differ.")

    add("data",
        "Write code that computes recall at k for a recommender.",
        "```python\nimport numpy as np\n\ndef recall_at_k(actual: list[set], recommended: list[list], k: int) -> float:\n    hits = [\n        len(set(rec[:k]) & act) / len(act) if act else 0.0\n        for rec, act in zip(recommended, actual)\n    ]\n    return float(np.mean(hits))\n```\nGuard against empty `actual` per user; otherwise division explodes.")

    add("data",
        "How do I create lag features for a time series?",
        "```python\nimport pandas as pd\n\nfor lag in [1, 7, 28]:\n    df[f'lag_{lag}'] = df['value'].shift(lag)\ndf = df.dropna()\n```\nDrop the resulting NaN rows or impute -- most models hate them. Lag features are the bread and butter of time-series forecasting.")

    add("data",
        "Write a numpy snippet that computes the moving average using cumulative sum.",
        "```python\nimport numpy as np\n\ndef moving_avg(a: np.ndarray, w: int) -> np.ndarray:\n    cs = np.cumsum(a, dtype=np.float64)\n    cs[w:] = cs[w:] - cs[:-w]\n    return cs[w-1:] / w\n```\nO(n) regardless of window size; for huge arrays this is much faster than `np.convolve`.")

    add("data",
        "How do I encode an ordinal variable preserving order?",
        "```python\nimport pandas as pd\n\norder = ['cold', 'warm', 'hot']\ndf['temp'] = pd.Categorical(df['temp'], categories=order, ordered=True).codes\n```\nThe `.codes` attribute returns integer indices that respect the declared order.")

    add("data",
        "Write code that bootstraps a 95% confidence interval for the mean.",
        "```python\nimport numpy as np\n\ndef bootstrap_mean_ci(x: np.ndarray, n_iter: int = 10_000, seed: int = 0) -> tuple[float, float]:\n    rng = np.random.default_rng(seed)\n    idx = rng.integers(0, len(x), size=(n_iter, len(x)))\n    samples = x[idx].mean(axis=1)\n    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))\n```\nVectorized resampling beats a Python loop by orders of magnitude.")

    add("data",
        "How do I plot multiple lines from a long-form DataFrame?",
        "```python\nimport seaborn as sns\nimport matplotlib.pyplot as plt\n\nfig, ax = plt.subplots(figsize=(10, 5))\nsns.lineplot(data=df, x='date', y='value', hue='series', ax=ax)\nfig.autofmt_xdate()\n```\nLong-form data + `hue=` is far less code than looping and plotting one series at a time.")

    add("data",
        "Write a numpy snippet that creates a meshgrid for plotting a 2D function.",
        "```python\nimport numpy as np\nimport matplotlib.pyplot as plt\n\nx = np.linspace(-3, 3, 200)\ny = np.linspace(-3, 3, 200)\nX, Y = np.meshgrid(x, y)\nZ = np.exp(-(X**2 + Y**2))\nfig, ax = plt.subplots()\nax.contourf(X, Y, Z, levels=20, cmap='viridis')\nfig.colorbar(ax.collections[0], ax=ax)\n```\n`indexing='xy'` is the default; pass `'ij'` if you want array-style row/column indexing.")

    add("data",
        "How do I deal with class imbalance in scikit-learn?",
        "```python\nfrom sklearn.linear_model import LogisticRegression\n\nclf = LogisticRegression(class_weight='balanced').fit(X, y)\n```\nFor severe imbalance, pair this with stratified sampling and metrics like ROC-AUC or precision/recall instead of plain accuracy.")

    add("data",
        "Write a function that computes the sample mean and standard error.",
        "```python\nimport numpy as np\n\ndef mean_se(x: np.ndarray) -> tuple[float, float]:\n    n = len(x)\n    if n < 2:\n        raise ValueError('need at least two samples')\n    mean = x.mean()\n    se = x.std(ddof=1) / np.sqrt(n)\n    return float(mean), float(se)\n```\n`ddof=1` for the sample (Bessel-corrected) estimator -- the default `ddof=0` is biased.")

    add("data",
        "How do I plot a scatter matrix in pandas?",
        "```python\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\npd.plotting.scatter_matrix(df.select_dtypes(include='number'), alpha=0.5, figsize=(10, 10))\nplt.tight_layout()\n```\nFor more control reach for `seaborn.pairplot`, which adds KDEs on the diagonal and `hue=` support.")

    add("data",
        "Write a snippet that splits a time series chronologically.",
        "```python\nimport pandas as pd\n\ndef chrono_split(df: pd.DataFrame, fraction: float = 0.8, time_col: str = 'date') -> tuple[pd.DataFrame, pd.DataFrame]:\n    df = df.sort_values(time_col)\n    cutoff = int(len(df) * fraction)\n    return df.iloc[:cutoff], df.iloc[cutoff:]\n```\nFor TS forecasting never use `train_test_split` with shuffling -- it leaks future into the train set.")

    add("data",
        "How do I deduplicate rows by approximate match on a string column?",
        "```python\nimport pandas as pd\nfrom rapidfuzz import process, fuzz\n\ndef fuzzy_dedupe(df: pd.DataFrame, col: str, threshold: int = 90) -> pd.DataFrame:\n    seen: list[str] = []\n    keep: list[bool] = []\n    for s in df[col].astype(str):\n        match = process.extractOne(s, seen, scorer=fuzz.ratio) if seen else None\n        if match is None or match[1] < threshold:\n            seen.append(s)\n            keep.append(True)\n        else:\n            keep.append(False)\n    return df[keep].reset_index(drop=True)\n```\n`rapidfuzz` is the modern, fast successor to `fuzzywuzzy` and has the same API surface.")

    add("data",
        "Write a numpy snippet that computes the cosine similarity between two vectors.",
        "```python\nimport numpy as np\n\ndef cosine(a: np.ndarray, b: np.ndarray) -> float:\n    na = np.linalg.norm(a)\n    nb = np.linalg.norm(b)\n    if na == 0 or nb == 0:\n        return 0.0\n    return float(a @ b / (na * nb))\n```\nFor matrices use `(A @ B.T) / (||A||_row * ||B||_row.T)`; same idea, vectorized.")

    add("data",
        "How do I plot a horizontal bar chart sorted by value?",
        "```python\nimport matplotlib.pyplot as plt\n\nseries = df['count'].sort_values()\nfig, ax = plt.subplots(figsize=(8, max(4, len(series) * 0.3)))\nax.barh(series.index, series.values)\nax.set_xlabel('Count')\nfig.tight_layout()\n```\nDynamic figure height keeps labels readable when there are many categories.")

    add("data",
        "Write a function that tests whether two means differ significantly.",
        "```python\nfrom scipy.stats import ttest_ind\n\ndef significant_diff(a, b, alpha: float = 0.05) -> bool:\n    stat, p = ttest_ind(a, b, equal_var=False)\n    return p < alpha\n```\n`equal_var=False` (Welch's t-test) is the safer default -- it doesn't assume equal variances.")

    add("data",
        "How do I compute monthly aggregates in pandas without losing zero-row months?",
        "```python\nimport pandas as pd\n\nmonthly = (\n    df.set_index('date')\n      .resample('MS')\n      ['amount'].sum()\n      .reindex(pd.date_range('2026-01-01', '2026-12-01', freq='MS'), fill_value=0)\n)\n```\nReindex to a full month range so you keep zero-activity months in the output.")

    add("data",
        "Write a pandas snippet that finds rows where a value crosses a threshold.",
        "```python\nimport pandas as pd\n\ncrossings = df[(df['signal'].shift(1) <= 0) & (df['signal'] > 0)]\n```\nThe two-row condition catches transitions; for the descending direction flip the comparison.")

    add("data",
        "How do I encode timestamps as features for a model?",
        "```python\nimport pandas as pd\n\nts = pd.to_datetime(df['ts'])\ndf['hour'] = ts.dt.hour\ndf['dow'] = ts.dt.dayofweek\ndf['is_weekend'] = ts.dt.dayofweek >= 5\ndf['month'] = ts.dt.month\n```\nCombine with cyclical (sin/cos) encodings for hour and month -- linear models love them.")

    add("data",
        "Write a numpy snippet that computes the determinant and inverse of a matrix.",
        "```python\nimport numpy as np\n\ndef det_inv(m: np.ndarray) -> tuple[float, np.ndarray]:\n    det = float(np.linalg.det(m))\n    if abs(det) < 1e-12:\n        raise ValueError('matrix is singular')\n    return det, np.linalg.inv(m)\n```\nFor solving `Ax = b` use `np.linalg.solve(A, b)` -- it's faster and more numerically stable than computing the inverse.")

    add("data",
        "How do I read a SQL query into a DataFrame?",
        "```python\nimport pandas as pd\nfrom sqlalchemy import create_engine\n\nengine = create_engine('postgresql://user:pass@host/db')\ndf = pd.read_sql_query('SELECT * FROM users WHERE created_at >= %s', engine, params=('2026-01-01',))\n```\nPass parameters as a tuple; never f-string user input into SQL -- that's how SQL injection happens.")

    add("data",
        "Write a function that computes the mean absolute percentage error.",
        "```python\nimport numpy as np\n\ndef mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:\n    mask = y_true != 0\n    if not mask.any():\n        return float('nan')\n    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)\n```\nMASK out zeros first; MAPE is undefined for them. Use SMAPE if zeros are common.")

    add("data",
        "How do I plot a Q-Q plot to check normality?",
        "```python\nimport scipy.stats as stats\nimport matplotlib.pyplot as plt\n\nfig, ax = plt.subplots()\nstats.probplot(data, dist='norm', plot=ax)\nax.set_title('Q-Q plot vs Normal')\nfig.tight_layout()\n```\nPoints close to the diagonal mean approximately Normal; systematic deviations show fat tails or skew.")

    add("data",
        "Write a numpy snippet that smooths a curve with a Gaussian kernel.",
        "```python\nimport numpy as np\nfrom scipy.ndimage import gaussian_filter1d\n\nsmoothed = gaussian_filter1d(noisy_signal, sigma=2.0)\n```\nLarger `sigma` means more smoothing; pick it based on how much detail you need to preserve.")

    add("data",
        "How do I add error handling to a pandas pipeline?",
        "```python\nimport pandas as pd\n\ndef safe_load(path: str) -> pd.DataFrame:\n    try:\n        return pd.read_csv(path, parse_dates=['date'])\n    except FileNotFoundError:\n        return pd.DataFrame(columns=['id', 'date', 'amount'])\n    except pd.errors.EmptyDataError:\n        return pd.DataFrame(columns=['id', 'date', 'amount'])\n```\nReturn an empty DataFrame with the expected schema so downstream code keeps the same shape.")

    add("data",
        "Write a function that bins a column into custom intervals.",
        "```python\nimport pandas as pd\n\ndef custom_bins(s: pd.Series) -> pd.Series:\n    edges = [0, 18, 35, 60, 200]\n    labels = ['child', 'adult', 'middle', 'senior']\n    return pd.cut(s, bins=edges, labels=labels, right=False)\n```\n`right=False` makes intervals left-closed: `[0, 18)`, `[18, 35)`, etc.")

    add("data",
        "How do I plot multiple histograms on the same axes for comparison?",
        "```python\nimport matplotlib.pyplot as plt\n\nfig, ax = plt.subplots()\nfor label, group in df.groupby('cohort'):\n    ax.hist(group['value'], bins=30, alpha=0.5, label=label)\nax.legend()\nfig.tight_layout()\n```\n`alpha=0.5` lets you see overlapping distributions; use `density=True` for unequal sample sizes.")

    add("data",
        "Write a numpy snippet that computes the rank of a matrix.",
        "```python\nimport numpy as np\n\ndef rank(m: np.ndarray, tol: float | None = None) -> int:\n    return int(np.linalg.matrix_rank(m, tol=tol))\n```\nThe optional tolerance lets you treat near-zero singular values as zero -- useful for floating-point matrices.")

    add("data",
        "How do I compute the percentage of NaN per column?",
        "```python\nimport pandas as pd\n\npct_missing = df.isna().mean().sort_values(ascending=False) * 100\n```\nA quick triage view; a column with 90%+ missing usually deserves to be dropped or studied separately.")

    add("data",
        "Write a snippet that computes monthly cohort retention.",
        "```python\nimport pandas as pd\n\nfirst = df.groupby('user_id')['date'].min().rename('cohort')\ndf = df.join(first, on='user_id')\ndf['cohort_month'] = df['cohort'].dt.to_period('M')\ndf['active_month'] = df['date'].dt.to_period('M')\nretention = (\n    df.groupby(['cohort_month', 'active_month'])['user_id'].nunique().unstack(fill_value=0)\n)\n```\nThe rectangular `unstack` form is what most retention dashboards consume.")

    add("data",
        "How do I plot a 3D surface in matplotlib?",
        "```python\nimport numpy as np\nimport matplotlib.pyplot as plt\n\nx = np.linspace(-3, 3, 100)\ny = np.linspace(-3, 3, 100)\nX, Y = np.meshgrid(x, y)\nZ = np.sin(np.sqrt(X**2 + Y**2))\nfig = plt.figure(figsize=(8, 6))\nax = fig.add_subplot(projection='3d')\nax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')\nfig.tight_layout()\n```\nFor anything interactive (zoom, rotate) reach for plotly -- matplotlib 3D plots are static and slow.")

    add("data",
        "Write a function that computes precision and recall at k.",
        "```python\nimport numpy as np\n\ndef precision_recall_at_k(actual: set, predicted: list, k: int) -> tuple[float, float]:\n    pred_k = predicted[:k]\n    hits = len(set(pred_k) & actual)\n    precision = hits / k if k else 0.0\n    recall = hits / len(actual) if actual else 0.0\n    return precision, recall\n```\nFor a single user this is straightforward; aggregate across users with macro or weighted means.")

    add("data",
        "How do I save a pandas DataFrame to S3?",
        "```python\nimport pandas as pd\n\ndf.to_parquet('s3://my-bucket/path/file.parquet', storage_options={'key': 'AKI...', 'secret': '...'})\n```\nBetter still, set credentials via environment variables or an IAM role and let pandas/`s3fs` pick them up automatically.")

    add("data",
        "Write a numpy snippet that finds the closest pair of values in two arrays.",
        "```python\nimport numpy as np\n\ndef closest_pair(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:\n    a_sorted = np.sort(a)\n    idx = np.searchsorted(a_sorted, b)\n    idx_left = np.clip(idx - 1, 0, len(a_sorted) - 1)\n    idx_right = np.clip(idx, 0, len(a_sorted) - 1)\n    diff = np.abs(np.stack([a_sorted[idx_left] - b, a_sorted[idx_right] - b]))\n    pos = diff.argmin(axis=0)\n    closest_a = np.where(pos == 0, a_sorted[idx_left], a_sorted[idx_right])\n    j = np.abs(closest_a - b).argmin()\n    return float(closest_a[j]), float(b[j])\n```\nO((n+m) log n) using `searchsorted`.")

    add("data",
        "How do I add annotations to a matplotlib chart?",
        "```python\nimport matplotlib.pyplot as plt\n\nfig, ax = plt.subplots()\nax.plot([1, 2, 3], [4, 1, 5])\nax.annotate('peak', xy=(3, 5), xytext=(2.5, 5.5),\n            arrowprops=dict(arrowstyle='->'))\nfig.tight_layout()\n```\n`xy` is the data point being annotated; `xytext` is where the label sits.")

    add("data",
        "Write a function that scales features using a robust scaler.",
        "```python\nimport numpy as np\n\ndef robust_scale(a: np.ndarray) -> np.ndarray:\n    median = np.median(a, axis=0)\n    q1, q3 = np.percentile(a, [25, 75], axis=0)\n    iqr = np.where(q3 - q1 == 0, 1, q3 - q1)\n    return (a - median) / iqr\n```\nMore stable than min/max scaling on data with outliers; `sklearn.preprocessing.RobustScaler` is the production version.")

    add("data",
        "How do I check whether a Series is monotonically increasing?",
        "```python\nimport pandas as pd\n\nis_inc = df['value'].is_monotonic_increasing\n```\nThe property is O(n); for time-series ordering checks it's the cleanest test.")

    add("data",
        "Write a snippet that creates a sparse matrix from a long-form DataFrame.",
        "```python\nimport pandas as pd\nfrom scipy.sparse import coo_matrix\n\ncodes_user = df['user_id'].astype('category').cat.codes\ncodes_item = df['item_id'].astype('category').cat.codes\nm = coo_matrix(\n    (df['rating'], (codes_user, codes_item)),\n    shape=(codes_user.max() + 1, codes_item.max() + 1),\n)\n```\nUseful for collaborative filtering -- materialising the dense version often blows memory.")

    add("data",
        "How do I plot a time series with shaded recession bands?",
        "```python\nimport matplotlib.pyplot as plt\n\nfig, ax = plt.subplots(figsize=(10, 4))\nax.plot(df.index, df['value'])\nfor start, end in recessions:\n    ax.axvspan(start, end, color='gray', alpha=0.3)\nfig.autofmt_xdate()\n```\n`axvspan` works in data coordinates; layer it before the line so the line sits on top.")

    add("data",
        "Write a numpy snippet that computes a percentile per row of a 2D array.",
        "```python\nimport numpy as np\n\np90 = np.percentile(arr, 90, axis=1)\n```\n`axis=1` collapses columns into one value per row. Pass a list `[5, 50, 95]` to get multiple percentiles in one call.")

    add("data",
        "How do I compute weekly retention as a percentage of cohort size?",
        "```python\nimport pandas as pd\n\nfirst = df.groupby('user_id')['week'].min().rename('cohort')\ndf = df.join(first, on='user_id')\nretention = (\n    df.groupby(['cohort', 'week'])['user_id'].nunique().unstack(fill_value=0)\n)\nretention_pct = retention.div(retention.iloc[:, 0], axis=0) * 100\n```\nFirst column of each cohort is 100% by construction; subsequent columns measure decay.")

    # ----- debugging / refactoring (25) -----
    add("debug",
        "I keep getting `RuntimeError: dictionary changed size during iteration`. How do I fix it?",
        "You're modifying a dict while iterating over it. Iterate over a snapshot:\n```python\nfor key in list(d.keys()):\n    if should_remove(key):\n        del d[key]\n```\nOr build the result functionally and replace at the end:\n```python\nd = {k: v for k, v in d.items() if not should_remove(k)}\n```\nThe second form is cleaner and harder to break.")

    add("debug",
        "My function returns `None` instead of the expected list. What should I check?",
        "Three usual culprits:\n1. You wrote `xs.sort()` or `xs.append(x)` and returned the result -- both return None and mutate in place. Use `sorted(xs)` and `[*xs, x]` if you want a value.\n2. A `return` statement is missing inside a branch.\n3. A method call returns None implicitly. Add `print(result, type(result))` at the bottom and at every return -- you'll see which one is wrong.")

    add("debug",
        "I'm getting `UnicodeDecodeError` when reading a file. How do I diagnose it?",
        "Most often the file isn't UTF-8. Detect the encoding:\n```python\nfrom chardet.universaldetector import UniversalDetector\nd = UniversalDetector()\nfor line in open(path, 'rb'):\n    d.feed(line)\n    if d.done: break\nd.close()\nprint(d.result)\n```\nThen open with the right encoding: `open(path, encoding='cp1252')`. If you genuinely don't care about the bad bytes, `errors='replace'` is the pragmatic escape hatch.")

    add("debug",
        "My script works on macOS but fails on Linux with file-not-found. Why?",
        "Almost always case sensitivity. macOS HFS+/APFS is case-insensitive by default; Linux is case-sensitive. `Path('Data/File.csv')` and `Path('data/file.csv')` are the same on Mac but different on Linux. Standardize on lowercase in the repo and trust the OS error.")

    add("debug",
        "I'm getting `RecursionError: maximum recursion depth exceeded`. What now?",
        "Either rewrite iteratively with an explicit stack/queue, or raise the limit with `sys.setrecursionlimit(10_000)` -- but only as a stopgap. Deep recursion in CPython risks a hard segfault on the C stack. For tree/graph problems, BFS or an explicit DFS stack is the production-grade answer.")

    add("debug",
        "Why does `if x is True` behave differently from `if x`?",
        "`is` checks object identity; `if x is True` only matches the singleton `True`. The truthy values 1, 'yes', `[1]` all evaluate `if x` as true but fail `is True`. Rule of thumb: use `if x:` for truthiness, `is None` / `is not None` for None, and never `is True`/`is False`.")

    add("debug",
        "How do I refactor a function that has too many parameters?",
        "Group related parameters into a dataclass. Before:\n```python\ndef ship(name, addr1, addr2, city, state, zip): ...\n```\nAfter:\n```python\nfrom dataclasses import dataclass\n\n@dataclass\nclass Address:\n    name: str\n    line1: str\n    line2: str\n    city: str\n    state: str\n    zip: str\n\ndef ship(addr: Address): ...\n```\nThe call site reads better and adding a field doesn't change every signature.")

    add("debug",
        "My logger isn't printing anything. What's wrong?",
        "Default root logger is WARNING and has no handler. Configure once at app entry:\n```python\nimport logging\nlogging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')\n```\nDon't call `basicConfig` from libraries -- only from the application entry point. Use `logger = logging.getLogger(__name__)` everywhere else.")

    add("debug",
        "I'm getting `TypeError: unhashable type: 'dict'`. How do I fix it?",
        "You're trying to use a dict (or list) as a dict key or in a set. Convert to an immutable structure first:\n```python\nfrozenset(d.items())   # for shallow dicts\ntuple(sorted(d.items()))   # if order doesn't matter\n```\nOr pick a hashable identifier from the dict, e.g. `d['id']`. `frozendict` from PyPI works too.")

    add("debug",
        "Why does my list comprehension see a stale variable from the enclosing scope?",
        "In Python 3 list comprehensions have their own scope, so this rarely happens. The classic trap is closures over a loop variable:\n```python\nfns = [lambda: i for i in range(3)]\n[f() for f in fns]   # [2, 2, 2]\n```\nFix with a default argument: `lambda i=i: i`. The default captures the current value at definition time.")

    add("debug",
        "How do I refactor nested if-else into something cleaner?",
        "Replace nested branches with early returns and small helper predicates. Before:\n```python\ndef price(user, cart):\n    if user.is_active:\n        if cart.total > 100:\n            if user.has_coupon:\n                return cart.total * 0.7\n            return cart.total * 0.9\n    return cart.total\n```\nAfter:\n```python\ndef price(user, cart):\n    if not user.is_active or cart.total <= 100:\n        return cart.total\n    return cart.total * (0.7 if user.has_coupon else 0.9)\n```\nGuard clauses keep the happy path on the left margin.")

    add("debug",
        "I'm getting `OSError: too many open files`. How do I find the leak?",
        "Use a context manager (`with open(...)`) so files close automatically. To find existing leaks:\n```python\nimport psutil\nproc = psutil.Process()\nprint(len(proc.open_files()))\n```\nLog the count at suspicious points; the place where it climbs without falling is the culprit. `lsof -p <pid>` is the same diagnosis from the shell.")

    add("debug",
        "Why does my mutable default argument keep getting old values?",
        "Default values are evaluated once at function definition. Don't put mutable defaults in signatures:\n```python\ndef append(item, target=None):\n    if target is None:\n        target = []\n    target.append(item)\n    return target\n```\nThe `None` sentinel pattern is the canonical fix. Linters (`ruff B006`) flag this automatically.")

    add("debug",
        "I'm seeing `pickle.UnpicklingError` after a class refactor. What happened?",
        "Pickle stores the import path of the class. If you renamed or moved the class, old pickles can't find it. Two fixes: keep a compatibility shim at the old import path that re-exports the new class, or migrate the old data to the new format with a one-time script. Long-term, prefer JSON/Parquet/Avro -- pickle is fragile across versions.")

    add("debug",
        "How do I refactor a 200-line function into something maintainable?",
        "Three steps: (1) Identify natural sections by comment lines and extract each into its own helper. (2) Replace shared mutable state with parameters and return values. (3) Pull the orchestration into one short function that names the steps. The function should read like a paragraph of pseudocode, with each helper handling a single responsibility.")

    add("debug",
        "Why is `requests.get(...)` hanging?",
        "You forgot a timeout. `requests` defaults to no timeout, so a hung server hangs your process forever. Always:\n```python\nrequests.get(url, timeout=(3, 10))   # connect, read\n```\nFor robustness, wrap with `tenacity.retry` and a circuit breaker. For async code use `httpx` with `timeout=httpx.Timeout(...)`.")

    add("debug",
        "I get `AttributeError: 'NoneType' object has no attribute 'foo'` deep in the call stack.",
        "Some function returned `None` where you expected an object. Add the boundary check at the call site:\n```python\nresult = do_thing()\nif result is None:\n    raise RuntimeError('expected a Thing, got None')\nresult.foo()\n```\nFor systematic prevention enable `mypy --strict` so optional return types must be unwrapped explicitly.")

    add("debug",
        "How do I fix a function that mutates its input list as a side effect?",
        "Decide whether mutation is intentional or accidental, then commit to one or the other.\nIntentional (in-place sort): `def sort_in_place(xs: list) -> None: xs.sort()` -- return None to signal mutation.\nAccidental: copy first.\n```python\ndef sorted_unique(xs: list) -> list:\n    return sorted(set(xs))\n```\nDocument the choice in the docstring; surprise mutation is one of the worst bug categories.")

    add("debug",
        "My pytest fixture leaks state between tests. How do I scope it?",
        "Default scope is `function`, which runs setup/teardown for each test -- the safe choice. If you bumped to `module` or `session` for speed, your fixture must reset state explicitly:\n```python\n@pytest.fixture(scope='module')\ndef db():\n    db = make_db()\n    yield db\n    db.clear()\n```\nIf reset is expensive, drop back to `function` scope and accept the slowdown. Speed isn't worth flaky tests.")

    add("debug",
        "Why does my datetime arithmetic give wrong results across DST?",
        "You're using naive datetimes. Always use timezone-aware ones for anything that matters:\n```python\nfrom datetime import datetime, timezone\nnow = datetime.now(timezone.utc)\n```\nFor wall-clock arithmetic in a particular zone use `zoneinfo.ZoneInfo('America/New_York')`. Adding a `timedelta(hours=24)` does NOT necessarily mean 'tomorrow at the same wall time' across DST -- it means exactly 24 hours.")

    add("debug",
        "How do I refactor a class that's doing too many things?",
        "Apply Single Responsibility. List the public methods and group them by what data they touch. If you see two clear groups, split into two classes. Compose them via dependency injection so tests can stub one half. The signal you're done: each class fits on one screen and its name describes one concern.")

    add("debug",
        "My async test never finishes. What's happening?",
        "Common causes: (1) you forgot `@pytest.mark.asyncio` and the coroutine returns a coroutine object instead of running, (2) you `await` something that never resolves (no timeout, deadlocked queue), (3) you're using `time.sleep` instead of `asyncio.sleep` and blocking the event loop. Add `pytest-timeout` and a per-test timeout so hangs become failures.")

    add("debug",
        "I'm getting `ImportError: cannot import name X from partially initialized module`.",
        "Circular imports. Two fixes that work: (1) move the conflicting import inside the function that uses it (lazy import), or (2) extract the shared symbols into a third module that neither depends on the others. The structural fix is better -- circular imports usually mean two modules are coupled and want to be one or three.")

    add("debug",
        "Why do my unit tests pass locally but fail in CI?",
        "Common causes: (1) tests depend on an environment variable set in your shell but not in CI, (2) tests rely on timezone (CI is usually UTC), (3) tests rely on filesystem ordering (use sorted listings), (4) tests rely on network access or DNS, (5) randomness without a fixed seed. The mitigation is to make tests deterministic and self-contained.")

    add("debug",
        "How do I refactor a long if/elif chain?",
        "Replace it with a dict dispatch:\n```python\nHANDLERS = {\n    'a': handle_a,\n    'b': handle_b,\n    'c': handle_c,\n}\n\ndef dispatch(kind, payload):\n    handler = HANDLERS.get(kind)\n    if handler is None:\n        raise ValueError(f'unknown kind: {kind}')\n    return handler(payload)\n```\nAdding a new branch is now a one-line dict update. For complex logic, switch to a strategy pattern (one class per branch).")
