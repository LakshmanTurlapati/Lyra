# SPDX-License-Identifier: MIT
"""Web (1) + data science (72) + debugging (28). Wave 4 part 4."""


def register_all(add):
    # ---- web (1) ----
    add("web",
        "Write a function that returns the public IP via a service.",
        "```python\nimport httpx\n\ndef public_ip() -> str:\n    r = httpx.get('https://api.ipify.org', timeout=5.0)\n    r.raise_for_status()\n    return r.text.strip()\n```\nUseful in scripts; cache the value if you need it many times in a run.")

    # ---- data science (72) ----
    add("data",
        "Write a function that loads a numpy array from disk lazily.",
        "```python\nimport numpy as np\n\ndef load_mmap(path: str) -> np.ndarray:\n    return np.load(path, mmap_mode='r')\n```\n`mmap_mode='r'` lets the OS page in chunks on demand -- great for arrays larger than RAM.")

    add("data",
        "Implement a function that scales a feature to [0, 1].",
        "```python\nimport numpy as np\n\ndef minmax(arr: np.ndarray) -> np.ndarray:\n    lo, hi = arr.min(), arr.max()\n    if hi == lo:\n        return np.zeros_like(arr, dtype=float)\n    return (arr - lo) / (hi - lo)\n```\nGuard the constant-array case to avoid divide-by-zero NaNs.")

    add("data",
        "Write a function that computes Pearson correlation between two arrays.",
        "```python\nimport numpy as np\n\ndef pearson(a: np.ndarray, b: np.ndarray) -> float:\n    return float(np.corrcoef(a, b)[0, 1])\n```\n`corrcoef` is C-fast and handles the centering/normalization for you.")

    add("data",
        "Implement a function that pivots a tidy DataFrame into a heatmap.",
        "```python\nimport matplotlib.pyplot as plt\nimport pandas as pd\n\ndef heatmap(df: pd.DataFrame, idx: str, col: str, val: str, out: str) -> None:\n    pivot = df.pivot_table(index=idx, columns=col, values=val, aggfunc='mean')\n    fig, ax = plt.subplots(figsize=(8, 6))\n    cax = ax.imshow(pivot.values, aspect='auto')\n    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=45, ha='right')\n    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)\n    fig.colorbar(cax)\n    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)\n```\nFor publication-quality heatmaps, `seaborn.heatmap` adds annotations cleanly.")

    add("data",
        "Write a function that returns rows with duplicate keys.",
        "```python\nimport pandas as pd\n\ndef dups(df: pd.DataFrame, on: str) -> pd.DataFrame:\n    return df[df.duplicated(subset=on, keep=False)].sort_values(on)\n```\n`keep=False` flags every duplicated row, not just the second occurrence.")

    add("data",
        "Implement a function that merges multiple DataFrames on a common key.",
        "```python\nimport pandas as pd\nfrom functools import reduce\n\ndef merge_all(dfs: list[pd.DataFrame], on: str) -> pd.DataFrame:\n    return reduce(lambda a, b: a.merge(b, on=on, how='outer'), dfs)\n```\nOuter join is the safe default if you don't know which key set is canonical.")

    add("data",
        "Write a function that returns the moving median of a Series.",
        "```python\nimport pandas as pd\n\ndef moving_median(s: pd.Series, window: int) -> pd.Series:\n    return s.rolling(window=window, min_periods=1).median()\n```\nMedian is more robust to outliers than mean for noisy data.")

    add("data",
        "Implement a function that computes percentile ranks of a column.",
        "```python\nimport pandas as pd\n\ndef percentile_rank(s: pd.Series) -> pd.Series:\n    return s.rank(pct=True)\n```\n`pct=True` returns ranks in [0, 1].")

    add("data",
        "Write a function that adds a numeric noise column for differential privacy.",
        "```python\nimport numpy as np\nimport pandas as pd\n\ndef noisy(df: pd.DataFrame, col: str, eps: float, sensitivity: float = 1.0, seed: int | None = None) -> pd.DataFrame:\n    rng = np.random.default_rng(seed)\n    df = df.copy()\n    df[col] = df[col] + rng.laplace(0, sensitivity / eps, size=len(df))\n    return df\n```\nLaplace mechanism for epsilon-DP -- the canonical noise distribution.")

    add("data",
        "Implement a function that computes histogram bin edges from quantiles.",
        "```python\nimport numpy as np\n\ndef quantile_edges(arr: np.ndarray, n_bins: int) -> np.ndarray:\n    return np.quantile(arr, np.linspace(0, 1, n_bins + 1))\n```\nEqual-frequency bins; useful when the distribution is heavily skewed.")

    add("data",
        "Write a function that returns per-row z-scores within groups.",
        "```python\nimport pandas as pd\n\ndef group_zscore(df: pd.DataFrame, group: str, col: str) -> pd.Series:\n    grouped = df.groupby(group)[col]\n    return (df[col] - grouped.transform('mean')) / grouped.transform('std')\n```\n`transform` aligns the result back to the original index.")

    add("data",
        "Implement a function that drops constant columns.",
        "```python\nimport pandas as pd\n\ndef drop_constant(df: pd.DataFrame) -> pd.DataFrame:\n    keep = [c for c in df.columns if df[c].nunique(dropna=False) > 1]\n    return df[keep]\n```\nIncluding NaN-only as 'one value' is intentional; tweak `dropna` if you disagree.")

    add("data",
        "Write a function that loads an Excel file's specific sheet.",
        "```python\nimport pandas as pd\n\ndef read_sheet(path: str, sheet: str) -> pd.DataFrame:\n    return pd.read_excel(path, sheet_name=sheet, engine='openpyxl')\n```\n`openpyxl` is the standard engine for `.xlsx` files.")

    add("data",
        "Implement a function that exports a DataFrame to JSON lines.",
        "```python\nimport pandas as pd\n\ndef to_jsonl(df: pd.DataFrame, path: str) -> None:\n    df.to_json(path, orient='records', lines=True)\n```\nNDJSON output streams well into downstream tools.")

    add("data",
        "Write a function that downsamples an image-like array by factor n.",
        "```python\nimport numpy as np\n\ndef downsample(arr: np.ndarray, factor: int) -> np.ndarray:\n    h, w = arr.shape[:2]\n    h2, w2 = h - h % factor, w - w % factor\n    cropped = arr[:h2, :w2]\n    return cropped.reshape(h2 // factor, factor, w2 // factor, factor, *arr.shape[2:]).mean(axis=(1, 3))\n```\nReshape-then-mean trick avoids loops.")

    add("data",
        "Implement a function that computes RMSE between predictions and targets.",
        "```python\nimport numpy as np\n\ndef rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:\n    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))\n```\nNo need to pull in scikit-learn for a one-liner.")

    add("data",
        "Write a function that performs a train/test split deterministically.",
        "```python\nimport numpy as np\n\ndef split(arr: np.ndarray, frac: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:\n    rng = np.random.default_rng(seed)\n    idx = rng.permutation(len(arr))\n    n_test = int(len(arr) * frac)\n    test, train = arr[idx[:n_test]], arr[idx[n_test:]]\n    return train, test\n```\nFix the seed so train/test composition is reproducible across runs.")

    add("data",
        "Implement a function that creates a categorical color map.",
        "```python\nimport matplotlib.pyplot as plt\nimport matplotlib.colors as mcolors\n\ndef cat_cmap(n: int):\n    base = plt.get_cmap('tab20')\n    return mcolors.ListedColormap(base.colors[:n])\n```\nCategorical palettes go up to 20 distinct colors -- past that, switch encoding.")

    add("data",
        "Write a function that interpolates missing values in a Series.",
        "```python\nimport pandas as pd\n\ndef interp(s: pd.Series) -> pd.Series:\n    return s.interpolate(method='linear', limit_direction='both')\n```\nFor irregular time series use `method='time'` with a DatetimeIndex.")

    add("data",
        "Implement a function that computes column-wise summary stats with custom percentiles.",
        "```python\nimport pandas as pd\n\ndef stats(df: pd.DataFrame) -> pd.DataFrame:\n    return df.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])\n```\nThe 5th/95th give a quick sense of tail behavior.")

    add("data",
        "Write a function that converts a wide DataFrame to long with melt.",
        "```python\nimport pandas as pd\n\ndef to_long(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:\n    return df.melt(id_vars=id_cols, var_name='variable', value_name='value')\n```\nLong format is what most plotting libraries expect.")

    add("data",
        "Implement a function that scatter-plots two columns with a regression line.",
        "```python\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\n\ndef scatter_with_fit(df: pd.DataFrame, x: str, y: str, out: str) -> None:\n    fig, ax = plt.subplots()\n    ax.scatter(df[x], df[y], alpha=0.5)\n    coef = np.polyfit(df[x], df[y], 1)\n    xs = np.linspace(df[x].min(), df[x].max(), 50)\n    ax.plot(xs, np.polyval(coef, xs), color='red')\n    ax.set_xlabel(x); ax.set_ylabel(y)\n    fig.savefig(out, dpi=150); plt.close(fig)\n```\nFor nonlinear fits, switch to `scipy.optimize.curve_fit`.")

    add("data",
        "Write a function that calculates rolling correlation between two Series.",
        "```python\nimport pandas as pd\n\ndef rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:\n    return a.rolling(window=window).corr(b)\n```\nRolling-window correlation surfaces regime changes the static correlation hides.")

    add("data",
        "Implement a function that converts a numpy array to a torch tensor.",
        "```python\nimport numpy as np\nimport torch\n\ndef to_tensor(arr: np.ndarray) -> torch.Tensor:\n    return torch.from_numpy(arr.copy())\n```\nUse `.copy()` to break the ndarray's read-only flag if it came from `mmap`.")

    add("data",
        "Write a function that returns the top-k features by variance.",
        "```python\nimport pandas as pd\n\ndef top_var(df: pd.DataFrame, k: int) -> list[str]:\n    return df.var(numeric_only=True).nlargest(k).index.tolist()\n```\nLow-variance features rarely help models; this is a quick filter.")

    add("data",
        "Implement a function that returns rows where any column is missing.",
        "```python\nimport pandas as pd\n\ndef any_missing(df: pd.DataFrame) -> pd.DataFrame:\n    return df[df.isna().any(axis=1)]\n```\nUseful for inspecting missingness patterns before deciding how to handle them.")

    add("data",
        "Write a function that creates a confusion matrix for binary predictions.",
        "```python\nimport numpy as np\n\ndef confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:\n    tp = int(((y_true == 1) & (y_pred == 1)).sum())\n    tn = int(((y_true == 0) & (y_pred == 0)).sum())\n    fp = int(((y_true == 0) & (y_pred == 1)).sum())\n    fn = int(((y_true == 1) & (y_pred == 0)).sum())\n    return np.array([[tn, fp], [fn, tp]])\n```\nHand-rolled is fine for binary; for multi-class use `sklearn.metrics.confusion_matrix`.")

    add("data",
        "Implement a function that computes precision/recall/F1.",
        "```python\ndef prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:\n    p = tp / (tp + fp) if tp + fp else 0.0\n    r = tp / (tp + fn) if tp + fn else 0.0\n    f1 = 2 * p * r / (p + r) if p + r else 0.0\n    return p, r, f1\n```\nGuard each denominator -- the all-zero case is real.")

    add("data",
        "Write a function that computes a weighted mean.",
        "```python\nimport numpy as np\n\ndef weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:\n    return float(np.average(values, weights=weights))\n```\n`np.average` handles weighting; `np.mean` does not.")

    add("data",
        "Implement a function that compares two DataFrames structurally.",
        "```python\nimport pandas as pd\n\ndef diff(a: pd.DataFrame, b: pd.DataFrame) -> dict:\n    return {\n        'shape_a': a.shape,\n        'shape_b': b.shape,\n        'cols_only_in_a': sorted(set(a.columns) - set(b.columns)),\n        'cols_only_in_b': sorted(set(b.columns) - set(a.columns)),\n        'dtype_diffs': {c: (str(a[c].dtype), str(b[c].dtype)) for c in a.columns & b.columns if a[c].dtype != b[c].dtype},\n    }\n```\nQuick structural diff before diving into row-level comparison.")

    add("data",
        "Write a function that adds a 7-day moving average column.",
        "```python\nimport pandas as pd\n\ndef add_ma7(df: pd.DataFrame, col: str) -> pd.DataFrame:\n    df = df.copy()\n    df[f'{col}_ma7'] = df[col].rolling(7, min_periods=1).mean()\n    return df\n```\nReturning a copy keeps the original DataFrame unchanged.")

    add("data",
        "Implement a function that loads CSV with date parsing.",
        "```python\nimport pandas as pd\n\ndef load_with_dates(path: str, date_cols: list[str]) -> pd.DataFrame:\n    df = pd.read_csv(path)\n    for c in date_cols:\n        df[c] = pd.to_datetime(df[c], errors='coerce', utc=True)\n    return df\n```\nExplicit per-column parsing is more predictable than the deprecated `parse_dates=` shortcut.")

    add("data",
        "Write a function that pivots time series to a wide format by date.",
        "```python\nimport pandas as pd\n\ndef daily_wide(df: pd.DataFrame, date_col: str, key_col: str, val_col: str) -> pd.DataFrame:\n    df = df.copy()\n    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()\n    return df.pivot_table(index=date_col, columns=key_col, values=val_col, aggfunc='sum')\n```\n`normalize()` strips time-of-day so the same date groups together.")

    add("data",
        "Implement a function that adds a date-derived feature column.",
        "```python\nimport pandas as pd\n\ndef add_dow(df: pd.DataFrame, col: str) -> pd.DataFrame:\n    df = df.copy()\n    df['dow'] = pd.to_datetime(df[col]).dt.day_name()\n    return df\n```\n`dt.day_name()` gives 'Monday', 'Tuesday', ... -- often more useful than the integer.")

    add("data",
        "Write a function that returns the moving sum over a rolling time window.",
        "```python\nimport pandas as pd\n\ndef rolling_window_sum(s: pd.Series, window: str) -> pd.Series:\n    return s.rolling(window=window).sum()\n```\nString windows (`'7D'`, `'1H'`) require a DatetimeIndex.")

    add("data",
        "Implement a function that computes a weighted moving average.",
        "```python\nimport numpy as np\nimport pandas as pd\n\ndef wma(s: pd.Series, weights: np.ndarray) -> pd.Series:\n    weights = weights / weights.sum()\n    return s.rolling(window=len(weights)).apply(lambda w: np.dot(w, weights), raw=True)\n```\n`raw=True` passes a numpy array, which lets `np.dot` work in C.")

    add("data",
        "Write a function that returns a count of distinct values per column.",
        "```python\nimport pandas as pd\n\ndef nunique_per_col(df: pd.DataFrame) -> pd.Series:\n    return df.nunique().sort_values(ascending=False)\n```\nUseful diagnostic when sizing up a new dataset.")

    add("data",
        "Implement a function that creates a stacked bar chart.",
        "```python\nimport matplotlib.pyplot as plt\nimport pandas as pd\n\ndef stacked_bar(df: pd.DataFrame, idx: str, cols: list[str], out: str) -> None:\n    fig, ax = plt.subplots()\n    df.set_index(idx)[cols].plot(kind='bar', stacked=True, ax=ax)\n    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)\n```\nDataFrame.plot is fine for quick exploratory charts.")

    add("data",
        "Write a function that returns the rolling exponential weighted mean.",
        "```python\nimport pandas as pd\n\ndef ewm_mean(s: pd.Series, span: int) -> pd.Series:\n    return s.ewm(span=span, adjust=False).mean()\n```\n`adjust=False` matches the recursive definition you'd write by hand.")

    add("data",
        "Implement a function that returns a stratified sample of a DataFrame.",
        "```python\nimport pandas as pd\n\ndef stratified_sample(df: pd.DataFrame, by: str, frac: float, seed: int = 0) -> pd.DataFrame:\n    return df.groupby(by, group_keys=False).apply(lambda g: g.sample(frac=frac, random_state=seed))\n```\nProportional sampling preserves class balance.")

    add("data",
        "Write a function that exports a DataFrame to a SQLite database.",
        "```python\nimport pandas as pd\nimport sqlite3\n\ndef to_sqlite(df: pd.DataFrame, path: str, table: str) -> None:\n    with sqlite3.connect(path) as conn:\n        df.to_sql(table, conn, if_exists='replace', index=False)\n```\n`if_exists='replace'` blows the table away; use `'append'` for incremental writes.")

    add("data",
        "Implement a function that computes the cumulative sum by group.",
        "```python\nimport pandas as pd\n\ndef cumsum_by_group(df: pd.DataFrame, group: str, col: str) -> pd.Series:\n    return df.groupby(group)[col].cumsum()\n```\nGroup-wise cumulative aggregations are one-liners thanks to `groupby`.")

    add("data",
        "Write a function that filters outliers via IQR.",
        "```python\nimport pandas as pd\n\ndef drop_iqr_outliers(df: pd.DataFrame, col: str, k: float = 1.5) -> pd.DataFrame:\n    q1, q3 = df[col].quantile([0.25, 0.75])\n    iqr = q3 - q1\n    return df[(df[col] >= q1 - k * iqr) & (df[col] <= q3 + k * iqr)]\n```\nTukey's k=1.5 is the textbook default; bump to 3 for more permissive filtering.")

    add("data",
        "Implement a function that converts a CSV to parquet for faster reads.",
        "```python\nimport pandas as pd\n\ndef csv_to_parquet(src: str, dst: str) -> None:\n    pd.read_csv(src).to_parquet(dst, compression='snappy', index=False)\n```\nFor very large files, iterate with `chunksize=` and write each chunk to a partitioned dataset.")

    add("data",
        "Write a function that draws a 95% confidence interval as a shaded band.",
        "```python\nimport matplotlib.pyplot as plt\nimport numpy as np\n\ndef plot_ci(x, y, lo, hi, out: str) -> None:\n    fig, ax = plt.subplots()\n    ax.plot(x, y)\n    ax.fill_between(x, lo, hi, alpha=0.2)\n    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)\n```\n`fill_between` is the matplotlib idiom for confidence bands.")

    add("data",
        "Implement a function that computes the kurtosis of a numeric column.",
        "```python\nimport pandas as pd\n\ndef kurtosis(s: pd.Series) -> float:\n    return float(s.kurt())\n```\nPandas reports excess kurtosis (subtracts 3) -- 0 for a normal distribution.")

    add("data",
        "Write a function that returns the most-correlated feature pairs.",
        "```python\nimport pandas as pd\n\ndef top_correlations(df: pd.DataFrame, n: int = 10) -> pd.Series:\n    corr = df.select_dtypes('number').corr().abs()\n    pairs = corr.where(~pd.np.tril(pd.np.ones(corr.shape, dtype=bool))).stack()\n    return pairs.nlargest(n)\n```\nMask the lower triangle so each pair appears once.")

    add("data",
        "Implement a function that makes a numpy array contiguous.",
        "```python\nimport numpy as np\n\ndef ensure_contiguous(arr: np.ndarray) -> np.ndarray:\n    return np.ascontiguousarray(arr)\n```\nNeeded before passing to libraries that require row-major memory (e.g. C extensions).")

    add("data",
        "Write a function that computes mean-by-group and joins it back.",
        "```python\nimport pandas as pd\n\ndef join_group_mean(df: pd.DataFrame, group: str, col: str) -> pd.DataFrame:\n    df = df.copy()\n    df[f'{col}_group_mean'] = df.groupby(group)[col].transform('mean')\n    return df\n```\n`transform` keeps the result aligned with the original index.")

    add("data",
        "Implement a function that bins a continuous column into named ranges.",
        "```python\nimport pandas as pd\n\ndef bucket(s: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:\n    return pd.cut(s, bins=edges, labels=labels, include_lowest=True)\n```\n`include_lowest=True` makes the leftmost edge inclusive.")

    add("data",
        "Write a function that returns the best linear fit slope and intercept.",
        "```python\nimport numpy as np\n\ndef linfit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:\n    slope, intercept = np.polyfit(x, y, 1)\n    return float(slope), float(intercept)\n```\nFor more sophisticated fits use `scipy.stats.linregress` -- it returns p-values too.")

    add("data",
        "Implement a function that computes geometric mean.",
        "```python\nimport numpy as np\n\ndef geomean(arr: np.ndarray) -> float:\n    arr = np.asarray(arr, dtype=float)\n    return float(np.exp(np.log(arr).mean()))\n```\nLog-then-mean avoids numerical overflow on large inputs.")

    add("data",
        "Write a function that computes Spearman rank correlation.",
        "```python\nimport pandas as pd\n\ndef spearman(a: pd.Series, b: pd.Series) -> float:\n    return float(a.corr(b, method='spearman'))\n```\nSpearman is robust to monotonic but nonlinear relationships -- Pearson misses those.")

    add("data",
        "Implement a function that returns a one-step ahead naive forecast.",
        "```python\nimport pandas as pd\n\ndef naive_forecast(s: pd.Series) -> pd.Series:\n    return s.shift(1)\n```\nThe naive baseline beats many fancy models on stable series.")

    add("data",
        "Write a function that decomposes a time series into trend/seasonal/residual.",
        "```python\nimport pandas as pd\nfrom statsmodels.tsa.seasonal import STL\n\ndef decompose(s: pd.Series, period: int) -> pd.DataFrame:\n    stl = STL(s, period=period).fit()\n    return pd.DataFrame({'trend': stl.trend, 'seasonal': stl.seasonal, 'resid': stl.resid})\n```\nSTL is more robust than classical decomposition for irregular series.")

    add("data",
        "Implement a function that converts categorical labels to integer codes.",
        "```python\nimport pandas as pd\n\ndef encode_codes(s: pd.Series) -> tuple[pd.Series, pd.Index]:\n    cat = s.astype('category')\n    return cat.cat.codes, cat.cat.categories\n```\nReturn the categories so you can decode predictions back to labels.")

    add("data",
        "Write a function that drops columns with too many missing values.",
        "```python\nimport pandas as pd\n\ndef drop_high_na(df: pd.DataFrame, thresh: float = 0.5) -> pd.DataFrame:\n    keep = [c for c in df.columns if df[c].isna().mean() < thresh]\n    return df[keep]\n```\nDefault drops columns more than half missing; tweak per dataset.")

    add("data",
        "Implement a function that creates a balanced binary dataset.",
        "```python\nimport pandas as pd\n\ndef balance(df: pd.DataFrame, label: str, seed: int = 0) -> pd.DataFrame:\n    counts = df[label].value_counts()\n    n = counts.min()\n    return df.groupby(label, group_keys=False).apply(lambda g: g.sample(n=n, random_state=seed))\n```\nDownsamples the majority class to match the minority count.")

    add("data",
        "Write a function that converts a DataFrame to a numpy feature matrix.",
        "```python\nimport pandas as pd\nimport numpy as np\n\ndef to_matrix(df: pd.DataFrame, label_col: str) -> tuple[np.ndarray, np.ndarray]:\n    y = df[label_col].to_numpy()\n    X = df.drop(columns=[label_col]).select_dtypes('number').to_numpy()\n    return X, y\n```\nFor mixed types, encode categoricals with `OneHotEncoder` first.")

    add("data",
        "Implement a function that exports a DataFrame to gzip-compressed CSV.",
        "```python\nimport pandas as pd\n\ndef to_csv_gz(df: pd.DataFrame, path: str) -> None:\n    df.to_csv(path, index=False, compression='gzip')\n```\nPandas infers compression from the file extension if you skip the parameter.")

    add("data",
        "Write a function that loads a Hugging Face dataset.",
        "```python\nfrom datasets import load_dataset\n\ndef load(name: str, split: str = 'train'):\n    return load_dataset(name, split=split)\n```\nDatasets are Arrow-backed, memory-efficient, and stream from disk.")

    add("data",
        "Implement a function that adds a one-hot encoding to a Series.",
        "```python\nimport pandas as pd\n\ndef onehot(s: pd.Series) -> pd.DataFrame:\n    return pd.get_dummies(s, prefix=s.name)\n```\n`prefix=s.name` namespaces the dummy columns nicely.")

    add("data",
        "Write a function that returns columns ordered by missing-value rate.",
        "```python\nimport pandas as pd\n\ndef missing_rate(df: pd.DataFrame) -> pd.Series:\n    return df.isna().mean().sort_values(ascending=False)\n```\nFirst-pass diagnostic: which columns are mostly empty?")

    add("data",
        "Implement a function that returns groups whose count exceeds a threshold.",
        "```python\nimport pandas as pd\n\ndef large_groups(df: pd.DataFrame, group: str, n: int) -> pd.DataFrame:\n    counts = df.groupby(group).size()\n    return df[df[group].isin(counts[counts >= n].index)]\n```\nFilter pattern works for any group-wise condition.")

    add("data",
        "Write a function that removes duplicate rows by all columns.",
        "```python\nimport pandas as pd\n\ndef dedupe(df: pd.DataFrame) -> pd.DataFrame:\n    return df.drop_duplicates()\n```\nFor large data, `drop_duplicates(subset=key_cols, ignore_index=True)` is usually what you want.")

    add("data",
        "Implement a function that counts occurrences with `value_counts(normalize=True)`.",
        "```python\nimport pandas as pd\n\ndef rates(s: pd.Series) -> pd.Series:\n    return s.value_counts(normalize=True, dropna=False)\n```\n`dropna=False` includes NaN as its own row -- often what you want for diagnostics.")

    add("data",
        "Write a function that loads CSV with custom NA values.",
        "```python\nimport pandas as pd\n\ndef load(path: str) -> pd.DataFrame:\n    return pd.read_csv(path, na_values=['', 'NA', 'N/A', '?', '-', 'null'])\n```\nReal-world data has many flavors of 'missing'.")

    add("data",
        "Implement a function that fills missing values forward then backward.",
        "```python\nimport pandas as pd\n\ndef fill_ffill_bfill(s: pd.Series) -> pd.Series:\n    return s.ffill().bfill()\n```\nForward then backward handles trailing missing values too.")

    add("data",
        "Write a function that creates a pivoted summary of two categoricals.",
        "```python\nimport pandas as pd\n\ndef cross_tab(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:\n    return pd.crosstab(df[a], df[b], normalize='index')\n```\n`normalize='index'` gives row probabilities -- useful for proportion comparisons.")

    add("data",
        "Implement a function that identifies multimodal distributions roughly.",
        "```python\nimport numpy as np\nfrom scipy.signal import find_peaks\n\ndef has_multiple_modes(values: np.ndarray, bins: int = 30) -> bool:\n    counts, _ = np.histogram(values, bins=bins)\n    peaks, _ = find_peaks(counts, prominence=counts.max() * 0.1)\n    return len(peaks) > 1\n```\nProminence threshold of 10% of max prevents counting noise-level bumps.")

    add("data",
        "Write a function that returns a DataFrame's schema as a dict.",
        "```python\nimport pandas as pd\n\ndef schema(df: pd.DataFrame) -> dict:\n    return {c: str(df[c].dtype) for c in df.columns}\n```\nFor JSON/YAML serialization of the schema.")

    add("data",
        "Implement a function that returns rolling quantiles.",
        "```python\nimport pandas as pd\n\ndef rolling_quantile(s: pd.Series, window: int, q: float) -> pd.Series:\n    return s.rolling(window).quantile(q)\n```\nRolling p95 etc. is a one-liner; useful for monitoring tails.")

    add("data",
        "Write a function that computes a feature's information value.",
        "```python\nimport numpy as np\nimport pandas as pd\n\ndef iv(df: pd.DataFrame, feature: str, label: str) -> float:\n    grp = df.groupby(feature)[label].agg(['count', 'sum'])\n    grp.columns = ['total', 'pos']\n    grp['neg'] = grp['total'] - grp['pos']\n    grp['pos_rate'] = grp['pos'] / grp['pos'].sum()\n    grp['neg_rate'] = grp['neg'] / grp['neg'].sum()\n    grp['woe'] = np.log((grp['pos_rate'] + 1e-9) / (grp['neg_rate'] + 1e-9))\n    return float(((grp['pos_rate'] - grp['neg_rate']) * grp['woe']).sum())\n```\nClassic credit-risk feature ranking; small epsilon avoids log(0).")

    add("data",
        "Implement a function that creates lag features.",
        "```python\nimport pandas as pd\n\ndef lag_features(df: pd.DataFrame, col: str, lags: list[int]) -> pd.DataFrame:\n    df = df.copy()\n    for k in lags:\n        df[f'{col}_lag_{k}'] = df[col].shift(k)\n    return df\n```\nLags are fundamental for time-series modeling.")

    add("data",
        "Write a function that returns a Series describing dtype distribution.",
        "```python\nimport pandas as pd\n\ndef dtype_counts(df: pd.DataFrame) -> pd.Series:\n    return df.dtypes.value_counts()\n```\nQuick check before deciding which columns to numerically encode.")

    # ---- debugging (28) ----
    add("debug",
        "Why does this list comprehension throw `NameError: name 'x' is not defined`?\n```python\nresult = [x for _ in range(3)]\nprint(x)\n```",
        "The `print(x)` references a variable that was never bound at module scope. The list comprehension's `x` (if it existed) would only live in the comprehension's own scope in Python 3. As written, the loop variable is `_`, so even renaming wouldn't help.\n\nFix: bind `x` outside, or use the value you actually want to print. If you wanted the last loop value, set it explicitly: `xs = [i for i in range(3)]; print(xs[-1])`.")

    add("debug",
        "I'm seeing `RuntimeError: dictionary changed size during iteration`. What's wrong?\n```python\nfor k, v in d.items():\n    if v == 0:\n        del d[k]\n```",
        "You can't mutate a dict while iterating its items view. Iterate over a snapshot of the keys: `for k in list(d.keys()):` or build a new dict: `d = {k: v for k, v in d.items() if v != 0}`. The dict-comprehension form is the more idiomatic fix.")

    add("debug",
        "This default-argument list keeps growing across calls. Why?\n```python\ndef append_to(item, target=[]):\n    target.append(item)\n    return target\n```",
        "Mutable default arguments are evaluated once at function-definition time, not per call. Every call without an explicit `target` shares the same list. Fix by sentinel: `def append_to(item, target=None): target = [] if target is None else target; ...`. Ruff's `B006` rule flags this automatically.")

    add("debug",
        "My closure captures the wrong loop variable. Why?\n```python\nfns = [lambda: i for i in range(3)]\nprint([f() for f in fns])  # [2, 2, 2]\n```",
        "Closures capture variables by reference, not value, and `i` keeps being rebound through the loop. By the time you call the lambdas, `i == 2`. Fix with a default-argument trick: `[lambda i=i: i for i in range(3)]`, which captures the current value via the default-argument-evaluation rule.")

    add("debug",
        "This recursive function hits `RecursionError`. The input isn't even that big.\n```python\ndef sum_to(n):\n    return 0 if n == 0 else n + sum_to(n - 1)\n```",
        "Python's default recursion limit is around 1000. For n above that, recursion overflows. Two fixes: convert to iteration (`sum(range(n + 1))`), or raise the limit with `sys.setrecursionlimit` if you really need recursion. Iteration is almost always the right answer in Python -- there's no tail-call optimization.")

    add("debug",
        "`json.dumps` raises `TypeError: Object of type Decimal is not JSON serializable`. What now?",
        "`json.dumps` doesn't know about `decimal.Decimal`. Pass a `default=` callback that returns a serializable form: `json.dumps(obj, default=lambda x: str(x) if isinstance(x, Decimal) else (_ for _ in ()).throw(TypeError))`. Or use `simplejson` which supports Decimal natively.")

    add("debug",
        "I get `UnicodeDecodeError` reading a file. How do I fix it?",
        "Open with explicit encoding instead of the platform default: `open(path, encoding='utf-8')`. If the file might contain invalid bytes, use `errors='replace'` (subs in U+FFFD) or `errors='ignore'`. For unknown encodings, sniff with `chardet`/`charset-normalizer`.")

    add("debug",
        "My `requests.get` hangs forever. What did I miss?",
        "You didn't pass a `timeout`. The default is no timeout; a hanging server hangs your code indefinitely. Always: `requests.get(url, timeout=10.0)`. Same applies to httpx and the stdlib `urlopen` -- explicit timeouts are non-negotiable in production.")

    add("debug",
        "`os.path.join('/etc', '/passwd')` returns `'/passwd'`. Why?",
        "`os.path.join` discards earlier parts when a later one is absolute -- it follows the convention used by the shell. If you want strict child-of-base joins, use `pathlib`: `Path('/etc') / 'passwd'` works the same way, but `Path('/etc').joinpath('passwd')` is explicit. For sanitizing user-supplied paths, resolve and verify ancestry.")

    add("debug",
        "Why does `==` work on my objects but `in list` doesn't find them?",
        "`list.__contains__` uses `==` by default, but if you've defined `__eq__` you also need `__hash__` for set/dict membership. For lists specifically, the cause is usually that `__eq__` works on identity (the default) instead of value. Define `__eq__` and `__hash__` consistently -- the `@dataclass(eq=True, frozen=True)` decorator does this for you.")

    add("debug",
        "`datetime.now()` returns naive datetimes that confuse my code. What do I do?",
        "Always use timezone-aware datetimes: `datetime.now(tz=timezone.utc)`. Mixing naive and aware values raises `TypeError` on comparison. Set a project-wide rule: every datetime that crosses a function boundary is UTC-aware. The `whenever` library on PyPI enforces this at the type level.")

    add("debug",
        "My `pytest` test passes individually but fails in the suite. Why?",
        "Test isolation issue. Likely culprits: module-level state (caches, singletons), file system mutations, environment variables, or random seeds set globally. Use `monkeypatch` for env vars, `tmp_path` for files, and reset shared state in fixtures. Run with `-p no:randomly` or `--randomize` to surface order dependencies.")

    add("debug",
        "I'm getting `ModuleNotFoundError` even though pip says the package is installed. Why?",
        "You're running with a different Python than `pip install` used. Run `python -m pip install ...` (not bare `pip`) to install into the interpreter you'll actually run. Confirm with `python -c 'import sys; print(sys.executable)'`. Virtual environments solve this once you're consistent.")

    add("debug",
        "Why is my `for x in queryset:` running multiple SQL queries?",
        "Django's `QuerySet` lazily fetches related fields; iterating triggers N+1 queries when accessing FK relationships. Use `select_related` for one-to-one/FK and `prefetch_related` for many-to-many or reverse FK. Enable Django Debug Toolbar in dev to see this happening live.")

    add("debug",
        "`AttributeError: 'NoneType' object has no attribute 'X'` -- but I returned the value. Why?",
        "A function with a missing `return` returns `None` implicitly. Either you forgot `return` or there's an early branch that exits without one. Run `mypy --strict` to surface the type-narrowing issue, or add an explicit `assert result is not None` to fail fast at the right line.")

    add("debug",
        "My async code runs sequentially despite `asyncio.gather`. What's wrong?",
        "You're awaiting each call before passing to `gather`: `await asyncio.gather(await foo(), await bar())` is sequential. Pass coroutines, not awaited results: `await asyncio.gather(foo(), bar())`. Same trap with comprehensions: `[await f() for f in fns]` is serial; use `asyncio.gather(*[f() for f in fns])`.")

    add("debug",
        "I'm seeing different floating-point results across machines. Why?",
        "Floating-point operations are deterministic on a single CPU but reduction order (sums of large arrays, parallel kernels) can differ across hardware. Pin BLAS thread count, sort before summing if precision matters, or use `math.fsum` for exact sums. For ML workloads, also pin CUDA's deterministic algorithms.")

    add("debug",
        "`subprocess.run([cmd], shell=True)` is dropping arguments. Why?",
        "`shell=True` expects a single string, not a list. Either pass a string (`shell=True`, with shell-escape risks) or a list (`shell=False`, which is the safer default). Mixing them silently passes only the first list element to the shell. Almost always you want `subprocess.run([cmd, arg1, arg2])` without `shell=True`.")

    add("debug",
        "Why does my `try/except` not catch the exception in a thread?",
        "Exceptions in threads don't propagate to the parent. Use `concurrent.futures.ThreadPoolExecutor` and call `future.result()` -- that re-raises. With raw threads, set up `threading.excepthook` or a result queue that captures exceptions. The `concurrent.futures` API was created exactly to fix this.")

    add("debug",
        "My logger doesn't show DEBUG messages despite `level=logging.DEBUG`. Why?",
        "You probably set the logger level but the root handler is at WARNING. Either configure both (`logging.basicConfig(level=logging.DEBUG)` configures the root) or set the level on the handler too. The hierarchy is logger -> handler -> filter; messages must pass each gate.")

    add("debug",
        "`NameError: name '__file__' is not defined` -- what's going on?",
        "`__file__` doesn't exist in interactive REPL or `exec()`'d code. Guard with `__file__ if '__file__' in globals() else None`. For locating package data, prefer `importlib.resources` -- it works in zipped packages and frozen builds where `__file__` doesn't.")

    add("debug",
        "Pickle is failing to load my saved object. What likely changed?",
        "Pickle stores fully-qualified class paths. If you renamed/moved the class or changed the package layout, unpickling fails. Either keep the path stable, set `__reduce__` for forward-compat, or use a stable serialization (JSON, msgpack, protobuf). Don't unpickle untrusted data -- pickle can execute arbitrary code.")

    add("debug",
        "My script works on macOS but `path.glob('**/*')` misses files on Linux. Why?",
        "`pathlib.Path.glob` is case-sensitive; macOS's default filesystem is case-insensitive. Match case explicitly, or use `rglob` with normalized lowercase comparisons. Also remember `**` only matches directories; `**/*` is correct but `**` alone is not.")

    add("debug",
        "I'm getting `SSL: CERTIFICATE_VERIFY_FAILED` for a valid HTTPS URL. Why?",
        "Often a stale system CA bundle. Update with `pip install --upgrade certifi` and ensure your code uses it (`requests` and `httpx` already do). On macOS, run `Install Certificates.command` from the Python install. Never disable verification (`verify=False`) in production -- it makes you vulnerable to MITM attacks.")

    add("debug",
        "My regex `\\d+` is matching across newlines unexpectedly. Why?",
        "It's not -- `\\d+` doesn't match newlines. The likely culprit is `re.DOTALL` or another regex (often `.+`) being too greedy and the `\\d+` part matching across what looks like a wrap. Print the exact match (`match.group()`) and check whether the input has CR/LF that you didn't expect.")

    add("debug",
        "`pd.read_csv` is reading my numeric column as object. Why?",
        "There's a non-numeric value somewhere in the column -- often whitespace, a stray comma, or a literal `'NA'` not in the default NA values. Pass `na_values=['NA', '?', '']`, or post-process with `pd.to_numeric(s, errors='coerce')` to turn unparseable values into NaN.")

    add("debug",
        "My `multiprocessing.Pool` workers keep dying with `Can't pickle...` errors.",
        "Multiprocessing pickles the function and its arguments to send to the worker. Closures, lambdas, and locally-defined classes can't be pickled. Move the worker function to module scope, and use `functools.partial` for binding parameters. Or switch to `multiprocessing.get_context('forkserver')` to avoid pickling on Linux.")

    add("debug",
        "`assert isinstance(x, list)` works in tests but fails in production. Why?",
        "Python ignores `assert` when run with `-O` (optimized mode). Don't use `assert` for runtime invariants -- only for documentation/development checks. Use explicit `if not isinstance(x, list): raise TypeError(...)` for production guards.")
