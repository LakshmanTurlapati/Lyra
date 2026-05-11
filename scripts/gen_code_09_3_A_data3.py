# SPDX-License-Identifier: MIT
"""Data science / numpy / pandas / matplotlib entries (100 total)."""


def register_all(add):
    # ---- 3-msg (60) ----
    add("data",
        "Write a function that loads a CSV with pandas and returns the first 5 rows.",
        "```python\nimport pandas as pd\n\ndef preview(path: str) -> pd.DataFrame:\n    return pd.read_csv(path).head()\n```\n`head()` defaults to 5 rows. For a quick look at a large file, also pass `nrows=5` to avoid loading everything.")

    add("data",
        "Write a numpy expression that creates a 5x5 identity matrix.",
        "```python\nimport numpy as np\n\nI = np.eye(5)\n```\n`np.eye(n)` is the standard. Use `np.eye(n, k=1)` for a shifted diagonal (super-diagonal of ones).")

    add("data",
        "Write a function that computes the column-wise mean of a numpy array.",
        "```python\nimport numpy as np\n\ndef col_means(a: np.ndarray) -> np.ndarray:\n    return a.mean(axis=0)\n```\n`axis=0` collapses rows (computes per-column). `axis=1` would give per-row means.")

    add("data",
        "Write a pandas one-liner that adds a 'total' column equal to the sum of two existing columns.",
        "```python\ndf['total'] = df['a'] + df['b']\n```\nElement-wise arithmetic across columns is vectorized — much faster than `apply`.")

    add("data",
        "Write a function that filters a DataFrame to rows where a column matches a value.",
        "```python\nimport pandas as pd\n\ndef filter_by(df: pd.DataFrame, col: str, value) -> pd.DataFrame:\n    return df[df[col] == value]\n```\nBoolean indexing is the fundamental pandas pattern.")

    add("data",
        "Write a function that returns the unique values of a DataFrame column sorted.",
        "```python\nimport pandas as pd\n\ndef sorted_unique(df: pd.DataFrame, col: str) -> list:\n    return sorted(df[col].dropna().unique())\n```\nDropping NaN avoids comparison errors when sorting mixed dtypes.")

    add("data",
        "Write a function that drops rows with any null values in a DataFrame.",
        "```python\nimport pandas as pd\n\ndef drop_nulls(df: pd.DataFrame) -> pd.DataFrame:\n    return df.dropna()\n```\nDefault `dropna()` drops any row containing a NaN. Pass `subset=` to limit which columns count.")

    add("data",
        "Write a function that fills NaN values in a DataFrame with the column mean.",
        "```python\nimport pandas as pd\n\ndef fill_with_mean(df: pd.DataFrame) -> pd.DataFrame:\n    return df.fillna(df.mean(numeric_only=True))\n```\n`numeric_only=True` avoids errors on string columns.")

    add("data",
        "Write a function that groups a DataFrame by one column and sums another.",
        "```python\nimport pandas as pd\n\ndef sum_by(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:\n    return df.groupby(group_col)[value_col].sum()\n```\nThe result is a Series indexed by the group values.")

    add("data",
        "Write a function that computes the correlation matrix of a DataFrame.",
        "```python\nimport pandas as pd\n\ndef corr_matrix(df: pd.DataFrame) -> pd.DataFrame:\n    return df.corr(numeric_only=True)\n```\nDefault is Pearson correlation. Pass `method='spearman'` for rank correlation.")

    add("data",
        "Write a function that one-hot encodes a categorical column.",
        "```python\nimport pandas as pd\n\ndef one_hot(df: pd.DataFrame, col: str) -> pd.DataFrame:\n    return pd.get_dummies(df, columns=[col])\n```\n`get_dummies` is the easy path. For sklearn pipelines, prefer `OneHotEncoder` since it remembers the categories at fit time.")

    add("data",
        "Write a numpy expression that normalizes an array to mean 0 and standard deviation 1.",
        "```python\nimport numpy as np\n\ndef standardize(a: np.ndarray) -> np.ndarray:\n    return (a - a.mean()) / a.std()\n```\nVectorized and broadcast-friendly. Use `ddof=1` if you want sample std.")

    add("data",
        "Write a function that creates a pandas DataFrame from a list of dicts.",
        "```python\nimport pandas as pd\n\ndef from_records(rows: list[dict]) -> pd.DataFrame:\n    return pd.DataFrame(rows)\n```\nThe DataFrame constructor accepts dicts directly — column order follows insertion order.")

    add("data",
        "Write a numpy function that computes Euclidean distance between two points.",
        "```python\nimport numpy as np\n\ndef euclid(a: np.ndarray, b: np.ndarray) -> float:\n    return float(np.linalg.norm(a - b))\n```\n`np.linalg.norm` defaults to L2. Wrap with `float()` to return a Python float.")

    add("data",
        "Write a function that returns the row with the highest value in a column.",
        "```python\nimport pandas as pd\n\ndef row_with_max(df: pd.DataFrame, col: str) -> pd.Series:\n    return df.loc[df[col].idxmax()]\n```\n`idxmax()` returns the index label of the max — pair with `.loc[]` to get the row.")

    add("data",
        "Write a function that pivots a DataFrame from long to wide format.",
        "```python\nimport pandas as pd\n\ndef to_wide(df: pd.DataFrame, index: str, columns: str, values: str) -> pd.DataFrame:\n    return df.pivot(index=index, columns=columns, values=values)\n```\nUse `pivot_table` instead if there are duplicate (index, column) pairs you want to aggregate.")

    add("data",
        "Write a function that converts a DataFrame to a CSV string.",
        "```python\nimport pandas as pd\n\ndef to_csv_string(df: pd.DataFrame) -> str:\n    return df.to_csv(index=False)\n```\nPassing `index=False` skips the unnamed first column most consumers don't want.")

    add("data",
        "Write a function that plots a histogram of a DataFrame column with matplotlib.",
        "```python\nimport matplotlib.pyplot as plt\nimport pandas as pd\n\ndef hist(df: pd.DataFrame, col: str, bins: int = 30) -> None:\n    df[col].hist(bins=bins)\n    plt.xlabel(col)\n    plt.ylabel('count')\n    plt.show()\n```\nFor publication plots, replace `plt.show()` with `plt.savefig(path)`.")

    add("data",
        "Write a function that resamples a time-indexed DataFrame to daily means.",
        "```python\nimport pandas as pd\n\ndef daily_mean(df: pd.DataFrame) -> pd.DataFrame:\n    return df.resample('D').mean()\n```\nThe DataFrame must have a `DatetimeIndex` for `resample` to work; convert via `pd.to_datetime` and `set_index` if needed.")

    add("data",
        "Write a numpy snippet that creates a 100x100 array of random floats from N(0,1).",
        "```python\nimport numpy as np\n\nrng = np.random.default_rng(seed=0)\nA = rng.standard_normal((100, 100))\n```\nThe newer `default_rng` API is preferred over `np.random.randn` for reproducible RNGs.")

    add("data",
        "Write a function that computes the rolling 7-day mean for a Series.",
        "```python\nimport pandas as pd\n\ndef rolling_7d(s: pd.Series) -> pd.Series:\n    return s.rolling(window=7, min_periods=1).mean()\n```\n`min_periods=1` gives values from the first element onward, instead of NaN for the first 6.")

    add("data",
        "Write a function that returns the top-N rows by a given column.",
        "```python\nimport pandas as pd\n\ndef top_n(df: pd.DataFrame, col: str, n: int = 10) -> pd.DataFrame:\n    return df.nlargest(n, col)\n```\n`nlargest` is faster than `sort_values().head(n)` because it avoids fully sorting.")

    add("data",
        "Write a function that converts a DataFrame's date column to datetime dtype.",
        "```python\nimport pandas as pd\n\ndef parse_date(df: pd.DataFrame, col: str) -> pd.DataFrame:\n    df = df.copy()\n    df[col] = pd.to_datetime(df[col], errors='coerce')\n    return df\n```\n`errors='coerce'` turns unparseable values into NaT, which is usually safer than raising.")

    add("data",
        "Write a function that returns the percentage change of a Series.",
        "```python\nimport pandas as pd\n\ndef pct_change(s: pd.Series) -> pd.Series:\n    return s.pct_change()\n```\nThe first element is always NaN since there's no prior value.")

    add("data",
        "Write a function that loads multiple CSV files into one DataFrame.",
        "```python\nimport pandas as pd\nfrom pathlib import Path\n\ndef load_csvs(directory: str) -> pd.DataFrame:\n    files = sorted(Path(directory).glob('*.csv'))\n    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)\n```\n`ignore_index=True` renumbers rows so duplicate indices don't collide.")

    add("data",
        "Write a function that returns the Pearson correlation of two columns.",
        "```python\nimport pandas as pd\n\ndef corr(df: pd.DataFrame, a: str, b: str) -> float:\n    return df[a].corr(df[b])\n```\nDefault is Pearson; pass `method='spearman'` or `'kendall'` for alternatives.")

    add("data",
        "Write a numpy function that returns the indices of the N largest values in an array.",
        "```python\nimport numpy as np\n\ndef top_n_indices(a: np.ndarray, n: int) -> np.ndarray:\n    idx = np.argpartition(a, -n)[-n:]\n    return idx[np.argsort(a[idx])[::-1]]\n```\n`argpartition` is O(n); the small subsequent sort costs O(n log n).")

    add("data",
        "Write a function that scatter-plots two columns of a DataFrame.",
        "```python\nimport matplotlib.pyplot as plt\nimport pandas as pd\n\ndef scatter(df: pd.DataFrame, x: str, y: str) -> None:\n    plt.scatter(df[x], df[y])\n    plt.xlabel(x)\n    plt.ylabel(y)\n    plt.show()\n```\nFor large datasets, switch to `plt.hexbin` to avoid overplotting.")

    add("data",
        "Write a function that computes a frequency table of a categorical column.",
        "```python\nimport pandas as pd\n\ndef freq(df: pd.DataFrame, col: str) -> pd.Series:\n    return df[col].value_counts()\n```\n`value_counts` already sorts descending. Pass `normalize=True` for proportions.")

    add("data",
        "Write a function that joins two DataFrames on a column.",
        "```python\nimport pandas as pd\n\ndef inner_join(a: pd.DataFrame, b: pd.DataFrame, key: str) -> pd.DataFrame:\n    return a.merge(b, on=key, how='inner')\n```\nThe `how=` parameter switches between inner/left/right/outer.")

    add("data",
        "Write a function that creates a DataFrame summary (mean, std, min, max).",
        "```python\nimport pandas as pd\n\ndef summary(df: pd.DataFrame) -> pd.DataFrame:\n    return df.describe()\n```\n`describe()` returns a multi-stat summary by default. Pass `include='all'` for object columns too.")

    add("data",
        "Write a function that calculates the minimum, mean, median, max of a numpy array.",
        "```python\nimport numpy as np\n\ndef stats(a: np.ndarray) -> dict:\n    return {\n        'min': float(a.min()),\n        'mean': float(a.mean()),\n        'median': float(np.median(a)),\n        'max': float(a.max()),\n    }\n```\nWrapping with `float()` returns native Python floats, which JSON-serialize cleanly.")

    add("data",
        "Write a numpy function that flattens a 2D matrix to 1D.",
        "```python\nimport numpy as np\n\ndef flatten(a: np.ndarray) -> np.ndarray:\n    return a.ravel()\n```\n`ravel` returns a view when possible (cheaper than `flatten`, which always copies).")

    add("data",
        "Write a pandas one-liner that renames the column 'old' to 'new'.",
        "```python\ndf = df.rename(columns={'old': 'new'})\n```\nUse `inplace=True` if you want to mutate, but new pandas idioms tend to favor immutable returns.")

    add("data",
        "Write a function that returns a DataFrame with only numeric columns.",
        "```python\nimport pandas as pd\n\ndef numeric_only(df: pd.DataFrame) -> pd.DataFrame:\n    return df.select_dtypes(include='number')\n```\n`select_dtypes` is the standard way to pick columns by dtype.")

    add("data",
        "Write a function that computes the cumulative sum of a Series.",
        "```python\nimport pandas as pd\n\ndef cumsum(s: pd.Series) -> pd.Series:\n    return s.cumsum()\n```\nComplements `cumprod`, `cummax`, `cummin`.")

    add("data",
        "Write a function that bins a Series into 10 equal-width buckets.",
        "```python\nimport pandas as pd\n\ndef bin10(s: pd.Series) -> pd.Series:\n    return pd.cut(s, bins=10)\n```\n`pd.cut` does equal-width bins; `pd.qcut` does equal-frequency (deciles).")

    add("data",
        "Write a function that returns the value at the 95th percentile of a Series.",
        "```python\nimport pandas as pd\n\ndef p95(s: pd.Series) -> float:\n    return float(s.quantile(0.95))\n```\nPandas defaults to linear interpolation between data points.")

    add("data",
        "Write a numpy snippet that sets all negative values to 0.",
        "```python\nimport numpy as np\n\na[a < 0] = 0\n```\nBoolean indexing with assignment is the standard 'clip from below' idiom; `np.maximum(a, 0)` is the non-mutating equivalent.")

    add("data",
        "Write a function that produces a side-by-side bar chart from two pandas Series.",
        "```python\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\n\ndef bar_compare(a: pd.Series, b: pd.Series) -> None:\n    x = np.arange(len(a))\n    width = 0.4\n    plt.bar(x - width/2, a.values, width, label=a.name)\n    plt.bar(x + width/2, b.values, width, label=b.name)\n    plt.xticks(x, a.index)\n    plt.legend()\n    plt.show()\n```\nThe `width/2` offset puts the bars on either side of each tick.")

    add("data",
        "Write a function that returns rows where any column is null.",
        "```python\nimport pandas as pd\n\ndef rows_with_nulls(df: pd.DataFrame) -> pd.DataFrame:\n    return df[df.isna().any(axis=1)]\n```\n`isna().any(axis=1)` produces a boolean per row, which we index with.")

    add("data",
        "Write a function that converts a DataFrame's float columns to float32 to save memory.",
        "```python\nimport pandas as pd\n\ndef shrink_floats(df: pd.DataFrame) -> pd.DataFrame:\n    out = df.copy()\n    for c in out.select_dtypes(include='float64').columns:\n        out[c] = out[c].astype('float32')\n    return out\n```\nHalves memory at the cost of precision; check that your downstream usage is OK with that.")

    add("data",
        "Write a numpy expression that computes pairwise distances between rows.",
        "```python\nimport numpy as np\nfrom scipy.spatial.distance import cdist\n\ndef pairwise(a: np.ndarray) -> np.ndarray:\n    return cdist(a, a)\n```\nFor large `a`, scipy's `cdist` is much faster than a manual loop and supports many distance metrics.")

    add("data",
        "Write a function that adds a row to a DataFrame.",
        "```python\nimport pandas as pd\n\ndef append_row(df: pd.DataFrame, row: dict) -> pd.DataFrame:\n    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)\n```\n`DataFrame.append` was deprecated; use `pd.concat` with a single-row DataFrame.")

    add("data",
        "Write a function that returns the columns of a DataFrame containing more than 50% NaNs.",
        "```python\nimport pandas as pd\n\ndef sparse_columns(df: pd.DataFrame, threshold: float = 0.5) -> list[str]:\n    null_frac = df.isna().mean()\n    return list(null_frac[null_frac > threshold].index)\n```\n`isna().mean()` gives null fraction per column.")

    add("data",
        "Write a numpy function that computes the moving average with a kernel of length k.",
        "```python\nimport numpy as np\n\ndef moving_average(a: np.ndarray, k: int) -> np.ndarray:\n    kernel = np.ones(k) / k\n    return np.convolve(a, kernel, mode='valid')\n```\n`mode='valid'` outputs only positions with full overlap; `'same'` pads to keep length.")

    add("data",
        "Write a function that extracts year and month columns from a date column.",
        "```python\nimport pandas as pd\n\ndef add_year_month(df: pd.DataFrame, col: str) -> pd.DataFrame:\n    df = df.copy()\n    dt = pd.to_datetime(df[col])\n    df[f'{col}_year'] = dt.dt.year\n    df[f'{col}_month'] = dt.dt.month\n    return df\n```\nThe `.dt` accessor exposes datetime fields on a Series.")

    add("data",
        "Write a function that returns the largest correlation pair (excluding the diagonal).",
        "```python\nimport pandas as pd\n\ndef top_correlation(df: pd.DataFrame) -> tuple[str, str, float]:\n    corr = df.corr(numeric_only=True).abs()\n    upper = corr.where(~pd.np.tri(*corr.shape, dtype=bool))\n    pair = upper.stack().idxmax()\n    return pair[0], pair[1], float(upper.stack().max())\n```\nMasking the lower triangle (and diagonal) avoids reporting `(x, x)` and counting each pair twice.")

    add("data",
        "Write a numpy expression that softmaxes each row of a 2D array.",
        "```python\nimport numpy as np\n\ndef softmax(x: np.ndarray) -> np.ndarray:\n    z = x - x.max(axis=1, keepdims=True)\n    e = np.exp(z)\n    return e / e.sum(axis=1, keepdims=True)\n```\nSubtracting the max prevents overflow in `exp` for large inputs.")

    add("data",
        "Write a function that linearly interpolates missing values in a Series.",
        "```python\nimport pandas as pd\n\ndef interpolate(s: pd.Series) -> pd.Series:\n    return s.interpolate(method='linear')\n```\nHandy for time-series gaps. For seasonal data, try `method='time'` with a DatetimeIndex.")

    add("data",
        "Write a function that returns the kurtosis and skewness of each numeric column.",
        "```python\nimport pandas as pd\n\ndef shape_stats(df: pd.DataFrame) -> pd.DataFrame:\n    num = df.select_dtypes(include='number')\n    return pd.DataFrame({'skew': num.skew(), 'kurtosis': num.kurt()})\n```\nPositive skew = right tail. Positive excess kurtosis = heavier tails than normal.")

    add("data",
        "Write a function that clusters numeric data with KMeans (sklearn) and returns the labels.",
        "```python\nimport numpy as np\nfrom sklearn.cluster import KMeans\n\ndef cluster(x: np.ndarray, k: int = 3, seed: int = 0) -> np.ndarray:\n    return KMeans(n_clusters=k, random_state=seed, n_init='auto').fit_predict(x)\n```\n`n_init='auto'` is the new sklearn default that runs k-means++ with multiple seeds.")

    add("data",
        "Write a function that splits a dataset into train/test with sklearn.",
        "```python\nfrom sklearn.model_selection import train_test_split\n\ndef split(x, y, test_size: float = 0.2, seed: int = 0):\n    return train_test_split(x, y, test_size=test_size, random_state=seed)\n```\nReturns `(x_train, x_test, y_train, y_test)`. Use `stratify=y` for classification to preserve class balance.")

    add("data",
        "Write a function that returns the top-K terms by TF-IDF for a corpus.",
        "```python\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nimport numpy as np\n\ndef top_terms(corpus: list[str], k: int = 10) -> list[str]:\n    vec = TfidfVectorizer(stop_words='english')\n    matrix = vec.fit_transform(corpus)\n    scores = np.asarray(matrix.sum(axis=0)).ravel()\n    terms = vec.get_feature_names_out()\n    idx = np.argsort(scores)[::-1][:k]\n    return [terms[i] for i in idx]\n```\nSumming TF-IDF scores across docs surfaces terms that are both frequent and distinctive.")

    add("data",
        "Write a function that fits a linear regression with sklearn and returns the coefficients.",
        "```python\nimport numpy as np\nfrom sklearn.linear_model import LinearRegression\n\ndef fit_linreg(x: np.ndarray, y: np.ndarray) -> dict:\n    model = LinearRegression().fit(x, y)\n    return {'coef': model.coef_.tolist(), 'intercept': float(model.intercept_)}\n```\n`coef_` is per-feature; `intercept_` is the bias term.")

    add("data",
        "Write a function that turns a DataFrame's column into a sklearn-ready feature matrix and target vector.",
        "```python\nimport pandas as pd\nimport numpy as np\n\ndef Xy(df: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:\n    y = df[target].to_numpy()\n    X = df.drop(columns=[target]).to_numpy()\n    return X, y\n```\n`to_numpy()` is preferred over `.values` (more explicit and a stable API).")

    add("data",
        "Write a function that computes precision, recall, and F1 for binary predictions.",
        "```python\nfrom sklearn.metrics import precision_recall_fscore_support\n\ndef metrics(y_true, y_pred) -> dict:\n    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')\n    return {'precision': p, 'recall': r, 'f1': f1}\n```\nUse `average='macro'` or `'weighted'` for multi-class.")

    add("data",
        "Write a numpy snippet that draws 1000 samples from a Poisson(lambda=3) distribution.",
        "```python\nimport numpy as np\n\nrng = np.random.default_rng(seed=42)\nsamples = rng.poisson(lam=3, size=1000)\n```\nThe new `Generator` API is the recommended path; legacy `np.random.poisson` still works.")

    add("data",
        "Write a function that produces a confusion matrix using sklearn.",
        "```python\nfrom sklearn.metrics import confusion_matrix\nimport numpy as np\n\ndef cm(y_true, y_pred) -> np.ndarray:\n    return confusion_matrix(y_true, y_pred)\n```\nPair with `ConfusionMatrixDisplay` for plotting.")

    # ---- 5-msg with follow-ups (40) ----
    add("data",
        "Write a function that loads a CSV file with pandas.",
        "```python\nimport pandas as pd\n\ndef load(path: str) -> pd.DataFrame:\n    return pd.read_csv(path)\n```\nMinimal wrapper.",
        "Add type inference for date columns and handle a comma decimal separator.",
        "```python\nimport pandas as pd\n\ndef load(path: str, date_cols: list[str] | None = None) -> pd.DataFrame:\n    df = pd.read_csv(path, decimal=',')\n    if date_cols:\n        for c in date_cols:\n            df[c] = pd.to_datetime(df[c], errors='coerce')\n    return df\n```\n`decimal=','` is essential for European CSVs. Parsing dates after load gives more control over `errors` than `parse_dates=`.")

    add("data",
        "Write a function that returns the mean of a numpy array.",
        "```python\nimport numpy as np\n\ndef mean(a: np.ndarray) -> float:\n    return float(a.mean())\n```\nWrapped to a Python float for JSON-friendliness.",
        "What if I want to ignore NaNs?",
        "```python\nimport numpy as np\n\ndef mean(a: np.ndarray) -> float:\n    return float(np.nanmean(a))\n```\nThe `nan*` family of numpy functions skips NaNs. If the entire array is NaN, you'll get a NaN result and a runtime warning.")

    add("data",
        "Write a function that returns the column means of a DataFrame.",
        "```python\nimport pandas as pd\n\ndef means(df: pd.DataFrame) -> pd.Series:\n    return df.mean(numeric_only=True)\n```\n`numeric_only=True` is needed since pandas 2.x to suppress warnings on object columns.",
        "Group by a category column and return per-group means.",
        "```python\nimport pandas as pd\n\ndef group_means(df: pd.DataFrame, by: str) -> pd.DataFrame:\n    return df.groupby(by).mean(numeric_only=True)\n```\nThe groupby result is itself a DataFrame indexed by the grouping value.")

    add("data",
        "Write a function that creates a line chart of a Series.",
        "```python\nimport matplotlib.pyplot as plt\nimport pandas as pd\n\ndef line_chart(s: pd.Series) -> None:\n    s.plot()\n    plt.show()\n```\nPandas' `.plot()` uses matplotlib under the hood.",
        "Save it to a PNG instead of showing.",
        "```python\nimport matplotlib.pyplot as plt\nimport pandas as pd\n\ndef line_chart(s: pd.Series, path: str) -> None:\n    fig, ax = plt.subplots()\n    s.plot(ax=ax)\n    fig.savefig(path, dpi=150, bbox_inches='tight')\n    plt.close(fig)\n```\n`bbox_inches='tight'` trims whitespace; `plt.close(fig)` releases memory in batch jobs.")

    add("data",
        "Write a numpy function that rotates a 2D array 90 degrees.",
        "```python\nimport numpy as np\n\ndef rot90(a: np.ndarray) -> np.ndarray:\n    return np.rot90(a)\n```\n`np.rot90` does the right thing without manual transposition.",
        "Generalize to N rotations of 90 degrees.",
        "```python\nimport numpy as np\n\ndef rot90n(a: np.ndarray, n: int = 1) -> np.ndarray:\n    return np.rot90(a, k=n)\n```\nPositive k rotates counter-clockwise; negative goes clockwise.")

    add("data",
        "Write a function that returns the rolling mean of a Series.",
        "```python\nimport pandas as pd\n\ndef rolling_mean(s: pd.Series, window: int) -> pd.Series:\n    return s.rolling(window=window).mean()\n```\nFirst `window-1` values will be NaN.",
        "Make it weight recent values more heavily — exponential weighted mean.",
        "```python\nimport pandas as pd\n\ndef ema(s: pd.Series, span: int) -> pd.Series:\n    return s.ewm(span=span, adjust=False).mean()\n```\n`adjust=False` matches the classic EMA recurrence used in finance.")

    add("data",
        "Write a function that returns rows of a DataFrame where col > threshold.",
        "```python\nimport pandas as pd\n\ndef filter_above(df: pd.DataFrame, col: str, threshold: float) -> pd.DataFrame:\n    return df[df[col] > threshold]\n```\nBoolean indexing — the bread and butter of pandas filtering.",
        "Combine with another filter on a different column using AND.",
        "```python\nimport pandas as pd\n\ndef filter_two(df: pd.DataFrame, col_a: str, gt_a: float, col_b: str, eq_b) -> pd.DataFrame:\n    return df[(df[col_a] > gt_a) & (df[col_b] == eq_b)]\n```\nParenthesize each condition: `&` has higher precedence than `>` in Python, so missing parens here is a classic bug.")

    add("data",
        "Write a function that plots two columns of a DataFrame as a scatter.",
        "```python\nimport matplotlib.pyplot as plt\nimport pandas as pd\n\ndef scatter(df: pd.DataFrame, x: str, y: str) -> None:\n    df.plot.scatter(x=x, y=y)\n    plt.show()\n```\nPandas' built-in plotter saves a few imports.",
        "Color the points by a third column.",
        "```python\nimport matplotlib.pyplot as plt\nimport pandas as pd\n\ndef scatter_colored(df: pd.DataFrame, x: str, y: str, hue: str) -> None:\n    df.plot.scatter(x=x, y=y, c=hue, colormap='viridis')\n    plt.show()\n```\nUsing the `c=` parameter with a column name maps values to a colormap automatically.")

    add("data",
        "Write a function that converts a DataFrame to JSON records.",
        "```python\nimport pandas as pd\n\ndef to_records(df: pd.DataFrame) -> str:\n    return df.to_json(orient='records')\n```\n`orient='records'` produces a list of dicts — the most common shape for APIs.",
        "Make dates ISO 8601 instead of Unix milliseconds.",
        "```python\nimport pandas as pd\n\ndef to_records(df: pd.DataFrame) -> str:\n    return df.to_json(orient='records', date_format='iso')\n```\n`date_format='iso'` is much more interoperable than the default Unix-millis format.")

    add("data",
        "Write a numpy snippet that creates a 5x5 array filled with the value 7.",
        "```python\nimport numpy as np\n\nA = np.full((5, 5), 7)\n```\n`np.full` is more explicit than `np.zeros + 7`.",
        "What if I want a different dtype, say float32?",
        "```python\nimport numpy as np\n\nA = np.full((5, 5), 7, dtype=np.float32)\n```\nThe `dtype` argument controls memory layout and arithmetic behavior downstream.")

    add("data",
        "Write a function that finds the duplicate rows in a DataFrame.",
        "```python\nimport pandas as pd\n\ndef dupes(df: pd.DataFrame) -> pd.DataFrame:\n    return df[df.duplicated()]\n```\nDefault keeps the first occurrence (so we see only the second+ copies).",
        "Show all copies, not just the duplicates.",
        "```python\nimport pandas as pd\n\ndef all_dupes(df: pd.DataFrame) -> pd.DataFrame:\n    return df[df.duplicated(keep=False)]\n```\n`keep=False` flags every row that appears more than once.")

    add("data",
        "Write a function that scales features to [0, 1] range.",
        "```python\nimport numpy as np\n\ndef minmax(x: np.ndarray) -> np.ndarray:\n    lo, hi = x.min(axis=0), x.max(axis=0)\n    return (x - lo) / (hi - lo)\n```\nClassic min-max scaling per column.",
        "Use sklearn's MinMaxScaler so I can re-apply the same transform to test data.",
        "```python\nimport numpy as np\nfrom sklearn.preprocessing import MinMaxScaler\n\ndef fit_scaler(x_train: np.ndarray) -> MinMaxScaler:\n    return MinMaxScaler().fit(x_train)\n\ndef apply_scaler(scaler: MinMaxScaler, x: np.ndarray) -> np.ndarray:\n    return scaler.transform(x)\n```\nFitting on train then applying to test prevents data leakage of test-set min/max into training.")

    add("data",
        "Write a function that loads parquet files lazily.",
        "```python\nimport pandas as pd\n\ndef load_parquet(path: str) -> pd.DataFrame:\n    return pd.read_parquet(path)\n```\npandas reads parquet via pyarrow or fastparquet.",
        "What if the file is huge — can I read just one column?",
        "```python\nimport pandas as pd\n\ndef load_column(path: str, column: str) -> pd.Series:\n    return pd.read_parquet(path, columns=[column])[column]\n```\nParquet's columnar format means reading one column doesn't touch the others on disk — a big win for wide tables.")

    add("data",
        "Write a function that fits a polynomial to (x, y) data.",
        "```python\nimport numpy as np\n\ndef fit_poly(x, y, deg: int = 2) -> np.ndarray:\n    return np.polyfit(x, y, deg)\n```\nReturns coefficients in descending degree order: `c_n, c_{n-1}, ..., c_0`.",
        "Switch to numpy.polynomial which uses ascending order and is the recommended modern API.",
        "```python\nimport numpy as np\nfrom numpy.polynomial import Polynomial\n\ndef fit_poly(x, y, deg: int = 2) -> Polynomial:\n    return Polynomial.fit(x, y, deg=deg).convert()\n```\n`Polynomial` objects can be evaluated with `()`, summed, multiplied, etc. — much more ergonomic than raw coefficient arrays.")

    add("data",
        "Write a function that computes the area under a curve given (x, y) samples.",
        "```python\nimport numpy as np\n\ndef auc(x, y) -> float:\n    return float(np.trapz(y, x))\n```\n`np.trapz` uses the trapezoidal rule. For smoother data, `scipy.integrate.simpson` is more accurate.",
        "Use scipy's simpson rule for higher accuracy.",
        "```python\nfrom scipy.integrate import simpson\n\ndef auc(x, y) -> float:\n    return float(simpson(y=y, x=x))\n```\nSimpson's rule fits parabolas to triplets of points; for smooth functions it's much more accurate at the same sample density.")

    add("data",
        "Write a function that returns the indices of NaN values in a Series.",
        "```python\nimport pandas as pd\n\ndef nan_indices(s: pd.Series) -> list:\n    return s.index[s.isna()].tolist()\n```\nUses boolean indexing of the Index to extract labels.",
        "Now do the same for a 2D DataFrame, returning (row, col) pairs.",
        "```python\nimport pandas as pd\n\ndef nan_locations(df: pd.DataFrame) -> list[tuple]:\n    mask = df.isna()\n    rows, cols = mask.values.nonzero()\n    return [(df.index[r], df.columns[c]) for r, c in zip(rows, cols)]\n```\n`mask.values.nonzero()` gives the underlying int positions; we map them back to labels.")

    add("data",
        "Write a function that bins a Series and returns counts per bin.",
        "```python\nimport pandas as pd\n\ndef bin_counts(s: pd.Series, bins: int = 10) -> pd.Series:\n    return pd.cut(s, bins=bins).value_counts().sort_index()\n```\n`sort_index` keeps bins in interval order rather than count order.",
        "Use custom bin edges instead of equal width.",
        "```python\nimport pandas as pd\n\ndef bin_counts(s: pd.Series, edges: list[float]) -> pd.Series:\n    return pd.cut(s, bins=edges, include_lowest=True).value_counts().sort_index()\n```\n`include_lowest=True` makes the first bin closed on the left so the minimum value isn't excluded.")

    add("data",
        "Write a function that converts a DataFrame to a long format using melt.",
        "```python\nimport pandas as pd\n\ndef to_long(df: pd.DataFrame, id_vars: list[str]) -> pd.DataFrame:\n    return df.melt(id_vars=id_vars, var_name='variable', value_name='value')\n```\n`melt` is the inverse of `pivot`. The non-id columns become rows.",
        "Filter the melted result to drop rows where value is NaN.",
        "```python\nimport pandas as pd\n\ndef to_long(df: pd.DataFrame, id_vars: list[str]) -> pd.DataFrame:\n    melted = df.melt(id_vars=id_vars, var_name='variable', value_name='value')\n    return melted.dropna(subset=['value'])\n```\nDropping NaN values after melt is a common cleanup step when columns sparsely have data.")

    add("data",
        "Write a function that computes the IQR (interquartile range) of a Series.",
        "```python\nimport pandas as pd\n\ndef iqr(s: pd.Series) -> float:\n    return float(s.quantile(0.75) - s.quantile(0.25))\n```\nIQR is robust to outliers, unlike standard deviation.",
        "Use it to detect outliers (Tukey's rule).",
        "```python\nimport pandas as pd\n\ndef outliers(s: pd.Series, k: float = 1.5) -> pd.Series:\n    q1, q3 = s.quantile(0.25), s.quantile(0.75)\n    iqr = q3 - q1\n    return s[(s < q1 - k * iqr) | (s > q3 + k * iqr)]\n```\nValues beyond `Q1 - 1.5*IQR` or `Q3 + 1.5*IQR` are flagged. `k=3` gives Tukey's 'far out' threshold.")

    add("data",
        "Write a function that loads a JSON file as a DataFrame.",
        "```python\nimport pandas as pd\n\ndef load_json(path: str) -> pd.DataFrame:\n    return pd.read_json(path)\n```\n`read_json` handles both records and split orientations automatically in many cases.",
        "What if the JSON is nested and I need to flatten it?",
        "```python\nimport json\nimport pandas as pd\nfrom pathlib import Path\n\ndef load_json(path: str) -> pd.DataFrame:\n    data = json.loads(Path(path).read_text())\n    return pd.json_normalize(data)\n```\n`json_normalize` flattens nested dicts using dot-separated column names; great for API responses.")

    add("data",
        "Write a function that fits a logistic regression with sklearn.",
        "```python\nfrom sklearn.linear_model import LogisticRegression\n\ndef fit_logreg(x, y, seed: int = 0):\n    return LogisticRegression(max_iter=1000, random_state=seed).fit(x, y)\n```\nThe default `max_iter=100` often warns; bumping to 1000 is a sensible default.",
        "Add cross-validated AUC to evaluate it.",
        "```python\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import cross_val_score\n\ndef fit_and_score(x, y, seed: int = 0) -> float:\n    model = LogisticRegression(max_iter=1000, random_state=seed)\n    scores = cross_val_score(model, x, y, scoring='roc_auc', cv=5)\n    return float(scores.mean())\n```\n`scoring='roc_auc'` is appropriate for binary classification with class imbalance.")

    add("data",
        "Write a function that converts a DataFrame to a Markdown table.",
        "```python\nimport pandas as pd\n\ndef to_md(df: pd.DataFrame) -> str:\n    return df.to_markdown(index=False)\n```\nRequires `tabulate`. Hide the index unless it's meaningful.",
        "Use prettytable or another fallback if tabulate isn't installed.",
        "```python\nimport pandas as pd\n\ndef to_md(df: pd.DataFrame) -> str:\n    try:\n        return df.to_markdown(index=False)\n    except ImportError:\n        # Manual fallback\n        cols = df.columns.tolist()\n        header = '| ' + ' | '.join(cols) + ' |'\n        sep = '| ' + ' | '.join(['---'] * len(cols)) + ' |'\n        rows = [\n            '| ' + ' | '.join(str(v) for v in row) + ' |'\n            for row in df.itertuples(index=False, name=None)\n        ]\n        return '\\n'.join([header, sep, *rows])\n```\nA hand-rolled fallback keeps the function dependency-free when needed.")

    add("data",
        "Write a function that creates a heatmap from a correlation matrix.",
        "```python\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport pandas as pd\n\ndef corr_heatmap(df: pd.DataFrame) -> None:\n    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', vmin=-1, vmax=1)\n    plt.show()\n```\nFixing `vmin/vmax` to [-1, 1] makes color comparisons meaningful across plots.",
        "Skip the seaborn dependency.",
        "```python\nimport matplotlib.pyplot as plt\nimport pandas as pd\n\ndef corr_heatmap(df: pd.DataFrame) -> None:\n    corr = df.corr(numeric_only=True)\n    fig, ax = plt.subplots()\n    im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)\n    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')\n    ax.set_yticks(range(len(corr.columns)), corr.columns)\n    fig.colorbar(im)\n    plt.tight_layout()\n    plt.show()\n```\nMatplotlib alone is enough for a basic heatmap, just more verbose.")

    add("data",
        "Write a function that returns the daily returns of a price Series.",
        "```python\nimport pandas as pd\n\ndef returns(prices: pd.Series) -> pd.Series:\n    return prices.pct_change().dropna()\n```\n`pct_change` divides each row by the previous one and subtracts 1.",
        "I want log-returns instead, which are more additive.",
        "```python\nimport numpy as np\nimport pandas as pd\n\ndef log_returns(prices: pd.Series) -> pd.Series:\n    return np.log(prices / prices.shift(1)).dropna()\n```\nLog-returns sum across periods to give the cumulative log-return — convenient for time-series modeling.")

    add("data",
        "Write a numpy function that computes the cosine similarity between two vectors.",
        "```python\nimport numpy as np\n\ndef cos_sim(a: np.ndarray, b: np.ndarray) -> float:\n    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))\n```\nIdentical vectors give 1.0; orthogonal give 0; anti-parallel give -1.",
        "Generalize to a (n, d) matrix vs a (d,) vector, returning the (n,) similarity vector.",
        "```python\nimport numpy as np\n\ndef cos_sim_batch(M: np.ndarray, v: np.ndarray) -> np.ndarray:\n    M_norm = np.linalg.norm(M, axis=1)\n    v_norm = np.linalg.norm(v)\n    return (M @ v) / (M_norm * v_norm + 1e-12)\n```\nThe `+ 1e-12` prevents division by zero on rows that happen to be the zero vector.")

    add("data",
        "Write a function that produces a quick sanity-check report of a DataFrame.",
        "```python\nimport pandas as pd\n\ndef sanity(df: pd.DataFrame) -> dict:\n    return {\n        'shape': df.shape,\n        'dtypes': {c: str(t) for c, t in df.dtypes.items()},\n        'nulls': df.isna().sum().to_dict(),\n        'duplicates': int(df.duplicated().sum()),\n    }\n```\nGreat first call on a fresh dataset.",
        "Add per-column basic stats (min/max/unique counts).",
        "```python\nimport pandas as pd\n\ndef sanity(df: pd.DataFrame) -> dict:\n    cols = {}\n    for c in df.columns:\n        s = df[c]\n        info = {\n            'dtype': str(s.dtype),\n            'nulls': int(s.isna().sum()),\n            'unique': int(s.nunique()),\n        }\n        if pd.api.types.is_numeric_dtype(s):\n            info['min'] = float(s.min()) if s.notna().any() else None\n            info['max'] = float(s.max()) if s.notna().any() else None\n        cols[c] = info\n    return {'shape': df.shape, 'duplicates': int(df.duplicated().sum()), 'columns': cols}\n```\nGuarding `min/max` with `notna().any()` avoids errors on all-null columns.")

    add("data",
        "Write a function that creates a histogram with KDE overlay.",
        "```python\nimport seaborn as sns\nimport matplotlib.pyplot as plt\nimport pandas as pd\n\ndef hist_kde(s: pd.Series) -> None:\n    sns.histplot(s, kde=True)\n    plt.show()\n```\nseaborn's `histplot(kde=True)` is the cleanest one-liner.",
        "Without seaborn please.",
        "```python\nimport matplotlib.pyplot as plt\nimport numpy as np\nfrom scipy.stats import gaussian_kde\n\ndef hist_kde(values) -> None:\n    fig, ax = plt.subplots()\n    ax.hist(values, bins=30, density=True, alpha=0.4)\n    xs = np.linspace(min(values), max(values), 200)\n    ax.plot(xs, gaussian_kde(values)(xs))\n    plt.show()\n```\n`density=True` rescales the histogram bars to integrate to 1 so they're comparable to the KDE curve.")

    add("data",
        "Write a function that returns the n most recent rows of a time-indexed DataFrame.",
        "```python\nimport pandas as pd\n\ndef recent(df: pd.DataFrame, n: int) -> pd.DataFrame:\n    return df.sort_index().tail(n)\n```\nSorting first guarantees correctness even if the index isn't ordered.",
        "Just last 30 days instead of last n rows.",
        "```python\nimport pandas as pd\n\ndef last_30_days(df: pd.DataFrame) -> pd.DataFrame:\n    cutoff = df.index.max() - pd.Timedelta(days=30)\n    return df[df.index > cutoff]\n```\nRelative to the data's most recent timestamp, not 'now', so it works for offline analysis.")

    add("data",
        "Write a function that bins a series into deciles.",
        "```python\nimport pandas as pd\n\ndef deciles(s: pd.Series) -> pd.Series:\n    return pd.qcut(s, q=10, labels=False)\n```\n`qcut` gives equal-frequency bins; `labels=False` returns integer labels 0-9.",
        "Handle the case where there are too many ties to form 10 bins.",
        "```python\nimport pandas as pd\n\ndef deciles(s: pd.Series) -> pd.Series:\n    return pd.qcut(s, q=10, labels=False, duplicates='drop')\n```\n`duplicates='drop'` collapses duplicate bin edges; you may end up with fewer than 10 bins, which is the honest answer for tied data.")

    add("data",
        "Write a function that applies a row-wise transformation efficiently.",
        "```python\nimport pandas as pd\n\ndef row_apply(df: pd.DataFrame, fn) -> pd.Series:\n    return df.apply(fn, axis=1)\n```\nSlow because it boxes each row as a Series. Use only when vectorizing isn't possible.",
        "How can I vectorize instead of using apply?",
        "If `fn` is a simple arithmetic combination, you can express it directly on columns:\n```python\n# Slow:\n# df.apply(lambda r: r['a'] + r['b'] * 2, axis=1)\n# Fast:\nresult = df['a'] + df['b'] * 2\n```\nFor more complex element-wise logic, `np.where`, `np.select`, or `df.eval()` typically run 10-100x faster than `apply(axis=1)`. Reach for `apply` only when the row-level logic genuinely requires Python control flow that can't be expressed in column terms.")

    add("data",
        "Write a function that joins three DataFrames on a common key.",
        "```python\nimport pandas as pd\n\ndef triple_join(a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame, key: str) -> pd.DataFrame:\n    return a.merge(b, on=key).merge(c, on=key)\n```\nSequential merges. The default join type is inner.",
        "Make all joins outer (no rows lost).",
        "```python\nimport pandas as pd\n\ndef triple_outer(a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame, key: str) -> pd.DataFrame:\n    return a.merge(b, on=key, how='outer').merge(c, on=key, how='outer')\n```\nOuter joins keep all keys from all sides, filling missing values with NaN.")

    add("data",
        "Write a function that computes the precision at K for ranked predictions.",
        "```python\ndef precision_at_k(predicted: list, relevant: set, k: int) -> float:\n    top_k = predicted[:k]\n    return sum(1 for p in top_k if p in relevant) / k if k else 0.0\n```\nReturns the fraction of the top-K predictions that are in the relevant set.",
        "Now compute mean average precision across many queries.",
        "```python\ndef average_precision(predicted: list, relevant: set) -> float:\n    if not relevant:\n        return 0.0\n    hits = 0\n    total = 0.0\n    for i, p in enumerate(predicted, 1):\n        if p in relevant:\n            hits += 1\n            total += hits / i\n    return total / len(relevant)\n\ndef mean_average_precision(queries: list[tuple[list, set]]) -> float:\n    if not queries:\n        return 0.0\n    return sum(average_precision(p, r) for p, r in queries) / len(queries)\n```\nMAP rewards both finding relevant items and ranking them early.")

    add("data",
        "Write a function that returns the top correlated features against a target.",
        "```python\nimport pandas as pd\n\ndef top_corr(df: pd.DataFrame, target: str, n: int = 10) -> pd.Series:\n    corrs = df.corr(numeric_only=True)[target].drop(target)\n    return corrs.abs().sort_values(ascending=False).head(n)\n```\nAbsolute value treats positive and negative correlation as equally informative.",
        "Show both correlation magnitude and sign in the output.",
        "```python\nimport pandas as pd\n\ndef top_corr(df: pd.DataFrame, target: str, n: int = 10) -> pd.DataFrame:\n    corrs = df.corr(numeric_only=True)[target].drop(target)\n    out = pd.DataFrame({'corr': corrs, 'abs_corr': corrs.abs()})\n    return out.sort_values('abs_corr', ascending=False).head(n).drop(columns='abs_corr')\n```\nSort by absolute value but report the signed value — preserves direction information for interpretation.")

    add("data",
        "Write a function that computes the cumulative distribution function (CDF) of a sample.",
        "```python\nimport numpy as np\n\ndef ecdf(values) -> tuple[np.ndarray, np.ndarray]:\n    x = np.sort(values)\n    y = np.arange(1, len(x) + 1) / len(x)\n    return x, y\n```\nSorted values on x; rank/N on y. The classic empirical CDF.",
        "Plot it.",
        "```python\nimport matplotlib.pyplot as plt\nimport numpy as np\n\ndef plot_ecdf(values) -> None:\n    x = np.sort(values)\n    y = np.arange(1, len(x) + 1) / len(x)\n    plt.step(x, y, where='post')\n    plt.xlabel('value')\n    plt.ylabel('F(x)')\n    plt.show()\n```\n`where='post'` produces the staircase shape that's correct for empirical CDFs.")

    add("data",
        "Write a function that computes the silhouette score for a clustering.",
        "```python\nfrom sklearn.metrics import silhouette_score\n\ndef silhouette(x, labels) -> float:\n    return float(silhouette_score(x, labels))\n```\nValues near 1: well-clustered. Near 0: borderline. Negative: likely wrong cluster.",
        "Use it to pick the best K for KMeans across a range.",
        "```python\nimport numpy as np\nfrom sklearn.cluster import KMeans\nfrom sklearn.metrics import silhouette_score\n\ndef best_k(x: np.ndarray, k_range=range(2, 11)) -> int:\n    best, best_score = None, -1.0\n    for k in k_range:\n        labels = KMeans(n_clusters=k, n_init='auto', random_state=0).fit_predict(x)\n        score = silhouette_score(x, labels)\n        if score > best_score:\n            best, best_score = k, score\n    return best  # type: ignore[return-value]\n```\nNote silhouette is undefined for k=1, hence starting the range at 2.")

    add("data",
        "Write a function that prints a confusion matrix nicely.",
        "```python\nfrom sklearn.metrics import confusion_matrix\n\ndef show_cm(y_true, y_pred, labels=None) -> None:\n    cm = confusion_matrix(y_true, y_pred, labels=labels)\n    if labels is None:\n        labels = sorted(set(y_true) | set(y_pred))\n    width = max(len(str(l)) for l in labels) + 2\n    print(' ' * width + ''.join(f'{l:>{width}}' for l in labels))\n    for i, row in enumerate(cm):\n        print(f'{labels[i]:>{width}}' + ''.join(f'{v:>{width}}' for v in row))\n```\nAligns columns by label width for readability.",
        "Use seaborn instead for a heatmap.",
        "```python\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.metrics import confusion_matrix\n\ndef plot_cm(y_true, y_pred, labels=None) -> None:\n    cm = confusion_matrix(y_true, y_pred, labels=labels)\n    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')\n    plt.xlabel('predicted')\n    plt.ylabel('actual')\n    plt.show()\n```\n`fmt='d'` keeps cell values as integers; `cmap='Blues'` is a friendly diverging colormap.")

    add("data",
        "Write a function that returns the index where a Series first crosses a threshold.",
        "```python\nimport pandas as pd\n\ndef first_crossing(s: pd.Series, threshold: float):\n    above = s > threshold\n    if not above.any():\n        return None\n    return above.idxmax()  # idxmax returns first True\n```\n`idxmax` on a boolean Series returns the label of the first True (since True > False).",
        "Now find every crossing (both up and down), not just the first.",
        "```python\nimport pandas as pd\n\ndef crossings(s: pd.Series, threshold: float) -> pd.Index:\n    above = s > threshold\n    flips = above.ne(above.shift(fill_value=above.iloc[0]))\n    return s.index[flips]\n```\n`ne(shift)` flags positions where the boolean changes from the previous step.")

    add("data",
        "Write a function that downsamples an image with numpy.",
        "```python\nimport numpy as np\n\ndef downsample(img: np.ndarray, factor: int = 2) -> np.ndarray:\n    return img[::factor, ::factor]\n```\nSimplest form: stride-based subsampling. Doesn't antialias.",
        "Add antialiasing with a box filter.",
        "```python\nimport numpy as np\n\ndef downsample(img: np.ndarray, factor: int = 2) -> np.ndarray:\n    h, w = img.shape[:2]\n    h2, w2 = h // factor, w // factor\n    cropped = img[: h2 * factor, : w2 * factor]\n    if img.ndim == 2:\n        return cropped.reshape(h2, factor, w2, factor).mean(axis=(1, 3))\n    return cropped.reshape(h2, factor, w2, factor, -1).mean(axis=(1, 3))\n```\nReshape into blocks and average each block. For higher quality, use `cv2.resize` or `PIL.Image.thumbnail`.")

    add("data",
        "Write a function that imputes missing values using the column median.",
        "```python\nimport pandas as pd\n\ndef impute_median(df: pd.DataFrame) -> pd.DataFrame:\n    out = df.copy()\n    num = out.select_dtypes(include='number').columns\n    out[num] = out[num].fillna(out[num].median())\n    return out\n```\nMedian is robust to outliers; mean is more efficient when data is symmetric.",
        "Use sklearn's SimpleImputer instead so the strategy is reusable for test data.",
        "```python\nimport pandas as pd\nfrom sklearn.impute import SimpleImputer\n\ndef fit_imputer(df: pd.DataFrame) -> SimpleImputer:\n    return SimpleImputer(strategy='median').fit(df.select_dtypes(include='number'))\n\ndef apply_imputer(imputer: SimpleImputer, df: pd.DataFrame) -> pd.DataFrame:\n    out = df.copy()\n    num = out.select_dtypes(include='number').columns\n    out[num] = imputer.transform(out[num])\n    return out\n```\nThe imputer remembers the medians from training data so test rows are filled with the same values, avoiding leakage.")

    add("data",
        "Write a function that returns the geometric mean of a Series.",
        "```python\nimport numpy as np\nimport pandas as pd\n\ndef geomean(s: pd.Series) -> float:\n    return float(np.exp(np.log(s).mean()))\n```\nLog-mean-exp form is numerically stable for products of many small or large numbers.",
        "Handle zeros gracefully (their log is -inf).",
        "```python\nimport numpy as np\nimport pandas as pd\n\ndef geomean(s: pd.Series, eps: float = 1e-12) -> float:\n    if (s <= 0).any():\n        return 0.0\n    return float(np.exp(np.log(s + eps).mean()))\n```\nIf any value is zero (or negative), the strict geometric mean is 0/undefined. The `eps` is just a numerical safety net.")

    add("data",
        "Write a function that returns the entropy of a probability distribution.",
        "```python\nimport numpy as np\n\ndef entropy(p) -> float:\n    p = np.asarray(p, dtype=float)\n    p = p[p > 0]  # log(0) is -inf\n    return float(-(p * np.log2(p)).sum())\n```\nIn bits (using log2). For a uniform distribution over n outcomes, entropy = log2(n).",
        "Add cross-entropy for two distributions.",
        "```python\nimport numpy as np\n\ndef cross_entropy(p, q, eps: float = 1e-12) -> float:\n    p = np.asarray(p, dtype=float)\n    q = np.asarray(q, dtype=float)\n    return float(-(p * np.log2(q + eps)).sum())\n```\nMeasures how many bits are needed to encode samples from `p` using a code optimized for `q`. The `eps` shields against `log(0)`.")
