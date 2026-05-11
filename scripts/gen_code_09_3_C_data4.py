# SPDX-License-Identifier: MIT
"""Data science / numpy / pandas / matplotlib entries (~105). Wave 3 part 4."""


def register_all(add):
    add("data",
        "Show how to load a CSV into a DataFrame and inspect it.",
        "```python\nimport pandas as pd\n\ndf = pd.read_csv('data.csv')\nprint(df.shape, df.dtypes)\nprint(df.head())\n```\nCheck `dtypes` early -- silently inferred string columns are a common source of bugs.")

    add("data",
        "Show how to filter a DataFrame by multiple conditions.",
        "```python\nimport pandas as pd\n\nmask = (df['age'] >= 18) & (df['country'] == 'US')\nadults_us = df.loc[mask]\n```\nUse `&` and `|` (not `and`/`or`) and parenthesize each comparison -- precedence will burn you otherwise.")

    add("data",
        "Show how to compute group-wise aggregates with pandas.",
        "```python\nimport pandas as pd\n\nsummary = df.groupby('region', as_index=False).agg(\n    total_sales=('sales', 'sum'),\n    avg_sales=('sales', 'mean'),\n    n=('sales', 'size'),\n)\n```\nNamed aggregation keeps column names readable.")

    add("data",
        "Show how to read a Parquet file with pandas.",
        "```python\nimport pandas as pd\n\ndf = pd.read_parquet('events.parquet')\n```\nParquet is much faster than CSV and preserves dtypes -- prefer it for analytics.")

    add("data",
        "Show how to merge two DataFrames on a common key.",
        "```python\nimport pandas as pd\n\nmerged = pd.merge(orders, customers, on='customer_id', how='left')\n```\n`how='left'` keeps every order even if the customer is missing; check for nulls afterwards.")

    add("data",
        "Show how to pivot a long DataFrame to wide.",
        "```python\nimport pandas as pd\n\nwide = df.pivot_table(index='date', columns='product', values='sales', aggfunc='sum').fillna(0)\n```\n`pivot_table` aggregates duplicates; raw `pivot` errors on duplicates.")

    add("data",
        "Show how to compute a rolling 7-day mean.",
        "```python\nimport pandas as pd\n\ndf = df.sort_values('date')\ndf['sales_7d'] = df['sales'].rolling(window=7, min_periods=1).mean()\n```\n`min_periods=1` lets you compute an estimate before the window is full.")

    add("data",
        "Show how to bin numeric values into categories.",
        "```python\nimport pandas as pd\n\ndf['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 120], labels=['minor','young','mid','senior'])\n```\n`pd.qcut` instead if you want equal-frequency bins.")

    add("data",
        "Show how to compute summary statistics across all numeric columns.",
        "```python\nimport pandas as pd\n\nprint(df.describe(include='all'))\n```\n`include='all'` adds non-numeric columns to the summary.")

    add("data",
        "Show how to fill missing values with the column mean.",
        "```python\nimport pandas as pd\n\nfor col in df.select_dtypes('number'):\n    df[col] = df[col].fillna(df[col].mean())\n```\nFor production, persist the imputation values so test-time rows get the same fill.")

    add("data",
        "Show how to one-hot encode a categorical column.",
        "```python\nimport pandas as pd\n\ndf_encoded = pd.get_dummies(df, columns=['city'], drop_first=True)\n```\n`drop_first=True` avoids the dummy-variable trap in linear models.")

    add("data",
        "Show how to convert a date column to datetime and extract month.",
        "```python\nimport pandas as pd\n\ndf['date'] = pd.to_datetime(df['date'])\ndf['month'] = df['date'].dt.month\n```\nUse `errors='coerce'` if you expect malformed dates -- they become NaT and you can filter them.")

    add("data",
        "Show how to set a column as a DataFrame index for time-series ops.",
        "```python\nimport pandas as pd\n\nts = df.set_index('timestamp').sort_index()\nresampled = ts['value'].resample('1H').mean()\n```\nResampling requires a sorted DatetimeIndex; sort first to avoid silent surprises.")

    add("data",
        "Show how to create a numpy array of zeros with a given shape.",
        "```python\nimport numpy as np\n\na = np.zeros((3, 4), dtype=np.float32)\n```\nBe explicit about dtype; the default `float64` doubles memory for many workloads.")

    add("data",
        "Show how to compute the mean and std of a numpy array.",
        "```python\nimport numpy as np\n\nmean = a.mean(axis=0)\nstd = a.std(axis=0, ddof=1)\n```\n`ddof=1` gives the unbiased estimate (sample std); the default 0 is the population formula.")

    add("data",
        "Show how to perform vectorized arithmetic on numpy arrays.",
        "```python\nimport numpy as np\n\nz = (x ** 2 + y ** 2) ** 0.5\n```\nVectorized ops are dramatically faster than Python loops; avoid `for` over numpy arrays.")

    add("data",
        "Show how to plot a histogram with matplotlib.",
        "```python\nimport matplotlib.pyplot as plt\n\nplt.hist(values, bins=30, edgecolor='black')\nplt.xlabel('value'); plt.ylabel('count'); plt.title('Distribution')\nplt.tight_layout(); plt.savefig('hist.png', dpi=150)\n```\nSet `dpi=150` for readable PNGs; the default 100 looks blurry on retina screens.")

    add("data",
        "Show how to plot a line chart of two series in matplotlib.",
        "```python\nimport matplotlib.pyplot as plt\n\nfig, ax = plt.subplots()\nax.plot(x, y_actual, label='actual')\nax.plot(x, y_pred, label='predicted', linestyle='--')\nax.legend(); ax.grid(True, alpha=0.3)\n```\nUse the explicit `fig, ax = plt.subplots()` form; pyplot global state is brittle.")

    add("data",
        "Show how to compute a confusion matrix with sklearn.",
        "```python\nfrom sklearn.metrics import confusion_matrix\n\ncm = confusion_matrix(y_true, y_pred, labels=[0, 1])\n```\nPassing `labels` explicitly stops the row/column order from depending on the data.")

    add("data",
        "Show how to scale features with StandardScaler.",
        "```python\nfrom sklearn.preprocessing import StandardScaler\n\nscaler = StandardScaler()\nX_train_s = scaler.fit_transform(X_train)\nX_test_s = scaler.transform(X_test)\n```\nFit on train only; calling `fit` on test leaks information from test back into preprocessing.")

    add("data",
        "Show how to split a dataset into train and test.",
        "```python\nfrom sklearn.model_selection import train_test_split\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)\n```\n`stratify=y` keeps class proportions; essential for imbalanced labels.")

    add("data",
        "Show how to fit a logistic regression and report accuracy.",
        "```python\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import accuracy_score\n\nmodel = LogisticRegression(max_iter=1000)\nmodel.fit(X_train, y_train)\nprint(accuracy_score(y_test, model.predict(X_test)))\n```\nDefault `max_iter=100` often fails to converge on real data; bump it.")

    add("data",
        "Show how to compute correlation between two pandas columns.",
        "```python\nimport pandas as pd\n\nprint(df['x'].corr(df['y']))\n# or for the whole frame:\nprint(df.corr(numeric_only=True))\n```\nPearson by default; pass `method='spearman'` for rank-based correlation that's robust to outliers.")

    add("data",
        "Show how to write a DataFrame to Parquet partitioned by a column.",
        "```python\nimport pandas as pd\n\ndf.to_parquet('output/', partition_cols=['year', 'month'])\n```\nPartitioning makes downstream readers prune to relevant directories -- huge speedups on big data.")

    add("data",
        "Show how to plot a scatter plot with point sizes.",
        "```python\nimport matplotlib.pyplot as plt\n\nfig, ax = plt.subplots()\nax.scatter(df['x'], df['y'], s=df['size'] * 5, alpha=0.6)\nax.set_xlabel('x'); ax.set_ylabel('y')\n```\n`alpha < 1` reveals overplotting density; `s` is in points^2.")

    add("data",
        "Show how to read JSON lines into a DataFrame.",
        "```python\nimport pandas as pd\n\ndf = pd.read_json('events.jsonl', lines=True)\n```\nFor very large files, iterate with `chunksize=N` and concat downstream-summarized results.")

    add("data",
        "Show how to compute a cross-tabulation in pandas.",
        "```python\nimport pandas as pd\n\nprint(pd.crosstab(df['gender'], df['churned'], normalize='index'))\n```\n`normalize='index'` shows row proportions; useful for diagnosing class imbalance per segment.")

    add("data",
        "Show how to drop duplicate rows in a DataFrame.",
        "```python\nimport pandas as pd\n\ndf_unique = df.drop_duplicates(subset=['user_id', 'date'], keep='last')\n```\nBe specific about subset; otherwise full-row dedup may not match intent.")

    add("data",
        "Show how to use `iloc` and `loc` correctly.",
        "```python\nimport pandas as pd\n\ndf.iloc[0]                 # first row by position\ndf.loc['2024-05-01']       # row by label\ndf.loc[df['x'] > 0, 'y']   # boolean mask + column selection\n```\n`iloc` is integer-position-based, `loc` is label-based -- mixing them causes off-by-one bugs.")

    add("data",
        "Show how to apply a function to every element of a Series.",
        "```python\nimport pandas as pd\n\ndf['name_upper'] = df['name'].str.upper()  # vectorized for strings\ndf['squared'] = df['x'] * df['x']           # vectorized for numerics\n```\nReach for `.apply` only when no vectorized op exists; pure-Python apply is slow.")

    add("data",
        "Show how to compute a histogram with numpy.",
        "```python\nimport numpy as np\n\ncounts, edges = np.histogram(values, bins=30)\n```\nReturns a tuple; for visualization pass to `plt.bar` or `plt.stairs`.")

    add("data",
        "Show how to load images into a numpy array with PIL.",
        "```python\nfrom PIL import Image\nimport numpy as np\n\narr = np.asarray(Image.open('photo.jpg').convert('RGB'))\n```\n`np.asarray` shares memory with the PIL buffer; convert with `np.array` if you need a writable copy.")

    add("data",
        "Show how to vectorize an if/else with numpy.",
        "```python\nimport numpy as np\n\nresult = np.where(x > 0, np.sqrt(x), 0)\n```\n`np.where(cond, a, b)` is the elementwise ternary; use `np.select` for more than two branches.")

    add("data",
        "Show how to compute pandas value counts and proportions.",
        "```python\nimport pandas as pd\n\nprint(df['status'].value_counts(normalize=True))\n```\n`normalize=True` returns proportions instead of raw counts.")

    add("data",
        "Show how to evaluate a model with cross-validation.",
        "```python\nfrom sklearn.model_selection import cross_val_score\nfrom sklearn.ensemble import RandomForestClassifier\n\nmodel = RandomForestClassifier(n_estimators=200, random_state=42)\nscores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')\nprint(scores.mean(), scores.std())\n```\nReport mean and std together; a single point estimate hides variance.")

    add("data",
        "Show how to load a NumPy array from disk.",
        "```python\nimport numpy as np\n\nnp.save('embeddings.npy', arr)\nloaded = np.load('embeddings.npy')\n```\nFor multiple arrays, use `np.savez_compressed`; for huge arrays use `np.memmap`.")

    add("data",
        "Show how to use seaborn to make a styled scatter plot.",
        "```python\nimport seaborn as sns\nimport matplotlib.pyplot as plt\n\nsns.set_theme(style='whitegrid')\nax = sns.scatterplot(data=df, x='x', y='y', hue='group', size='count')\nplt.tight_layout(); plt.savefig('scatter.png', dpi=150)\n```\nSeaborn handles legends, color palettes, and themes that matplotlib leaves to you.")

    add("data",
        "Show how to load a SQL query into a DataFrame.",
        "```python\nimport pandas as pd\nfrom sqlalchemy import create_engine\n\nengine = create_engine('postgresql+psycopg://user:pass@host/db')\ndf = pd.read_sql('SELECT * FROM events WHERE date >= :d', engine, params={'d': '2025-01-01'})\n```\nUse parameter binding rather than string formatting -- safer and cacheable on the DB side.")

    add("data",
        "Show how to compute a moving median in pandas.",
        "```python\nimport pandas as pd\n\ndf['x_med7'] = df['x'].rolling(7, min_periods=3).median()\n```\nMedian rolling is robust to outliers compared to rolling mean.")

    add("data",
        "Show how to assemble a numpy array from a list of arrays.",
        "```python\nimport numpy as np\n\nstacked = np.stack(arrays, axis=0)\nconcatenated = np.concatenate(arrays, axis=0)\n```\n`stack` adds a new axis; `concatenate` joins along an existing one.")

    add("data",
        "Show how to compute Euclidean distance with numpy.",
        "```python\nimport numpy as np\n\ndist = np.linalg.norm(a - b)\n```\nSpecify `axis` for batched distances: `np.linalg.norm(A - B, axis=1)`.")

    add("data",
        "Show how to detect outliers with the IQR method.",
        "```python\nimport pandas as pd\n\nq1, q3 = df['x'].quantile([0.25, 0.75])\niqr = q3 - q1\nmask = (df['x'] < q1 - 1.5 * iqr) | (df['x'] > q3 + 1.5 * iqr)\noutliers = df.loc[mask]\n```\n1.5 * IQR is the Tukey rule; for very heavy-tailed data this flags too aggressively.")

    add("data",
        "Show how to plot subplots with shared axes.",
        "```python\nimport matplotlib.pyplot as plt\n\nfig, axes = plt.subplots(1, 2, sharey=True, figsize=(8, 4))\naxes[0].plot(x1, y1)\naxes[1].plot(x2, y2)\nplt.tight_layout()\n```\nShared axes make side-by-side comparisons honest.")

    add("data",
        "Show how to encode timestamps as Unix epoch seconds.",
        "```python\nimport pandas as pd\n\ndf['ts_epoch'] = pd.to_datetime(df['ts']).astype('int64') // 10**9\n```\nPandas timestamps are nanoseconds; divide by 1e9 for seconds.")

    add("data",
        "Show how to fit a linear regression and inspect coefficients.",
        "```python\nfrom sklearn.linear_model import LinearRegression\n\nmodel = LinearRegression().fit(X, y)\nprint(dict(zip(feature_names, model.coef_)))\nprint(model.intercept_)\n```\nFor inference (CIs, p-values), `statsmodels.OLS` is more appropriate than sklearn.")

    add("data",
        "Show how to plot a heatmap of a correlation matrix.",
        "```python\nimport seaborn as sns\nimport matplotlib.pyplot as plt\n\nsns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm', center=0)\nplt.tight_layout(); plt.savefig('corr.png', dpi=150)\n```\nCenter the colormap at zero so positive and negative correlations get opposite hues.")

    add("data",
        "Show how to load a HuggingFace dataset.",
        "```python\nfrom datasets import load_dataset\n\nds = load_dataset('imdb', split='train')\nprint(ds.features)\nprint(ds[0])\n```\n`ds[0]` shows row 0; `ds.features` shows the schema. Datasets are Arrow-backed and memory-mapped.")

    add("data",
        "Show how to filter and map a HuggingFace dataset.",
        "```python\nfrom datasets import load_dataset\n\nds = load_dataset('imdb', split='train')\nds = ds.filter(lambda r: len(r['text']) < 5000)\nds = ds.map(lambda r: {'text': r['text'].lower()}, num_proc=4)\n```\n`num_proc>1` parallelizes the map; `batched=True` is even faster for vectorizable transforms.")

    add("data",
        "Show how to compute a percentile of a Series.",
        "```python\nimport pandas as pd\n\np95 = df['latency_ms'].quantile(0.95)\n```\nLatency distributions are usually log-normal -- always look at percentiles, not means.")

    add("data",
        "Show how to write a DataFrame to a CSV without the index.",
        "```python\nimport pandas as pd\n\ndf.to_csv('out.csv', index=False)\n```\nLeaving the index in is a common cause of mysterious 'Unnamed: 0' columns when re-reading.")

    add("data",
        "Show how to plot grouped bars in matplotlib.",
        "```python\nimport numpy as np\nimport matplotlib.pyplot as plt\n\nx = np.arange(len(labels))\nwidth = 0.35\nfig, ax = plt.subplots()\nax.bar(x - width/2, group_a, width, label='A')\nax.bar(x + width/2, group_b, width, label='B')\nax.set_xticks(x); ax.set_xticklabels(labels)\nax.legend()\n```\nOffset the x positions by half the width to put the bars side by side.")

    add("data",
        "Show how to do a left anti-join in pandas.",
        "```python\nimport pandas as pd\n\nm = df_a.merge(df_b, on='id', how='left', indicator=True)\nleft_only = m[m['_merge'] == 'left_only'].drop(columns='_merge')\n```\n`indicator=True` adds a column showing which side each row came from.")

    add("data",
        "Show how to use a numpy boolean mask to update values.",
        "```python\nimport numpy as np\n\narr[arr < 0] = 0\n```\nMasked assignment is fast and clearer than `np.where`+reassign for in-place updates.")

    add("data",
        "Show how to construct a 2D numpy array from a function.",
        "```python\nimport numpy as np\n\ngrid = np.fromfunction(lambda i, j: i + j, shape=(4, 4), dtype=int)\n```\n`np.fromfunction` calls the function with broadcastable index arrays once, not in a Python loop.")

    add("data",
        "Show how to plot a boxplot from a DataFrame.",
        "```python\nimport seaborn as sns\nimport matplotlib.pyplot as plt\n\nsns.boxplot(data=df, x='category', y='value')\nplt.xticks(rotation=30)\nplt.tight_layout()\n```\nRotate x labels when categories are long; otherwise they overlap.")

    add("data",
        "Show how to fit a decision tree and visualize it.",
        "```python\nfrom sklearn.tree import DecisionTreeClassifier, export_text\n\nclf = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)\nprint(export_text(clf, feature_names=feature_names))\n```\n`export_text` is enough for a quick sanity check; `plot_tree` produces a graph for presentations.")

    add("data",
        "Show how to compute precision, recall, and F1.",
        "```python\nfrom sklearn.metrics import precision_recall_fscore_support\n\np, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')\nprint({'precision': p, 'recall': r, 'f1': f1})\n```\nFor multiclass use `average='macro'` (treat classes equally) or `'weighted'` (weight by support).")

    add("data",
        "Show how to plot a ROC curve.",
        "```python\nfrom sklearn.metrics import roc_curve, auc\nimport matplotlib.pyplot as plt\n\nfpr, tpr, _ = roc_curve(y_true, y_score)\nplt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.3f}')\nplt.plot([0,1],[0,1], '--', color='gray')\nplt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend()\n```\nUse `roc_auc_score` if you only need the area; `roc_curve` gives you the points to plot.")

    add("data",
        "Show how to use `np.einsum` for a matrix multiplication.",
        "```python\nimport numpy as np\n\nC = np.einsum('ij,jk->ik', A, B)\n```\n`einsum` is great for non-trivial contractions; for plain matmul `A @ B` is faster.")

    add("data",
        "Show how to compute pairwise cosine similarity.",
        "```python\nfrom sklearn.metrics.pairwise import cosine_similarity\n\nsim = cosine_similarity(embeddings)\n```\nFor large N, batch the computation -- the full N\u00b2 matrix grows fast.")

    add("data",
        "Show how to use `pandas.cut` with custom labels.",
        "```python\nimport pandas as pd\n\nlabels = ['low', 'medium', 'high']\ndf['bucket'] = pd.cut(df['value'], bins=[0, 10, 100, 1000], labels=labels)\n```\nThe number of labels must equal `len(bins) - 1`.")

    add("data",
        "Show how to convert a pandas DataFrame to a list of dicts.",
        "```python\nimport pandas as pd\n\nrecords = df.to_dict(orient='records')\n```\n`'records'` is the most useful orient for serialization (e.g., to JSON).")

    add("data",
        "Show how to detect missing values per column.",
        "```python\nimport pandas as pd\n\nprint(df.isna().sum().sort_values(ascending=False))\n```\nSorting surfaces problem columns first.")

    add("data",
        "Show how to handle a Polars DataFrame for a CSV.",
        "```python\nimport polars as pl\n\ndf = pl.read_csv('big.csv')\nresult = df.group_by('category').agg(pl.col('amount').sum()).sort('amount', descending=True)\n```\nPolars is multi-threaded and lazy by default; for big CSVs it's much faster than pandas.")

    add("data",
        "Show how to compute the elbow method for KMeans.",
        "```python\nfrom sklearn.cluster import KMeans\n\ninertias = []\nfor k in range(1, 11):\n    km = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(X)\n    inertias.append(km.inertia_)\nplt.plot(range(1, 11), inertias, 'o-')\nplt.xlabel('k'); plt.ylabel('inertia')\n```\nThe 'elbow' is subjective; silhouette score is more rigorous.")

    add("data",
        "Show how to compute a confusion matrix display in sklearn.",
        "```python\nfrom sklearn.metrics import ConfusionMatrixDisplay\n\nConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize='true')\n```\n`normalize='true'` shows row-wise proportions, which highlights class-conditional error rates.")

    add("data",
        "Show how to sample a fraction of a DataFrame.",
        "```python\nimport pandas as pd\n\nsubset = df.sample(frac=0.1, random_state=42)\n```\nSet `random_state` so the sample is reproducible.")

    add("data",
        "Show how to apply a function across DataFrame rows.",
        "```python\nimport pandas as pd\n\ndf['full_name'] = df.apply(lambda r: f\"{r['first']} {r['last']}\", axis=1)\n```\nFor pure string ops, `df['first'] + ' ' + df['last']` is faster -- vectorized vs per-row.")

    add("data",
        "Show how to compute permutation feature importance.",
        "```python\nfrom sklearn.inspection import permutation_importance\n\nresult = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)\nfor name, mean, std in sorted(zip(feature_names, result.importances_mean, result.importances_std),\n                              key=lambda x: -x[1]):\n    print(f'{name}: {mean:.3f} \u00b1 {std:.3f}')\n```\nMore reliable than tree feature importances, which are biased toward high-cardinality columns.")

    add("data",
        "Show how to compute a Spearman correlation matrix.",
        "```python\nimport pandas as pd\n\nprint(df.corr(method='spearman', numeric_only=True))\n```\nSpearman is robust to outliers and monotonic-but-nonlinear relationships.")

    add("data",
        "Show how to use a Pipeline with a scaler and classifier.",
        "```python\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\n\npipe = Pipeline([('scale', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))])\npipe.fit(X_train, y_train)\n```\nPipelines prevent leakage by ensuring `fit_transform` happens only on training data.")

    add("data",
        "Show how to grid-search hyperparameters.",
        "```python\nfrom sklearn.model_selection import GridSearchCV\nfrom sklearn.ensemble import RandomForestClassifier\n\nparam_grid = {'n_estimators': [100, 300], 'max_depth': [None, 5, 10]}\ngs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)\ngs.fit(X, y)\nprint(gs.best_params_, gs.best_score_)\n```\nFor large grids prefer `RandomizedSearchCV` -- random samples often beat exhaustive search per CPU-hour.")

    add("data",
        "Show how to plot a stacked area chart.",
        "```python\nimport matplotlib.pyplot as plt\n\nfig, ax = plt.subplots()\nax.stackplot(x, y1, y2, y3, labels=['A','B','C'])\nax.legend(loc='upper left')\n```\nStacked areas obscure individual trends -- use a regular line chart if relative position matters.")

    add("data",
        "Show how to read an Excel file with multiple sheets.",
        "```python\nimport pandas as pd\n\nsheets = pd.read_excel('book.xlsx', sheet_name=None)\nfor name, df in sheets.items():\n    print(name, df.shape)\n```\n`sheet_name=None` returns a dict of name -> DataFrame.")

    add("data",
        "Show how to use `numpy.random.default_rng` for reproducibility.",
        "```python\nimport numpy as np\n\nrng = np.random.default_rng(seed=42)\nsample = rng.normal(0, 1, size=1000)\n```\nThe new Generator API is preferred over `np.random.seed`; it's stateful only within the `rng` object.")

    add("data",
        "Show how to plot a CDF in matplotlib.",
        "```python\nimport numpy as np\nimport matplotlib.pyplot as plt\n\nsorted_x = np.sort(values)\ncdf = np.arange(1, len(sorted_x) + 1) / len(sorted_x)\nplt.plot(sorted_x, cdf)\nplt.xlabel('value'); plt.ylabel('CDF')\n```\nCDFs make tail behavior obvious; histograms can mislead with bin choice.")

    add("data",
        "Show how to convert a list of records into a DataFrame and infer dtypes.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame.from_records(records)\nprint(df.dtypes)\n```\n`from_records` accepts a list of tuples or dicts; works without specifying column names if records are dicts.")

    add("data",
        "Show how to use `np.clip` to bound values.",
        "```python\nimport numpy as np\n\nclamped = np.clip(values, 0, 100)\n```\nGreat for sanitizing predictions to a valid range without conditionals.")

    add("data",
        "Show how to compute a TF-IDF representation.",
        "```python\nfrom sklearn.feature_extraction.text import TfidfVectorizer\n\nvec = TfidfVectorizer(min_df=5, ngram_range=(1, 2))\nX = vec.fit_transform(corpus)\n```\n`min_df` drops very rare terms; bigrams help for short documents.")

    add("data",
        "Show how to plot a violin plot.",
        "```python\nimport seaborn as sns\nimport matplotlib.pyplot as plt\n\nsns.violinplot(data=df, x='group', y='score')\nplt.tight_layout()\n```\nViolin = boxplot + KDE; richer than box but harder to read for many groups.")

    add("data",
        "Show how to compute a Pearson correlation with statsmodels for p-values.",
        "```python\nfrom scipy.stats import pearsonr\n\nr, pvalue = pearsonr(df['x'], df['y'])\nprint(f'r={r:.3f} p={pvalue:.4g}')\n```\np-values from huge samples are usually tiny -- judge effect size, not just significance.")

    add("data",
        "Show how to one-hot encode with sklearn.",
        "```python\nfrom sklearn.preprocessing import OneHotEncoder\n\nenc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)\nenc.fit(X_train_cat)\nX_train_oh = enc.transform(X_train_cat)\n```\n`handle_unknown='ignore'` so unseen categories at inference produce a row of zeros instead of an error.")

    add("data",
        "Show how to compute a hash of a numpy array for caching.",
        "```python\nimport hashlib\nimport numpy as np\n\ndef array_hash(a: np.ndarray) -> str:\n    return hashlib.sha1(a.tobytes()).hexdigest()\n```\nUseful for cache keys; handles dtype + shape implicitly through the byte representation.")

    add("data",
        "Show how to plot multiple lines with a colormap.",
        "```python\nimport matplotlib.pyplot as plt\nimport matplotlib.cm as cm\n\nfig, ax = plt.subplots()\ncolors = cm.viridis([i / (n - 1) for i in range(n)])\nfor i, y in enumerate(series):\n    ax.plot(x, y, color=colors[i])\n```\nPerceptually-uniform colormaps like `viridis` are safer than `jet`.")

    add("data",
        "Show how to expand a list-valued column into rows.",
        "```python\nimport pandas as pd\n\ndf_long = df.explode('tags').reset_index(drop=True)\n```\n`explode` turns each list element into its own row; one-line replacement for a manual loop.")

    add("data",
        "Show how to read CSVs in chunks for memory efficiency.",
        "```python\nimport pandas as pd\n\ntotals = pd.Series(dtype=float)\nfor chunk in pd.read_csv('huge.csv', chunksize=100_000):\n    totals = totals.add(chunk.groupby('cat')['amt'].sum(), fill_value=0)\n```\nAccumulate per-chunk aggregates instead of loading the whole file.")

    add("data",
        "Show how to compute a numpy moving average via convolution.",
        "```python\nimport numpy as np\n\ndef moving_avg(x: np.ndarray, w: int) -> np.ndarray:\n    return np.convolve(x, np.ones(w) / w, mode='valid')\n```\n`mode='valid'` returns the central window-aligned result; `'same'` pads to keep length equal.")

    add("data",
        "Show how to compute mean absolute error.",
        "```python\nfrom sklearn.metrics import mean_absolute_error\n\nprint(mean_absolute_error(y_true, y_pred))\n```\nMAE is interpretable in the units of `y` and is robust to outliers compared to MSE.")

    add("data",
        "Show how to standardize column names.",
        "```python\nimport re\nimport pandas as pd\n\ndf.columns = [re.sub(r'[^a-z0-9]+', '_', c.strip().lower()).strip('_') for c in df.columns]\n```\nLowercase + underscore-separated columns avoid quoting hassles in SQL exports later.")

    add("data",
        "Show how to convert a pandas Series of strings to categorical for memory.",
        "```python\nimport pandas as pd\n\ndf['country'] = df['country'].astype('category')\n```\nCategorical dtype dramatically reduces memory for low-cardinality string columns.")

    add("data",
        "Show how to plot a histogram with seaborn and a KDE overlay.",
        "```python\nimport seaborn as sns\nimport matplotlib.pyplot as plt\n\nsns.histplot(values, bins=30, kde=True)\nplt.tight_layout()\n```\nThe KDE is a smoothed estimate; bin count still matters for the histogram.")

    add("data",
        "Show how to use `np.unique` to count distinct values.",
        "```python\nimport numpy as np\n\nvalues, counts = np.unique(arr, return_counts=True)\n```\nOne-pass alternative to `Counter` for numpy arrays; preserves dtype.")

    add("data",
        "Show how to compute the rank of values in a Series.",
        "```python\nimport pandas as pd\n\ndf['rank'] = df['score'].rank(method='dense', ascending=False)\n```\n`'dense'` skips no rank values on ties; `'min'`/`'max'`/`'average'` choose different tie-breaking behavior.")

    add("data",
        "Show how to use `apply` on a grouped DataFrame.",
        "```python\nimport pandas as pd\n\ndef top_n(g, n=3):\n    return g.nlargest(n, 'score')\n\nresult = df.groupby('category', group_keys=False).apply(top_n, n=3)\n```\n`group_keys=False` avoids the duplicated index level pandas adds otherwise.")

    add("data",
        "Show how to compute a cumulative sum within groups.",
        "```python\nimport pandas as pd\n\ndf['cum_sales'] = df.sort_values('date').groupby('product')['sales'].cumsum()\n```\nSort first; cumsum within unsorted groups produces nonsense.")

    add("data",
        "Show how to plot a bar chart with error bars.",
        "```python\nimport matplotlib.pyplot as plt\n\nfig, ax = plt.subplots()\nax.bar(labels, means, yerr=stds, capsize=5)\nax.set_ylabel('value')\n```\nError bars communicate uncertainty; bare bars suggest false precision.")

    add("data",
        "Show how to convert a DataFrame to a sparse matrix.",
        "```python\nimport scipy.sparse as sp\nimport pandas as pd\n\nsparse_arr = sp.csr_matrix(df.values)\n```\nFor mostly-zero data (e.g., one-hot or TF-IDF) the memory savings are huge.")

    add("data",
        "Show how to time a code block with `%timeit` in a script.",
        "```python\nimport timeit\n\nelapsed = timeit.timeit('sorted(xs)', globals={'xs': list(range(1000))}, number=1000)\nprint(f'{elapsed/1000*1e6:.1f} us per call')\n```\nThe Jupyter `%timeit` magic delegates to `timeit.timeit` under the hood.")

    add("data",
        "Show how to use `pd.read_csv` to parse dates with multiple formats.",
        "```python\nimport pandas as pd\n\ndf = pd.read_csv('events.csv', parse_dates=['ts'])\ndf['ts'] = pd.to_datetime(df['ts'], errors='coerce')\n```\n`errors='coerce'` turns malformed values into `NaT`; you can then filter or impute.")

    add("data",
        "Show how to create a DataFrame from a dict of Series.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({\n    'a': pd.Series([1, 2, 3]),\n    'b': pd.Series([4, 5, 6]),\n})\n```\nIndex-based alignment kicks in if the Series have indexes; mismatched indexes produce NaN.")

    add("data",
        "Show how to compute a Z-score per group.",
        "```python\nimport pandas as pd\n\ndf['z'] = df.groupby('group')['value'].transform(lambda s: (s - s.mean()) / s.std(ddof=0))\n```\n`transform` returns a Series aligned with the original index -- ideal for assigning back.")

    add("data",
        "Show how to plot a 2D density via hexbin.",
        "```python\nimport matplotlib.pyplot as plt\n\nfig, ax = plt.subplots()\nax.hexbin(x, y, gridsize=40, mincnt=1, cmap='viridis')\n```\nHexbin scales to millions of points where scatter starts to overplot.")

    add("data",
        "Show how to read a Feather file with pandas.",
        "```python\nimport pandas as pd\n\ndf = pd.read_feather('events.feather')\n```\nFeather is a fast, language-agnostic format good for short-lived intermediates.")

    add("data",
        "Show how to assemble train/val/test splits with stratification.",
        "```python\nfrom sklearn.model_selection import train_test_split\n\nX_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)\nX_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)\n```\nTwo splits give you 70/15/15; pass the same random_state for repeatability.")

    add("data",
        "Show how to sort a DataFrame by multiple columns.",
        "```python\nimport pandas as pd\n\ndf_sorted = df.sort_values(by=['date', 'priority'], ascending=[True, False])\n```\nLengths of `by` and `ascending` must match; passing a single bool applies to all.")

    add("data",
        "Show how to compute MAE and MAPE side by side.",
        "```python\nimport numpy as np\n\ndef mae(y, yhat): return np.mean(np.abs(y - yhat))\ndef mape(y, yhat): return np.mean(np.abs((y - yhat) / np.where(y == 0, 1, y))) * 100\n```\nMAPE breaks when truth is zero; the `np.where` guard avoids divide-by-zero warnings.")

    add("data",
        "Show how to compute a feature's mutual information with target.",
        "```python\nfrom sklearn.feature_selection import mutual_info_classif\n\nmi = mutual_info_classif(X, y, discrete_features='auto', random_state=42)\nfor name, score in sorted(zip(feature_names, mi), key=lambda x: -x[1]):\n    print(f'{name}: {score:.4f}')\n```\nMI catches non-linear relationships that correlation misses.")

    add("data",
        "Show how to inspect memory usage of a DataFrame.",
        "```python\nimport pandas as pd\n\ndf.info(memory_usage='deep')\n```\n`'deep'` accounts for object-dtype string sizes; the default counts pointer size only.")

    add("data",
        "Show how to compute confusion matrix metrics for multiclass.",
        "```python\nfrom sklearn.metrics import classification_report\n\nprint(classification_report(y_true, y_pred, target_names=classes, digits=3))\n```\nClassification report gives per-class precision/recall/F1 plus macro and weighted averages.")

    add("data",
        "Show how to cache an expensive computation to disk.",
        "```python\nfrom joblib import Memory\n\nmemory = Memory('.cache', verbose=0)\n\n@memory.cache\ndef expensive(arg: str):\n    return slow_compute(arg)\n```\njoblib's Memory handles arrays well; functools.lru_cache is process-only.")

    add("data",
        "Show how to convert datetimes to a specified timezone.",
        "```python\nimport pandas as pd\n\ndf['ts'] = pd.to_datetime(df['ts'], utc=True).dt.tz_convert('America/Los_Angeles')\n```\nAlways store UTC and convert at display time -- mixing local times causes silent bugs.")

    add("data",
        "Show how to plot a calibration curve for a classifier.",
        "```python\nfrom sklearn.calibration import calibration_curve\nimport matplotlib.pyplot as plt\n\nfrac, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)\nplt.plot(mean_pred, frac, marker='o')\nplt.plot([0,1],[0,1],'--',color='gray')\nplt.xlabel('mean predicted'); plt.ylabel('observed fraction')\n```\nIf the curve isn't on the diagonal, your probabilities aren't well calibrated -- use Platt or isotonic.")
