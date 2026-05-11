# SPDX-License-Identifier: MIT
"""Data science entries: numpy, pandas, matplotlib, sklearn (~100 total)."""


def register_all(add):
    add("data",
        "Create a 5x5 numpy array of zeros and set its diagonal to 1.",
        "```python\nimport numpy as np\n\na = np.zeros((5, 5))\nnp.fill_diagonal(a, 1)\n```\nOr just `np.eye(5)` -- shorter and idiomatic for the identity matrix.")

    add("data",
        "Compute the column means of a 2D numpy array, ignoring NaNs.",
        "```python\nimport numpy as np\n\nmeans = np.nanmean(arr, axis=0)\n```\n`nanmean` skips NaNs without you needing to mask manually.")

    add("data",
        "Read a CSV with pandas and parse a date column.",
        "```python\nimport pandas as pd\n\ndf = pd.read_csv('events.csv', parse_dates=['ts'])\n```\n`parse_dates=` triggers automatic datetime conversion -- much faster than post-hoc `pd.to_datetime`.")

    add("data",
        "Group a DataFrame by 'user' and compute mean of 'value'.",
        "```python\ndf.groupby('user')['value'].mean()\n```\nA tight, idiomatic groupby aggregation. Returns a Series indexed by user.")

    add("data",
        "Filter a DataFrame to rows where value > threshold.",
        "```python\ndf_filtered = df[df['value'] > 100]\n```\nBoolean indexing is the standard pandas idiom for row filtering.")

    add("data",
        "Plot a line chart with matplotlib showing two series.",
        "```python\nimport matplotlib.pyplot as plt\n\nplt.plot(x, y1, label='train')\nplt.plot(x, y2, label='val')\nplt.xlabel('epoch'); plt.ylabel('loss'); plt.legend()\nplt.show()\n```\nAlways add labels and a legend; saves your future self from deciphering unlabeled plots.")

    add("data",
        "Train a logistic regression on a small dataset using sklearn.",
        "```python\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import train_test_split\n\nX_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)\nclf = LogisticRegression(max_iter=1000)\nclf.fit(X_tr, y_tr)\nprint(clf.score(X_te, y_te))\n```\nIncrease `max_iter` past the default 100 to avoid convergence warnings on harder datasets.")

    add("data",
        "Compute the Pearson correlation between two columns of a DataFrame.",
        "```python\nr = df['x'].corr(df['y'])\n```\nDefaults to Pearson; pass `method='spearman'` for rank correlation.")

    add("data",
        "Reshape a numpy array of length 12 into a 3x4 matrix.",
        "```python\nimport numpy as np\n\nm = np.arange(12).reshape(3, 4)\n```\n`reshape` returns a view when possible -- no copy unless the strides demand one.")

    add("data",
        "Convert a pandas Series of strings to lowercase.",
        "```python\ndf['name'] = df['name'].str.lower()\n```\nThe `.str` accessor vectorizes string methods.")

    add("data",
        "Use numpy to generate 1000 samples from a normal distribution.",
        "```python\nimport numpy as np\nrng = np.random.default_rng(seed=42)\nsamples = rng.normal(loc=0, scale=1, size=1000)\n```\nUse `default_rng` rather than `np.random.normal` directly -- it's the modern, thread-safe API.")

    add("data",
        "Drop rows with any NaN from a DataFrame.",
        "```python\ndf_clean = df.dropna()\n```\nUse `subset=['col']` to drop only when specific columns are NaN.")

    add("data",
        "Compute a rolling 7-day mean on a time-indexed DataFrame.",
        "```python\ndf['ma7'] = df['value'].rolling(window=7).mean()\n```\nThe first 6 rows will be NaN; that's expected since the window isn't full yet.")

    add("data",
        "Sort a DataFrame by column 'created' descending.",
        "```python\ndf_sorted = df.sort_values('created', ascending=False)\n```\nReturns a new DataFrame; `inplace=True` modifies the original.")

    add("data",
        "Plot a histogram with 30 bins.",
        "```python\nimport matplotlib.pyplot as plt\n\nplt.hist(values, bins=30)\nplt.xlabel('x'); plt.ylabel('count')\nplt.show()\n```\nBin count is a tradeoff -- too few hides structure, too many shows noise. 30 is a good default.")

    add("data",
        "Standardize features using sklearn's StandardScaler.",
        "```python\nfrom sklearn.preprocessing import StandardScaler\n\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\n```\nFit on training data only; use the same scaler to transform validation/test.")

    add("data",
        "Concatenate two DataFrames vertically.",
        "```python\nimport pandas as pd\n\ndf = pd.concat([df1, df2], ignore_index=True)\n```\n`ignore_index=True` resets the row index; useful when you don't care about source positions.")

    add("data",
        "Count unique values in a Series.",
        "```python\nn = df['category'].nunique()\n```\nFor counts per value, use `value_counts()` instead.")

    add("data",
        "Compute the elementwise product of two numpy arrays.",
        "```python\nimport numpy as np\n\nout = a * b  # NOT a.dot(b) -- that's matrix mul\n```\nThe `*` operator on numpy arrays is elementwise. Use `@` or `dot` for matrix product.")

    add("data",
        "Slice the first 10 rows of a DataFrame.",
        "```python\nfirst10 = df.head(10)\n```\nClearer than `df.iloc[:10]` for the common case.")

    add("data",
        "Plot a scatter of x vs y with point sizes proportional to z.",
        "```python\nimport matplotlib.pyplot as plt\n\nplt.scatter(x, y, s=z * 50, alpha=0.5)\nplt.xlabel('x'); plt.ylabel('y')\n```\n`alpha=0.5` keeps overlapping points visible.")

    add("data",
        "Train a random forest classifier with sklearn.",
        "```python\nfrom sklearn.ensemble import RandomForestClassifier\n\nclf = RandomForestClassifier(n_estimators=200, random_state=0)\nclf.fit(X_train, y_train)\nprint(clf.feature_importances_)\n```\n`feature_importances_` is a useful first sanity check on what the model learned.")

    add("data",
        "Convert a DataFrame to a numpy array.",
        "```python\nX = df.to_numpy()\n```\nFaster and clearer than the legacy `.values`.")

    add("data",
        "Pivot a long DataFrame to wide format.",
        "```python\nwide = df.pivot(index='user', columns='metric', values='value')\n```\nFor aggregation during pivot, use `pivot_table` with `aggfunc=`.")

    add("data",
        "Compute the cumulative sum of a numpy array.",
        "```python\nimport numpy as np\n\ncs = np.cumsum(arr)\n```\nFor 2D arrays specify `axis=` to control direction.")

    add("data",
        "Sort a numpy array along the last axis.",
        "```python\nimport numpy as np\n\nsorted_arr = np.sort(arr, axis=-1)\n```\n`axis=-1` is the default; included for clarity.")

    add("data",
        "Read JSON-lines into a DataFrame.",
        "```python\nimport pandas as pd\n\ndf = pd.read_json('events.jsonl', lines=True)\n```\nNDJSON is a common log format; `lines=True` parses one record per line.")

    add("data",
        "Compute the mean squared error between two arrays.",
        "```python\nfrom sklearn.metrics import mean_squared_error\n\nmse = mean_squared_error(y_true, y_pred)\n```\nUse the sklearn metric so the implementation matches what training pipelines use.")

    add("data",
        "Plot a heatmap of a 2D numpy array.",
        "```python\nimport matplotlib.pyplot as plt\n\nplt.imshow(matrix, aspect='auto', cmap='viridis')\nplt.colorbar()\nplt.show()\n```\n`viridis` is perceptually uniform and color-blind friendly.")

    add("data",
        "Compute the median of a numpy array along axis 0.",
        "```python\nimport numpy as np\n\nmedians = np.median(arr, axis=0)\n```\nFor large arrays consider `np.percentile(arr, 50, axis=0)` -- slightly faster path internally.")

    add("data",
        "Use pandas to fill missing values with the column mean.",
        "```python\ndf = df.fillna(df.mean(numeric_only=True))\n```\nMean imputation is a simple baseline; for production use sklearn's IterativeImputer or domain-aware logic.")

    add("data",
        "Train/test split with stratification on the target.",
        "```python\nfrom sklearn.model_selection import train_test_split\n\nX_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)\n```\nStratification preserves class proportions in both splits.")

    add("data",
        "Compute the inverse of a square numpy matrix.",
        "```python\nimport numpy as np\n\ninv = np.linalg.inv(M)\n```\nFor solving Mx = b, prefer `np.linalg.solve(M, b)` -- numerically more stable than computing the inverse.")

    add("data",
        "Plot multiple subplots in a 2x2 grid.",
        "```python\nimport matplotlib.pyplot as plt\n\nfig, axes = plt.subplots(2, 2, figsize=(10, 8))\nfor ax, data in zip(axes.flat, datasets):\n    ax.plot(data)\nplt.tight_layout()\nplt.show()\n```\n`tight_layout` prevents overlapping axis labels.")

    add("data",
        "Compute the F1 score on binary predictions.",
        "```python\nfrom sklearn.metrics import f1_score\n\nf1 = f1_score(y_true, y_pred)\n```\nFor multi-class pass `average='macro'` or `'weighted'`.")

    add("data",
        "Generate a numpy array of evenly spaced numbers.",
        "```python\nimport numpy as np\n\nx = np.linspace(0, 1, num=50)\n```\nIncludes both endpoints. Use `np.arange(0, 1, step)` if you want fixed step instead of fixed count.")

    add("data",
        "Convert a pandas Timestamp column to UNIX epoch seconds.",
        "```python\ndf['epoch'] = df['ts'].astype('int64') // 10**9\n```\nAssumes the column is tz-aware or already UTC; otherwise normalize first with `dt.tz_localize`.")

    add("data",
        "Compute confusion matrix on classifier output.",
        "```python\nfrom sklearn.metrics import confusion_matrix\n\ncm = confusion_matrix(y_true, y_pred)\n```\nRows are true labels, columns are predicted. Use `labels=` to fix order for multiclass.")

    add("data",
        "Save a numpy array to a .npy file.",
        "```python\nimport numpy as np\n\nnp.save('embeddings.npy', arr)\n# Load with: np.load('embeddings.npy')\n```\nBinary format is much faster than CSV for numeric data.")

    add("data",
        "Use sklearn pipelines for scaling + classifier.",
        "```python\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\n\npipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])\npipe.fit(X_train, y_train)\n```\nPipelines prevent test-set leakage during cross-validation.")

    add("data",
        "Filter a DataFrame to rows matching multiple conditions.",
        "```python\nmask = (df['age'] > 18) & (df['country'] == 'US')\ndf[mask]\n```\nUse `&` and `|` (not `and`/`or`) and parenthesize each comparison.")

    add("data",
        "Compute argmax of each row in a 2D numpy array.",
        "```python\nimport numpy as np\n\nidx = np.argmax(arr, axis=1)\n```\nReturns row-wise index of max. For ties, returns the first occurrence.")

    add("data",
        "Vectorize a Python function with numpy.",
        "```python\nimport numpy as np\n\nf = np.vectorize(lambda x: x*x + 1)\nout = f(np.arange(10))\n```\nNote: `np.vectorize` is convenience, not speed -- it loops in Python. Rewrite using array ops for true vectorization.")

    add("data",
        "Plot a bar chart from a Series.",
        "```python\nimport matplotlib.pyplot as plt\n\nseries.plot(kind='bar')\nplt.xticks(rotation=45)\nplt.tight_layout()\nplt.show()\n```\npandas wraps matplotlib so you can plot directly from a Series or DataFrame.")

    add("data",
        "Compute the dot product of two numpy vectors.",
        "```python\nimport numpy as np\n\nresult = a @ b\n```\nThe `@` operator (PEP 465) is matmul; for 1D arrays it's the dot product.")

    add("data",
        "Encode categorical strings as integers with sklearn.",
        "```python\nfrom sklearn.preprocessing import LabelEncoder\n\nle = LabelEncoder()\ny_encoded = le.fit_transform(y_strings)\n```\nGood for the target. For features, prefer `OneHotEncoder` or pandas' `get_dummies` to avoid imposing a false ordinal relationship.")

    add("data",
        "Read parquet into a DataFrame.",
        "```python\nimport pandas as pd\n\ndf = pd.read_parquet('data.parquet')\n```\nParquet is columnar and compressed -- much faster and smaller than CSV for analytical workloads.")

    add("data",
        "Compute weekly resampling on a time-indexed DataFrame.",
        "```python\nweekly = df.resample('W').mean()\n```\nFrequency strings: 'D' day, 'W' week, 'ME' month-end, 'YE' year-end.")

    add("data",
        "Plot a 3D surface with matplotlib.",
        "```python\nimport numpy as np\nimport matplotlib.pyplot as plt\n\nX, Y = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))\nZ = np.sin(np.sqrt(X**2 + Y**2))\nfig = plt.figure()\nax = fig.add_subplot(projection='3d')\nax.plot_surface(X, Y, Z, cmap='viridis')\nplt.show()\n```\n`projection='3d'` requires the mpl_toolkits.mplot3d module which ships with matplotlib.")

    add("data",
        "Compute a one-hot encoding of categorical column.",
        "```python\nimport pandas as pd\n\nencoded = pd.get_dummies(df['color'], prefix='color')\n```\nFor sklearn-compatible pipelines, prefer `OneHotEncoder` instead.")

    add("data",
        "Use joblib to parallelize a function over a list.",
        "```python\nfrom joblib import Parallel, delayed\n\nresults = Parallel(n_jobs=-1)(delayed(work)(x) for x in items)\n```\n`n_jobs=-1` uses all cores. Joblib is what sklearn uses internally.")

    add("data",
        "Compute the silhouette score for KMeans clustering.",
        "```python\nfrom sklearn.cluster import KMeans\nfrom sklearn.metrics import silhouette_score\n\nkm = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)\nprint(silhouette_score(X, km.labels_))\n```\nSilhouette ranges -1 to 1; higher = better-separated clusters. Beware: it favors convex/equal-size clusters.")

    add("data",
        "Compute element-wise log of a numpy array, handling zeros.",
        "```python\nimport numpy as np\n\nlog_arr = np.log(arr, where=arr > 0, out=np.full_like(arr, -np.inf, dtype=float))\n```\nThe `where=` mask keeps zeros from generating warnings; out array supplies the default.")

    add("data",
        "Get a contingency table (crosstab) of two columns.",
        "```python\nimport pandas as pd\n\nct = pd.crosstab(df['country'], df['plan'])\n```\nGreat for cross-tabulating two categorical variables.")

    add("data",
        "Normalize a feature column to [0, 1].",
        "```python\nfrom sklearn.preprocessing import MinMaxScaler\n\nscaler = MinMaxScaler()\ndf[['x']] = scaler.fit_transform(df[['x']])\n```\nUse double brackets so the result is 2D as scaler expects.")

    add("data",
        "Compute mean and std of a numpy array.",
        "```python\nimport numpy as np\n\nmean = arr.mean(); std = arr.std(ddof=1)\n```\n`ddof=1` for sample std (Bessel's correction); default 0 is the population std.")

    add("data",
        "Use pandas to compute weekday from a datetime column.",
        "```python\ndf['dow'] = df['ts'].dt.day_name()\n```\nReturns 'Monday', 'Tuesday', ... For numeric weekday use `.dt.weekday`.")

    add("data",
        "Apply a function to a DataFrame column without a Python loop.",
        "```python\ndf['cleaned'] = df['raw'].str.strip().str.lower()\n```\nChained `.str` methods are vectorized in pandas. `df.apply` is the escape hatch but slower.")

    add("data",
        "Use cross-validation to estimate model accuracy.",
        "```python\nfrom sklearn.model_selection import cross_val_score\nfrom sklearn.linear_model import LogisticRegression\n\nscores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5)\nprint(scores.mean(), scores.std())\n```\nReport mean and std together so readers can see variability across folds.")

    add("data",
        "Use numpy to find indices where a condition is true.",
        "```python\nimport numpy as np\n\nidx = np.where(arr > 5)[0]\n```\nFor 1D, slice [0] to get the indices array. For nD, the tuple has one array per axis.")

    add("data",
        "Use pandas pivot_table with multiple aggregations.",
        "```python\nimport numpy as np\n\ntbl = df.pivot_table(\n    index='dept', values='salary',\n    aggfunc=['mean', 'median', np.std]\n)\n```\nReturns a DataFrame with a MultiIndex column for the agg names.")

    add("data",
        "Compute pairwise cosine similarity of rows.",
        "```python\nfrom sklearn.metrics.pairwise import cosine_similarity\n\nS = cosine_similarity(X)\n```\nReturns an n x n symmetric matrix. For very large n consider FAISS or sparse representations.")

    add("data",
        "Save a DataFrame to compressed CSV.",
        "```python\ndf.to_csv('out.csv.gz', index=False, compression='gzip')\n```\nPandas detects compression from extension, but being explicit is clearer.")

    add("data",
        "Plot ROC curve for a binary classifier.",
        "```python\nfrom sklearn.metrics import roc_curve, auc\nimport matplotlib.pyplot as plt\n\nfpr, tpr, _ = roc_curve(y_true, y_scores)\nplt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.3f}')\nplt.plot([0,1], [0,1], '--', alpha=0.3)\nplt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend()\nplt.show()\n```\nFor heavily imbalanced data, precision-recall curves are more informative than ROC.")

    add("data",
        "Use numpy to compute the eigenvalues of a square matrix.",
        "```python\nimport numpy as np\n\nvals, vecs = np.linalg.eig(M)\n```\nFor symmetric/Hermitian matrices use `eigh` -- it's faster and returns real eigenvalues.")

    add("data",
        "Group a DataFrame by month from a datetime column.",
        "```python\nmonthly = df.groupby(df['ts'].dt.to_period('M'))['value'].sum()\n```\n`to_period('M')` gives a PeriodIndex; convert to timestamps via `to_timestamp()` if you need a regular date.")

    add("data",
        "Use sklearn's GridSearchCV for hyperparameter tuning.",
        "```python\nfrom sklearn.model_selection import GridSearchCV\nfrom sklearn.svm import SVC\n\ngs = GridSearchCV(SVC(), {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}, cv=5, n_jobs=-1)\ngs.fit(X, y)\nprint(gs.best_params_, gs.best_score_)\n```\nFor large search spaces use `RandomizedSearchCV` -- often nearly as good with far less compute.")

    add("data",
        "Use pandas to read SQL into a DataFrame.",
        "```python\nimport pandas as pd\nfrom sqlalchemy import create_engine\n\nengine = create_engine('postgresql://user:pw@host/db')\ndf = pd.read_sql('SELECT * FROM events WHERE ts > now() - interval \\'1 day\\'', engine)\n```\nPass the engine, not a raw connection -- pandas' SQLAlchemy path supports more dialects cleanly.")

    add("data",
        "Compute the elementwise max of two numpy arrays.",
        "```python\nimport numpy as np\n\nout = np.maximum(a, b)\n```\nBroadcasts shapes when compatible. `np.max` is the reducer (single value) -- different operation.")

    add("data",
        "Use plt.subplots to make sharex aligned plots.",
        "```python\nimport matplotlib.pyplot as plt\n\nfig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))\naxes[0].plot(t, signal)\naxes[1].plot(t, derivative)\nplt.tight_layout(); plt.show()\n```\n`sharex=True` keeps both axes locked when zooming/panning interactively.")

    add("data",
        "Use sklearn KFold for manual cross-validation.",
        "```python\nfrom sklearn.model_selection import KFold\n\nkf = KFold(n_splits=5, shuffle=True, random_state=0)\nfor tr_idx, te_idx in kf.split(X):\n    X_tr, X_te = X[tr_idx], X[te_idx]\n    # train and evaluate per fold\n```\n`shuffle=True` is important when data has ordering structure.")

    add("data",
        "Group and apply a custom function in pandas.",
        "```python\nresult = df.groupby('cat').apply(lambda g: (g['x'] * g['w']).sum() / g['w'].sum())\n```\nWeighted average per group. For one-off, this is fine; for hot paths convert to a vectorized form.")

    add("data",
        "Use numpy where to conditionally fill an array.",
        "```python\nimport numpy as np\n\nout = np.where(mask, a, b)\n```\nElement-wise: pick `a[i]` if `mask[i]`, else `b[i]`. Like a vectorized ternary.")

    add("data",
        "Compute the CDF (sorted percentile) of a numpy array.",
        "```python\nimport numpy as np\n\nsorted_x = np.sort(arr)\ncdf = np.arange(1, len(arr) + 1) / len(arr)\n```\nPlot `sorted_x` vs `cdf` for an empirical CDF.")

    add("data",
        "Use pandas merge to join two DataFrames on a key.",
        "```python\nmerged = df1.merge(df2, on='user_id', how='left')\n```\n`how='left'` keeps all rows from df1; switch to 'inner' or 'outer' as needed.")

    add("data",
        "Plot a confusion matrix as a heatmap.",
        "```python\nfrom sklearn.metrics import ConfusionMatrixDisplay\nimport matplotlib.pyplot as plt\n\nConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap='Blues')\nplt.show()\n```\nThe sklearn helper handles labels and axis layout for you.")

    add("data",
        "Compute the principal components of a matrix using PCA.",
        "```python\nfrom sklearn.decomposition import PCA\n\npca = PCA(n_components=2)\nX_2d = pca.fit_transform(X)\nprint(pca.explained_variance_ratio_)\n```\nThe ratio array tells you how much variance each component captures.")

    add("data",
        "Vectorized cosine similarity between two numpy vectors.",
        "```python\nimport numpy as np\n\ncos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))\n```\nFor batches use `sklearn.metrics.pairwise.cosine_similarity` -- it normalizes for you.")

    add("data",
        "Use pandas to drop duplicates based on a subset of columns.",
        "```python\ndf_unique = df.drop_duplicates(subset=['user_id', 'ts'])\n```\nKeeps the first occurrence by default; use `keep='last'` to keep the last.")

    # ---- 5-msg follow-ups ----
    add("data",
        "Train a sklearn model and serialize it to disk.",
        "```python\nimport joblib\nfrom sklearn.ensemble import GradientBoostingClassifier\n\nclf = GradientBoostingClassifier(random_state=0)\nclf.fit(X_train, y_train)\njoblib.dump(clf, 'model.joblib')\n```\njoblib uses numpy-aware serialization that's faster than pickle for sklearn models.",
        "Is joblib backwards-compatible across versions?",
        "Not reliably across major sklearn versions -- pickled estimators can break when sklearn's internals change. Treat saved models as production artifacts pinned to specific sklearn/numpy versions, and consider exporting to ONNX or PMML if you need portability across environments.")

    add("data",
        "Use pandas to compute a 7-day rolling correlation between two columns.",
        "```python\nrolling_corr = df['x'].rolling(window=7).corr(df['y'])\n```\nFirst 6 values will be NaN since the window isn't full.",
        "What if the data has gaps?",
        "Use `min_periods=` to compute on shorter windows where data exists, e.g. `rolling(window=7, min_periods=3)`. For irregular timestamps, use `rolling('7D')` (time-based offset) so the window is wall-clock time rather than row count -- gaps are then handled correctly.")

    add("data",
        "Train a neural network using sklearn's MLPClassifier.",
        "```python\nfrom sklearn.neural_network import MLPClassifier\n\nclf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=0)\nclf.fit(X_tr, y_tr)\nprint(clf.score(X_te, y_te))\n```\nFine for small problems; for deep learning use PyTorch or JAX.",
        "When should I switch to PyTorch?",
        "When you need GPU acceleration, custom architectures (transformers, CNNs), or training loops with weird loss functions. MLPClassifier is fine for tabular data with up to ~100k rows and standard MLP architecture; beyond that, the ergonomics and performance of PyTorch outweigh the boilerplate.")

    add("data",
        "Compute word frequencies in a text column.",
        "```python\nfrom collections import Counter\n\nwords = df['text'].str.lower().str.split().sum()\nfreq = Counter(words)\nprint(freq.most_common(20))\n```\nFast for moderate corpora; for large text use sklearn's CountVectorizer.",
        "Why CountVectorizer for large text?",
        "It produces a sparse matrix instead of a Counter, so memory scales with non-zero counts rather than total tokens. It also handles tokenization, n-grams, and stopword filtering in one pass and integrates cleanly with sklearn pipelines for downstream classification or topic modeling.")

    add("data",
        "Plot training loss over epochs from a list.",
        "```python\nimport matplotlib.pyplot as plt\n\nplt.plot(losses, marker='.')\nplt.xlabel('epoch'); plt.ylabel('loss'); plt.title('Training Loss')\nplt.grid(alpha=0.3); plt.show()\n```\nGrid lines with low alpha are easy on the eye and helpful for reading values.",
        "How do I add a smoothed version of the curve?",
        "Compute an exponential moving average: `import pandas as pd; smooth = pd.Series(losses).ewm(alpha=0.1).mean()` and plot both the raw and smoothed series. EMA is preferred to a simple moving average because it reacts faster to recent changes -- valuable when you're trying to spot when training plateaus.")

    add("data",
        "Use sklearn to one-hot encode features in a pipeline.",
        "```python\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.linear_model import LogisticRegression\n\npre = ColumnTransformer([\n    ('cat', OneHotEncoder(handle_unknown='ignore'), ['country']),\n    ('num', StandardScaler(), ['age', 'income']),\n])\npipe = Pipeline([('pre', pre), ('clf', LogisticRegression())])\n```\n`handle_unknown='ignore'` means unseen categories at test time become all-zero vectors instead of crashing.",
        "What's a better alternative for high-cardinality categoricals?",
        "Target encoding (encode each category as the mean of the target conditional on that category) shines when categories are many but each has substantial data. Use `category_encoders.TargetEncoder` with cross-validation to avoid leakage, or for tree models try `HistGradientBoostingClassifier` which natively handles categoricals.")

    add("data",
        "Compute a histogram efficiently with numpy.",
        "```python\nimport numpy as np\n\ncounts, edges = np.histogram(arr, bins=50)\n```\nReturns counts and bin edges. For visualization use `plt.hist` directly.",
        "How can I plot it without re-binning?",
        "Use `plt.bar`: `plt.bar(edges[:-1], counts, width=np.diff(edges), align='edge')`. `np.diff(edges)` gives correct widths for irregular bins; `align='edge'` puts each bar at its left edge so it matches the histogram convention.")

    add("data",
        "Use sklearn to train a SVM with grid search.",
        "```python\nfrom sklearn.svm import SVC\nfrom sklearn.model_selection import GridSearchCV\n\nparam_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.1]}\ngs = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1')\ngs.fit(X, y)\nprint(gs.best_params_)\n```\n`scoring='f1'` tunes for F1 instead of accuracy -- often what you want for imbalanced data.",
        "Why is SVC slow on big datasets?",
        "SVC's training is O(n^2) to O(n^3) in samples because it solves a quadratic program. For >50k rows, switch to `LinearSVC` (linear kernel only, much faster) or use kernel approximations via `Nystroem` followed by a linear model. SGD-based classifiers like `SGDClassifier(loss='hinge')` give a similar margin objective and scale to millions of rows.")

    add("data",
        "Resample an irregular time series to 1-minute bars.",
        "```python\nminute = df.set_index('ts')['price'].resample('1min').agg(['first', 'max', 'min', 'last'])\nminute.columns = ['open', 'high', 'low', 'close']\n```\nClassic OHLC bars from tick data. Add `.dropna()` to remove minutes with no trades.",
        "How do I handle gaps in the bar series?",
        "Forward-fill is wrong for high/low (they should be NaN if no trades), but valid for close (last known price). The cleanest pattern: keep NaNs to represent 'no trades', and downstream consumers decide whether to skip or impute. For visualization, gaps usually look better as missing bars than as flat lines.")

    add("data",
        "Plot multiple series with seaborn.",
        "```python\nimport seaborn as sns\nimport matplotlib.pyplot as plt\n\nsns.lineplot(data=df, x='date', y='value', hue='series')\nplt.show()\n```\nSeaborn handles legend, color cycling, and confidence intervals in one line.",
        "When do I prefer pure matplotlib?",
        "When you need full control over every element (custom annotations, weird axes transforms, non-standard layouts), or when you're plotting in a tight loop where seaborn's overhead matters. Use seaborn for exploratory analysis where defaults look great, matplotlib when you're producing publication or production-quality figures with strict requirements.")

    add("data",
        "Compute feature importance using permutation importance.",
        "```python\nfrom sklearn.inspection import permutation_importance\n\nresult = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=0)\nfor i in result.importances_mean.argsort()[::-1]:\n    print(f'{X_test.columns[i]:30s} {result.importances_mean[i]:.3f}')\n```\nPermutation importance is model-agnostic and more reliable than tree feature_importances_, which are biased toward high-cardinality features.",
        "Why is it more reliable?",
        "Tree-based feature importance counts how often a feature splits and how much it reduces impurity. High-cardinality features (continuous, many unique values) get more split opportunities and inflate their importance. Permutation importance shuffles a feature's values and measures the drop in model performance -- it's unbiased and tied directly to the model's predictive power, not to dataset structure.")

    add("data",
        "Use pandas to compute a pivot table with custom aggfunc.",
        "```python\ntbl = df.pivot_table(\n    index='product', columns='month',\n    values='revenue',\n    aggfunc=lambda x: x.quantile(0.95)\n)\n```\nP95 revenue per product per month -- helpful for outlier-aware reporting.",
        "Is there a more efficient way for percentiles?",
        "Yes -- pandas has `np.percentile` and a Cython-fast `quantile` agg. Use `aggfunc=('p95', lambda x: np.quantile(x, 0.95))` or precompute with `groupby(['product', 'month'])['revenue'].quantile(0.95).unstack()`. The lambda path is correct but slower because it can't drop into Cython.")

    add("data",
        "Train an isolation forest for anomaly detection.",
        "```python\nfrom sklearn.ensemble import IsolationForest\n\nclf = IsolationForest(contamination=0.05, random_state=0)\nclf.fit(X)\nscores = -clf.score_samples(X)  # higher = more anomalous\n```\n`contamination` is your prior on the proportion of outliers.",
        "How do I choose contamination?",
        "If you have a labeled validation set, sweep contamination and pick by precision/recall on outlier labels. Without labels, treat scores as continuous and pick a threshold from the score distribution (e.g. top 1% as anomalies). The `contamination='auto'` setting uses scoring without a strict threshold and is often a fine default.")
