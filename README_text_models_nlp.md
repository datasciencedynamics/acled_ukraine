# Text Vectorization Pipeline (Non-CatBoost Models)

## Overview

This pipeline prepares and integrates cleaned text features for models that **do not support native text handling**, such as:

- Logistic Regression
- Linear SVM
- Naive Bayes
- Random Forest
- XGBoost
- LightGBM
- Neural networks requiring numeric inputs

Unlike CatBoost, these models require text to be converted into numeric features (e.g., TF-IDF).

The reusable artifact:

```
data/text_base.parquet
```

Contains:

| Column            | Description                   |
|-------------------|-------------------------------|
| `event_id_cnty`   | Primary key                   |
| `notes_clean_ml`  | Cleaned text for modeling     |

---

## Why Vectorization Is Required

Most ML models operate on numeric matrices.

Therefore, text must be transformed into structured numeric features using:

- TF-IDF (recommended baseline)
- Hashing Vectorizer
- Word embeddings
- Sentence embeddings

This document focuses on TF-IDF as the default baseline.

---

## Step 1 — Build the Text Artifact

Generate the cleaned text file:

```bash
python src/preprocessing/build_text_base.py
```

Output:

```
data/text_base.parquet
```

This script:

- Reads raw data
- Removes leading date phrases
- Cleans URLs, emails, and unwanted characters
- Normalizes punctuation
- Strips leading commas
- Outputs a deduplicated file keyed by `event_id_cnty`

---

## Step 2 — Merge Text With Tabular Features

Text must be merged **before vectorization** to guarantee row alignment.

```python
import pandas as pd

PK = "event_id_cnty"

text_df = pd.read_parquet("data/text_base.parquet")
feat_df = pd.read_parquet("data/features.parquet")

df = feat_df.merge(
    text_df[[PK, "notes_clean_ml"]],
    on=PK,
    how="left",
    validate="one_to_one"
)

df["notes_clean_ml"] = df["notes_clean_ml"].fillna("")
```

At this stage:

- Each row represents one modeling instance
- Primary key alignment is guaranteed
- Text and tabular features share identical row order

---

## Step 3 — Train/Test Split (Prevent Leakage)

Always split before fitting the vectorizer.

```python
from sklearn.model_selection import train_test_split

X_train_df, X_test_df, y_train, y_test = train_test_split(
    df,
    df["target_col"],
    test_size=0.2,
    random_state=42,
    stratify=df["target_col"]
)
```

---

## Step 4 — TF-IDF Vectorization

Convert text into numeric features.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    min_df=2,
    max_df=0.95,
    ngram_range=(1, 2)
)

X_train_text = tfidf.fit_transform(X_train_df["notes_clean_ml"])
X_test_text  = tfidf.transform(X_test_df["notes_clean_ml"])
```

Important:

- `fit()` must be called only on training data.
- `transform()` is used on test data.
- This prevents information leakage.

---

## Step 5 — Combine With Tabular Features

If tabular features are numeric:

```python
from scipy import sparse

X_train_tab = sparse.csr_matrix(
    X_train_df.drop(columns=["target_col", "notes_clean_ml"]).to_numpy()
)

X_test_tab = sparse.csr_matrix(
    X_test_df.drop(columns=["target_col", "notes_clean_ml"]).to_numpy()
)

from scipy.sparse import hstack

X_train = hstack([X_train_tab, X_train_text]).tocsr()
X_test  = hstack([X_test_tab, X_test_text]).tocsr()
```

Row order remains aligned because merging occurred before splitting.

---

## Recommended Production Pattern (ColumnTransformer)

For cleaner, more maintainable pipelines, use `ColumnTransformer` and `Pipeline`.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

text_col = "notes_clean_ml"
cat_cols = ["categorical_feature"]
num_cols = ["numeric_feature_1", "numeric_feature_2"]

preprocess = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1,2)), text_col),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

model = Pipeline(
    steps=[
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=2000))
    ]
)

model.fit(df, df["target_col"])
```

Benefits:

- Automatic feature alignment
- No manual matrix stacking
- Cleaner experimentation workflow
- Reproducible preprocessing
- Easier hyperparameter tuning

---

## Key Design Principles

- Text artifacts are stored separately from modeling logic.
- Merging happens before vectorization to preserve row alignment.
- Vectorizers are fitted only on training data.
- Sparse matrices are used for memory efficiency.
- Pipelines are preferred for maintainability and production readiness.

---

## Artifact Structure

```
data/
    raw/
        acled_ukraine_data_YYYY_MM_DD.parquet
    text_base.parquet
    catboost_text.parquet
    features.parquet
```

---

## When to Use This Pipeline

Use this pipeline when:

- Training sklearn-based classifiers
- Running XGBoost or LightGBM without native text handling
- Building neural networks that require numeric features
- Benchmarking against CatBoost

---

## When NOT to Use This Pipeline

Do not use this pipeline if:

- Training CatBoost with native text features
- Leveraging models that internally tokenize text

For those cases, use the **README_catboost_nlp** documentation.
