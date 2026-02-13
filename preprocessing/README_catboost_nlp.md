# CatBoost Text Pipeline

## Overview

CatBoost supports native text features and does **not** require TF-IDF or manual vectorization.

This pipeline prepares cleaned text data for CatBoost models and stores it as a reusable artifact keyed by the primary key.

The output file:

```
data/catboost_text.parquet
```

Contains:

| Column                  | Description                                |
|--------------------------|--------------------------------------------|
| `event_id_cnty`          | Primary key                                |
| `catboost_notes_clean`   | Cleaned text for CatBoost consumption      |

---

## Why CatBoost Is Different

Unlike sklearn-based models, CatBoost:

- Handles raw text directly
- Performs internal tokenization
- Generates text features internally (BM25, embeddings, etc.)
- Does not require TF-IDF preprocessing

Because of this, we only need to provide:

- Cleaned text
- Proper primary key alignment

---

## Step 1 — Build the CatBoost Text Artifact

Generate the cleaned text file:

```bash
python src/preprocessing/build_catboost_text.py
```

Output:

```
data/catboost_text.parquet
```

This script:

- Reads raw data
- Removes leading date phrases
- Cleans URLs, emails, and unwanted characters
- Normalizes punctuation
- Strips leading commas
- Outputs a deduplicated file keyed by `event_id_cnty`

---

## Step 2 — Merge With Tabular Features

Merge text features into your modeling dataset.

```python
import pandas as pd

PK = "event_id_cnty"

text_df = pd.read_parquet("data/catboost_text.parquet")
feat_df = pd.read_parquet("data/features.parquet")

df = feat_df.merge(
    text_df[[PK, "catboost_notes_clean"]],
    on=PK,
    how="left",
    validate="one_to_one"
)

df["catboost_notes_clean"] = df["catboost_notes_clean"].fillna("")
```

At this point:

- Each row corresponds to one modeling instance
- Primary key alignment is guaranteed
- Text and tabular features share identical row order

---

## Step 3 — Train CatBoost With Text Feature

Example classification model:

```python
from catboost import CatBoostClassifier

text_features = ["catboost_notes_clean"]

model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    verbose=100
)

model.fit(
    df.drop(columns=["target_col"]),
    df["target_col"],
    text_features=text_features
)
```

Important:

- Pass the column name(s) in `text_features`
- Do not manually vectorize
- Do not apply TF-IDF

---

## Train/Test Split Example

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=["target_col"]),
    df["target_col"],
    test_size=0.2,
    random_state=42,
    stratify=df["target_col"]
)

model.fit(
    X_train,
    y_train,
    text_features=text_features
)
```

---

## Key Design Principles

- Text preprocessing is minimal and deterministic.
- Vectorization is handled internally by CatBoost.
- Merging is performed before modeling to ensure row alignment.
- The artifact is keyed by `event_id_cnty` for safe joins.
- Text cleaning removes noise but preserves semantic meaning.

---

## Artifact Structure

```
data/
    raw/
        acled_ukraine_data_YYYY_MM_DD.parquet
    catboost_text.parquet
    text_base.parquet
    features.parquet
```

---

## When to Use This Pipeline

Use this pipeline when:

- Training CatBoost models
- Leveraging CatBoost native text handling
- Avoiding manual TF-IDF feature engineering
- Combining structured and text data seamlessly

---

## When NOT to Use This Pipeline

Do not use this pipeline if:

- Training sklearn models (Logistic Regression, SVM, etc.)
- Using XGBoost without manual text vectorization
- Building neural models requiring embeddings

For those cases, use the **README_text_models_nlp** documentation.
