# ACLED Ukraine Fatalities Modeling Pipeline

## Overview

This project implements an end-to-end machine learning pipeline to model and rank conflict events in Ukraine by predicted fatalities using ACLED event-level data.

Rather than framing the task as a binary classification problem (fatal vs non-fatal), fatalities are modeled as a continuous outcome. Model performance is evaluated not only with standard regression metrics, but also by how effectively models capture cumulative fatalities when events are ranked by predicted severity.

This framing is designed for real-world use cases such as humanitarian prioritization, conflict monitoring, and early-warning systems.

---

## Project Structure

```
acled_ukraine/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   │   └── inference/
│
├── preprocessing/
│   ├── preprocessing.py
│   └── feat_gen.py
│
├── modeling/
│   ├── train.py
│   ├── evaluation.py
│   ├── predict.py
│   ├── explainer.py
│   └── explanations_*.py
│
├── models/
│   ├── results/
│   └── eval/
│
├── mlruns/
├── Makefile
├── requirements.txt
└── README.md
```

---

## Environment Setup

### Conda (recommended)

```bash
conda create -n acled_conda python=3.11
conda activate acled_conda
pip install -r requirements.txt
```

### Virtual Environment (alternative)

```bash
make create_venv
source acled_venv/bin/activate
make requirements
```

---

## MLflow Tracking

MLflow is used throughout the pipeline to log:
- metrics,
- plots,
- CSV artifacts,
- and model parameters.

To start the MLflow UI:

```bash
make mlflow_ui
```

Then open: http://localhost:5501

---

## Data Pipeline

### 1. Raw Data Ingestion

```bash
make data_gen
```

Converts raw ACLED CSV files into Parquet format for efficient downstream processing.

---

### 2. Preprocessing (Training)

```bash
make data_prep_preprocessing_training
```

Key steps:
- filtering invalid or zero-information records,
- normalizing actor identifiers,
- handling missingness flags,
- preparing a clean modeling dataset.

---

### 3. Feature Engineering

```bash
make feat_gen_training
```

Generated features include:
- actor embeddings (Actor 1 and Actor 2),
- interaction flags,
- categorical encodings,
- structural metadata.

Embedding difference features are intentionally excluded by default to reduce feature duplication and overfitting risk.

---

### Full Preprocessing Pipeline

```bash
make preproc_pipeline
```

---

## Model Training

The target variable is defined as:

```
fatalities = log(1 + observed fatalities)
```

This transformation stabilizes variance while preserving severity ordering.

### Models Trained

- Linear Regression
- Lasso Regression
- XGBoost Regression
- CatBoost Regression

Each model is trained under two pipelines:
- `orig`
- `orig_rfe` (recursive feature elimination)

### Train All Models

```bash
make train_all_models
```

---

## Model Evaluation

Evaluation is performed on train, validation, and test splits.

### Regression Metrics

For each split:
- R²
- Explained Variance
- Mean Absolute Error (MAE)
- Median Absolute Error
- Mean Squared Error (MSE)
- RMSE

All metrics are logged to MLflow.

---

### Evaluation Plots

For each data split, the following artifacts are generated and logged.

#### 1. Actual vs Predicted (log scale)

A scatter plot comparing observed vs predicted log-scale fatalities, used to assess calibration and bias.

#### 2. Cumulative Fatalities Capture Curve

This is the primary evaluation visualization.

It answers the question:
"If events are ranked by predicted severity, how quickly are total fatalities captured?"

- X-axis: fraction of events ranked
- Y-axis: fraction of total fatalities captured

A random baseline is shown for comparison.

The Area Under the Cumulative Capture Curve (AUC-CC) is reported as a summary statistic. This is not ROC AUC, but a ranking-quality measure tailored to severity capture.

---

### Capture Tables (CSV Artifacts)

For each split, a corresponding table is produced with:
- event_fraction
- cumulative_fatalities
- cumulative_fraction

These tables are:
- directly aligned with the plotted curves,
- logged to MLflow as CSV artifacts,
- summarized in the terminal at key cutoffs (e.g., top 1%, 5%, 10%).

---

### Run Evaluation

```bash
make eval_all_models
```

---

## Model Selection and Explainability

### Best Model Selection

```bash
make model_explainer
```

Selects the best-performing model based on validation metrics.

---

### SHAP Explanations (Training)

```bash
make model_explanations_training
```

Produces SHAP values, feature rankings, and explanation tables for the selected model.

---

### SHAP Explanations (Inference)

```bash
make model_explanations_inference
```

Used to explain predictions on new, unseen data.

---

## Production Inference

### Preprocess New Data

```bash
make preproc_pipeline_inf
```

### Generate Predictions

```bash
make predict
```

Outputs ranked event-level fatality predictions suitable for downstream analysis or deployment.

---

## Design Philosophy

- Regression over classification: severity matters more than binary outcomes.
- Ranking quality over point accuracy: capturing the majority of fatalities early is operationally meaningful.
- Transparent evaluation: every plot has a corresponding table.
- Reproducibility first: Makefile-driven, MLflow-tracked, environment-agnostic.

---

## Status

The pipeline is fully operational, reproducible end-to-end, and suitable for research, policy analysis, and operational risk assessment.
