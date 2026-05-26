<p align="left">
  <img src="https://raw.githubusercontent.com/datasciencedynamics/datasciencedynamics.github.io/refs/heads/main/data_science_dynamics_logo.svg" alt="Data Science Dynamics" width="200"/>
</p>
 
# Predicting Conflict Event Fatalities in Ukraine Using Machine Learning for Humanitarian Risk Assessment

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Environment Setup](#environment-setup)
4. [MLflow Tracking](#mlflow-tracking)
5. [Core Module](#core-module)
6. [Data Pipeline](#data-pipeline)
7. [Model Training](#model-training)
8. [Model Evaluation](#model-evaluation)
9. [Bootstrap Evaluation](#bootstrap-evaluation)
10. [Build Predictions](#build-predictions)
11. [Model Selection and Explainability](#model-selection-and-explainability)
12. [Production Inference](#production-inference)
13. [Notebooks](#notebooks)
14. [Design Philosophy](#design-philosophy)
15. [Status](#status)

---

## Overview

This project implements an end-to-end machine learning pipeline to model and rank conflict events in Ukraine by predicted fatalities using ACLED event-level data.

Rather than framing the task as a binary classification problem (fatal vs. non-fatal), fatalities are modeled as a continuous outcome. Model performance is evaluated not only with standard regression metrics, but also by how effectively models capture cumulative fatalities when events are ranked by predicted severity.

This framing is designed for real-world use cases such as humanitarian prioritization, conflict monitoring, and early-warning systems.

---

## Project Structure

```
acled_ukraine/
├── core/
│   ├── __init__.py
│   ├── config.py           # Project-wide configuration and paths
│   ├── constants.py        # Shared constants (feature lists, column names, etc.)
│   └── functions.py        # Shared utility functions
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   │   └── inference/
│
├── images/
│   ├── html_images/
│   ├── png_images/
│   └── svg_images/
│
├── modeling/
│   ├── train.py
│   ├── evaluation.py
│   ├── bootstrap_evaluation.py
│   ├── build_predictions.py
│   ├── explainer.py
│   ├── explanations_training.py
│   ├── explanations_inference.py
│   └── predict.py
│
├── models/
│   ├── results/
│   └── eval/
│
├── notebooks/
│   ├── catboost_clean_text.ipynb
│   ├── data_exploration.ipynb
│   ├── embeddings_eda.ipynb
│   ├── event_date_statistics.ipynb
|   ├── map_visualizations.ipynb
│   ├── partial_dependence.ipynb
│   ├── performance_assessment.ipynb 
│   └── theoretical_auc.ipynb
│
├── preprocessing/
│   ├── preprocessing.py
│   └── feat_gen.py
│
├── mlruns/
├── Makefile
├── README.md
├── requirements.txt
└── setup.py
```

---

## Environment Setup

### Conda (recommended)

```bash
conda create -n acled_conda python=3.12.7
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

MLflow is used throughout the pipeline to log metrics, plots, CSV artifacts, and model parameters.

To start the MLflow UI:

```bash
make mlflow_ui
```

Then open: http://localhost:5501

---

## Core Module

The `core/` directory contains shared infrastructure used across preprocessing, training, and evaluation.

- **`config.py`**: Project-wide configuration including data paths, model output directories, and pipeline settings.
- **`constants.py`**: Shared constants such as feature column names, categorical variable lists, and target variable definitions.
- **`functions.py`**: General-purpose utility functions reused across multiple pipeline scripts.

---

## Data Pipeline

### 1. Raw Data Ingestion

```bash
make data_gen
```

Converts raw ACLED CSV files into Parquet format for efficient downstream processing.

---

### 2. Temporal Splits

```bash
make temporal_splits
```

Partitions the dataset into train, validation, and test splits using chronological boundaries to prevent data leakage.

---

### 3. Preprocessing (Training)

```bash
make data_prep_preprocessing_training
```

Key steps include filtering invalid or zero-information records, normalizing actor identifiers, handling missingness flags, and preparing a clean modeling dataset.

---

### 4. Feature Engineering

```bash
make feat_gen_training
```

Generated features include actor embeddings (Actor 1 and Actor 2), interaction flags, categorical encodings, and structural metadata. Embedding difference features are intentionally excluded by default to reduce feature duplication and overfitting risk.

---

### Full Preprocessing Pipeline

```bash
make preproc_pipeline
```

Runs data ingestion, temporal splits, preprocessing, and feature engineering in sequence.

---

## Model Training

The target variable is defined as:

```
fatalities = log(1 + observed fatalities)
```

This log transformation stabilizes variance while preserving severity ordering. Predictions are back-transformed via `expm1` at evaluation and inference time.

### Models Trained

| Model | Type | Pipelines |
|---|---|---|
| Linear Regression | Linear | `orig`, `orig_rfe` |
| Lasso Regression | Regularized linear (L1) | `orig`, `orig_rfe` |
| Ridge Regression | Regularized linear (L2) | `orig`, `orig_rfe` |
| ElasticNet | Regularized linear (L1+L2) | `orig`, `orig_rfe` |
| XGBoost | Gradient-boosted trees | `orig`, `orig_rfe` |
| CatBoost | Gradient-boosted trees | `orig`, `orig_rfe` |

Each model is trained under two pipeline variants: `orig` (full feature set) and `orig_rfe` (recursive feature elimination).

### Train All Models

```bash
make train_all_models
```

Individual model targets are also available:

```bash
make train_lr
make train_lasso
make train_ridge
make train_enet
make train_xgb
make train_cat
```

---

## Model Evaluation

Evaluation is performed on train, validation, and test splits. All metrics and artifacts are logged to MLflow.

### Regression Metrics

For each split: R², Explained Variance, Mean Absolute Error (MAE), Median Absolute Error, Mean Squared Error (MSE), and RMSE.

---

### Evaluation Plots

#### 1. Actual vs. Predicted (log scale)

A scatter plot comparing observed vs. predicted log-scale fatalities, used to assess calibration and bias.

#### 2. Cumulative Fatalities Capture Curve

The primary evaluation visualization. It answers the question: *If events are ranked by predicted severity, how quickly are total fatalities captured?*

- X-axis: fraction of events ranked (by descending predicted fatality)
- Y-axis: fraction of total fatalities captured
- A random baseline is shown for comparison

The Area Under the Cumulative Capture Curve (AUC-CC) is reported as a summary statistic. This is not ROC AUC; it is a ranking-quality measure tailored to severity capture.

---

### Capture Tables (CSV Artifacts)

For each split, a corresponding table is produced with `event_fraction`, `cumulative_fatalities`, and `cumulative_fraction`. These tables are directly aligned with the plotted curves, logged to MLflow as CSV artifacts, and summarized in the terminal at key cutoffs (e.g., top 1%, 5%, 10%).

---

### Run Evaluation

```bash
make eval_all_models
```

Individual model evaluation targets:

```bash
make eval_lr
make eval_lasso
make eval_ridge
make eval_enet
make eval_xgb
make eval_cat
```

### Full Training and Evaluation Pipeline

```bash
make modeling_train_eval_pipeline
```

---

## Bootstrap Evaluation

Bootstrap evaluation produces confidence intervals for validation and test set metrics across all models.

```bash
make bootstrap_evaluation
```

Runs 5,000 bootstrap resamples on each split. Results are written to `models/bootstrap/<outcome>/bootstrap_results_<split>.csv`. This produces uncertainty estimates for R², MAE, RMSE, and AUC-CC, and is the basis for the bootstrap performance table reported in the paper.

---

## Build Predictions

Generates split-level prediction files used for downstream explanation and analysis.

```bash
make build_predictions
```

Output files are written to `data/processed/` and named by outcome and pipeline type.

---

## Model Selection and Explainability

### Best Model Selection

```bash
make model_explainer
```

Selects the best-performing model based on validation R² across all trained models and pipelines.

---

### SHAP Explanations (Training)

```bash
make model_explanations_training
```

Produces SHAP values, feature importance rankings, and explanation tables for the selected model evaluated on the test set. A residual-based tolerance threshold (default: δ = 5 fatalities) is used to construct approximate confusion matrices from regression residuals.

---

### Combined Explaining Pipeline

```bash
make model_explaining_training
```

Runs model selection followed by SHAP explanation generation in sequence.

---

## Production Inference

### Preprocess New Data

```bash
make data_prep_preprocessing_inference
make feat_gen_inference
```

Or run the full inference preprocessing pipeline:

```bash
make preproc_pipeline_inference
```

### Generate Predictions

```bash
make predict
```

Outputs ranked event-level fatality predictions to `data/processed/inference/predictions_<outcome>.csv`, suitable for downstream analysis or operational deployment.

### SHAP Explanations (Inference)

```bash
make model_explanations_inference
```

Explains predictions on new, unseen data using the best model identified at training time.

---

## Notebooks

Exploratory and analytical notebooks are located in `notebooks/`. These are not part of the automated pipeline but support development, diagnosis, and manuscript preparation.

| Notebook | Purpose |
|---|---|
| `data_exploration.ipynb` | Initial EDA on the raw ACLED dataset |
| `embeddings_eda.ipynb` | Analysis of actor embedding features |
| `event_date_statistics.ipynb` | Temporal distribution of conflict events |
| `catboost_clean_text.ipynb` | CatBoost experiments with text features |
| `partial_dependence.ipynb` | Partial dependence plots for feature interpretation |
| `performance_assessment.ipynb` | Cross-model performance comparison |
| `theoretical_auc.ipynb` | Theoretical analysis of the AUC-CC metric |
| `visualizations.ipynb` | Manuscript and presentation figure generation |

---

## Design Philosophy

- **Regression over classification**: severity matters more than binary outcomes.
- **Ranking quality over point accuracy**: capturing the majority of fatalities early is operationally meaningful.
- **Transparent evaluation**: every plot has a corresponding table; every metric has a bootstrap confidence interval.
- **Reproducibility first**: Makefile-driven, MLflow-tracked, environment-agnostic.

---

## Status

The pipeline is fully operational, reproducible end-to-end, and suitable for research, policy analysis, and operational risk assessment.