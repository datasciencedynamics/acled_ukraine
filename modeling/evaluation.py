from pathlib import Path
import typer
from loguru import logger
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

# ------------------------------------------------------------------
# Imports from your codebase (unchanged)
# ------------------------------------------------------------------
from core.functions import (
    plot_actual_vs_predicted,
    adjusted_r2,
    plot_cumulative_fatalities_captured,
    print_capture_summary,
    create_shap_plots,
    mlflow_load_model,
    log_mlflow_metrics,
    mlflow_log_parameters_model,
    mlflow_dumpArtifact,
)

from core.constants import (
    target_log_outcome,
    mlflow_models_data,
)

from core.config import (
    PROCESSED_DATA_DIR,
    model_definitions,
    categorical_cols,
)

app = typer.Typer()

# ==================================================================
# Main CLI entry
# ==================================================================


@app.command()
def main(
    model_type: str = "xgb",
    pipeline_type: str = "orig",
    outcome: str = target_log_outcome,
    outcome_name: str = None,  # Optional: defaults to outcome
    data_path: Path = PROCESSED_DATA_DIR,
):
    """
    Evaluate trained model on train/valid/test splits.

    Args:
        model_type: Model type (lr, lasso, xgb, cat)
        pipeline_type: Pipeline type (orig, orig_rfe)
        outcome: Outcome variable (fatalities or log_fatalities)
        outcome_name: Optional override for outcome column name
        data_path: Path to processed data directory
    """

    # --------------------------------------------------------------
    # STEP 1: Resolve model metadata
    # --------------------------------------------------------------
    estimator_name = model_definitions[model_type]["estimator_name"]

    # Default outcome_name to outcome if not provided
    if outcome_name is None:
        outcome_name = outcome

    run_name = f"{estimator_name}_{pipeline_type}_training"
    experiment_name = f"{outcome}_model"

    print("\n" + "=" * 80)
    print(f"EVALUATION: {run_name}")
    print(f"Outcome: {outcome}")
    print("=" * 80)

    # --------------------------------------------------------------
    # STEP 2: Load trained model from MLflow
    # --------------------------------------------------------------
    print("\nLoading model from MLflow...")
    model = mlflow_load_model(
        experiment_name=experiment_name,
        run_name=run_name,
        model_name=f"{estimator_name}_{outcome}",
    )
    print("Model loaded successfully")

    # --------------------------------------------------------------
    # STEP 3: Load temporal splits
    # --------------------------------------------------------------
    print("\nLoading temporal splits...")

    # Load X splits
    X_train = pd.read_parquet(data_path / "X_train.parquet")
    X_valid = pd.read_parquet(data_path / "X_valid.parquet")
    X_test = pd.read_parquet(data_path / "X_test.parquet")

    if model_type == "xgb" and pipeline_type == "orig":
        for col in categorical_cols:
            combined_cats = pd.Categorical(
                pd.concat([X_train[col], X_valid[col], X_test[col]])
            ).categories
            X_train[col] = pd.Categorical(X_train[col], categories=combined_cats)
            X_valid[col] = pd.Categorical(X_valid[col], categories=combined_cats)
            X_test[col] = pd.Categorical(X_test[col], categories=combined_cats)

    # Load y splits based on outcome
    y_train = pd.read_parquet(data_path / f"y_train_{outcome}.parquet").iloc[:, 0]
    y_valid = pd.read_parquet(data_path / f"y_valid_{outcome}.parquet").iloc[:, 0]
    y_test = pd.read_parquet(data_path / f"y_test_{outcome}.parquet").iloc[:, 0]

    print(f"Loaded splits:")
    print(f"  Train: {X_train.shape[0]:,} samples")
    print(f"  Valid: {X_valid.shape[0]:,} samples")
    print(f"  Test:  {X_test.shape[0]:,} samples")

    # --------------------------------------------------------------
    # STEP 4: Store splits in dictionary
    # --------------------------------------------------------------
    splits = {
        "train": (X_train, y_train),
        "valid": (X_valid, y_valid),
        "test": (X_test, y_test),
    }

    # --------------------------------------------------------------
    # STEP 5: Metrics by split
    # --------------------------------------------------------------
    split_metrics = {}
    log_r2_results = {}
    n_features = X_train.shape[1]

    for split, (X_s, y_s) in splits.items():
        print("\n" + "*" * 80)
        print(f"METRICS ({split.upper()})")
        print("*" * 80)

        metrics = model.return_metrics(
            X=X_s,
            y=y_s,
            model_metrics=True,
            return_dict=True,
        )

        numeric_metrics = {
            k: v
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }

        # Compute log-scale R² and Adjusted R²
        y_pred_s = model.predict(X_s)
        r2_log = r2_score(y_s, y_pred_s)
        adj_r2_log = adjusted_r2(r2_log, n=len(y_s), p=n_features)

        numeric_metrics["r2"] = r2_log
        numeric_metrics["adj_r2"] = adj_r2_log
        log_r2_results[split] = r2_log

        print(f"  R²:      {r2_log:.4f}")
        print(f"  Adj R²:  {adj_r2_log:.4f}")

        for k, v in numeric_metrics.items():
            split_metrics[f"{split}_{k}"] = v

    # Log metrics once
    print("\n" + "=" * 80)
    print("Logging metrics to MLflow...")
    print("=" * 80)
    log_mlflow_metrics(
        experiment_name=experiment_name,
        run_name=run_name,
        metrics=pd.Series(split_metrics),
    )
    print("Metrics logged")

    # --------------------------------------------------------------
    # STEP 6: Calculate and Display Log-Scale R-Squared Summary
    # --------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE (Log Scale - Training Objective)")
    print("=" * 80)

    log_r2_results = {}

    for split, (X_s, y_s) in splits.items():
        y_pred_s = model.predict(X_s)
        r2_log = r2_score(y_s, y_pred_s)
        log_r2_results[split] = r2_log
        print(f"{split.capitalize()} R-Squared (log): {r2_log:.4f}")

    print("=" * 80)

    # --------------------------------------------------------------
    # STEP 7: Plots and capture tables by split
    # --------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Generating evaluation plots...")
    print("=" * 80)

    all_figs = {}
    capture_tables = {}

    for split, (X_s, y_s) in splits.items():

        # Generate predictions
        y_pred_s = pd.Series(
            model.predict(X_s),
            index=y_s.index,
        )

        # ---- Actual vs Predicted (show log-scale metrics)
        fig_avp = plot_actual_vs_predicted(
            y_true=y_s,
            y_pred=y_pred_s,
            title=f"Actual vs Predicted Fatalities ({split})",
            log_scale=False,
            show_log_metrics=True,  # Show log-scale R-Squared (consistent)
        )
        all_figs[f"actual_vs_predicted_{split}"] = fig_avp

        # ---- Cumulative fatalities captured
        fig_cap, capture_df = plot_cumulative_fatalities_captured(
            y_true_log=y_s,
            y_pred_log=y_pred_s,
            model_name=estimator_name,
            return_table=True,
        )

        all_figs[f"cumulative_fatalities_captured_{split}"] = fig_cap
        capture_tables[split] = capture_df

        # ---- Terminal summary
        print_capture_summary(capture_df, split.upper())

    # --------------------------------------------------------------
    # STEP 8: Log PNG plots
    # --------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Logging PNG plots to MLflow...")
    print("=" * 80)

    png_figs = {f"{name}.png": fig for name, fig in all_figs.items()}

    log_mlflow_metrics(
        experiment_name=experiment_name,
        run_name=run_name,
        images=png_figs,
    )
    print(f"Logged {len(png_figs)} PNG plots")

    # --------------------------------------------------------------
    # STEP 8b: Log SVG plots
    # --------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Logging SVG plots to MLflow...")
    print("=" * 80)

    for name, fig in all_figs.items():
        mlflow_dumpArtifact(
            experiment_name=experiment_name,
            run_name=run_name,
            obj_name=name,
            obj=fig,
            artifacts_data_path=mlflow_models_data,
            artifact_format="svg",
        )
    print(f"Logged {len(all_figs)} SVG plots")

    # --------------------------------------------------------------
    # STEP 9: Log capture tables as CSV artifacts
    # --------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Logging capture tables to MLflow...")
    print("=" * 80)
    for split, capture_df in capture_tables.items():
        mlflow_dumpArtifact(
            experiment_name=experiment_name,
            run_name=run_name,
            obj_name=f"cumulative_fatalities_capture_{split}",
            obj=capture_df.round(4),
            artifacts_data_path=mlflow_models_data,
            artifact_format="csv",
        )
    print("Capture tables logged")

    # --------------------------------------------------------------
    # STEP 10: Log model parameters
    # --------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Logging model parameters to MLflow...")
    print("=" * 80)
    mlflow_log_parameters_model(
        experiment_name=experiment_name,
        run_name=run_name,
        model_name=f"{estimator_name}_{outcome}",
        model=model,
    )
    print("Model parameters logged")

    # --------------------------------------------------------------
    # STEP 11: SHAP Analysis
    # --------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Running SHAP analysis...")
    print("=" * 80)

    # Create output directory for SHAP plots
    shap_output_dir = Path("./models/eval") / outcome / estimator_name
    shap_output_dir.mkdir(parents=True, exist_ok=True)

    _, shap_importance, shap_figs = create_shap_plots(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        output_dir=shap_output_dir,
        max_display=20,
        sample_size=100,
    )

    # Log SHAP PNG plots
    print("\nLogging SHAP PNG plots to MLflow...")
    shap_png_dict = {f"{name}.png": fig for name, fig in shap_figs.items()}

    log_mlflow_metrics(
        experiment_name=experiment_name,
        run_name=run_name,
        images=shap_png_dict,
    )
    print(f"Logged {len(shap_png_dict)} SHAP PNG plots")

    # Log SHAP SVG plots
    print("Logging SHAP SVG plots to MLflow...")
    for name, fig in shap_figs.items():
        mlflow_dumpArtifact(
            experiment_name=experiment_name,
            run_name=run_name,
            obj_name=name,
            obj=fig,
            artifacts_data_path=mlflow_models_data,
            artifact_format="svg",
        )
    print(f"Logged {len(shap_figs)} SHAP SVG plots")

    # Log SHAP importance CSV
    mlflow_dumpArtifact(
        experiment_name=experiment_name,
        run_name=run_name,
        obj_name="shap_feature_importance",
        obj=shap_importance,
        artifacts_data_path=mlflow_models_data,
        artifact_format="csv",
    )

    print("SHAP analysis complete - plots and data logged to MLflow")

    logger.success("Model evaluation complete.")


# ==================================================================
if __name__ == "__main__":
    app()
