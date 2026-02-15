from pathlib import Path
import typer
from loguru import logger
import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# Imports from your codebase (unchanged)
# ------------------------------------------------------------------
from core.functions import (
    plot_actual_vs_predicted,
    plot_cumulative_fatalities_captured,
    print_capture_summary,
    mlflow_load_model,
    log_mlflow_metrics,
    mlflow_log_parameters_model,
    mlflow_dumpArtifact,
)

from core.constants import (
    target_outcome,
    target_log_outcome,
    mlflow_models_data,
)

from core.config import (
    PROCESSED_DATA_DIR,
    model_definitions,
)

app = typer.Typer()

# ==================================================================
# Main CLI entry
# ==================================================================


@app.command()
def main(
    model_type: str = "lr",
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
    # STEP 6: Plots and capture tables by split
    # --------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Generating evaluation plots...")
    print("=" * 80)

    all_figs = {}
    capture_tables = {}

    for split, (X_s, y_s) in splits.items():

        y_pred_s = pd.Series(
            model.predict(X_s),
            index=y_s.index,
        )

        # ---- Actual vs Predicted (log scale)
        fig_avp = plot_actual_vs_predicted(
            y_true=y_s,
            y_pred=y_pred_s,
            title=f"Actual vs Predicted Fatalities ({split})",
        )
        all_figs[f"actual_vs_predicted_{split}.png"] = fig_avp

        # ---- Cumulative fatalities captured
        fig_cap, capture_df = plot_cumulative_fatalities_captured(
            y_true_log=y_s,
            y_pred_log=y_pred_s,
            model_name=estimator_name,
            return_table=True,
        )

        all_figs[f"cumulative_fatalities_captured_{split}.png"] = fig_cap
        capture_tables[split] = capture_df

        # ---- Terminal summary
        print_capture_summary(capture_df, split.upper())

    # --------------------------------------------------------------
    # STEP 7: Log plots (figures only)
    # --------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Logging plots to MLflow...")
    print("=" * 80)
    log_mlflow_metrics(
        experiment_name=experiment_name,
        run_name=run_name,
        images=all_figs,
    )
    print("Plots logged")

    # --------------------------------------------------------------
    # STEP 8: Log capture tables as CSV artifacts
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
    # STEP 9: Log model parameters
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

    logger.success("Model evaluation complete.")


# ==================================================================
if __name__ == "__main__":
    app()
