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
    outcome: str = target_outcome,
    features_path: Path = PROCESSED_DATA_DIR / "X.parquet",
    labels_path: Path = PROCESSED_DATA_DIR / f"y_{target_outcome}.parquet",
    outcome_name: str = target_outcome,
):

    # --------------------------------------------------------------
    # STEP 1: Resolve model metadata
    # --------------------------------------------------------------
    estimator_name = model_definitions[model_type]["estimator_name"]

    run_name = f"{estimator_name}_{pipeline_type}_training"
    experiment_name = f"{outcome}_model"

    print(run_name)
    print(f"{estimator_name}_{outcome}")

    # --------------------------------------------------------------
    # STEP 2: Load trained model from MLflow
    # --------------------------------------------------------------
    model = mlflow_load_model(
        experiment_name=experiment_name,
        run_name=run_name,
        model_name=f"{estimator_name}_{outcome}",
    )

    # --------------------------------------------------------------
    # STEP 3: Load data
    # --------------------------------------------------------------
    X = pd.read_parquet(features_path)
    y_all = pd.read_parquet(labels_path)
    y = y_all[outcome_name]

    # --------------------------------------------------------------
    # STEP 4: Retrieve train, valid, test splits from model
    # --------------------------------------------------------------
    X_train, y_train = model.get_train_data(X, y)
    X_valid, y_valid = model.get_valid_data(X, y)
    X_test, y_test   = model.get_test_data(X, y)

    splits = {
        "train": (X_train, y_train),
        "valid": (X_valid, y_valid),
        "test":  (X_test,  y_test),
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
    log_mlflow_metrics(
        experiment_name=experiment_name,
        run_name=run_name,
        metrics=pd.Series(split_metrics),
    )

    # --------------------------------------------------------------
    # STEP 6: Plots and capture tables by split
    # --------------------------------------------------------------
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
    log_mlflow_metrics(
        experiment_name=experiment_name,
        run_name=run_name,
        images=all_figs,
    )

    # --------------------------------------------------------------
    # STEP 8: Log capture tables as CSV artifacts
    # --------------------------------------------------------------
    for split, capture_df in capture_tables.items():
        mlflow_dumpArtifact(
            experiment_name=experiment_name,
            run_name=run_name,
            obj_name=f"cumulative_fatalities_capture_{split}",
            obj=capture_df.round(4),
            artifacts_data_path=mlflow_models_data,
            artifact_format="csv",
        )

    # --------------------------------------------------------------
    # STEP 9: Log model parameters
    # --------------------------------------------------------------
    mlflow_log_parameters_model(
        experiment_name=experiment_name,
        run_name=run_name,
        model_name=f"{estimator_name}_{outcome}",
        model=model,
    )

    logger.success("Model evaluation complete.")

# ==================================================================
if __name__ == "__main__":
    app()
