################################################################################
# STEP 1: Import Required Libraries and Modules
################################################################################

from pathlib import Path
import typer
from loguru import logger
import mlflow
import pandas as pd
import numpy as np
import os

from core.constants import mlflow_models_data, target_log_outcome
from core.config import PROCESSED_DATA_DIR_INFER, PROCESSED_DATA_DIR, categorical_cols
from core.functions import mlflow_load_model

app = typer.Typer()

print("\n" + "#" * 80)
print(f"Running script: {os.path.basename(__file__)}")
print("#" * 80 + "\n")

################################################################################
# STEP 2: Define Function to Find the Best Model from MLflow Experiment
################################################################################


def find_best_model(
    experiment_name: str,
    metric_name: str,
    mode: str = "min",
    mlruns_location: str = None,
) -> str:
    """
    Finds the best model from a given MLflow experiment based on a specified
    metric.

    :param experiment_name: The name of the MLflow experiment to search in.
    :param metric_name: The metric used to determine the best model.
    :param mode: Specify "max" to select model based on maximum metric value
                 or "min" for minimum. Default is "min" (appropriate for
                 error metrics like RMSE).
    :return: Tuple of (run_name, estimator_name) for the best model.
    :raises ValueError: If the experiment does not exist.
    """
    if mlruns_location is None:
        abs_mlflow_data = os.path.abspath(mlflow_models_data)

    mlflow.set_tracking_uri(f"file://{abs_mlflow_data}")

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' does not exist.")

    experiment_id = experiment.experiment_id

    order_clause = (
        f"metrics.`{metric_name}` DESC"
        if mode == "max"
        else f"metrics.`{metric_name}` ASC"
    )

    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        order_by=[order_clause],
    )
    if runs.empty:
        raise ValueError(f"No runs found for experiment '{experiment_name}'")

    best_run = runs.iloc[0]
    best_run_id = runs.iloc[0]["run_id"]
    best_metric_value = runs.iloc[0][f"metrics.{metric_name}"]
    print(f"Best Run ID: {best_run_id}, Best {metric_name}: {best_metric_value}")

    run_name = best_run["tags.mlflow.runName"]
    estimator_name = run_name.split("_")[0]
    return run_name, estimator_name


@app.command()
def main(
    input_data_file: Path = PROCESSED_DATA_DIR_INFER / "X.parquet",
    predictions_path: Path = "predictions.csv",
    outcome: str = target_log_outcome,
    metric_name: str = "test_R2",
    mode: str = "max",
    data_path: Path = PROCESSED_DATA_DIR_INFER,
    training_data_path: Path = PROCESSED_DATA_DIR,
):
    """
    Generate fatality predictions using the best trained regression model.

    Args:
        input_data_file: Path to the input feature parquet file.
        predictions_path: Path to save the output predictions CSV.
        outcome: Target variable name (default: log_fatalities).
        metric_name: MLflow metric to select the best model.
        mode: "max" for metrics where higher is better (e.g., R²),
              "min" for metrics where lower is better (e.g., RMSE).
        data_path: Path to inference data directory.
        training_data_path: Path to training data directory (for categorical
              alignment).
    """

    ############################################################################
    # STEP 3: Load Input Data for Prediction
    ############################################################################

    print("Loading input data...")
    X = pd.read_parquet(input_data_file)
    print(f"Input data shape: {X.shape}")

    ############################################################################
    # STEP 3b: Align Categorical Dtypes
    ############################################################################
    # XGBoost native categorical support requires consistent category codes
    # across splits. Load the training and validation splits to derive
    # combined_cats, then apply to the inference data.
    ############################################################################

    X_train = pd.read_parquet(training_data_path / "X_train.parquet")
    X_valid = pd.read_parquet(training_data_path / "X_valid.parquet")

    for col in categorical_cols:
        combined_cats = pd.Categorical(
            pd.concat([X_train[col], X_valid[col], X[col]])
        ).categories
        X[col] = pd.Categorical(X[col], categories=combined_cats)

    ############################################################################
    # STEP 4: Perform Inference Using the Best Model
    ############################################################################

    logger.info("Loading best model for inference...")

    experiment_name = f"{outcome}_model"

    run_name, estimator_name = find_best_model(
        experiment_name,
        metric_name,
        mode,
    )

    model_name = f"{estimator_name}_{outcome}"

    best_model = mlflow_load_model(
        experiment_name,
        run_name,
        model_name,
    )

    print(f"The best model is {best_model.name}.")

    ############################################################################
    # STEP 5: Generate Predictions
    ############################################################################
    # Generate log-scale predictions and back-transform to raw fatality
    # counts using expm1 (inverse of log1p). Predictions are clipped at
    # zero since negative fatality counts are not meaningful.
    ############################################################################

    y_pred_log = best_model.predict(X)

    X["predicted_fatalities"] = (
        np.clip(np.expm1(y_pred_log), a_min=0, a_max=None).round().astype(int)
    )

    # Load actual target if available
    y_log_path = data_path / f"y_{outcome}.parquet"
    if y_log_path.exists():
        y_log = pd.read_parquet(y_log_path).iloc[:, 0]
        X["actual_fatalities"] = (
            np.clip(np.expm1(y_log), a_min=0, a_max=None).round().astype(int)
        )

    logger.success("Inference complete.")

    print(f"\nPrediction summary (raw fatality counts):")
    print(f"  Mean:   {X['predicted_fatalities'].mean():.2f}")
    print(f"  Median: {X['predicted_fatalities'].median():.2f}")
    print(f"  Max:    {X['predicted_fatalities'].max():.2f}")
    print(f"  Zeros:  {(X['predicted_fatalities'] == 0).sum():,}")

    ############################################################################
    # STEP 6: Save Predictions to File
    ############################################################################

    output_cols = ["predicted_fatalities"]
    if "actual_fatalities" in X.columns:
        output_cols.insert(0, "actual_fatalities")
    X[output_cols].to_csv(predictions_path)
    print(f"\nPredictions saved to: {predictions_path}")


if __name__ == "__main__":
    app()
